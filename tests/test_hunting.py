"""Аффорданс ОХОТА v0.1 (Фрай hunting.md 11.06): Адам→всеядный, ловит мелкую
добычу за мясо. Client-half: diet-флип (→server kill-energy) + DS-hunt-ATTACK
(BS-prey-ATTACK инертна у single-Adam) + meat-дима GC. dopamine уже wired
(apply_kill_prey, Хьюберт). Kill-switch client_flag hunting (дефолт OFF).
"""
from __future__ import annotations
import sys, types
from pathlib import Path
import pytest
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
pytest.importorskip("torch")
from utopia_client.local_compute import LocalColonyCompute  # noqa: E402


def _c():
    return LocalColonyCompute(device="cpu")


# ── diet kill-switch ────────────────────────────────────────────────────

def test_set_hunting_makes_adam_omnivore():
    c = _c()
    c._single_organism = True
    c.organisms["a"] = types.SimpleNamespace()
    assert c.set_hunting(True) is True
    assert c._hunting_enabled is True
    assert c.traits["a"]["diet_gene"] == c._OMNIVORE_DIET   # 0.618 всеядный
    assert "a" in c._skill_changed_cids                     # re-announce P40
    # off → травоядный (текущее поведение)
    c.set_hunting(False)
    assert c.traits["a"]["diet_gene"] == 0.0


def test_hunting_skips_non_single_organism():
    c = _c()
    c._single_organism = False
    c.organisms["a"] = types.SimpleNamespace()
    c.set_hunting(True)
    assert c._hunting_enabled is True            # флаг ставится
    assert "a" not in c.traits or "diet_gene" not in (c.traits.get("a") or {})  # diet НЕ трогаем


# ── meat-дима GC (hunt-outcome, изолированно от plant-income) ────────────

def test_sample_includes_meat_cum():
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(cortisol=60, glucose=80, hydration=90)
    c._beh_meat_cum["a"] = 35.0
    s = c._beh_gc_sample("a")
    assert s["meat_cum"] == 35.0                  # сырьё для meat_rate
    assert "meat" in c._BEH_GC_MDE_TARGET         # power-target есть


# ── DS-hunt-ATTACK (single-Adam BS=0) ───────────────────────────────────

def test_hunt_attack_when_omnivore_prey_reachable():
    import torch
    c = _c()
    logits = torch.zeros(16)
    obs = [0.0] * 64
    obs[56] = 0.8                                 # prey-направление (nav тянет move)
    obs[58] = 0.9                                 # добыча НА КОНТАКТЕ (p_prox)
    # diet>0.5 (всеядный), хищника нет (obs[61]=0)
    c._shape_action_logits(logits, obs, diet=0.618, energy_ratio=1.0)
    assert float(logits[5]) > 0                   # ATTACK по добыче
    # КОНТАКТ-COMMIT: ATTACK ПОБЕЖДАЕТ argmax (не «наезжает» move'ом на добычу)
    assert int(logits.argmax()) == 5


def test_herbivore_no_hunt_attack():
    import torch
    c = _c()
    logits = torch.zeros(16)
    obs = [0.0] * 64
    obs[58] = 0.9                                 # добыча достижима
    c._shape_action_logits(logits, obs, diet=0.0, energy_ratio=1.0)  # травоядный
    # травоядный НЕ охотится: hunt-ATTACK не добавлен (§4-suppress держит ≤0)
    assert float(logits[5]) <= 0


def test_hunt_suppressed_near_predator():
    import torch
    c = _c()
    logits_hunt = torch.zeros(16)
    obs = [0.0] * 64
    obs[58] = 0.9                                 # добыча достижима
    obs[61] = 0.5                                 # НО хищник близко (d_prox>0.3)
    c._shape_action_logits(logits_hunt, obs, diet=0.618, energy_ratio=1.0)
    # сравнение: без хищника hunt-ATTACK сильнее (выживание > охота)
    logits_safe = torch.zeros(16)
    obs2 = [0.0] * 64; obs2[58] = 0.9
    c._shape_action_logits(logits_safe, obs2, diet=0.618, energy_ratio=1.0)
    assert float(logits_hunt[5]) < float(logits_safe[5])   # хищник подавил охоту


# ── hunt-when-starving = life_critical (Фрай, ДО grass-cut) ──────────────

def _hunt_entry(action, energy, prey_prox, hunting=True, single=True):
    import utopia_client.ws_client as wsm
    ws = wsm.ColonyWSClient(server="https://e.com", token="t", colony_name="cheef",
                            client_version="0.0.0", estimated_population=0)
    comp = types.SimpleNamespace()
    comp._single_organism = single
    comp._hunting_enabled = hunting
    comp.biochem = {"a": types.SimpleNamespace(energy=energy)}
    comp.handle_tick = lambda *a, **k: {"a": {"action": action}}
    comp.get_phase_emas = lambda cid: None
    ws.compute = comp
    obs = [0.0] * 64
    obs[58] = prey_prox
    out = ws._run_tick_and_build({"a": obs}, {}, {}, world_tick=0)
    return {e["cid"]: e for e in (out or [])}["a"]


def test_hunt_starving_is_life_critical():
    e = _hunt_entry(action=5, energy=50, prey_prox=0.9)
    assert e.get("life_critical") is True


def test_hunt_fed_not_life_critical():
    e = _hunt_entry(action=5, energy=500, prey_prox=0.9)
    assert "life_critical" not in e


def test_hunt_no_prey_not_life_critical():
    e = _hunt_entry(action=5, energy=50, prey_prox=0.0)
    assert "life_critical" not in e


def test_hunt_starving_off_when_hunting_disabled():
    e = _hunt_entry(action=5, energy=50, prey_prox=0.9, hunting=False)
    assert "life_critical" not in e


# ── hunt-seek: голод-модуляция + baseline-опортунизм (Фрай 12.06) ────────

def _prey_nav(energy_ratio, p_prox, diet=0.618):
    import torch
    c = _c()
    l = torch.zeros(16)
    obs = [0.0] * 64
    obs[56] = 1.0           # prey-направление
    obs[58] = p_prox
    c._shape_action_logits(l, obs, diet=diet, energy_ratio=energy_ratio)
    return float(l[1])      # prey-nav move-компонента


def test_huntseek_hungry_stronger_than_fed():
    # голод-модуляция: голодный активно охотится, сытый форажит траву
    far_fed = _prey_nav(0.8, 0.1)       # сытый, далёкая добыча
    far_hungry = _prey_nav(0.2, 0.1)    # голодный, далёкая добыча
    assert far_hungry > far_fed
    assert far_hungry > 4.0             # ≈4.47·DS (активная охота)
    assert far_fed < 2.76               # < grass-нав (форажит траву)


def test_huntseek_opportunism_close_prey():
    # baseline-опортунизм: близкая добыча берётся даже сытым (контакт-commit добьёт)
    far_fed = _prey_nav(0.8, 0.1)       # сытый, далеко → нет опортунизма
    close_fed = _prey_nav(0.8, 0.5)     # сытый, близко (p_prox≥0.2) → опортунизм
    assert close_fed > far_fed
    assert close_fed > 2.76             # > grass-нав → сворачивает к близкой добыче
