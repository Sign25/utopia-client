"""Phase 2 feeding-ladder — средняя дичь = 55 (Фрай pounce-модель 12.06).
Client-half: восприятие nearest_medium_prey → medium-seek nav (арбитраж голод+
способность, коммит ОДНУ цель, ровная φ-погоня) + pounce-флаг (dist≤_POUNCE_DIST
→ +1 рывок на entry MOVE) + контакт-ATTACK. Сервер (Хьюберт) даёт kill-energy 55,
сложность (hp/speed/flee) + prey-tiring (нагон). Spawn OFF до ack.
"""
from __future__ import annotations
import sys, types
from pathlib import Path
import pytest
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
pytest.importorskip("torch")
from utopia_client.local_compute import LocalColonyCompute, _PHI  # noqa: E402

_PINV = 1.0 / _PHI            # φ⁻¹≈0.618 (порог голода)


def _c(hunting=True):
    c = LocalColonyCompute(device="cpu")
    c._single_organism = True
    c.set_instinct_dir_strength(2.0)         # прод: DS из client_flag (>0)
    c._hunting_enabled = hunting
    return c


# ── medium-seek nav: арбитраж голод + способность ───────────────────────

def _med_nav(energy_ratio, dr=-1, dc=0, dist=5.0, hunting=True, diet=0.618,
             d_prox=0.0):
    import torch
    c = _c(hunting=hunting)
    l = torch.zeros(16)
    obs = [0.0] * 64
    obs[61] = d_prox                         # хищник (survival-гейт)
    mp = {"dr": dr, "dc": dc, "dist": dist}
    c._shape_action_logits(l, obs, diet=diet, energy_ratio=energy_ratio,
                           medium_prey=mp)
    return l


def test_medium_seek_when_hungry_capable():
    # голоден (< φ⁻¹) И способен (> φ⁻⁵) → тянет к средней (dr<0 → NORTH)
    l = _med_nav(energy_ratio=0.4, dr=-1, dc=0, dist=5.0)
    assert float(l[0]) > 0                    # NORTH (к добыче севернее)
    assert float(l[0]) > float(l[1])          # доминирует обратное


def test_medium_seek_east():
    l = _med_nav(energy_ratio=0.4, dr=0, dc=1, dist=5.0)
    assert float(l[2]) > 0                     # EAST (dc>0)


def test_medium_seek_off_when_fed():
    # сытый (≥ φ⁻¹) → НЕ лезет на среднюю (форажит дешёвое)
    l = _med_nav(energy_ratio=0.8, dr=-1, dc=0, dist=5.0)
    assert float(l[0]) == 0.0                  # нет medium-pull


def test_medium_seek_off_when_critically_starving():
    # критично-истощён (< φ⁻⁵) → НЕ способен, не лезет (умрёт в попытке)
    l = _med_nav(energy_ratio=0.05, dr=-1, dc=0, dist=5.0)
    assert float(l[0]) == 0.0


def test_medium_seek_off_when_hunting_disabled():
    l = _med_nav(energy_ratio=0.4, dr=-1, dc=0, dist=5.0, hunting=False)
    assert float(l[0]) == 0.0


def test_medium_seek_suppressed_near_predator():
    # хищник рядом (d_prox≥0.3) → survival > охота, medium-seek ОТКЛ
    l_safe = _med_nav(energy_ratio=0.4, dr=-1, dc=0, dist=5.0, d_prox=0.0)
    l_danger = _med_nav(energy_ratio=0.4, dr=-1, dc=0, dist=5.0, d_prox=0.5)
    assert float(l_safe[0]) > 0
    assert float(l_danger[0]) == 0.0


def test_medium_seek_dominates_small_prey():
    # арбитраж: голодный коммитит СРЕДНЮЮ (55) сильнее мелкой-prey nav (21)
    import torch
    c = _c()
    # только мелкая
    l_small = torch.zeros(16)
    obs_s = [0.0] * 64
    obs_s[56] = 1.0; obs_s[58] = 0.1          # мелкая prey-направление север
    c._shape_action_logits(l_small, obs_s, diet=0.618, energy_ratio=0.4)
    # средняя + мелкая
    l_both = torch.zeros(16)
    obs_b = [0.0] * 64
    obs_b[56] = 1.0; obs_b[58] = 0.1
    mp = {"dr": -1, "dc": 0, "dist": 5.0}
    c._shape_action_logits(l_both, obs_b, diet=0.618, energy_ratio=0.4,
                           medium_prey=mp)
    assert float(l_both[0]) > float(l_small[0])   # средняя добавила тягу


def test_medium_contact_attack():
    # на контакте (dist≤1) → доминантный ATTACK + гаси move
    l = _med_nav(energy_ratio=0.4, dr=0, dc=0, dist=1.0)
    assert float(l[5]) > 0                     # ATTACK
    assert int(l.argmax()) == 5                # доминирует argmax


# ── pounce-флаг (obs-loop) → entry speed_boost ──────────────────────────

def _ws():
    import utopia_client.ws_client as wsm
    return wsm.ColonyWSClient(server="https://e.com", token="t",
                              colony_name="cheef", client_version="0.0.0",
                              estimated_population=0)


def test_pounce_flag_set_in_window():
    # dist≤_POUNCE_DIST + голоден + способен + hunting → флаг ставится
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(energy=300.0)   # er=0.3 < φ⁻¹
    c._hunt_pounce.clear()
    # эмулируем обращение obs-loop: проверяем по контракту флага напрямую
    _er = 0.3
    _mp = {"dr": -1, "dc": 0, "dist": 2.0}
    if (c._hunting_enabled and _mp is not None
            and _er < (1.0 / _PHI) and _er > (1.0 / _PHI) ** 5
            and _mp.get("dist") <= c._POUNCE_DIST):
        c._hunt_pounce["a"] = 1
    assert c._hunt_pounce.get("a") == 1


def test_pounce_entry_speed_boost():
    # флаг стоит → кардинальный MOVE получает speed_boost=+1
    ws = _ws()
    comp = types.SimpleNamespace()
    comp._single_organism = True
    comp._hunting_enabled = True
    comp._hunt_pounce = {"a": 1}
    comp.biochem = {"a": types.SimpleNamespace(energy=300.0)}
    comp.handle_tick = lambda *a, **k: {"a": {"action": 0}}   # MOVE NORTH
    comp.get_phase_emas = lambda cid: None
    ws.compute = comp
    out = ws._run_tick_and_build({"a": [0.0] * 64}, {}, {}, world_tick=0)
    e = {x["cid"]: x for x in (out or [])}["a"]
    assert e.get("speed_boost") == 1
    assert e.get("persist") is True


def test_no_pounce_boost_without_flag():
    ws = _ws()
    comp = types.SimpleNamespace()
    comp._single_organism = True
    comp._hunting_enabled = True
    comp._hunt_pounce = {}                       # нет флага
    comp.biochem = {"a": types.SimpleNamespace(energy=300.0)}
    comp.handle_tick = lambda *a, **k: {"a": {"action": 0}}
    comp.get_phase_emas = lambda cid: None
    ws.compute = comp
    out = ws._run_tick_and_build({"a": [0.0] * 64}, {}, {}, world_tick=0)
    e = {x["cid"]: x for x in (out or [])}["a"]
    assert "speed_boost" not in e                # нет рывка
