"""Body Migration метаболизм client-side (31.05.2026, Бендер; контракт Хьюберт).

P40 шлёт effective rates (step_cost_now/telomere_decay_now/thirst_now), client
интегрирует energy/hydration/telomere + death-check (голод energy<=0 / старость
telomere AGONY) → projection alive=False → P40 убирает. Закрывает death pressure
(P40 phase-out не тикал owned → 9× перенаселение).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402
from utopia_client.biochemistry import make_default  # noqa: E402


def _compute_with_org(cid="c1", energy=500.0, telomere=1.0, hydration=100.0):
    c = LocalColonyCompute(device="cpu")
    org = types.SimpleNamespace(generation=0, telomere=telomere)
    c.organisms[cid] = org
    bc = make_default()
    bc.energy = energy
    bc.hydration = hydration
    c.biochem[cid] = bc
    return c, org, bc


def test_energy_decay():
    c, org, bc = _compute_with_org(energy=500.0)
    c._apply_metabolism("c1", {"step_cost_now": 30.0,
                               "telomere_decay_now": 0.0, "thirst_now": 0.0})
    assert bc.energy == 470.0
    assert "c1" not in c._dead_cids


def test_metab_guard_one_drain_per_server_tick():
    # Contract ×2 fix (Хьюберт): дубль handle_tick того же world_tick не дренит.
    c, org, bc = _compute_with_org(energy=500.0)
    rates = {"step_cost_now": 10.0, "telomere_decay_now": 0.0, "thirst_now": 0.0}
    c._last_world_tick = 100
    c._apply_metabolism("c1", rates)
    assert bc.energy == 490.0 and c._metab_applies == 1
    c._apply_metabolism("c1", rates)          # тот же server-тик → skip
    assert bc.energy == 490.0 and c._metab_skips == 1
    c._last_world_tick = 101                  # новый server-тик → дренит
    c._apply_metabolism("c1", rates)
    assert bc.energy == 480.0 and c._metab_applies == 2


def test_thirst_decays_calibration_mode():
    """Калибровка 31.05: thirst-декей ВКЛ для наблюдения баланса (для активных
    cid). Смерть от жажды ОТКЛ (см. ниже). Аккумулятор thirst_sum растёт."""
    c, org, bc = _compute_with_org(hydration=80.0)
    c._hydration_active.add("c1")
    c._apply_metabolism("c1", {"step_cost_now": 0.0,
                               "telomere_decay_now": 0.0, "thirst_now": 15.0})
    assert bc.hydration == 65.0  # декей применён (калибровка)
    assert c._hyd_thirst_sum == 15.0  # аккумулятор для calib-лога


def test_thirst_unconditional_now():
    """01.06: водный контур client-authoritative — жажда БЕЗУСЛОВНА (доход
    теперь client-side из террейна, gate _hydration_active убран)."""
    c, org, bc = _compute_with_org(hydration=80.0)
    c._apply_metabolism("c1", {"step_cost_now": 0.0,
                               "telomere_decay_now": 0.0, "thirst_now": 15.0})
    assert bc.hydration == 65.0  # декей применён даже без _hydration_active


def test_no_dehydration_death_while_damage_disabled():
    """Death-урон ОТКЛ (флаг False): hydration=0, energy=342, telomere=0.999
    → НЕ умирает, энергия не тронута. (Защитный путь — урок 0.11.24.)"""
    c, org, bc = _compute_with_org(energy=342.0, hydration=0.0)
    c.organisms["c1"].telomere = 0.999
    c._dehydration_damage_enabled = False  # явно (дефолт 0.11.34 = True)
    c._apply_metabolism("c1", {"step_cost_now": 0.0,
                               "telomere_decay_now": 0.0, "thirst_now": 30.0})
    assert "c1" not in c._dead_cids  # ЖИВ (energy_drain выключен)
    assert bc.energy == 342.0       # энергия не тронута жаждой


def test_dehydration_damage_disabled_by_default():
    """0.11.38: death-урон ОТКЛ — дрейн (даже ×0.1) опрокидывал маргинальный
    eat-income → energy<порог-репро → вымирание (3×). Контроль населения теперь
    через популяционный кэп в размножении, не death-налог."""
    c, org, bc = _compute_with_org()
    assert c._dehydration_damage_enabled is False


def test_shape_logits_flora_gradient_pulls_movement():
    """Phase 4: флора-градиент NS>0 → bias к SOUTH (logits[1]) над NORTH[0]."""
    import torch
    c = LocalColonyCompute(device="cpu")
    logits = torch.zeros(16)
    obs = [0.0] * 64
    obs[33] = 1.0  # флора-градиент NS
    c._shape_action_logits(logits, obs, diet=0.5, energy_ratio=1.0)
    assert float(logits[1]) > float(logits[0])  # S притянут, N оттолкнут


def test_shape_logits_penalizes_stay_reproduce_dig():
    import torch
    c = LocalColonyCompute(device="cpu")
    logits = torch.zeros(16)
    c._shape_action_logits(logits, [0.0] * 64, diet=0.5, energy_ratio=1.0)
    assert float(logits[4]) < 0    # STAY
    assert float(logits[9]) < 0    # REPRODUCE
    assert float(logits[11]) < 0   # DIG
    assert float(logits[0]) == 0.0  # move без градиента не тронут


def test_shape_logits_attack_context():
    import torch
    c = LocalColonyCompute(device="cpu")
    # нет prey → ATTACK штраф
    l1 = torch.zeros(16)
    c._shape_action_logits(l1, [0.0] * 64, diet=0.5, energy_ratio=1.0)
    assert float(l1[5]) < 0
    # добыча достижима (prox>0.3) → ATTACK буст
    l2 = torch.zeros(16)
    obs = [0.0] * 64; obs[58] = 0.5
    c._shape_action_logits(l2, obs, diet=0.5, energy_ratio=1.0)
    assert float(l2[5]) > 0
    # добыча видна но далеко (prox 0.1..0.3) → нейтрально (ни штраф, ни буст)
    l3 = torch.zeros(16)
    obs = [0.0] * 64; obs[58] = 0.2
    c._shape_action_logits(l3, obs, diet=0.5, energy_ratio=1.0)
    assert float(l3[5]) == 0.0  # prey-градиент подведёт, ATTACK не форсим


def test_shape_logits_flee_near_predator():
    import torch
    c = LocalColonyCompute(device="cpu")
    logits = torch.zeros(16)
    obs = [0.0] * 64; obs[61] = 0.5  # pred_prox > 0.15
    c._shape_action_logits(logits, obs, diet=0.5, energy_ratio=1.0)
    assert float(logits[10]) > 0   # FLEE буст у хищника


def test_skill_growth_efficiency_movespeed():
    """F5 (Фрай): eat>10 → efficiency+1, move>100 → move_speed+1, reset+changed."""
    c, org, bc = _compute_with_org()
    c.traits["c1"] = {"efficiency": 6, "attack_power": 3, "move_speed": 4}
    c._skill_eat["c1"] = 15
    c._skill_move["c1"] = 120
    c._skill_growth_step("c1")
    assert c.traits["c1"]["efficiency"] == 7
    assert c.traits["c1"]["move_speed"] == 5
    assert "c1" in c._skill_changed_cids
    assert c._skill_eat["c1"] == 0  # окно сброшено


def test_skill_decay_on_disuse():
    """eat<=2 → efficiency-1 (min5), move<30 → move_speed-1 (min2)."""
    c, org, bc = _compute_with_org()
    c.traits["c1"] = {"efficiency": 8, "attack_power": 3, "move_speed": 6}
    c._skill_eat["c1"] = 1
    c._skill_move["c1"] = 10
    c._skill_growth_step("c1")
    assert c.traits["c1"]["efficiency"] == 7
    assert c.traits["c1"]["move_speed"] == 5


def test_skill_efficiency_floor_no_spurious_change():
    """efficiency на полу 5 + disuse → не падает ниже 5, changed НЕ ставится."""
    c, org, bc = _compute_with_org()
    c.traits["c1"] = {"efficiency": 5, "attack_power": 3, "move_speed": 5}
    c._skill_eat["c1"] = 0
    c._skill_move["c1"] = 50  # 30..100 → move без изменений
    c._skill_growth_step("c1")
    assert c.traits["c1"]["efficiency"] == 5
    assert "c1" not in c._skill_changed_cids


def test_skill_attack_power_phi_threshold():
    """kill>=max(2,round(atk/φ)) → attack_power+1 (φ-порог)."""
    c, org, bc = _compute_with_org()
    c.traits["c1"] = {"efficiency": 6, "attack_power": 3, "move_speed": 5}
    c._skill_kill["c1"] = 2   # порог для atk=3: max(2,round(1.85))=2
    c._skill_eat["c1"] = 5    # без изменения efficiency
    c._skill_move["c1"] = 50  # без изменения move_speed
    c._skill_growth_step("c1")
    assert c.traits["c1"]["attack_power"] == 4


def test_infection_ticks_and_drains_energy():
    """Фрай: owned-инфекцию тикает клиент. severity +0.005/тик, energy -=
    severity*2.0 (болезнь)."""
    c, org, bc = _compute_with_org(energy=100.0)
    bc.infection_severity = 0.5
    c._apply_metabolism("c1", {"step_cost_now": 0.0,
                               "telomere_decay_now": 0.0, "thirst_now": 0.0})
    assert abs(bc.infection_severity - 0.505) < 1e-6
    assert abs(bc.energy - (100.0 - 0.505 * 2.0)) < 1e-6


def test_infection_death_at_severity_1():
    """severity>=1.0 → death (cause=infection) через единый death-envelope."""
    c, org, bc = _compute_with_org(energy=500.0)
    c.organisms["c1"].telomere = 0.999
    bc.infection_severity = 0.999
    c._apply_metabolism("c1", {"step_cost_now": 0.0,
                               "telomere_decay_now": 0.0, "thirst_now": 0.0})
    assert bc.infection_severity >= 1.0
    assert "c1" in c._dead_cids


def test_infection_contact_event():
    """P40 шлёт infected/infection_contact → начальная severity 0.05."""
    pytest.importorskip("environment.biochemistry")
    c, org, bc = _compute_with_org()
    bc.infection_severity = 0.0
    c._apply_biochem_events("c1", {"infected": True})
    assert abs(bc.infection_severity - 0.05) < 1e-9
    assert bc.infected is True


def test_infection_contact_list_format():
    """Формат Хьюберта: infection_contact=[{from_cid, severity_hint}] →
    infected=True, severity=max(severity_hint)."""
    pytest.importorskip("environment.biochemistry")
    c, org, bc = _compute_with_org()
    bc.infection_severity = 0.0
    c._apply_biochem_events("c1", {"infection_contact": [
        {"from_cid": "z9", "severity_hint": 0.05}]})
    assert abs(bc.infection_severity - 0.05) < 1e-9
    assert bc.infected is True


def test_projection_includes_infection():
    """projection_batch несёт infected + infection_severity (P40 зеркалит)."""
    c, org, bc = _compute_with_org()
    bc.infected = True
    bc.infection_severity = 0.3
    p = c.build_projection_batch()[0]
    assert p["infected"] is True
    assert abs(p["infection_severity"] - 0.3) < 1e-9


def test_speciation_empty_topology_collapses_to_founder():
    """Phase 4 fix (Фрай): пустая топология → founder-вид (1), НЕ новый каждый
    раз (баг 17/17). find_best_match симулируем как не-матчащий (баг ядра)."""
    pytest.importorskip("core.tissue_speciation")
    from utopia_client.speciation import assign_species

    class _Sp:
        def __init__(self, sid):
            self.species_id = sid
            self.extinct = False

    class _Reg:
        def __init__(self):
            self.sp = [_Sp(1), _Sp(5), _Sp(3)]
            self.created = 0

        def is_empty(self):
            return not self.sp

        def all(self):
            return self.sp

        def find_best_match(self, *a, **k):
            return None  # симуляция бага: пустая топология не матчится

        def create(self, **k):
            self.created += 1
            return _Sp(99)

        def revive(self, sid):
            pass

    reg = _Reg()
    sid, is_new = assign_species(reg, [], tick=1, founder_cid="x")
    assert sid == 1            # founder (min species_id)
    assert is_new is False     # НЕ новый вид
    assert reg.created == 0    # новый вид НЕ создан


def test_snapshot_elite_health_gate(tmp_path):
    """Elite (Фрай): снимок только при здоровой колонии (>=min_alive живых с
    energy>0) — не затирать elite умирающей колонией. Мёртвые/energy=0 не в счёт."""
    pytest.importorskip("environment.biochemistry")
    import types
    from utopia_client.biochemistry import make_default
    c = LocalColonyCompute(device="cpu")
    # c0 energy=0, c1 dead, c2/c3/c4 здоровы → alive-healthy = 3
    specs = [(0.0, False), (500.0, True), (500.0, False), (500.0, False),
             (500.0, False)]
    for i, (e, dead) in enumerate(specs):
        cid = f"c{i}"
        c.organisms[cid] = types.SimpleNamespace(generation=0, telomere=1.0)
        b = make_default(); b.energy = e
        c.biochem[cid] = b
        if dead:
            c._dead_cids.add(cid)
    assert c.snapshot_elite(str(tmp_path), min_alive=4) == 0  # 3 < 4 → гейт


def test_motor_reward_baseline_cleanup():
    """Phase 4 #1: REINFORCE-baseline dict существует и чистится в remove."""
    c = LocalColonyCompute(device="cpu")
    assert isinstance(c._motor_reward_baseline, dict)
    c._motor_reward_baseline["x"] = 1.5
    c.remove_creature("x")
    assert "x" not in c._motor_reward_baseline


def test_reproduction_population_cap():
    """0.11.38: размножение НЕ идёт при alive >= ёмкости (bounded self-
    sustaining цикл, без перенаселения)."""
    import types
    c = LocalColonyCompute(device="cpu")
    c._natural_selection_capacity = 2  # маленькая ёмкость для теста
    for cid in ("a", "b", "c"):  # 3 alive > cap 2
        c.organisms[cid] = types.SimpleNamespace(generation=0, telomere=1.0)
        c.biochem[cid] = make_default()
    born = c.detect_and_emit_mate_pairs(world_tick=1000, embodied_client=None)
    assert born == []  # кэп сработал — рождений нет


def test_dehydration_stage_thresholds():
    from utopia_client.local_compute import _dehydration_stage
    assert _dehydration_stage(80.0, 100.0) == 0   # >0.5 норма
    assert _dehydration_stage(40.0, 100.0) == 1   # 0.25–0.5 жажда
    assert _dehydration_stage(10.0, 100.0) == 2   # 0–0.25 обезвоживание
    assert _dehydration_stage(0.0, 100.0) == 3    # =0 критическое


def test_dehydration_energy_drain_when_enabled():
    """Флаг ВКЛ: hydration stage 2 (10/100) → energy -= φ²×0.1≈0.262 (мягкий
    рекалиброванный дрейн под client-tick, не полный φ²)."""
    c, org, bc = _compute_with_org(energy=100.0, hydration=10.0)
    c._dehydration_damage_enabled = True
    c._apply_metabolism("c1", {"step_cost_now": 0.0,
                               "telomere_decay_now": 0.0, "thirst_now": 0.0})
    assert abs(bc.energy - (100.0 - 1.618033988749895 ** 2 * 0.1)) < 1e-6


def test_dehydration_death_via_energy():
    """Флаг ВКЛ + почти нулевая энергия + крит-обезвоживание → energy<=0 →
    смерть через starvation (единая ось), не отдельной hydration-смертью.
    (Дрейн мягкий φ³×0.1≈0.424 → нужна низкая стартовая энергия для смерти
    за тик; в проде смерть наступает за минуты, не мгновенно.)"""
    c, org, bc = _compute_with_org(energy=0.3, hydration=0.0)
    c.organisms["c1"].telomere = 0.999
    c._dehydration_damage_enabled = True
    c._apply_metabolism("c1", {"step_cost_now": 0.0,
                               "telomere_decay_now": 0.0, "thirst_now": 0.0})
    assert bc.energy == 0.0          # 0.3 − φ³×0.1≈0.424 → clamp 0
    assert "c1" in c._dead_cids      # смерть через энергию


def test_delta_hydration_income_still_works():
    """income (delta_hydration) безвреден — hydration только растёт/стоит,
    ось активируется (для будущего re-enable), но смерть/декей отключены.
    (Нужен environment.biochemistry для apply_feed-импорта.)"""
    pytest.importorskip("environment.biochemistry")
    c, org, bc = _compute_with_org(hydration=50.0)
    c._apply_biochem_events("c1", {"delta_hydration": 20.0})
    assert "c1" in c._hydration_active
    assert bc.hydration == 70.0  # income применён


def test_telomere_decay():
    c, org, bc = _compute_with_org(telomere=0.5)
    c._apply_metabolism("c1", {"step_cost_now": 0.0,
                               "telomere_decay_now": 0.1, "thirst_now": 0.0})
    assert abs(org.telomere - 0.4) < 1e-9


def test_starvation_death():
    c, org, bc = _compute_with_org(energy=20.0)
    c._apply_metabolism("c1", {"step_cost_now": 30.0,  # > energy → 0
                               "telomere_decay_now": 0.0, "thirst_now": 0.0})
    assert bc.energy == 0.0
    assert "c1" in c._dead_cids  # голод


def test_telomere_agony_death():
    pytest.importorskip("core.telomere_phase")
    from core.telomere_phase import get_phase, TelomerePhase
    c, org, bc = _compute_with_org(telomere=0.02)
    # подберём telomere в фазе AGONY
    assert get_phase(0.0) == TelomerePhase.AGONY
    org.telomere = 0.0
    c._apply_metabolism("c1", {"step_cost_now": 0.0,
                               "telomere_decay_now": 0.0, "thirst_now": 0.0})
    assert "c1" in c._dead_cids  # старость


def test_no_rates_noop():
    c, org, bc = _compute_with_org(energy=500.0)
    c._apply_metabolism("c1", None)
    assert bc.energy == 500.0
    assert "c1" not in c._dead_cids


def test_clamp_energy_nonnegative():
    c, org, bc = _compute_with_org(energy=10.0)
    c._apply_metabolism("c1", {"step_cost_now": 999.0,
                               "telomere_decay_now": 0.0, "thirst_now": 0.0})
    assert bc.energy == 0.0  # не уходит в минус


def test_projection_includes_metabolic_fields():
    c, org, bc = _compute_with_org(energy=420.0, telomere=0.7, hydration=88.0)
    projs = c.build_projection_batch()
    assert len(projs) == 1
    p = projs[0]
    assert p["alive"] is True
    assert p["energy"] == 420.0
    assert p["hydration"] == 88.0
    assert abs(p["telomere_scale"] - 0.7) < 1e-9


def test_projection_alive_false_for_dead():
    c, org, bc = _compute_with_org(energy=0.0)
    c._dead_cids.add("c1")
    p = c.build_projection_batch()[0]
    assert p["alive"] is False


def test_dead_skipped_in_remove_cleanup():
    c, org, bc = _compute_with_org()
    c._dead_cids.add("c1")
    c.remove_creature("c1")
    assert "c1" not in c._dead_cids
    assert "c1" not in c.organisms
