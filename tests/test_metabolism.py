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
