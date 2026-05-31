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


def test_thirst_inactive_no_decay():
    """Hydration-ось гейтится: пока cid НЕ активен (P40 не шлёт
    delta_hydration), thirst не применяется — deploy-order-safe, нет коллапса."""
    c, org, bc = _compute_with_org(hydration=80.0)
    # cid НЕ в _hydration_active
    c._apply_metabolism("c1", {"step_cost_now": 0.0,
                               "telomere_decay_now": 0.0, "thirst_now": 15.0})
    assert bc.hydration == 80.0  # не тронут (ось не активна)
    assert "c1" not in c._dead_cids


def test_thirst_active_decays():
    """Когда питьё активно (P40 шлёт delta_hydration) — thirst применяется."""
    c, org, bc = _compute_with_org(hydration=80.0)
    c._hydration_active.add("c1")  # эмулируем активацию (P40 шлёт income)
    c._apply_metabolism("c1", {"step_cost_now": 0.0,
                               "telomere_decay_now": 0.0, "thirst_now": 15.0})
    assert bc.hydration == 65.0


def test_dehydration_death_when_active():
    c, org, bc = _compute_with_org(hydration=10.0)
    c._hydration_active.add("c1")
    c._apply_metabolism("c1", {"step_cost_now": 0.0,
                               "telomere_decay_now": 0.0, "thirst_now": 30.0})
    assert bc.hydration == 0.0
    assert "c1" in c._dead_cids  # жажда


def test_delta_hydration_activates_and_incomes():
    """delta_hydration в event → активирует hydration-ось + добавляет income.
    (Нужен environment.biochemistry для apply_feed-импорта в _apply_biochem_events.)"""
    pytest.importorskip("environment.biochemistry")
    c, org, bc = _compute_with_org(hydration=50.0)
    c._apply_biochem_events("c1", {"delta_hydration": 20.0})
    assert "c1" in c._hydration_active
    assert bc.hydration == 70.0  # income применён


def test_no_dehydration_death_when_inactive():
    """hydration=0 у НЕактивного cid НЕ убивает (ложная смерть без income)."""
    c, org, bc = _compute_with_org(hydration=0.0)
    # cid НЕ активен
    c._apply_metabolism("c1", {"step_cost_now": 5.0,
                               "telomere_decay_now": 0.0, "thirst_now": 30.0})
    assert "c1" not in c._dead_cids  # энергии хватает, жажда не считается


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
