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


def test_thirst_DISABLED_no_decay():
    """INCIDENT 31.05: hydration-ось ОТКЛЮЧЕНА. income питья недостаточен
    (drink < thirst_now) → жажда выкосила ВСЮ колонию при здоровых
    energy(183-357)+telomere(0.999). thirst_now НЕ применяется (даже если cid
    активен). Включить только когда drink-income покроет thirst (калибровка)."""
    c, org, bc = _compute_with_org(hydration=80.0)
    c._hydration_active.add("c1")  # даже активный
    c._apply_metabolism("c1", {"step_cost_now": 0.0,
                               "telomere_decay_now": 0.0, "thirst_now": 15.0})
    assert bc.hydration == 80.0  # НЕ тронут (ось отключена)


def test_no_dehydration_death_even_active_at_zero():
    """РОВНО баг, выкосивший колонию: hydration=0 + активна, но energy=342 +
    telomere=0.999 (здоров) → НЕ умирает (жажда-смерть отключена)."""
    c, org, bc = _compute_with_org(energy=342.0, hydration=0.0)
    c.organisms["c1"].telomere = 0.999
    c._hydration_active.add("c1")
    c._apply_metabolism("c1", {"step_cost_now": 0.0,
                               "telomere_decay_now": 0.0, "thirst_now": 30.0})
    assert "c1" not in c._dead_cids  # ЖИВ


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
