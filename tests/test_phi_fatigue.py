"""stamina φ-расход выносливости (Фрай §19, АКТИВНЫЙ) — fatigue от действий + дренаж→hp.

GAP закрыт: client раньше НЕ копил action-fatigue (apply_action_taken не звался →
Адам не уставал, fat=0). ON (phi_fatigue) → _apply_action_fatigue (re-use
apply_action_taken, ACTION_FATIGUE_GLUCOSE φ-лестница) per-tick → fatigue копится →
выносливость=0 (fatigue=max) → дренаж hp φ³. SELF-recovering (§3=force-STAY=отдых →
fatigue decay → recovery; passive-backstop не нужен). OFF → инертно (только decay).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_NEUROCORE = _ROOT.parent / "NeuroCore"
for _p in (str(_ROOT), str(_NEUROCORE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute, _HP_NEED_DRAIN, _HP_HEAL, _FATIGUE_MAX,
)
from utopia_client.biochemistry import ClientCreatureBiochem  # noqa: E402


def _compute(energy=500.0, hydration=100.0, hp=500.0, fatigue=0.0):
    c = LocalColonyCompute(device="cpu")
    c._single_organism = True
    c._damage_factor = 1.0
    bc = ClientCreatureBiochem()
    bc.energy, bc.hydration, bc.hp, bc.fatigue = energy, hydration, hp, fatigue
    c.biochem["c0"] = bc
    return c, bc


def _rates(**kw):
    base = {"step_cost_per_tick": 0.0, "basal_drain_per_tick": 0.0,
            "thirst_per_tick": 0.0, "telomere_decay_per_tick": 0.0}
    base.update(kw)
    return base


def test_default_off_and_setter():
    c = LocalColonyCompute(device="cpu")
    assert c._phi_fatigue_enabled is False
    assert c.set_phi_fatigue(True) is True
    assert c._phi_fatigue_enabled is True
    assert c.set_phi_fatigue(False) is False


def test_apply_action_fatigue_accumulates():
    """GAP закрыт: _apply_action_fatigue копит fatigue от действия (re-use)."""
    pytest.importorskip("environment.biochemistry")
    c, bc = _compute(fatigue=0.0)
    c._apply_action_fatigue("c0", 5)            # ATTACK
    assert bc.fatigue > 0.0                      # накопилось (раньше было 0 всегда)


def test_action_fatigue_ladder_ordering():
    """ATTACK (схватка, φ³b) копит ≥ движения (b) — φ-лестница (и legacy 0.5≥0.2)."""
    pytest.importorskip("environment.biochemistry")
    c1, bc1 = _compute(fatigue=0.0)
    c1._apply_action_fatigue("c0", 5)            # ATTACK
    c2, bc2 = _compute(fatigue=0.0)
    c2._apply_action_fatigue("c0", 0)            # move N
    assert bc1.fatigue >= bc2.fatigue            # схватка ≥ движение


def test_stamina_drain_to_hp_when_exhausted():
    """phi_fatigue ON + выносливость=0 (fatigue=max) → дренаж hp φ³ (3-я нужда)."""
    c, bc = _compute(energy=500.0, hydration=100.0, hp=500.0, fatigue=_FATIGUE_MAX)
    c.set_hp_authoritative(True)
    c.set_phi_fatigue(True)
    c._apply_metabolism("c0", _rates())
    assert bc.hp == pytest.approx(500.0 - _HP_NEED_DRAIN)   # дренаж φ³ от выносливости=0


def test_stamina_drain_off_when_phi_disabled():
    """phi_fatigue OFF → выносливость НЕ лимитирует (инертно): fatigue=max не дренит
    hp, heal штатно (нужды ок)."""
    c, bc = _compute(energy=500.0, hydration=100.0, hp=500.0, fatigue=_FATIGUE_MAX)
    c.set_hp_authoritative(True)                  # phi_fatigue OFF
    c._apply_metabolism("c0", _rates())
    assert bc.hp == pytest.approx(500.0 + _HP_HEAL)  # heal, НЕ дренаж (выносливость inert)


def test_heal_blocked_when_exhausted():
    """phi_fatigue ON + выносливость=0 → дренаж (НЕ heal), даже если energy/hyd ок."""
    c, bc = _compute(energy=500.0, hydration=100.0, hp=500.0, fatigue=_FATIGUE_MAX)
    c.set_hp_authoritative(True)
    c.set_phi_fatigue(True)
    c._apply_metabolism("c0", _rates())
    assert bc.hp < 500.0                          # дренаж выиграл (heal заблокирован)
