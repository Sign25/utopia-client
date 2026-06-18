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
    """GAP закрыт: _apply_action_fatigue копит fatigue от действия (client-φ-лестница,
    БЕЗ neurocore-re-use). ATTACK = b×φ³ ≈ 0.847 (b=0.2)."""
    from utopia_client.local_compute import _FATIGUE_B_DEFAULT
    import math
    phi = 1.6180339887498949
    c, bc = _compute(fatigue=0.0)
    c._apply_action_fatigue("c0", 5)            # ATTACK = b·φ³
    assert bc.fatigue > 0.0                      # накопилось (раньше было 0 всегда)
    assert math.isclose(bc.fatigue, _FATIGUE_B_DEFAULT * phi ** 3, rel_tol=1e-6)


def test_action_fatigue_phi_ladder():
    """φ-лестница тиров: ATTACK(φ³b) > рывки(φ²b) > координация(φb) > move(b) > STAY(0)."""
    def _fat(act):
        c, bc = _compute(fatigue=0.0)
        c._apply_action_fatigue("c0", act)
        return bc.fatigue
    f_stay, f_move, f_coord, f_burst, f_atk = (
        _fat(4), _fat(0), _fat(6), _fat(11), _fat(5))   # STAY/move/SIGNAL/DIG/ATTACK
    assert f_stay == 0.0                         # STAY = recovery (0 расход)
    assert f_atk > f_burst > f_coord > f_move > 0.0   # строгая φ-лестница


def test_fatigue_b_tunable():
    """b агильно tunable (DB fatigue_b, replay-pressure): delta = b×тир."""
    c, bc = _compute(fatigue=0.0)
    c.set_fatigue_b(0.5)                          # поднять b (давление)
    c._apply_action_fatigue("c0", 0)             # move = b
    assert bc.fatigue == pytest.approx(0.5)      # delta = b×1.0
    assert c.set_fatigue_b(10.0) == 5.0          # клемп [0,5]


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
