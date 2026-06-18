"""force_water_far (Фрай live-bar a', §18) — VALIDATION-ONLY: вода-недостижима.

ON → water-seek-навигация OFF (ws_client) + drink-income (delta_hydration>0) НЕ
применяется → thirst дренит hyd→0 → φ³ hp-дренаж → hp→§3 → passive_water ЕДИНСТВЕННЫЙ
recovery → живой тест absorbing-closure non-spiral под energy-guardrail. Обратимо,
ТОЛЬКО на тест. passive_water (paralysis backstop) НЕ затронут (его валидируем).
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

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402
from utopia_client.biochemistry import ClientCreatureBiochem  # noqa: E402


def _compute(hydration=50.0):
    c = LocalColonyCompute(device="cpu")
    c._single_organism = True
    bc = ClientCreatureBiochem()
    bc.hydration = float(hydration)
    c.biochem["c0"] = bc
    return c, bc


def test_default_off_and_setter():
    c = LocalColonyCompute(device="cpu")
    assert c._force_water_far_enabled is False
    assert c.set_force_water_far(True) is True
    assert c._force_water_far_enabled is True
    assert c.set_force_water_far(False) is False


def test_drink_income_suppressed_when_on():
    """force_water_far ON → положительный delta_hydration (питьё) НЕ применяется."""
    c, bc = _compute(hydration=50.0)
    c.set_force_water_far(True)
    c._apply_biochem_events("c0", {"delta_hydration": 20.0})
    assert bc.hydration == pytest.approx(50.0)        # питьё подавлено


def test_drink_income_applied_when_off():
    """OFF (default): питьё (delta_hydration>0) применяется штатно."""
    c, bc = _compute(hydration=50.0)
    c._apply_biochem_events("c0", {"delta_hydration": 20.0})
    assert bc.hydration == pytest.approx(70.0)        # питьё прошло


def test_negative_hydration_not_suppressed():
    """Отрицательный delta_hydration (потеря) проходит даже при force_water_far
    (подавляем только ВХОД-питьё, не потерю)."""
    c, bc = _compute(hydration=50.0)
    c.set_force_water_far(True)
    c._apply_biochem_events("c0", {"delta_hydration": -10.0})
    assert bc.hydration == pytest.approx(40.0)        # потеря применилась
