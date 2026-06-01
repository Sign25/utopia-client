"""bias_scale curriculum — порт server loop.py:607-636 (01.06.2026, Фрай).

Кроссфейд shaping↔motor: own_contribution=max(0,1-bias_scale) масштабирует
motor_delta. bias_scale стартует 1.0 (untrained → shaping ведёт), decay по
ratio n_alive/target. NAV-данные подтвердили: motor перебивал shaping → голод.
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


def _c(target=20):
    c = LocalColonyCompute(device="cpu")
    c._natural_selection_capacity = target  # _get_hw_capacity → target (без бенча)
    return c


def _populate(c, n):
    for i in range(n):
        c.organisms[f"a{i}"] = types.SimpleNamespace()


def test_starts_at_one():
    assert _c()._bias_scale == 1.0


def test_throttle_under_1000_ticks():
    c = _c()
    _populate(c, 20)
    c._update_bias_curriculum(500)        # <1000 с прошлого (0) → no-op
    assert c._bias_scale == 1.0
    assert c._bias_last_update_tick == 0


def test_decays_when_self_sustaining():
    c = _c()
    _populate(c, 20)
    c._last_window = {"ratio": 1.2}       # income>cost → self-sustaining
    c._update_bias_curriculum(1000)
    assert abs(c._bias_scale - 0.95) < 1e-9   # health>=1 → -0.05 (отпускаем motor)


def test_raises_when_starving():
    # Ключевой фикс: голодает (net<0) → bias РАСТЁТ (shaping доминирует).
    c = _c()
    c._bias_scale = 0.3                    # был отпущен
    _populate(c, 16)
    c._last_window = {"ratio": 0.1}       # income≪cost
    c._update_bias_curriculum(1000)
    assert abs(c._bias_scale - 0.4) < 1e-9   # health<1 → +0.1


def test_holds_at_one_when_starving():
    c = _c()
    _populate(c, 16)
    c._last_window = {"ratio": 0.05}      # голодает
    c._update_bias_curriculum(1000)
    assert c._bias_scale == 1.0           # старт 1.0, +0.1 capped → держится 1.0


def test_no_window_treated_as_starving():
    # _last_window=None (рано) → health=0 → bias держится наверху.
    c = _c()
    _populate(c, 16)
    c._update_bias_curriculum(1000)
    assert c._bias_scale == 1.0


def test_own_contribution_crossfade():
    # bias=1.0 → motor подавлен полностью; bias=0 → motor на полную.
    assert max(0.0, 1.0 - 1.0) == 0.0
    assert max(0.0, 1.0 - 0.0) == 1.0
    c = _c()
    c._bias_scale = 0.3
    assert abs(max(0.0, 1.0 - c._bias_scale) - 0.7) < 1e-9
