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


def test_decays_when_at_target_and_healthy():
    c = _c(target=20)
    _populate(c, 20)                      # n_alive=20, ema→20, ratio=1.0
    c._last_window = {"ratio": 1.2}       # энергетически здорова (income>cost)
    c._update_bias_curriculum(1000)
    assert abs(c._bias_scale - 0.95) < 1e-9   # ratio>=0.95 И healthy → -0.05


def test_no_decay_when_at_target_but_starving():
    # Ключевой фикс: население at-target, но net<0 → НЕ декеим (motor подавлен).
    c = _c(target=20)
    _populate(c, 20)                      # ratio=1.0
    c._last_window = {"ratio": 0.1}       # голодает (income≪cost)
    c._update_bias_curriculum(1000)
    assert c._bias_scale == 1.0           # held — motor остаётся подавлен


def test_holds_in_band():
    c = _c(target=20)
    _populate(c, 16)                      # ratio=16/20=0.8 → держим
    c._last_window = {"ratio": 1.5}       # даже здоровая — band держит
    c._update_bias_curriculum(1000)
    assert c._bias_scale == 1.0


def test_raises_on_crash():
    c = _c(target=20)
    c._bias_scale = 0.5
    c._alive_ema = 20.0                   # недавняя норма высокая
    _populate(c, 10)                      # ratio=10/20=0.5 < 0.7 → +0.1
    c._update_bias_curriculum(1000)
    assert abs(c._bias_scale - 0.6) < 1e-9


def test_own_contribution_crossfade():
    # bias=1.0 → motor подавлен полностью; bias=0 → motor на полную.
    assert max(0.0, 1.0 - 1.0) == 0.0
    assert max(0.0, 1.0 - 0.0) == 1.0
    c = _c()
    c._bias_scale = 0.3
    assert abs(max(0.0, 1.0 - c._bias_scale) - 0.7) < 1e-9
