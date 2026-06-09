"""§6 рост мозга durable (Фрай 09.06): прогресс петли роста переживает рестарт.

Вчерашнее отключение питания вскрыло gap — predictor/связи персистились
(save_state), но ПРОГРЕСС петли (growth_kept/reverted-счётчики + трейлинг-floor
deque + счётчик стагнации) был in-memory → сбрасывался на рестарте → KPI «связей
закреплено» врал «0» + PROPOSE ждал ~233-тик re-warm детектора-плато. Фикс:
персистить эти поля в save_state/restore_persisted_state. predictor цел (loss
непрерывен) → floor-история валидна → возобновлять, не re-warm'ить.
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

import types  # noqa: E402

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402


def _compute():
    c = LocalColonyCompute(device="cpu")
    c.organisms["a"] = types.SimpleNamespace(generation=0)
    return c


def test_growth_loop_round_trip():
    src = _compute()
    src._growth_kept = 12
    src._growth_reverted = 33
    src._growth_intr_hist["a"] = deque(
        [0.009, 0.008, 0.0079], maxlen=src._growth_intr_window)
    src._growth_stagnation_n["a"] = 40

    payload = src.save_state("a")
    assert payload is not None
    gl = payload["growth_loop"]
    assert gl["kept"] == 12 and gl["reverted"] == 33
    assert gl["intr_hist"] == [0.009, 0.008, 0.0079]
    assert gl["stagnation_n"] == 40

    # свежий compute (= рестарт): счётчики на 0, history пуст
    dst = _compute()
    assert dst._growth_kept == 0 and "a" not in dst._growth_intr_hist
    dst.restore_persisted_state("a", payload)

    assert dst._growth_kept == 12            # KPI «закреплено» не врёт «0»
    assert dst._growth_reverted == 33
    assert dst._growth_stagnation_n["a"] == 40   # стагнация продолжается, не re-warm
    hist = dst._growth_intr_hist["a"]
    assert isinstance(hist, deque)
    assert list(hist) == [0.009, 0.008, 0.0079]
    assert hist.maxlen == dst._growth_intr_window   # bounded


def test_growth_loop_absent_payload_safe():
    # Старый .pt без growth_loop → restore не падает, счётчики дефолтные.
    dst = _compute()
    dst.restore_persisted_state("a", {"predictor_loss_ema": 0.05})
    assert dst._growth_kept == 0 and "a" not in dst._growth_intr_hist


def test_growth_loop_empty_hist_not_written():
    # Нет истории (петля ещё не крутилась) → intr_hist пустой, restore не создаёт
    # deque (детектор начнёт копить заново — корректно, истории нет).
    src = _compute()
    payload = src.save_state("a")
    assert payload["growth_loop"]["intr_hist"] == []
    dst = _compute()
    dst.restore_persisted_state("a", payload)
    assert "a" not in dst._growth_intr_hist
