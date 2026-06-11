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


def test_beh_verdicts_round_trip():
    """Блок 7b: поведенческие вердикты переживают рестарт — секция
    «Поведенческая ценность тканей» не пустеет на self-update/сбое питания
    (Шеф 11.06: «ткань появляется, через время исчезает»)."""
    src = _compute()
    src._stat_beh_verdicts["grown4"] = {"verdict": "KEEP", "dims": {"cortisol": -2.4}}
    src._stat_beh_verdicts["grown92"] = {"verdict": "SOFT#1", "dims": {}}

    payload = src.save_state("a")
    assert payload["growth_loop"]["beh_verdicts"]["grown4"]["verdict"] == "KEEP"

    dst = _compute()                      # рестарт: секция пуста
    assert dst._stat_beh_verdicts == {}
    dst.restore_persisted_state("a", payload)

    assert dst._stat_beh_verdicts["grown4"]["verdict"] == "KEEP"
    assert dst._stat_beh_verdicts["grown92"]["verdict"] == "SOFT#1"
    # экспорт в diagnostics не пуст после рестарта
    owner = dst._stat_owner_extra()
    roles = {v["role"] for v in owner["beh_verdicts"]}
    assert "grown4" in roles and "grown92" in roles


def test_beh_gc_abort_cooldown_round_trip():
    """§3-abort escalating cooldown durable — иначе рестарт обнулял бы эскалацию
    и дестабилизирующий узел сразу вернулся бы в очередь GC (спираль 11.06)."""
    src = _compute()
    src._beh_gc_abort_count["a"] = {"grown93": 3}
    src._beh_gc_abort_cd["a"] = {"grown93": 123456}

    payload = src.save_state("a")
    gl = payload["growth_loop"]
    assert gl["beh_gc_abort_count"]["grown93"] == 3
    assert gl["beh_gc_abort_cd"]["grown93"] == 123456

    dst = _compute()
    assert "a" not in dst._beh_gc_abort_cd
    dst.restore_persisted_state("a", payload)
    assert dst._beh_gc_abort_count["a"]["grown93"] == 3
    assert dst._beh_gc_abort_cd["a"]["grown93"] == 123456
