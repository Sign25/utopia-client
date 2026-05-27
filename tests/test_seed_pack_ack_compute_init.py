"""Track 1 fix (27.05.2026, Бендер): `seed_pack_ack` с restored>0 → инициирует
`_ensure_compute()`.

Background: при service-restart с cached seeds P40 шлёт только
`seed_pack_ack: restored=N`, не fresh `seed_chunk`. До fix компьют
оставался None всю сессию (no Hebbian/embodied path). Fix трижды
проверяется: restored>0 → init, restored=0 → no init, idempotent при
повторном вызове.

Стиль mock-based как test_diagnostics_push.py, без heavy neurocore deps.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utopia_client.ws_client import ColonyWSClient  # noqa: E402


def _make_ws() -> ColonyWSClient:
    """Minimal ColonyWSClient instance без сети (нужны только handler-методы)."""
    return ColonyWSClient(
        server="https://test.local",
        token="fake_token",
        colony_name="test-colony",
        client_version="0.0.0-test",
    )


def _run(coro):
    """Запустить coroutine в fresh event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Track 1 fix: seed_pack_ack → _ensure_compute ────────────────────────────


def test_seed_pack_ack_restored_positive_triggers_ensure_compute():
    """`restored=10` → `_ensure_compute()` вызван ровно один раз."""
    ws = _make_ws()
    ws._ensure_compute = MagicMock(return_value=True)
    msg = {
        "type": "seed_pack_ack",
        "restored": 10,
        "requested": 10,
        "failed": 0,
        "error": None,
        "fallback_path": None,
    }
    _run(ws._handle(msg))
    ws._ensure_compute.assert_called_once()


def test_seed_pack_ack_restored_zero_does_not_trigger():
    """`restored=0` (server fallback path) → `_ensure_compute` НЕ вызван.

    Сервер ушёл в fallback (load_personal/genesis) и ждать обычный seed_*
    flow. `_ensure_compute` дёрнется внутри seed_chunk handler, не тут.
    """
    ws = _make_ws()
    ws._ensure_compute = MagicMock(return_value=True)
    msg = {
        "type": "seed_pack_ack",
        "restored": 0,
        "requested": 10,
        "failed": 0,
        "fallback_path": "load_personal",
    }
    _run(ws._handle(msg))
    ws._ensure_compute.assert_not_called()


def test_seed_pack_ack_missing_restored_treated_as_zero():
    """Отсутствующее поле `restored` → trеated как 0, нет init."""
    ws = _make_ws()
    ws._ensure_compute = MagicMock(return_value=True)
    msg = {"type": "seed_pack_ack", "fallback_path": "genesis"}
    _run(ws._handle(msg))
    ws._ensure_compute.assert_not_called()


def test_seed_pack_ack_non_int_restored_handled_safely():
    """Битый payload (`restored="abc"`) → не падает, не вызывает init."""
    ws = _make_ws()
    ws._ensure_compute = MagicMock(return_value=True)
    msg = {"type": "seed_pack_ack", "restored": "not-a-number"}
    # Не должно raise ServerError или ValueError
    _run(ws._handle(msg))
    ws._ensure_compute.assert_not_called()


def test_seed_pack_ack_ensure_compute_exception_doesnt_propagate():
    """Если `_ensure_compute` сам бросает — handler не падает (try/except)."""
    ws = _make_ws()
    ws._ensure_compute = MagicMock(side_effect=RuntimeError("torch unavailable"))
    msg = {"type": "seed_pack_ack", "restored": 5}
    # raise → внутренний log warning, не пропагирует наружу
    _run(ws._handle(msg))
    ws._ensure_compute.assert_called_once()


def test_seed_pack_ack_idempotent_double_call():
    """Повторный seed_pack_ack — `_ensure_compute` вызывается ещё раз, но
    сам он идемпотентен (early return на self.compute is not None — это
    внутренний контракт, проверяется в test_ensure_compute_idempotence)."""
    ws = _make_ws()
    ws._ensure_compute = MagicMock(return_value=True)
    msg = {"type": "seed_pack_ack", "restored": 8}
    _run(ws._handle(msg))
    _run(ws._handle(msg))
    assert ws._ensure_compute.call_count == 2  # handler не дедуплицирует
    # дедупликация — забота `_ensure_compute` (it checks `self.compute is not None`).


def test_ensure_compute_idempotence_contract():
    """`_ensure_compute` идемпотентен: если `self.compute` уже set — early return True."""
    ws = _make_ws()
    sentinel = MagicMock()  # симулируем готовый LocalColonyCompute
    ws.compute = sentinel
    # _ensure_compute должен видеть, что compute уже есть, и не пересоздать
    result = ws._ensure_compute()
    assert result is True
    assert ws.compute is sentinel  # не подменили
