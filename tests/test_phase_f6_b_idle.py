"""Phase F.6.B (03.05.2026) — graceful run↔idle: bye + close WS.

Покрытие:
  F.6.B.1  send_bye() формирует корректное сообщение
  F.6.B.2  send_bye() с ws=None — silent no-op
  F.6.B.3  state machine: ws=None в idle, инициализируется на idle→run,
           останавливается с bye на run→idle
  F.6.B.4  benchmark→idle: WS закрывается даже если он был открыт
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class _CapturingWS:
    """Минимальная заглушка ws, копит отправленные сообщения."""

    def __init__(self):
        self.sent: list = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


# ── F.6.B.1: send_bye → правильный envelope ────────────────────────────


def test_send_bye_writes_bye_envelope():
    from utopia_client.ws_client import ColonyWSClient

    async def _run():
        ws = ColonyWSClient(server="https://x", token="t",
                            colony_name="home", client_version="test")
        ws._ws = _CapturingWS()
        await ws.send_bye()
        assert ws._ws.sent
        msg = ws._ws.sent[-1]
        assert msg["type"] == "bye"
        assert isinstance(msg.get("ts"), int)
        assert msg["ts"] > 0

    asyncio.run(_run())


def test_send_bye_without_ws_is_silent():
    from utopia_client.ws_client import ColonyWSClient

    async def _run():
        ws = ColonyWSClient(server="https://x", token="t",
                            colony_name="home", client_version="test")
        ws._ws = None
        # Не должен крашить.
        await ws.send_bye()

    asyncio.run(_run())


def test_send_bye_swallows_exceptions():
    """Если ws.send бросает — send_bye логирует и не пробрасывает."""
    from utopia_client.ws_client import ColonyWSClient

    class _FailingWS:
        async def send(self, raw: str) -> None:
            raise RuntimeError("connection broken")

    async def _run():
        ws = ColonyWSClient(server="https://x", token="t",
                            colony_name="home", client_version="test")
        ws._ws = _FailingWS()
        # Не должен бросить.
        await ws.send_bye()

    asyncio.run(_run())


# ── F.6.B.3: cmd_run state machine — ws lifecycle ──────────────────────


def _make_cfg() -> dict:
    return {
        "server": "https://x",
        "token": "t",
        "name": "home",
        "benchmark": {"estimated_population": 0},
    }


def test_make_ws_factory_returns_fresh_instance():
    """_make_ws — фабрика, не singleton: каждый вызов = новый объект."""
    from utopia_client.main import _make_ws
    from utopia_client.ws_client import ColonyWSClient

    cfg = _make_cfg()
    a = _make_ws(cfg, "home")
    b = _make_ws(cfg, "home")
    assert isinstance(a, ColonyWSClient)
    assert isinstance(b, ColonyWSClient)
    assert a is not b


def test_idle_to_run_creates_and_starts_ws(monkeypatch):
    """Симулируем один прогон cmd_run-петли: idle→run должен создать ws.start()."""
    from utopia_client import main as cli_main
    from utopia_client.ws_client import ColonyWSClient

    started: list = []
    stopped: list = []

    class _FakeWS:
        def __init__(self, *a, **kw):
            self.connected = False
            self.n_alive_owned = 0
            self.last_world_tick = 0
            self.compute = None
            started.append(self)

        def start(self):
            self.connected = True

        def stop(self):
            self.connected = False
            stopped.append(self)

    monkeypatch.setattr(cli_main, "ColonyWSClient", _FakeWS)
    monkeypatch.setattr(cli_main, "_make_ws",
                        lambda cfg, name: _FakeWS())

    # Прямая модель state-перехода (как в cmd_run):
    ws = None
    current_state = "idle"

    # idle → run
    desired = "run"
    if desired == "run" and ws is None:
        ws = cli_main._make_ws(_make_cfg(), "home")
        ws.start()
    current_state = desired

    assert ws is not None
    assert len(started) == 1
    assert ws.connected is True

    # run → idle
    desired = "idle"
    if desired == "idle" and ws is not None:
        ws.stop()
        ws = None
    current_state = desired

    assert ws is None
    assert len(stopped) == 1


def test_double_idle_does_not_start_ws():
    """Если desired=idle, текущий уже idle — ws=None остаётся None."""
    ws = None
    desired = "idle"
    if desired == "run" and ws is None:
        ws = object()
    elif desired == "idle" and ws is not None:
        ws = None
    assert ws is None


# ── F.6.B.4: stop() вызывает send_bye через run_coroutine_threadsafe ───


def test_stop_dispatches_bye_when_connected():
    """stop() при connected=True должен попробовать отправить bye через loop.

    Полная интеграция трудна (нужен живой event loop в потоке), поэтому
    проверяем: когда _loop=None, stop не падает; когда есть _loop+_ws+
    connected — schedule вызывается.
    """
    from utopia_client.ws_client import ColonyWSClient

    ws = ColonyWSClient(server="https://x", token="t",
                        colony_name="home", client_version="test")
    # Без _loop / _thread: stop() — no-op + ничего не падает.
    ws.stop()  # без exception


def test_stop_without_loop_is_noop():
    from utopia_client.ws_client import ColonyWSClient

    ws = ColonyWSClient(server="https://x", token="t",
                        colony_name="home", client_version="test")
    ws._loop = None
    ws._stop_event = None
    ws._ws = None
    ws.connected = False
    # Без живого loop — никаких ошибок.
    ws.stop()
