"""Phase G.3 (04.05.2026) — empty-world watchdog.

Покрытие:
  G.3.1  watchdog молчит, пока n_alive_owned > 0
  G.3.2  watchdog молчит при welcome.mode="awaiting_seed_pack"
  G.3.3  watchdog молчит, пока since_alive < grace
  G.3.4  watchdog шлёт respawn_owned_request после grace
  G.3.5  cooldown — повторный запрос только после retry_sec
  G.3.6  colony_reset снимает awaiting_seed_pack и cooldown
  G.3.7  stats обновляет last_owned_alive_ts при n>0
  G.3.8  welcome логирует mode и обновляет last_owned_alive_ts при n>0
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class _CapturingWS:
    def __init__(self):
        self.sent: list = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _client():
    from utopia_client.ws_client import ColonyWSClient
    c = ColonyWSClient(server="https://x", token="t",
                       colony_name="home", client_version="test")
    c.connected = True
    c._ws = _CapturingWS()
    return c


# ── G.3.1: пока n_alive_owned > 0 — нет запросов ───────────────────────


def test_g3_no_request_when_alive():
    async def _run():
        c = _client()
        c.n_alive_owned = 3
        c.last_owned_alive_ts = time.time() - 9999.0
        sent = await c._maybe_request_respawn(time.time() - 9999.0)
        assert sent is False
        assert c._ws.sent == []
    asyncio.run(_run())


# ── G.3.2: awaiting_seed_pack — не дублируем ───────────────────────────


def test_g3_no_request_when_awaiting_seed_pack():
    async def _run():
        c = _client()
        c.n_alive_owned = 0
        c._welcome_mode = "awaiting_seed_pack"
        sent = await c._maybe_request_respawn(time.time() - 9999.0)
        assert sent is False
        assert c._ws.sent == []
    asyncio.run(_run())


# ── G.3.3: до grace — молчание ─────────────────────────────────────────


def test_g3_silent_within_grace():
    from utopia_client.ws_client import EMPTY_WORLD_GRACE_SEC

    async def _run():
        c = _client()
        c.n_alive_owned = 0
        sent = await c._maybe_request_respawn(time.time() - 1.0)
        assert sent is False
        sent = await c._maybe_request_respawn(
            time.time() - (EMPTY_WORLD_GRACE_SEC - 1.0))
        assert sent is False
        assert c._ws.sent == []
    asyncio.run(_run())


# ── G.3.4: после grace — респавн запрос ────────────────────────────────


def test_g3_respawn_request_after_grace():
    from utopia_client.ws_client import EMPTY_WORLD_GRACE_SEC

    async def _run():
        c = _client()
        c.n_alive_owned = 0
        c._welcome_mode = ""
        sent = await c._maybe_request_respawn(
            time.time() - (EMPTY_WORLD_GRACE_SEC + 5.0))
        assert sent is True
        assert len(c._ws.sent) == 1
        msg = c._ws.sent[0]
        assert msg["type"] == "respawn_owned_request"
        assert msg["colony_name"] == "home"
        assert msg["mode"] == "auto"
        assert msg["n"] == 5
        assert isinstance(msg["ts"], int)
        assert c._last_respawn_request_ts > 0.0
    asyncio.run(_run())


# ── G.3.5: cooldown ────────────────────────────────────────────────────


def test_g3_cooldown_blocks_second_request():
    from utopia_client.ws_client import (EMPTY_WORLD_GRACE_SEC,
                                          EMPTY_WORLD_RETRY_SEC)

    async def _run():
        c = _client()
        c.n_alive_owned = 0
        ct = time.time() - (EMPTY_WORLD_GRACE_SEC + 5.0)
        assert await c._maybe_request_respawn(ct) is True
        assert len(c._ws.sent) == 1
        # сразу после первого — cooldown
        assert await c._maybe_request_respawn(ct) is False
        assert len(c._ws.sent) == 1
        # эмулируем что cooldown истёк
        c._last_respawn_request_ts = time.time() - (EMPTY_WORLD_RETRY_SEC + 5.0)
        assert await c._maybe_request_respawn(ct) is True
        assert len(c._ws.sent) == 2
    asyncio.run(_run())


# ── G.3.6: colony_reset снимает флаги ──────────────────────────────────


def test_g3_colony_reset_clears_flags():
    async def _run():
        c = _client()
        c._welcome_mode = "awaiting_seed_pack"
        c._last_respawn_request_ts = time.time()
        c.last_owned_alive_ts = 0.0
        c._handle_colony_reset({"reason": "auto_respawn"})
        assert c._welcome_mode == ""
        assert c._last_respawn_request_ts == 0.0
        assert c.last_owned_alive_ts > 0.0
    asyncio.run(_run())


# ── G.3.7: stats обновляет last_owned_alive_ts ────────────────────────


def test_g3_stats_updates_last_owned_alive_ts():
    async def _run():
        c = _client()
        c.last_owned_alive_ts = 0.0
        await c._handle({"type": "stats", "n_alive_owned": 4,
                          "world_tick": 100})
        assert c.n_alive_owned == 4
        assert c.last_owned_alive_ts > 0.0
        prev = c.last_owned_alive_ts
        await c._handle({"type": "stats", "n_alive_owned": 0,
                          "world_tick": 101})
        assert c.n_alive_owned == 0
        assert c.last_owned_alive_ts == prev
    asyncio.run(_run())


# ── G.3.8: welcome.mode + n_creatures ─────────────────────────────────


def test_g3_welcome_records_mode_and_alive_ts():
    async def _run():
        c = _client()
        await c._handle({
            "type": "welcome", "world_tick": 10, "n_creatures": 3,
            "mode": "awaiting_seed_pack", "server_time": 1,
        })
        assert c._welcome_mode == "awaiting_seed_pack"
        assert c.n_alive_owned == 3
        assert c.last_owned_alive_ts > 0.0
        c._welcome_mode = "stale"
        c.last_owned_alive_ts = 0.0
        await c._handle({
            "type": "welcome", "world_tick": 11, "n_creatures": 1,
            "server_time": 1,
        })
        assert c._welcome_mode == ""
        assert c.last_owned_alive_ts > 0.0
    asyncio.run(_run())


# ── G.3.9: respawn_owned_ack корректно логируется (no-op handler) ─────


def test_g3_respawn_owned_ack_no_crash():
    async def _run():
        c = _client()
        await c._handle({
            "type": "respawn_owned_ack",
            "colony_name": "home",
            "scheduled": True,
        })
    asyncio.run(_run())
