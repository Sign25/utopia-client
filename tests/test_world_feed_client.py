"""Тесты для WorldFeedClient (Фаза 2 транспорта — WS-подписчик /ws/feed).

Покрывают:
- URL: https → wss, http → ws, без префикса остаётся как есть.
- `_consume_message`: snapshot → cache.apply_snap, прочее игнорируется.
- bootstrap-loop вызывает cache.bootstrap (через мок requests).
- start/stop устойчив к отсутствию сервера (websockets connect упадёт → backoff).
"""

from __future__ import annotations

import base64
import gzip
import json
import time
from unittest.mock import MagicMock

import pytest

from utopia_client.world_cache import WorldStateCache
from utopia_client.world_feed_client import WorldFeedClient


# ─────────────────────────── URL handling ────────────────────────────────


def test_url_https_to_wss():
    cache = WorldStateCache(base_url="https://divisci.com")
    c = WorldFeedClient(server="https://divisci.com", cache=cache)
    assert c.url == "wss://divisci.com/ws/feed"


def test_url_http_to_ws():
    cache = WorldStateCache(base_url="http://localhost:8001")
    c = WorldFeedClient(server="http://localhost:8001", cache=cache)
    assert c.url == "ws://localhost:8001/ws/feed"


def test_url_bare_passthrough():
    cache = WorldStateCache(base_url="https://x")
    c = WorldFeedClient(server="wss://example.com:9000/", cache=cache)
    assert c.url == "wss://example.com:9000/ws/feed"


# ─────────────────────────── _consume_message ────────────────────────────


def _make_snapshot_payload():
    """Минимальный snap с delta-блоками: один full-frame с одной флорой."""
    from environment.world_snapshot_delta import (
        ClientSnapshotState, build_snapshot_delta,
    )
    server_state = ClientSnapshotState(full_frame_interval=50)
    delta_block, _ = build_snapshot_delta(
        state=server_state, current_tick=0,
        flora_list=[(3, 5, 1)], fauna_list=[],
        signals_list=[], creatures=[{"id": "a", "x": 1, "y": 2}],
    )
    snap = {"world": {"tick": 0}}
    snap.update(delta_block)
    return snap


def test_consume_snapshot_applies_to_cache():
    cache = WorldStateCache(base_url="https://x")
    c = WorldFeedClient(server="https://x", cache=cache)
    payload = _make_snapshot_payload()
    raw = json.dumps({"type": "snapshot", "payload": payload,
                      "ts": int(time.time() * 1000)})
    assert c._consume_message(raw) is True
    assert c.snapshots_received == 1
    assert (3, 5, 1) in cache.flora
    assert cache.creature_pos.get("a") == (1, 2)


def test_consume_ignores_unknown_type():
    cache = WorldStateCache(base_url="https://x")
    c = WorldFeedClient(server="https://x", cache=cache)
    raw = json.dumps({"type": "hello", "payload": {}})
    assert c._consume_message(raw) is False
    assert c.snapshots_received == 0


def test_consume_ignores_missing_payload():
    cache = WorldStateCache(base_url="https://x")
    c = WorldFeedClient(server="https://x", cache=cache)
    raw = json.dumps({"type": "snapshot"})
    assert c._consume_message(raw) is False
    assert c.snapshots_received == 0


def test_consume_ignores_bad_payload_shape():
    cache = WorldStateCache(base_url="https://x")
    c = WorldFeedClient(server="https://x", cache=cache)
    raw = json.dumps({"type": "snapshot", "payload": "not a dict"})
    assert c._consume_message(raw) is False
    assert c.snapshots_received == 0


def test_consume_ignores_invalid_json():
    cache = WorldStateCache(base_url="https://x")
    c = WorldFeedClient(server="https://x", cache=cache)
    assert c._consume_message("{not json") is False
    assert c.snapshots_received == 0


def test_consume_snapshot_without_delta_block_not_counted():
    """Сервер без WORLD_SNAPSHOT_DELTAS_ENABLED шлёт snap без delta-блоков —
    cache.apply_snap вернёт False, snapshots_received не растёт."""
    cache = WorldStateCache(base_url="https://x")
    c = WorldFeedClient(server="https://x", cache=cache)
    raw = json.dumps({"type": "snapshot",
                      "payload": {"world": {"tick": 0}, "flora": []}})
    assert c._consume_message(raw) is False
    assert c.snapshots_received == 0


def test_consume_skips_when_apply_raises(monkeypatch):
    """Если cache.apply_snap бросит — _consume_message глотает и возвращает False."""
    cache = WorldStateCache(base_url="https://x")
    c = WorldFeedClient(server="https://x", cache=cache)
    monkeypatch.setattr(cache, "apply_snap",
                        MagicMock(side_effect=RuntimeError("boom")))
    raw = json.dumps({"type": "snapshot", "payload": {"world": {"tick": 0}}})
    assert c._consume_message(raw) is False
    assert c.snapshots_received == 0


# ─────────────────────────── bootstrap_loop ──────────────────────────────


@pytest.mark.asyncio
async def test_bootstrap_loop_calls_cache_bootstrap(monkeypatch):
    """Если cache не bootstrapped — _bootstrap_loop дёргает HTTP один раз."""
    import asyncio

    size = 4
    raw_terrain = bytes(range(size * size))
    raw_biomes = bytes(reversed(range(size * size)))
    fake_response = {
        "world": {"size": size, "smell_radius": 10, "max_energy": 100.0,
                  "max_hydration": 100.0, "full_frame_interval": 50},
        "tick": 7,
        "terrain_gz_b64": base64.b64encode(gzip.compress(raw_terrain)).decode("ascii"),
        "biomes_gz_b64": base64.b64encode(gzip.compress(raw_biomes)).decode("ascii"),
        "encoding": "gzip+base64", "dtype": "uint8",
    }

    class _FakeResp:
        status_code = 200
        def json(self): return fake_response
        def raise_for_status(self): pass

    monkeypatch.setattr(
        "utopia_client.world_cache.requests.get",
        lambda url, timeout: _FakeResp(),
    )

    cache = WorldStateCache(base_url="https://example.com")
    c = WorldFeedClient(server="https://example.com", cache=cache)
    c._loop = asyncio.get_running_loop()
    c._stop_event = asyncio.Event()
    await c._bootstrap_loop()

    assert cache.is_bootstrapped
    assert cache.terrain == raw_terrain


@pytest.mark.asyncio
async def test_bootstrap_loop_noop_when_already_bootstrapped(monkeypatch):
    """Если cache уже bootstrapped — _bootstrap_loop не дёргает HTTP."""
    import asyncio

    cache = WorldStateCache(base_url="https://x")
    # Имитируем уже-bootstrapped state.
    from utopia_client.world_cache import WorldConfigSnapshot
    cache._config = WorldConfigSnapshot(
        size=8, smell_radius=10, max_energy=100.0,
        max_hydration=100.0, full_frame_interval=50)

    called = {"n": 0}
    def _fake_bootstrap():
        called["n"] += 1
    monkeypatch.setattr(cache, "bootstrap", _fake_bootstrap)

    c = WorldFeedClient(server="https://x", cache=cache)
    c._loop = asyncio.get_running_loop()
    c._stop_event = asyncio.Event()
    await c._bootstrap_loop()

    assert called["n"] == 0


# ─────────────────────────── lifecycle ───────────────────────────────────


def test_start_stop_without_server():
    """start() поднимает thread, stop() корректно гасит даже без сервера.
    Внутри bootstrap упадёт (нет HTTP), но клиент должен пережить и stop."""
    cache = WorldStateCache(base_url="http://127.0.0.1:1")  # порт-заглушка
    c = WorldFeedClient(server="http://127.0.0.1:1", cache=cache)
    c.start()
    time.sleep(0.1)
    c.stop()
    assert c._thread is None


def test_stats_includes_cache_stats():
    cache = WorldStateCache(base_url="https://x")
    c = WorldFeedClient(server="https://x", cache=cache)
    s = c.stats()
    assert s["url"] == "wss://x/ws/feed"
    assert s["snapshots_received"] == 0
    # Поля из cache.stats() прокинуты.
    assert "is_bootstrapped" in s
    assert "snaps_applied" in s
