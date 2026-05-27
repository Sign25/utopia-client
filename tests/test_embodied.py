"""TZ Body Migration Phase 1: unit-тесты embodied WS клиента + EmbodiedOrganism.

Проверяем:
  - msgpack roundtrip Phase 1 payload (схема ТЗ §2.2 минимум)
  - ping/pong filter — server keepalive обрабатывается, не пробрасывается в callback
  - observation latency tracking — round-trip по cid
  - reconnect exponential backoff (mock websockets.connect)
  - throttle emit_alive_owned (не чаще period_sec)
  - schema валидация payload — все обязательные поля присутствуют

Стиль mock-based как `test_diagnostics_push.py` — не требует реального VPS.
"""
from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

msgpack = pytest.importorskip("msgpack")

from utopia_client.embodied import (  # noqa: E402
    PROTOCOL_VERSION,
    EmbodiedOrganism,
)
from utopia_client.embodied_ws import EmbodiedWSClient  # noqa: E402


# ── msgpack roundtrip ──────────────────────────────────────────────────────


def test_msgpack_roundtrip_phase1_payload():
    """Phase 1 payload pack→unpack → идентичен (msgpack 1.1.2)."""
    payload = {
        "protocol_version": 1,
        "cid": "c1234abc",
        "pos": [10, 20],
        "alive": True,
        "world_tick": 18_500_000,
        "ts_client": 1234567890.123,
    }
    packed = msgpack.packb(payload)
    unpacked = msgpack.unpackb(packed, raw=False)
    assert unpacked == payload


def test_msgpack_observation_echo_unpackable():
    """Echo от P40 (формат `_build_observation` в embodied_pusher.py) парсится."""
    obs_packed = msgpack.packb({
        "world_tick": 12345,
        "echo": True,
        "cid": "c-test",
        "pos": [10, 20],
        "protocol_version": 1,
    })
    unpacked = msgpack.unpackb(obs_packed, raw=False)
    assert unpacked["echo"] is True
    assert unpacked["cid"] == "c-test"
    assert unpacked["world_tick"] == 12345


# ── EmbodiedWSClient: ping/pong filter ─────────────────────────────────────


def _run(coro):
    """Helper: запустить coroutine в свежем event loop'е (без deprecated get_event_loop)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_ping_from_server_triggers_pong():
    """Server шлёт `{"type":"ping"}` → клиент отправляет `{"type":"pong"}`."""
    client = EmbodiedWSClient("https://test.local", "fake_token")
    client._ws = AsyncMock()
    ping_raw = msgpack.packb({"type": "ping"})
    _run(client._handle_msg(ping_raw))
    assert client.pings_handled == 1
    expected_pong = msgpack.packb({"type": "pong"})
    client._ws.send.assert_awaited_with(expected_pong)


def test_ping_not_forwarded_to_observation_callback():
    """Server keepalive ping НЕ доходит до on_observation callback."""
    cb = MagicMock()
    client = EmbodiedWSClient("https://test.local", "fake_token",
                              on_observation=cb)
    client._ws = AsyncMock()
    ping_raw = msgpack.packb({"type": "ping"})
    _run(client._handle_msg(ping_raw))
    cb.assert_not_called()


def test_non_dict_msg_silently_ignored():
    """Если broker прислал не-dict — клиент не падает, не вызывает callback."""
    cb = MagicMock()
    client = EmbodiedWSClient("https://test.local", "fake_token",
                              on_observation=cb)
    raw = msgpack.packb([1, 2, 3])  # list, not dict
    _run(client._handle_msg(raw))
    cb.assert_not_called()


def test_corrupted_msg_increments_errors():
    """Битый msgpack → errors_total++, no crash."""
    client = EmbodiedWSClient("https://test.local", "fake_token")
    _run(client._handle_msg(b"\xff\xff\xff\xff garbage"))
    assert client.errors_total >= 1


# ── EmbodiedWSClient: latency tracking ─────────────────────────────────────


def test_observation_latency_tracked_by_cid():
    """Echo с тем же cid → latency_ms записан в окно."""
    client = EmbodiedWSClient("https://test.local", "fake_token")
    cid = "c-roundtrip"
    client._sent_at[cid] = time.monotonic() - 0.080  # 80мс назад
    obs_raw = msgpack.packb({
        "cid": cid, "echo": True, "world_tick": 100,
    })
    _run(client._handle_msg(obs_raw))
    assert client.observations_received == 1
    assert len(client._latencies_ms) == 1
    lat = client._latencies_ms[0]
    assert 50 < lat < 200  # ~80мс ± jitter


def test_observation_without_matching_cid_no_latency():
    """Echo с cid, которого мы не отправляли → no latency sample."""
    client = EmbodiedWSClient("https://test.local", "fake_token")
    obs_raw = msgpack.packb({"cid": "unknown-cid", "echo": True})
    _run(client._handle_msg(obs_raw))
    assert client.observations_received == 1
    assert len(client._latencies_ms) == 0


def test_stats_returns_latency_aggregates_when_samples():
    """stats() возвращает mean/p50/p95/max когда есть samples."""
    client = EmbodiedWSClient("https://test.local", "fake_token")
    client._latencies_ms.extend([
        50.0, 80.0, 100.0, 120.0, 150.0,
        200.0, 250.0, 300.0, 400.0, 500.0,
    ])
    s = client.stats()
    assert "latency_mean_ms" in s
    assert s["latency_mean_ms"] == 215.0
    assert "latency_p50_ms" in s
    assert "latency_p95_ms" in s
    assert s["latency_max_ms"] == 500.0
    assert s["latency_samples"] == 10


def test_stats_no_latency_keys_when_no_samples():
    """stats() без latency-полей если ни одного round-trip ещё не было."""
    client = EmbodiedWSClient("https://test.local", "fake_token")
    s = client.stats()
    assert "latency_mean_ms" not in s
    assert s["latency_samples"] == 0


# ── EmbodiedWSClient: send + URL ───────────────────────────────────────────


def test_url_constructed_correctly():
    """URL должен быть wss://host/ws/embodied/client?token=<token>."""
    client = EmbodiedWSClient(
        "https://divisci.com", "abc123token",
    )
    assert client.url == "wss://divisci.com/ws/embodied/client?token=abc123token"
    # http:// → тоже wss
    client2 = EmbodiedWSClient("http://divisci.com/", "t")
    assert client2.url == "wss://divisci.com/ws/embodied/client?token=t"


def test_send_state_records_sent_at():
    """send_state с cid → запоминает ts в _sent_at для latency tracking."""
    client = EmbodiedWSClient("https://test.local", "fake_token")
    client._ws = AsyncMock()
    payload = {"cid": "c-1", "pos": [0, 0], "alive": True}
    _run(client._send_state_async(payload))
    assert "c-1" in client._sent_at
    assert client.states_sent == 1


def test_send_state_packs_msgpack():
    """send_state шлёт packed msgpack через ws.send."""
    client = EmbodiedWSClient("https://test.local", "fake_token")
    client._ws = AsyncMock()
    payload = {"cid": "c-1", "pos": [5, 5], "alive": True}
    _run(client._send_state_async(payload))
    sent_bytes = client._ws.send.call_args[0][0]
    assert isinstance(sent_bytes, bytes)
    # roundtrip
    decoded = msgpack.unpackb(sent_bytes, raw=False)
    assert decoded == payload


# ── EmbodiedOrganism: build_state + emit ───────────────────────────────────


def test_build_state_phase1_schema():
    """Payload содержит все обязательные Phase 1 поля по ТЗ §2.2 минимум."""
    compute = MagicMock()
    compute._obs_pos_cache = {"c-test": (10, 20)}
    ws = MagicMock()
    ws.on_observation = None
    eo = EmbodiedOrganism(compute, ws)
    payload = eo._build_state("c-test", world_tick=42)
    assert payload["protocol_version"] == PROTOCOL_VERSION
    assert payload["cid"] == "c-test"
    assert payload["pos"] == [10, 20]
    assert payload["alive"] is True
    assert payload["world_tick"] == 42
    assert "ts_client" in payload
    assert isinstance(payload["ts_client"], float)


def test_build_state_no_pos_cache_fallback_zero():
    """Если _obs_pos_cache нет / cid не в кеше — pos = [0, 0]."""
    compute = MagicMock(spec=[])  # no attributes
    ws = MagicMock()
    ws.on_observation = None
    eo = EmbodiedOrganism(compute, ws)
    payload = eo._build_state("missing-cid", world_tick=1)
    assert payload["pos"] == [0, 0]


def test_emit_alive_owned_sends_for_each_organism():
    """emit_alive_owned обходит compute.organisms, шлёт всем."""
    compute = MagicMock()
    compute.organisms = {"c1": object(), "c2": object(), "c3": object()}
    compute._obs_pos_cache = {}
    ws = MagicMock()
    ws.on_observation = None
    ws.send_state = MagicMock(return_value=True)
    eo = EmbodiedOrganism(compute, ws)
    n = eo.emit_alive_owned(world_tick=10, period_sec=0.01)
    assert n == 3
    assert ws.send_state.call_count == 3


def test_emit_alive_owned_throttle_within_period():
    """Повторный вызов в течение period_sec — throttle, 0 отправок."""
    compute = MagicMock()
    compute.organisms = {"c1": object(), "c2": object()}
    compute._obs_pos_cache = {}
    ws = MagicMock()
    ws.on_observation = None
    ws.send_state = MagicMock(return_value=True)
    eo = EmbodiedOrganism(compute, ws)
    n1 = eo.emit_alive_owned(world_tick=10, period_sec=1.0)
    assert n1 == 2
    # сразу же — throttled
    n2 = eo.emit_alive_owned(world_tick=11, period_sec=1.0)
    assert n2 == 0


def test_emit_alive_owned_empty_organisms_returns_zero():
    """Если organisms пуст — возвращает 0."""
    compute = MagicMock()
    compute.organisms = {}
    compute._obs_pos_cache = {}
    ws = MagicMock()
    ws.on_observation = None
    eo = EmbodiedOrganism(compute, ws)
    n = eo.emit_alive_owned(world_tick=10, period_sec=0.01)
    assert n == 0


# ── EmbodiedOrganism: observation callback ─────────────────────────────────


def test_on_observation_echo_increments_counter():
    """Echo observation → observations_with_echo += 1, world_tick обновлён."""
    compute = MagicMock()
    compute._obs_pos_cache = {}
    ws = MagicMock()
    ws.on_observation = None
    eo = EmbodiedOrganism(compute, ws)
    eo._on_observation({"cid": "c1", "echo": True, "world_tick": 12345})
    assert eo.observations_with_echo == 1
    assert eo.last_world_tick_from_p40 == 12345


def test_on_observation_no_echo_field_does_not_increment():
    """Observation без поля echo=True — НЕ инкрементит echo counter."""
    compute = MagicMock()
    compute._obs_pos_cache = {}
    ws = MagicMock()
    ws.on_observation = None
    eo = EmbodiedOrganism(compute, ws)
    eo._on_observation({"cid": "c1", "world_tick": 100})  # no echo field
    assert eo.observations_with_echo == 0


def test_on_observation_missing_world_tick_keeps_previous():
    """Если world_tick=0 или отсутствует — last_world_tick_from_p40 не сбрасывается."""
    compute = MagicMock()
    compute._obs_pos_cache = {}
    ws = MagicMock()
    ws.on_observation = None
    eo = EmbodiedOrganism(compute, ws)
    eo.last_world_tick_from_p40 = 999
    eo._on_observation({"cid": "c1", "echo": True, "world_tick": 0})
    # world_tick=0 не пишет
    assert eo.last_world_tick_from_p40 == 999


# ── EmbodiedOrganism: stats integration ────────────────────────────────────


def test_stats_includes_ws_substats_and_counters():
    """stats(): flat top-level + nested ws-блок для debugging.

    Convention (для VPS-side observability, Фрай review 27.05.2026):
    latency_* и основные счётчики на верхнем уровне `embodied.*`,
    granular ws-блок остаётся вложенным.
    """
    compute = MagicMock()
    compute._obs_pos_cache = {}
    ws = EmbodiedWSClient("https://test.local", "fake_token")
    eo = EmbodiedOrganism(compute, ws)
    s = eo.stats()
    # Flat top-level — convention для Хьюберта VPS-side
    assert "connected" in s
    assert "states_sent" in s
    assert "observations_received" in s
    assert "errors_total" in s
    # Этот слой
    assert s["observations_with_echo"] == 0
    assert s["last_world_tick_from_p40"] == 0
    # Granular ws-блок ещё доступен
    assert "ws" in s
    assert "url" in s["ws"]


def test_stats_flat_latency_when_samples_present():
    """С samples в EmbodiedWSClient — latency_mean_ms / latency_p95_ms
    появляются на верхнем уровне `embodied.*` (не только в ws-блоке)."""
    compute = MagicMock()
    compute._obs_pos_cache = {}
    ws = EmbodiedWSClient("https://test.local", "fake_token")
    ws._latencies_ms.extend([50.0, 100.0, 150.0, 200.0, 250.0])
    eo = EmbodiedOrganism(compute, ws)
    s = eo.stats()
    assert "latency_mean_ms" in s
    assert s["latency_mean_ms"] == 150.0
    assert "latency_p95_ms" in s
    assert "latency_max_ms" in s
    assert s["latency_samples"] == 5


def test_stats_flat_no_latency_keys_when_no_samples():
    """Без samples — latency_* отсутствуют на верхнем уровне (по contract
    EmbodiedWSClient.stats() который их не включает при пустом окне)."""
    compute = MagicMock()
    compute._obs_pos_cache = {}
    ws = EmbodiedWSClient("https://test.local", "fake_token")
    eo = EmbodiedOrganism(compute, ws)
    s = eo.stats()
    assert "latency_mean_ms" not in s
    assert s["latency_samples"] == 0


# ── Reconnect backoff (mock websockets.connect) ────────────────────────────


def test_reconnect_backoff_progression():
    """Backoff: 1s → 2s → 4s → ... → 30s (verified через локальную копию constants)."""
    # Это smoke-проверка постоянной — реальное reconnect-loop тестируется
    # интеграционно (с реальным сервером невыжно для unit).
    from utopia_client.embodied_ws import (
        RECONNECT_INITIAL_SEC, RECONNECT_MAX_SEC,
    )
    assert RECONNECT_INITIAL_SEC == 1.0
    assert RECONNECT_MAX_SEC == 30.0
    # progression simulation
    b = RECONNECT_INITIAL_SEC
    seq = [b]
    while b < RECONNECT_MAX_SEC:
        b = min(b * 2, RECONNECT_MAX_SEC)
        seq.append(b)
    assert seq == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0]
