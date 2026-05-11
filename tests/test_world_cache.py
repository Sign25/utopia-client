"""Тесты для WorldStateCache (Фаза 2 переноса obs на клиент).

Проверяют:
- bootstrap: распаковка terrain/biomes из gzip+base64, заполнение config
- apply_snap: применение дельт, full-frame override старого стейта
- roundtrip: сервер строит дельты → клиент применяет → состояния сходятся
- идемпотентность повторного apply с теми же дельтами

Серверная часть импортируется напрямую из neurocore (zip-зеркало 1.4.7+).
"""

from __future__ import annotations

import base64
import gzip

import pytest

# Импорт серверного модуля для генерации синтетических дельт.
from environment.world_snapshot_delta import (
    ClientSnapshotState,
    build_snapshot_delta,
)

from utopia_client.world_cache import WorldStateCache, WorldConfigSnapshot


# ─────────────────────────── bootstrap parsing ──────────────────────────


def test_bootstrap_decodes_terrain(monkeypatch):
    """WorldStateCache.bootstrap парсит gzip+base64 terrain из HTTP-ответа."""
    size = 8
    raw_terrain = bytes(range(size * size))  # 0..63
    raw_biomes = bytes(reversed(range(size * size)))

    fake_response = {
        "world": {
            "size": size,
            "smell_radius": 40,
            "max_energy": 1309.0,
            "max_hydration": 100.0,
            "full_frame_interval": 50,
        },
        "tick": 42,
        "terrain_gz_b64": base64.b64encode(gzip.compress(raw_terrain)).decode("ascii"),
        "biomes_gz_b64": base64.b64encode(gzip.compress(raw_biomes)).decode("ascii"),
        "encoding": "gzip+base64",
        "dtype": "uint8",
    }

    class _FakeResp:
        status_code = 200
        def json(self): return fake_response
        def raise_for_status(self): pass

    captured_url = []
    def _fake_get(url, timeout):
        captured_url.append(url)
        return _FakeResp()

    monkeypatch.setattr("utopia_client.world_cache.requests.get", _fake_get)

    cache = WorldStateCache(base_url="https://example.com")
    cache.bootstrap()

    assert captured_url == ["https://example.com/api/world/bootstrap"]
    assert cache.is_bootstrapped
    assert cache.terrain == raw_terrain
    assert cache.biomes == raw_biomes
    cfg = cache.config
    assert isinstance(cfg, WorldConfigSnapshot)
    assert cfg.size == size
    assert cfg.smell_radius == 40
    assert cfg.max_energy == 1309.0
    assert cfg.full_frame_interval == 50


def test_bootstrap_rejects_unknown_encoding(monkeypatch):
    class _FakeResp:
        status_code = 200
        def json(self): return {"encoding": "lz4+base64"}
        def raise_for_status(self): pass

    monkeypatch.setattr(
        "utopia_client.world_cache.requests.get",
        lambda url, timeout: _FakeResp(),
    )
    cache = WorldStateCache(base_url="https://x")
    with pytest.raises(ValueError, match="unsupported bootstrap encoding"):
        cache.bootstrap()


# ─────────────────────────── apply_snap basics ──────────────────────────


def _make_snap_with_deltas(server_state, tick, flora, fauna, signals, creatures):
    """Хелпер: серверная сборка snap с delta-блоками."""
    delta_block, new_state = build_snapshot_delta(
        state=server_state, current_tick=tick,
        flora_list=flora, fauna_list=fauna,
        signals_list=signals, creatures=creatures,
    )
    snap = {"world": {"tick": tick}}
    snap.update(delta_block)
    return snap, new_state


def test_apply_snap_no_delta_block_returns_false():
    cache = WorldStateCache(base_url="https://x")
    snap = {"world": {"tick": 0}, "flora": [], "fauna": []}  # старый формат
    assert cache.apply_snap(snap) is False
    assert cache.snaps_applied == 0


def test_apply_snap_first_call_is_full_frame():
    cache = WorldStateCache(base_url="https://x")
    server_state = ClientSnapshotState(full_frame_interval=50)
    snap, _ = _make_snap_with_deltas(
        server_state, tick=0,
        flora=[(0, 0, 1), (1, 1, 2)],
        fauna=[(10, 5, 5, 0, 1.0)],
        signals=[],
        creatures=[{"id": "a", "x": 3, "y": 4}],
    )
    assert cache.apply_snap(snap) is True
    assert cache.snaps_applied == 1
    assert cache.full_frames_received == 1
    assert cache.last_tick == 0
    assert (0, 0, 1) in cache.flora
    assert (1, 1, 2) in cache.flora
    assert 10 in cache.fauna
    assert cache.creature_pos.get("a") == (3, 4)


# ─────────────────────────── roundtrip ─────────────────────────────────


def test_roundtrip_server_to_client():
    """Сервер строит дельты, клиент применяет — состояния сходятся."""
    cache = WorldStateCache(base_url="https://x")
    server_state = ClientSnapshotState(full_frame_interval=50)

    sequence = [
        (0,
         [(0, 0, 1), (1, 1, 2)],
         [(10, 5, 5, 0, 1.0)],
         [(2, 2, 100, 0, 1)],
         [{"id": "a", "x": 3, "y": 4}]),
        (1,
         [(0, 0, 1)],
         [(10, 5, 6, 0, 0.5)],
         [(2, 2, 100, 0, 1), (5, 5, 110, 1, 2)],
         [{"id": "a", "x": 3, "y": 5}, {"id": "b", "x": 0, "y": 0}]),
        (2,
         [(0, 0, 1), (7, 7, 1)],
         [],
         [(5, 5, 110, 1, 2)],
         [{"id": "b", "x": 0, "y": 0}]),
    ]

    for (tick, flora, fauna, signals, creatures) in sequence:
        snap, server_state = _make_snap_with_deltas(
            server_state, tick, flora, fauna, signals, creatures)
        assert cache.apply_snap(snap)

        # Сходимость с серверным state.
        assert cache.flora == server_state.flora, f"tick={tick} flora"
        assert cache.fauna == server_state.fauna, f"tick={tick} fauna"
        assert cache.signals == server_state.signals, f"tick={tick} signals"
        assert cache.creature_pos == server_state.creature_pos, \
            f"tick={tick} creature_pos"

    assert cache.snaps_applied == 3
    assert cache.full_frames_received == 1  # только tick=0


def test_full_frame_resync_after_interval():
    """Когда сервер шлёт full_frame, клиент обнуляет старый стейт."""
    cache = WorldStateCache(base_url="https://x")
    # full_frame_interval=5 для быстрого триггера.
    server_state = ClientSnapshotState(full_frame_interval=5)

    # tick=0: full-frame.
    snap, server_state = _make_snap_with_deltas(
        server_state, 0, [(0, 0, 1)], [], [], [])
    cache.apply_snap(snap)
    assert (0, 0, 1) in cache.flora

    # tick=5: снова full-frame (другой набор флоры).
    snap, server_state = _make_snap_with_deltas(
        server_state, 5, [(9, 9, 9)], [], [], [])
    cache.apply_snap(snap)
    assert snap["flora_delta"]["full"] is True
    assert cache.flora == {(9, 9, 9)}  # старая (0,0,1) выкинута
    assert cache.full_frames_received == 2


# ─────────────────────────── stats ─────────────────────────────────────


def test_stats_counter():
    cache = WorldStateCache(base_url="https://x")
    server_state = ClientSnapshotState(full_frame_interval=50)
    snap, _ = _make_snap_with_deltas(
        server_state, 0, [(0, 0, 1)], [], [], [{"id": "a", "x": 1, "y": 2}])
    cache.apply_snap(snap)
    s = cache.stats()
    assert s["snaps_applied"] == 1
    assert s["full_frames_received"] == 1
    assert s["n_flora"] == 1
    assert s["n_creatures"] == 1
    assert s["last_tick"] == 0
    assert s["last_apply_error"] == ""


def test_apply_snap_bad_delta_records_error():
    """Сломанный delta-блок не падает наружу, записывается в last_apply_error."""
    cache = WorldStateCache(base_url="https://x")
    bad_snap = {
        "world": {"tick": 0},
        "flora_delta": "not a dict",  # type error inside apply
        "fauna_delta": {},
        "signals_delta": {},
        "creatures_delta": {},
    }
    ok = cache.apply_snap(bad_snap)
    assert ok is False
    assert cache.last_apply_error != ""
