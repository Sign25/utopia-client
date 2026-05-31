"""Тесты shadow-сборки obs клиентом параллельно серверному (Фаза 3.3A).

Проверяет `ColonyWSClient._compare_shadow_obs`:
  - Если кеш не bootstrap'нут — все cid идут в skipped.
  - Если meta для cid есть и view доступен — client_obs строится и
    сравнивается с серверным. Matched/mismatched корректно учитываются.
  - На live-сценарии (cache == server) earth_compat obs совпадает.

ТЗ: docs/tasks/tz_world_snapshot_to_client.md.
"""

from __future__ import annotations

import numpy as np

from environment.world import World, WorldConfig, _ObsBatchContext
from environment.world_snapshot_delta import ClientSnapshotState

from utopia_client.world_cache import WorldStateCache
from utopia_client.ws_client import ColonyWSClient

from tests.test_world_cache_obs import (
    _bootstrap_cache_from_world,
    _build_snap_from_world,
)


def _make_ws_stub() -> ColonyWSClient:
    """ColonyWSClient без I/O — голый объект для вызова _compare_shadow_obs."""
    ws = ColonyWSClient.__new__(ColonyWSClient)
    ws.world_cache = None
    ws._client_obs_built = 0
    ws._client_obs_skipped = 0
    ws._client_obs_match = 0
    ws._client_obs_mismatch = 0
    ws._client_obs_max_diff = 0.0
    ws._client_obs_local_built = 0
    return ws


def _strip_obs_from_payload(creatures: list) -> list:
    """Phase 3.3B: симулирует серверный payload без поля 'obs'."""
    return [{k: v for k, v in c.items() if k != "obs"} for c in creatures]


def _make_creatures_payload(w: World) -> tuple[list, dict]:
    """Серверный payload obs_batch для всех живых особей (earth_compat)."""
    creatures = []
    obs_per_cid: dict = {}
    ctx = _ObsBatchContext(w)
    for cr in w.creatures:
        if not cr.alive:
            continue
        obs = np.asarray(
            w.get_observation_earth_compat(cr, _ctx=ctx),
            dtype=np.float32,
        )
        cid = str(cr.creature_id)
        obs_per_cid[cid] = obs
        creatures.append({
            "cid": cid,
            "row": int(cr.row),
            "col": int(cr.col),
            "energy": float(cr.energy),
            "steps_taken": int(cr.steps_taken),
            "obs": obs.tolist(),
        })
    return creatures, obs_per_cid


def _make_world(size: int = 64, n_creatures: int = 3, seed: int = 1) -> World:
    cfg = WorldConfig(size=size, seed=seed)
    w = World(cfg)
    while len(w.creatures) < n_creatures:
        w.add_creature()
    return w


class TestCompareShadowObs:

    def test_no_cache_increments_skipped(self):
        ws = _make_ws_stub()
        w = _make_world()
        creatures, obs_per_cid = _make_creatures_payload(w)
        ws._compare_shadow_obs(creatures, obs_per_cid)
        assert ws._client_obs_skipped == len(obs_per_cid)
        assert ws._client_obs_built == 0
        assert ws._client_obs_match == 0
        assert ws._client_obs_mismatch == 0

    def test_cache_not_bootstrapped_skips(self):
        ws = _make_ws_stub()
        ws.world_cache = WorldStateCache(base_url="https://x")
        w = _make_world()
        creatures, obs_per_cid = _make_creatures_payload(w)
        ws._compare_shadow_obs(creatures, obs_per_cid)
        assert ws._client_obs_skipped == len(obs_per_cid)
        assert ws._client_obs_built == 0

    def test_meta_missing_skips_unmatched_cid(self):
        """cid без meta пропускается, обычные cid идут в build."""
        ws = _make_ws_stub()
        cache = WorldStateCache(base_url="https://x")
        w = _make_world(size=64, n_creatures=3, seed=5)
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(
            w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        ws.world_cache = cache

        target_cid = str(w.creatures[0].creature_id)
        cache._creature_meta.pop(target_cid, None)

        creatures, obs_per_cid = _make_creatures_payload(w)
        ws._compare_shadow_obs(creatures, obs_per_cid)

        assert ws._client_obs_skipped == 1
        assert ws._client_obs_built == len(obs_per_cid) - 1

    def test_match_on_consistent_cache(self):
        """Когда cache совпадает с server world, earth_compat obs совпадает."""
        ws = _make_ws_stub()
        cache = WorldStateCache(base_url="https://x")
        w = _make_world(size=64, n_creatures=3, seed=7)
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(
            w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        ws.world_cache = cache

        creatures, obs_per_cid = _make_creatures_payload(w)
        ws._compare_shadow_obs(creatures, obs_per_cid)

        assert ws._client_obs_built == len(obs_per_cid)
        assert ws._client_obs_mismatch == 0
        assert ws._client_obs_match == ws._client_obs_built
        assert ws._client_obs_max_diff < 1e-3

    def test_skip_when_tick_desync(self):
        """Phase 3.3 fix2: если |cache_tick - server_world_tick| > 1 → skip.

        В live snap-поток (~500мс) и obs_batch (~50мс) не синхронизированы;
        кеш может уйти на десятки/сотни тиков вперёд obs_batch. Без гейта
        prey/pred slots 56-61 дают огромные diffs из-за реальных шагов
        фауны между тиками — это не баг builder'а. Сравниваем только при
        |tick_diff| ≤ 1.
        """
        ws = _make_ws_stub()
        cache = WorldStateCache(base_url="https://x")
        w = _make_world(size=64, n_creatures=3, seed=17)
        w.tick = 500  # cache.last_tick = 500 после apply_snap
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(
            w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        ws.world_cache = cache
        assert cache.last_tick == 500

        creatures, obs_per_cid = _make_creatures_payload(w)
        # cache.last_tick=500, server отстал на 100 тиков
        server_world_tick = cache.last_tick - 100
        ws._compare_shadow_obs(creatures, obs_per_cid,
                                 server_world_tick=server_world_tick)

        assert ws._client_obs_built == 0
        assert ws._client_obs_match == 0
        assert ws._client_obs_mismatch == 0
        assert ws._client_obs_skipped == len(obs_per_cid)
        skip_info = ws._client_obs_last_tick_skip
        assert skip_info.get("tick_diff") == 100
        assert skip_info.get("cache_tick") == cache.last_tick
        assert skip_info.get("server_tick") == server_world_tick

    def test_tick_sync_exact_match_compares(self):
        """tick_diff == 0 — единственный валидный случай для сравнения.
        Любая рассинхронизация (даже 1 тик) даёт реальные расхождения
        на быстрой фауне (хищник speed=3 за тик)."""
        ws = _make_ws_stub()
        cache = WorldStateCache(base_url="https://x")
        w = _make_world(size=64, n_creatures=3, seed=19)
        w.tick = 500
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(
            w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        ws.world_cache = cache

        creatures, obs_per_cid = _make_creatures_payload(w)
        ws._compare_shadow_obs(creatures, obs_per_cid,
                                 server_world_tick=cache.last_tick)

        assert ws._client_obs_built == len(obs_per_cid)
        assert ws._client_obs_match == ws._client_obs_built

    def test_tick_sync_one_tick_lag_skips(self):
        """tick_diff == 1 — теперь тоже skip (ужесточили гейт)."""
        ws = _make_ws_stub()
        cache = WorldStateCache(base_url="https://x")
        w = _make_world(size=64, n_creatures=3, seed=23)
        w.tick = 500
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(
            w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        ws.world_cache = cache

        creatures, obs_per_cid = _make_creatures_payload(w)
        ws._compare_shadow_obs(creatures, obs_per_cid,
                                 server_world_tick=cache.last_tick + 1)

        assert ws._client_obs_built == 0
        assert ws._client_obs_skipped == len(obs_per_cid)
        assert ws._client_obs_last_tick_skip.get("tick_diff") == -1

    def test_mismatch_when_cache_stale(self):
        """Если cache отстал (особь пропала из creature_pos) → mismatch.

        Ставим cr0 и cr1 рядом на PLAIN-тайлах: сервер видит cr1 как
        poison-соседа cr0. После apply_snap удаляем cr1 из creature_pos
        кеша — клиент думает, что соседа нет → slot[3] = 0 vs 1.
        """
        from environment.world import Tile
        ws = _make_ws_stub()
        cache = WorldStateCache(base_url="https://x")
        w = _make_world(size=64, n_creatures=2, seed=11)

        cr0, cr1 = w.creatures[0], w.creatures[1]
        sz = w.config.size
        pos = None
        for r in range(2, sz - 2):
            for col in range(2, sz - 2):
                if (w.terrain[r][col] == Tile.PLAIN
                        and w.terrain[(r - 1) % sz][col] == Tile.PLAIN):
                    pos = (r, col)
                    break
            if pos:
                break
        assert pos is not None, "не нашли пару PLAIN-тайлов"
        r, col = pos
        cr0.row, cr0.col = r, col
        cr1.row, cr1.col = (r - 1) % sz, col

        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(
            w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        ws.world_cache = cache

        cache._state.creature_pos.pop(str(cr1.creature_id), None)

        creatures, obs_per_cid = _make_creatures_payload(w)
        ws._compare_shadow_obs(creatures, obs_per_cid)

        assert ws._client_obs_built >= 1
        assert ws._client_obs_mismatch >= 1
        assert ws._client_obs_max_diff > 1e-3


class TestCollectObsBatchPhase33B:
    """Phase 3.3B: клиент строит obs локально, если сервер не прислал."""

    def test_server_sends_obs_passthrough(self):
        """Старый сервер: obs есть в payload → используем его, локальный
        builder НЕ дёргается."""
        ws = _make_ws_stub()
        cache = WorldStateCache(base_url="https://x")
        w = _make_world(size=64, n_creatures=3, seed=31)
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(
            w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        ws.world_cache = cache

        creatures, _ = _make_creatures_payload(w)
        obs, events, intero, _rates = ws._collect_obs_batch(creatures)

        assert len(obs) == 3
        assert ws._client_obs_local_built == 0
        for cid in obs:
            assert obs[cid].shape == (64,)
            assert cid in events

    def test_server_omits_obs_local_builder_kicks_in(self):
        """Phase 3.3B: сервер не шлёт obs для owned → клиент строит локально.
        Результат сравнивается с тем, что построил бы сервер — должен совпасть."""
        ws = _make_ws_stub()
        cache = WorldStateCache(base_url="https://x")
        w = _make_world(size=64, n_creatures=3, seed=37)
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(
            w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        ws.world_cache = cache

        creatures, server_obs = _make_creatures_payload(w)
        creatures_no_obs = _strip_obs_from_payload(creatures)
        obs, events, intero, _rates = ws._collect_obs_batch(creatures_no_obs)

        assert len(obs) == 3
        assert ws._client_obs_local_built == 3
        # Локальный builder должен дать тот же obs, что и серверный.
        for cid, server_arr in server_obs.items():
            assert cid in obs
            diff = float(np.max(np.abs(obs[cid] - server_arr)))
            assert diff < 1e-3, f"cid={cid} diff={diff}"

    def test_no_cache_no_obs_skips_all(self):
        """Кеш не bootstrap'нут, obs нет → все cid отброшены."""
        ws = _make_ws_stub()
        w = _make_world(size=64, n_creatures=3, seed=41)
        creatures, _ = _make_creatures_payload(w)
        creatures_no_obs = _strip_obs_from_payload(creatures)
        obs, events, intero, _rates = ws._collect_obs_batch(creatures_no_obs)

        assert obs == {}
        assert events == {}
        assert ws._client_obs_local_built == 0
