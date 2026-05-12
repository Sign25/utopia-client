"""Тесты obs_world_view + equivalence client build_observation ↔ server
get_observation_earth_compat (Фаза 3.3).

Сценарий:
  1. Поднимаем синтетический `World` с creatures/flora/fauna.
  2. Серверным обогащением snap получаем delta-блоки + creatures-list.
  3. Применяем snap к `WorldStateCache`.
  4. `cache.obs_world_view(self_cid=...)` → ObsWorldView.
  5. `build_observation(creature_view, world_view)` на клиенте.
  6. Сравниваем с серверным `get_observation_earth_compat(creature, _ctx=...)`.

ТЗ: docs/tasks/tz_world_snapshot_to_client.md.
"""

from __future__ import annotations

import numpy as np
import pytest

from environment.observation_client import (
    ObsCreatureView,
    build_observation,
)
from environment.world import World, WorldConfig, _ObsBatchContext
from environment.world_snapshot_delta import (
    ClientSnapshotState,
    build_snapshot_delta,
)

from utopia_client.world_cache import WorldConfigSnapshot, WorldStateCache


# ─────────────────────────── helpers ────────────────────────────


def _bootstrap_cache_from_world(w: World, cache: WorldStateCache) -> None:
    """Имитирует bootstrap из реального мира (terrain/biomes/config)."""
    size = int(w.config.size)
    terrain_bytes = bytes(
        int(w.terrain[r][c]) & 0xFF
        for r in range(size) for c in range(size)
    )
    biomes_bytes = bytes(
        int(w.biomes[r][c]) & 0xFF
        for r in range(size) for c in range(size)
    )
    cache._config = WorldConfigSnapshot(
        size=size,
        smell_radius=int(w.config.smell_radius),
        max_energy=float(w.config.max_energy),
        max_hydration=float(w.config.max_hydration),
        full_frame_interval=50,
        signal_decay=int(w.config.signal_decay),
        night_vision_penalty=int(w.config.night_vision_penalty),
    )
    cache._terrain = terrain_bytes
    cache._biomes = biomes_bytes
    cache._terrain_np = None
    cache._state = ClientSnapshotState(full_frame_interval=50)


def _build_snap_from_world(w: World, server_state: ClientSnapshotState):
    """Серверная сборка snap (дельты + creatures с метаданными) из World."""
    flora_list = [
        (int(r), int(c), int(fl.kind))
        for (r, c), fl in w.flora.items() if fl.alive
    ]
    fauna_list = [
        (int(f.id), int(f.col), int(f.row), int(f.kind),
         float(f.hp) / max(1.0, float(f.max_hp)))
        for f in w.fauna if f.alive
    ]
    signals_list = [
        (int(r), int(c), int(st), int(stype), int(ch))
        for (r, c), (st, stype, ch) in w.signals.items()
    ]
    creatures_snap = [
        {"id": cr.creature_id, "x": int(cr.col), "y": int(cr.row),
         "clan": int(cr.clan_id), "sig": int(cr.signal_type)}
        for cr in w.creatures if cr.alive
    ]
    delta_block, new_state = build_snapshot_delta(
        state=server_state, current_tick=int(w.tick),
        flora_list=flora_list, fauna_list=fauna_list,
        signals_list=signals_list, creatures=creatures_snap,
    )
    snap = {
        "world": {"tick": int(w.tick), "is_night": bool(w.is_night)},
        "creatures": creatures_snap,
    }
    snap.update(delta_block)
    return snap, new_state


def _make_creature_view(creature) -> ObsCreatureView:
    return ObsCreatureView(
        row=int(creature.row),
        col=int(creature.col),
        energy=float(creature.energy),
        steps_taken=int(creature.steps_taken),
    )


def _server_obs(w: World, creature) -> np.ndarray:
    """Серверный earth_compat через `_ObsBatchContext` — путь продакшна."""
    ctx = _ObsBatchContext(w)
    return np.asarray(
        w.get_observation_earth_compat(creature, _ctx=ctx),
        dtype=np.float32,
    )


def _make_world(size: int = 96, n_creatures: int = 6, seed: int = 42) -> World:
    cfg = WorldConfig(size=size, seed=seed)
    w = World(cfg)
    while len(w.creatures) < n_creatures:
        w.add_creature()
    return w


# ─────────────────────────── tests ──────────────────────────────


class TestObsWorldView:
    """obs_world_view собирает корректные индексы из cache state."""

    def test_returns_obs_world_view(self):
        w = _make_world(size=64, n_creatures=3, seed=1)
        cache = WorldStateCache(base_url="https://x")
        _bootstrap_cache_from_world(w, cache)
        server_state = ClientSnapshotState(full_frame_interval=50)
        snap, _ = _build_snap_from_world(w, server_state)
        assert cache.apply_snap(snap)
        view = cache.obs_world_view()
        assert view.size == 64
        assert view.terrain.shape == (64, 64)
        assert view.terrain.dtype == np.uint8
        assert view.smell_radius == int(w.config.smell_radius)

    def test_flora_indices_populated(self):
        w = _make_world(size=64, n_creatures=2, seed=2)
        cache = WorldStateCache(base_url="https://x")
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        view = cache.obs_world_view()
        n_flora_server = sum(1 for fl in w.flora.values() if fl.alive)
        assert view.flora_fr.size == n_flora_server
        assert view.flora_fc.size == n_flora_server

    def test_creature_meta_extracted_from_creatures_list(self):
        """meta (clan/sig_type) сохраняется как opaque metadata.

        earth_compat layout их не использует, но meta остаётся в cache —
        вдруг пригодится для будущих фич (e.g. UI-агрегация по кланам).
        """
        w = _make_world(size=64, n_creatures=4, seed=3)
        for i, c in enumerate(w.creatures):
            c.clan_id = 100 + i
            c.signal_type = 200 + i
        cache = WorldStateCache(base_url="https://x")
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        for i, c in enumerate(w.creatures):
            assert cache.creature_meta[c.creature_id] == (100 + i, 200 + i)

    def test_self_excluded_from_creature_pos(self):
        """creature_pos в view не содержит self_cid."""
        w = _make_world(size=64, n_creatures=3, seed=4)
        cache = WorldStateCache(base_url="https://x")
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        self_cid = w.creatures[0].creature_id
        view = cache.obs_world_view(self_cid=self_cid)
        self_pos = (int(w.creatures[0].row), int(w.creatures[0].col))
        assert self_pos not in view.creature_pos
        # Другие особи должны быть.
        other_pos = (int(w.creatures[1].row), int(w.creatures[1].col))
        if other_pos != self_pos:
            assert other_pos in view.creature_pos

    def test_is_night_from_snap(self):
        """cache.is_night обновляется из snap.world.is_night."""
        cache = WorldStateCache(base_url="https://x")
        w = _make_world(size=64, n_creatures=2, seed=5)
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(w, ClientSnapshotState(full_frame_interval=50))
        snap["world"]["is_night"] = True
        cache.apply_snap(snap)
        assert cache.is_night is True

    def test_obs_world_view_before_bootstrap_raises(self):
        cache = WorldStateCache(base_url="https://x")
        with pytest.raises(RuntimeError, match="bootstrap"):
            cache.obs_world_view()


class TestBuildObservationEquivalence:
    """build_observation на клиенте == get_observation_earth_compat на сервере."""

    def _check_one_creature(self, w: World, creature, cache: WorldStateCache):
        view = cache.obs_world_view(self_cid=creature.creature_id)
        server_obs = _server_obs(w, creature)
        cv = _make_creature_view(creature)
        client_obs = build_observation(cv, view)
        diff = np.abs(server_obs - client_obs)
        return server_obs, client_obs, diff

    def test_equivalence_fresh_world(self):
        w = _make_world(size=96, n_creatures=6, seed=11)
        cache = WorldStateCache(base_url="https://x")
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        for c in w.creatures:
            if not c.alive:
                continue
            server_obs, client_obs, diff = self._check_one_creature(w, c, cache)
            assert diff.max() < 1e-6, (
                f"creature {c.creature_id}: max diff = {diff.max():.6f}, "
                f"slots = {np.where(diff > 1e-6)[0].tolist()}"
            )

    def test_equivalence_after_ticks(self):
        """После нескольких тиков мира — флора/фауна сдвинулись."""
        w = _make_world(size=96, n_creatures=6, seed=23)
        for _ in range(30):
            w.tick_world()
        cache = WorldStateCache(base_url="https://x")
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        for c in w.creatures:
            if not c.alive:
                continue
            _, _, diff = self._check_one_creature(w, c, cache)
            assert diff.max() < 1e-6, (
                f"creature {c.creature_id}: slots {np.where(diff > 1e-6)[0].tolist()}"
            )

    def test_equivalence_torus_wrap(self):
        w = _make_world(size=64, n_creatures=4, seed=7)
        c = next(x for x in w.creatures if x.alive)
        c.row, c.col = 1, 1
        cache = WorldStateCache(base_url="https://x")
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        _, _, diff = self._check_one_creature(w, c, cache)
        assert diff.max() < 1e-6

    def test_steps_taken_passes_through(self):
        """steps_taken из payload корректно попадает в slot 35."""
        w = _make_world(size=64, n_creatures=2, seed=13)
        c = w.creatures[0]
        c.steps_taken = 1000
        cache = WorldStateCache(base_url="https://x")
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        _, client_obs, diff = self._check_one_creature(w, c, cache)
        assert abs(float(client_obs[35]) - 0.2) < 1e-6  # 1000/5000
        assert diff.max() < 1e-6
