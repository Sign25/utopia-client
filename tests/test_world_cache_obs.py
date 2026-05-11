"""Тесты obs_world_view + equivalence client build_observation ↔ server get_observation.

Фаза 3.2 переноса obs-сборки на клиент. Сценарий:
  1. На сервере поднимаем синтетический `World` с известными creatures/flora/fauna.
  2. Серверным обогащением snap (через `_enrich_snap_with_deltas`-эквивалент)
     получаем delta-блоки + creatures-list.
  3. Применяем snap к `WorldStateCache`.
  4. `cache.obs_world_view(self_cid=...)` → ObsWorldView.
  5. `build_observation(creature_view, world_view)` на клиенте.
  6. Сравниваем со серверным `world.get_observation(creature)` слот-в-слот.

ТЗ: docs/tasks/tz_world_snapshot_to_client.md.
"""

from __future__ import annotations

import base64
import gzip

import numpy as np
import pytest

from environment.observation_client import (
    ObsCreatureView,
    build_observation,
)
from environment.world import SignalChannel, World, WorldConfig
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
    # Flora list [(r, c, kind), ...]
    flora_list = [
        (int(r), int(c), int(fl.kind))
        for (r, c), fl in w.flora.items() if fl.alive
    ]
    # Fauna list — формат сервера: (id, col, row, kind, hp_norm).
    fauna_list = [
        (int(f.id), int(f.col), int(f.row), int(f.kind),
         float(f.hp) / max(1.0, float(f.max_hp)))
        for f in w.fauna if f.alive
    ]
    # Signals list [(r, c, sig_tick, sig_type, channel)]
    signals_list = [
        (int(r), int(c), int(st), int(stype), int(ch))
        for (r, c), (st, stype, ch) in w.signals.items()
    ]
    # creatures: snap-формат (id, x=col, y=row, clan, sig).
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
        hydration=float(creature.hydration),
        vision_radius=int(creature.vision_radius),
        smell_radius=int(creature.smell_radius),
        signal_type=int(creature.signal_type),
        clan_id=int(creature.clan_id),
        camel=int(creature.camel),
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

    def test_flora_indices_populated(self):
        w = _make_world(size=64, n_creatures=2, seed=2)
        cache = WorldStateCache(base_url="https://x")
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        view = cache.obs_world_view()
        # Все живые flora-позиции должны быть в массивах.
        n_flora_server = sum(1 for fl in w.flora.values() if fl.alive)
        assert view.flora_fr.size == n_flora_server
        assert view.flora_fc.size == n_flora_server

    def test_creature_meta_extracted_from_creatures_list(self):
        w = _make_world(size=64, n_creatures=4, seed=3)
        # Назначим уникальные clan_id и signal_type.
        for i, c in enumerate(w.creatures):
            c.clan_id = 100 + i
            c.signal_type = 200 + i
        cache = WorldStateCache(base_url="https://x")
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        for i, c in enumerate(w.creatures):
            assert cache.creature_meta[c.creature_id] == (100 + i, 200 + i)

    def test_self_excluded_from_clan_by_pos(self):
        w = _make_world(size=64, n_creatures=3, seed=4)
        cache = WorldStateCache(base_url="https://x")
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        self_cid = w.creatures[0].creature_id
        view = cache.obs_world_view(self_cid=self_cid)
        self_pos = (int(w.creatures[0].row), int(w.creatures[0].col))
        assert self_pos not in view.creature_clan_by_pos
        # ally_positions при этом ВКЛЮЧАЕТ self (как на сервере).
        assert self_pos in view.ally_positions

    def test_is_night_from_snap(self):
        """cache.is_night обновляется из snap.world.is_night."""
        cache = WorldStateCache(base_url="https://x")
        w = _make_world(size=64, n_creatures=2, seed=5)
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(w, ClientSnapshotState(full_frame_interval=50))
        # Подменяем поле в snap (для теста — обычно сервер сам выставляет).
        snap["world"]["is_night"] = True
        cache.apply_snap(snap)
        assert cache.is_night is True
        view = cache.obs_world_view()
        assert view.is_night is True

    def test_obs_world_view_before_bootstrap_raises(self):
        cache = WorldStateCache(base_url="https://x")
        with pytest.raises(RuntimeError, match="bootstrap"):
            cache.obs_world_view()


class TestBuildObservationEquivalence:
    """build_observation на клиенте == get_observation на сервере.

    Замечание о порядке: серверный `_obs_flora_fr/fc` строится из
    `dict.items()` (insertion order), а клиентский — из `set` после
    apply_delta_reference (порядок теряется). При равных distance
    `np.argmin` возвращает разный idx → разные `food_dir` (slots 48-49,
    51-52). Чтобы тесты не зависели от tie-break, выравниваем порядок
    на сервере с порядком клиента перед вызовом `get_observation`.
    """

    def _check_one_creature(self, w: World, creature, cache: WorldStateCache):
        # Сначала строим клиентский view (стабильный порядок) и подменяем
        # серверные obs-индексы на ровно те же массивы → tie-break
        # одинаков.
        view = cache.obs_world_view(self_cid=creature.creature_id)
        w._prepare_obs_cache()
        w._obs_flora_fr = view.flora_fr
        w._obs_flora_fc = view.flora_fc
        w._obs_fauna_fr = view.fauna_fr
        w._obs_fauna_fc = view.fauna_fc
        w._obs_fauna_kind = view.fauna_kind
        server_obs = np.asarray(w.get_observation(creature), dtype=np.float32)
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

    def test_equivalence_with_signals(self):
        w = _make_world(size=96, n_creatures=6, seed=23)
        w.signals[(10, 10)] = (w.tick, 1, SignalChannel.FOOD)
        w.signals[(15, 20)] = (w.tick, 2, SignalChannel.DANGER)
        w.signals[(50, 50)] = (w.tick, 0, SignalChannel.GENERIC)
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

    def test_equivalence_kin_proximity(self):
        w = _make_world(size=96, n_creatures=4, seed=3)
        c1, c2 = w.creatures[0], w.creatures[1]
        c1.clan_id = c2.clan_id = 42
        c1.row, c1.col = 30, 30
        c2.row, c2.col = 32, 31
        cache = WorldStateCache(base_url="https://x")
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        server_obs, client_obs, diff = self._check_one_creature(w, c1, cache)
        assert server_obs[62] > 0.0, "kin_proximity должен быть > 0"
        assert diff.max() < 1e-6

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

    def test_equivalence_mixed_signal_types(self):
        """Сигналы разных type'ов: свой ×2, чужой ×0.5."""
        w = _make_world(size=64, n_creatures=2, seed=9)
        c = w.creatures[0]
        c.row, c.col = 30, 30
        c.signal_type = 5
        w.signals.clear()
        w.signals[(25, 30)] = (w.tick, 5, SignalChannel.FOOD)
        w.signals[(35, 30)] = (w.tick, 99, SignalChannel.FOOD)
        cache = WorldStateCache(base_url="https://x")
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        server_obs, client_obs, diff = self._check_one_creature(w, c, cache)
        # slot 58 — sig_food_NS, ожидаем > 0 т.к. свой северный ×2 больше чужого южного ×0.5.
        assert server_obs[58] > 0.0
        assert diff.max() < 1e-6
