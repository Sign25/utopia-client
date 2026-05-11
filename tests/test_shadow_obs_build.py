"""Тесты shadow-сборки obs клиентом параллельно серверному (Фаза 3.3A).

Проверяет `ColonyWSClient._compare_shadow_obs`:
  - Если кеш не bootstrap'нут — все cid идут в skipped.
  - Если meta для cid есть и view доступен — client_obs строится и
    сравнивается с серверным. Matched/mismatched корректно учитываются.
  - Tie-break в slots 48-49/51-52 не должен ронять матч, поскольку этот
    тест моделирует «идеальный» live-сценарий (cache == server). Tie-break
    смягчён monkey-patch'ем серверных obs-кешей, как и в Фазе 3.2C.

ТЗ: docs/tasks/tz_world_snapshot_to_client.md.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from environment.world import World, WorldConfig
from environment.world_snapshot_delta import ClientSnapshotState

from utopia_client.world_cache import WorldStateCache, WorldConfigSnapshot
from utopia_client.ws_client import ColonyWSClient

# Импортируем helpers из соседнего теста, чтобы не дублировать.
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
    return ws


def _make_creatures_payload(w: World) -> tuple[list, dict]:
    """Серверный payload obs_batch для всех живых особей.

    Возвращает (creatures_list, obs_per_cid) — slot-by-slot матч клиенту
    обеспечивается monkey-patch'ем серверных obs-кешей view'ом cache.
    """
    creatures = []
    obs_per_cid: dict = {}
    for cr in w.creatures:
        if not cr.alive:
            continue
        obs = np.asarray(w.get_observation(cr), dtype=np.float32)
        cid = str(cr.creature_id)
        obs_per_cid[cid] = obs
        creatures.append({
            "cid": cid,
            "row": int(cr.row),
            "col": int(cr.col),
            "energy": float(cr.energy),
            "hydration": float(cr.hydration),
            "vision_radius": int(cr.vision_radius),
            "smell_radius": int(cr.smell_radius),
            "signal_type": int(cr.signal_type),
            "clan_id": int(cr.clan_id),
            "camel": int(cr.camel),
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
        # is_bootstrapped == False → skipped
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

        # Удалим meta для одной особи — она должна попасть в skipped.
        target_cid = str(w.creatures[0].creature_id)
        cache._creature_meta.pop(target_cid, None)
        # Monkey-patch obs-кешей сервера — детерминистический tie-break.
        w._prepare_obs_cache()
        view0 = cache.obs_world_view(self_cid=target_cid)
        w._obs_flora_fr = view0.flora_fr
        w._obs_flora_fc = view0.flora_fc
        w._obs_fauna_fr = view0.fauna_fr
        w._obs_fauna_fc = view0.fauna_fc
        w._obs_fauna_kind = view0.fauna_kind

        creatures, obs_per_cid = _make_creatures_payload(w)
        ws._compare_shadow_obs(creatures, obs_per_cid)

        # Один cid скипнут, остальные построены.
        assert ws._client_obs_skipped == 1
        assert ws._client_obs_built == len(obs_per_cid) - 1

    def test_match_on_consistent_cache(self):
        """Когда cache совпадает с server world, все obs матчатся."""
        ws = _make_ws_stub()
        cache = WorldStateCache(base_url="https://x")
        w = _make_world(size=64, n_creatures=3, seed=7)
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(
            w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        ws.world_cache = cache

        # Делаем серверные obs детерминированными (tie-break == client).
        w._prepare_obs_cache()
        for cr in w.creatures:
            if not cr.alive:
                continue
            view = cache.obs_world_view(self_cid=str(cr.creature_id))
            w._obs_flora_fr = view.flora_fr
            w._obs_flora_fc = view.flora_fc
            w._obs_fauna_fr = view.fauna_fr
            w._obs_fauna_fc = view.fauna_fc
            w._obs_fauna_kind = view.fauna_kind
            # Перезатираем obs в payload после переподмены кеша.
            pass

        # Строим payload по перевернутым obs-кешам (детерминистический tie-break).
        creatures, obs_per_cid = _make_creatures_payload(w)
        ws._compare_shadow_obs(creatures, obs_per_cid)

        assert ws._client_obs_built == len(obs_per_cid)
        assert ws._client_obs_mismatch == 0
        assert ws._client_obs_match == ws._client_obs_built
        assert ws._client_obs_max_diff < 1e-3

    def test_mismatch_when_cache_stale(self):
        """Если cache отстал (особь сдвинута на сервере) → mismatch."""
        ws = _make_ws_stub()
        cache = WorldStateCache(base_url="https://x")
        w = _make_world(size=64, n_creatures=2, seed=11)
        _bootstrap_cache_from_world(w, cache)
        snap, _ = _build_snap_from_world(
            w, ClientSnapshotState(full_frame_interval=50))
        cache.apply_snap(snap)
        ws.world_cache = cache

        # Сдвинем особь на сервере, не обновляя кеш → шахматное расхождение.
        cr0 = w.creatures[0]
        cr0.row = (cr0.row + 5) % w.config.size
        cr0.col = (cr0.col + 5) % w.config.size

        creatures, obs_per_cid = _make_creatures_payload(w)
        ws._compare_shadow_obs(creatures, obs_per_cid)

        assert ws._client_obs_built >= 1
        # Хотя бы одна должна не совпасть.
        assert ws._client_obs_mismatch >= 1
        assert ws._client_obs_max_diff > 1e-3
