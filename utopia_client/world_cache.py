"""Клиентский кеш Мира — bootstrap + применение дельт (Фаза 2).

Клиент дёргает `/api/world/bootstrap` при подключении один раз (terrain,
biomes, конфиг), затем на каждом snap'е извлекает блоки `flora_delta`/
`fauna_delta`/`signals_delta`/`creatures_delta` и обновляет внутренний
`ClientSnapshotState` из `neurocore.environment.world_snapshot_delta`.

В Фазе 2 кеш только **хранит** актуальную модель Мира — obs из него ещё
не собирается (это Фаза 3).

ТЗ сервера: docs/tasks/tz_world_snapshot_to_client.md.
"""

from __future__ import annotations

import base64
import gzip
import logging
from dataclasses import dataclass
from typing import Optional

import requests

from environment.world_snapshot_delta import (
    ClientSnapshotState,
    apply_delta_reference,
)

logger = logging.getLogger("utopia_client.world_cache")


@dataclass(frozen=True)
class WorldConfigSnapshot:
    """Статичные параметры Мира, полученные из /api/world/bootstrap."""

    size: int
    smell_radius: int
    max_energy: float
    max_hydration: float
    full_frame_interval: int


class WorldStateCache:
    """Хранит локальную модель Мира на стороне клиента.

    Usage:
        cache = WorldStateCache(base_url="https://divisci.com")
        cache.bootstrap()  # один раз при подключении
        cache.apply_snap(snap_dict)  # на каждый принятый snap

    После apply_snap доступны:
        cache.flora      — set[(row, col, kind)]
        cache.fauna      — dict[fauna_id, (row, col, kind, hp_norm)]
        cache.signals    — dict[(row, col, channel), sig_tick]
        cache.creature_pos — dict[cid, (x, y)]
        cache.terrain    — bytes (size*size, row-major uint8)
        cache.biomes     — bytes (size*size)
        cache.config     — WorldConfigSnapshot
        cache.last_tick  — int
    """

    def __init__(self, base_url: str, *, request_timeout: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_timeout = float(request_timeout)
        # Внутренний state применения дельт.
        self._state = ClientSnapshotState()
        self._config: Optional[WorldConfigSnapshot] = None
        self._terrain: bytes = b""
        self._biomes: bytes = b""
        self._bootstrap_tick: int = -1
        # Счётчики наблюдаемости.
        self.snaps_applied: int = 0
        self.full_frames_received: int = 0
        self.last_apply_error: str = ""

    # ─────────────────────────── public views ──────────────────────────

    @property
    def config(self) -> Optional[WorldConfigSnapshot]:
        return self._config

    @property
    def terrain(self) -> bytes:
        return self._terrain

    @property
    def biomes(self) -> bytes:
        return self._biomes

    @property
    def flora(self) -> set[tuple[int, int, int]]:
        return self._state.flora

    @property
    def fauna(self) -> dict[int, tuple[int, int, int, float]]:
        return self._state.fauna

    @property
    def signals(self) -> dict[tuple[int, int, int], int]:
        return self._state.signals

    @property
    def creature_pos(self) -> dict[str, tuple[int, int]]:
        return self._state.creature_pos

    @property
    def last_tick(self) -> int:
        return self._state.last_tick

    @property
    def is_bootstrapped(self) -> bool:
        return self._config is not None

    # ─────────────────────────── bootstrap ─────────────────────────────

    def bootstrap(self) -> None:
        """Дёргает /api/world/bootstrap, распаковывает terrain и конфиг.

        Идемпотентен: повторный вызов перезатирает кеш (например, после
        переподключения или смены мира на сервере).
        """
        url = f"{self.base_url}/api/world/bootstrap"
        resp = requests.get(url, timeout=self.request_timeout)
        resp.raise_for_status()
        data = resp.json()

        encoding = data.get("encoding", "")
        if encoding != "gzip+base64":
            raise ValueError(f"unsupported bootstrap encoding: {encoding!r}")

        world = data.get("world") or {}
        self._config = WorldConfigSnapshot(
            size=int(world["size"]),
            smell_radius=int(world["smell_radius"]),
            max_energy=float(world["max_energy"]),
            max_hydration=float(world["max_hydration"]),
            full_frame_interval=int(world["full_frame_interval"]),
        )
        self._terrain = gzip.decompress(base64.b64decode(data["terrain_gz_b64"]))
        self._biomes = gzip.decompress(base64.b64decode(data["biomes_gz_b64"]))
        self._bootstrap_tick = int(data.get("tick", 0))

        # Reset state с правильным full_frame_interval.
        self._state = ClientSnapshotState(
            full_frame_interval=self._config.full_frame_interval,
        )
        logger.info(
            "world bootstrap: size=%d terrain=%d B biomes=%d B tick=%d",
            self._config.size, len(self._terrain), len(self._biomes),
            self._bootstrap_tick,
        )

    # ─────────────────────────── apply ────────────────────────────────

    def apply_snap(self, snap: dict) -> bool:
        """Применить delta-блоки из snap к локальной модели.

        Возвращает True, если snap содержал дельты и был применён, False иначе
        (например, сервер не включил `WORLD_SNAPSHOT_DELTAS_ENABLED`).
        """
        if not isinstance(snap, dict):
            return False
        flora_delta = snap.get("flora_delta")
        if flora_delta is None:
            return False

        tick = int((snap.get("world") or {}).get("tick", 0))
        try:
            self._state = apply_delta_reference(
                state=self._state,
                tick=tick,
                flora_delta=flora_delta,
                fauna_delta=snap.get("fauna_delta") or {},
                signals_delta=snap.get("signals_delta") or {},
                creatures_delta=snap.get("creatures_delta") or {},
            )
        except Exception as e:
            self.last_apply_error = f"{type(e).__name__}: {e}"
            logger.warning("apply_snap failed at tick=%d: %s", tick, e)
            return False

        self.snaps_applied += 1
        if flora_delta.get("full"):
            self.full_frames_received += 1
        return True

    # ─────────────────────────── observability ────────────────────────

    def stats(self) -> dict:
        return {
            "is_bootstrapped": self.is_bootstrapped,
            "last_tick": self.last_tick,
            "snaps_applied": self.snaps_applied,
            "full_frames_received": self.full_frames_received,
            "n_flora": len(self.flora),
            "n_fauna": len(self.fauna),
            "n_signals": len(self.signals),
            "n_creatures": len(self.creature_pos),
            "last_apply_error": self.last_apply_error,
        }
