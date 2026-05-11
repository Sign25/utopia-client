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

import numpy as np
import requests

from environment.observation_client import ObsWorldView
from environment.world_snapshot_delta import (
    ClientSnapshotState,
    apply_delta_reference,
)

logger = logging.getLogger("utopia_client.world_cache")


@dataclass(frozen=True)
class WorldConfigSnapshot:
    """Статичные параметры Мира, полученные из /api/world/bootstrap.

    `signal_decay` и `night_vision_penalty` добавлены в 1.4.9 для
    клиентской сборки obs (Фаза 3). Старые серверы их не присылают —
    подставляются дефолты из `environment.world.WorldConfig`.
    """

    size: int
    smell_radius: int
    max_energy: float
    max_hydration: float
    full_frame_interval: int
    signal_decay: int = 55
    night_vision_penalty: int = 4


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
        # Метаданные особей из snap.creatures[] — нужны obs_world_view.
        # creature_meta[cid] = (clan_id, signal_type)
        self._creature_meta: dict[str, tuple[int, int]] = {}
        # Время суток из snap.world.is_night (для night_vision_penalty).
        self._is_night: bool = False
        # Лёгкий кеш numpy-массива terrain (uint8 size×size).
        self._terrain_np: Optional[np.ndarray] = None
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

    @property
    def is_night(self) -> bool:
        return self._is_night

    @property
    def creature_meta(self) -> dict[str, tuple[int, int]]:
        """`{cid: (clan_id, signal_type)}` — обновляется из snap.creatures[]."""
        return self._creature_meta

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
            signal_decay=int(world.get("signal_decay", 55)),
            night_vision_penalty=int(world.get("night_vision_penalty", 4)),
        )
        self._terrain = gzip.decompress(base64.b64decode(data["terrain_gz_b64"]))
        self._biomes = gzip.decompress(base64.b64decode(data["biomes_gz_b64"]))
        self._bootstrap_tick = int(data.get("tick", 0))
        # Пересоберём numpy-вью terrain'а при следующем obs_world_view().
        self._terrain_np = None

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

        world_block = snap.get("world") or {}
        tick = int(world_block.get("tick", 0))
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

        # is_night из snap.world (меняется во времени).
        if "is_night" in world_block:
            self._is_night = bool(world_block["is_night"])

        # clan_id/signal_type — из snap.creatures[] (Фаза 3 obs).
        # creatures_delta содержит только id/x/y, метаданные полным списком.
        creatures_list = snap.get("creatures") or []
        if creatures_list:
            new_meta: dict[str, tuple[int, int]] = {}
            for c in creatures_list:
                cid = c.get("id")
                if cid is None:
                    continue
                new_meta[str(cid)] = (int(c.get("clan", 0)), int(c.get("sig", 0)))
            self._creature_meta = new_meta

        self.snaps_applied += 1
        if flora_delta.get("full"):
            self.full_frames_received += 1
        return True

    # ─────────────────────────── obs view ────────────────────────────

    def obs_world_view(self, *, self_cid: Optional[str] = None) -> ObsWorldView:
        """Собирает `ObsWorldView` для клиентского `build_observation` (Фаза 3).

        Аргументы:
            self_cid: ID особи, для которой строится obs. Если задан, она
                будет исключена из `creature_clan_by_pos` (но сохранится в
                `ally_positions` — серверный `get_observation` так и делает).

        Поля:
            - terrain — np.uint8[size, size], кешируется на жизнь bootstrap.
            - flora_fr/fc — np.int32 из `state.flora` (kind игнорируется).
            - fauna_fr/fc/kind — np.int32 из `state.fauna`.
            - fauna_by_pos — dict[(row, col), kind].
            - signals — dict[(r, c, ch), (sig_tick, sig_type)] (state.signals).
            - ally_positions — set[(row, col)] всех известных существ.
            - creature_clan_by_pos — dict[(row, col), clan_id], без self_cid.

        Размер ~O(N_flora + N_fauna + N_signals + N_creatures). На P40:
        6700 flora + 100 fauna + 21 creatures ~ 0.5 мс.
        """
        if self._config is None:
            raise RuntimeError("WorldStateCache: bootstrap() ещё не вызван")
        cfg = self._config
        size = int(cfg.size)

        # terrain — кешируем uint8 ndarray, пересобираем только при bootstrap'е.
        if self._terrain_np is None:
            if len(self._terrain) != size * size:
                raise RuntimeError(
                    f"terrain length mismatch: {len(self._terrain)} vs {size*size}"
                )
            self._terrain_np = np.frombuffer(
                self._terrain, dtype=np.uint8,
            ).reshape(size, size)
        terrain = self._terrain_np

        # Flora positions — kind игнорируем (obs использует только наличие).
        flora_set = self._state.flora
        if flora_set:
            fr_list = [r for (r, _c, _k) in flora_set]
            fc_list = [c for (_r, c, _k) in flora_set]
            flora_fr = np.asarray(fr_list, dtype=np.int32)
            flora_fc = np.asarray(fc_list, dtype=np.int32)
        else:
            flora_fr = np.empty(0, dtype=np.int32)
            flora_fc = np.empty(0, dtype=np.int32)

        # Fauna positions + kind + fast-lookup dict.
        fauna_state = self._state.fauna
        if fauna_state:
            fr_list = []
            fc_list = []
            kind_list = []
            fauna_by_pos: dict[tuple[int, int], int] = {}
            for (r, c, k, _hp) in fauna_state.values():
                fr_list.append(r)
                fc_list.append(c)
                kind_list.append(k)
                fauna_by_pos[(int(r), int(c))] = int(k)
            fauna_fr = np.asarray(fr_list, dtype=np.int32)
            fauna_fc = np.asarray(fc_list, dtype=np.int32)
            fauna_kind = np.asarray(kind_list, dtype=np.int32)
        else:
            fauna_fr = np.empty(0, dtype=np.int32)
            fauna_fc = np.empty(0, dtype=np.int32)
            fauna_kind = np.empty(0, dtype=np.int32)
            fauna_by_pos = {}

        # Сородичи: ally_positions включает всех живых (со self_cid),
        # creature_clan_by_pos — исключает self_cid (как на сервере при
        # переборе `for other in self.creatures: if other is creature: continue`).
        ally_positions: set[tuple[int, int]] = set()
        clan_by_pos: dict[tuple[int, int], int] = {}
        # snap.creatures даёт (x, y) — col, row. У state.creature_pos то же.
        for cid, (x, y) in self._state.creature_pos.items():
            pos = (int(y), int(x))  # (row, col)
            ally_positions.add(pos)
            if self_cid is not None and cid == self_cid:
                continue
            meta = self._creature_meta.get(cid)
            if meta is None:
                continue
            clan_by_pos[pos] = meta[0]

        return ObsWorldView(
            size=size,
            max_energy=float(cfg.max_energy),
            max_hydration=float(cfg.max_hydration),
            signal_decay=int(cfg.signal_decay),
            night_vision_penalty=int(cfg.night_vision_penalty),
            is_night=bool(self._is_night),
            tick=int(self._state.last_tick),
            terrain=terrain,
            flora_fr=flora_fr,
            flora_fc=flora_fc,
            fauna_fr=fauna_fr,
            fauna_fc=fauna_fc,
            fauna_kind=fauna_kind,
            fauna_by_pos=fauna_by_pos,
            signals=dict(self._state.signals),
            ally_positions=ally_positions,
            creature_clan_by_pos=clan_by_pos,
        )

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
