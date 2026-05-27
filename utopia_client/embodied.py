"""EmbodiedOrganism — клиентский слой Embodied API (Body Migration Этап 3a).

Phase 1 (skeleton): композиция с `LocalColonyCompute`. Эмитит минимальный
публичный payload `{cid, pos, alive, world_tick, ts_client,
protocol_version}` через `EmbodiedWSClient`, принимает echo observation
от P40.

Phase 2+ (см. `docs/tasks/tz_body_migration.md` §4):
  - Phase 2: биохимия Z7 на клиенте → mental_break ↔ visible_action
  - Phase 3: 21 ткань + Hebbian → r_cell_form computed locally
  - Phase 4: NEAT + memory + reproduction
  - Phase 5: Адам-Зодчий onboarding
  - Phase 6: legacy P40 cleanup

Этот класс — **композиция**, не наследование: `LocalColonyCompute` уже
3700 строк, добавлять туда embodied-логику плохо для тестируемости.
Composition позволяет переиспользовать compute (cids, тики, биохимию
в Phase 2+) без раздувания.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from .embodied_ws import EmbodiedWSClient

if TYPE_CHECKING:
    from .local_compute import LocalColonyCompute

logger = logging.getLogger("utopia_client.embodied")

PROTOCOL_VERSION = 1
DEFAULT_EMIT_PERIOD_SEC = 1.0  # ТЗ §3.5: client-server exchange 5 TPS (Phase 1 — реже)


class EmbodiedOrganism:
    """Клиентский embodied-слой поверх `LocalColonyCompute`.

    Phase 1 минимум: эмитит `{cid, pos, alive, world_tick}` для каждого
    живого owned cid'а раз в `period_sec`, регистрирует callback на echo
    observation от P40. Latency tracking ведётся в `EmbodiedWSClient`.

    Использование:
        eo = EmbodiedOrganism(compute, ws_client)
        # в daemon loop, каждые ~1с:
        eo.emit_alive_owned(world_tick=ws.last_world_tick)
        # eo.stats() — для diagnostics push
    """

    def __init__(
        self,
        compute: "LocalColonyCompute",
        ws_client: EmbodiedWSClient,
    ) -> None:
        self.compute = compute
        self.ws = ws_client
        self.ws.on_observation = self._on_observation
        # Phase 1 счётчики
        self.last_world_tick_from_p40: int = 0
        self.observations_with_echo: int = 0
        self._last_emit_at: float = 0.0

    # ── state building ──────────────────────────────────────────────────

    def _build_state(self, cid: str, world_tick: int = 0) -> dict:
        """Phase 1 минимальный публичный payload по ТЗ §2.2.

        В Phase 1 шлём только то, что нужно для echo: `cid`, `pos`,
        `alive`, `world_tick`. `ts_client` — для дополнительной round-trip
        диагностики (echo handler сейчас его не возвращает, но это
        forward-compat).

        В Phase 2+ добавятся: `r_cell_form`, `visible_action`,
        `perception_radius`, `acoustic_radius`, `attack_radius`,
        `speed_factor`, `size`, `bite_strength`, `outgoing_events`,
        `lineage`, `species_id`, `client_metabolism_hint`.
        """
        pos = [0, 0]
        # Phase 1 — pos best-effort из доступного compute state. Реальный
        # источник появится в Phase 2 (через obs кеш / serverside arbitr).
        try:
            cache = getattr(self.compute, "_obs_pos_cache", None)
            if cache and cid in cache:
                pos = list(cache[cid])
        except Exception:
            pass
        return {
            "protocol_version": PROTOCOL_VERSION,
            "cid": cid,
            "pos": pos,
            "alive": True,
            "world_tick": int(world_tick),
            "ts_client": time.time(),
        }

    # ── emit ────────────────────────────────────────────────────────────

    def emit_state(self, cid: str, world_tick: int = 0) -> bool:
        """Эмитит state по одному cid. True если поставлено в очередь WS."""
        payload = self._build_state(cid, world_tick=world_tick)
        return self.ws.send_state(payload)

    def emit_alive_owned(
        self,
        world_tick: int = 0,
        period_sec: float = DEFAULT_EMIT_PERIOD_SEC,
    ) -> int:
        """Эмитит state для всех owned cid'ов из `compute.organisms` раз
        в `period_sec`. Возвращает количество отправленных.

        Throttle: если с прошлого emit прошло меньше `period_sec` —
        возвращает 0, ничего не шлёт. Это позволяет звать метод из
        быстрого heartbeat цикла без перегрузки канала.
        """
        now = time.monotonic()
        if now - self._last_emit_at < period_sec:
            return 0
        self._last_emit_at = now
        try:
            cids = list(self.compute.organisms.keys())
        except Exception:
            return 0
        sent = 0
        for cid in cids:
            if self.emit_state(cid, world_tick=world_tick):
                sent += 1
        return sent

    # ── observation callback ────────────────────────────────────────────

    def _on_observation(self, obs: dict) -> None:
        """Phase 1: учитывает echo + обновляет last_world_tick_from_p40."""
        if obs.get("echo"):
            self.observations_with_echo += 1
        wt = obs.get("world_tick", 0)
        if isinstance(wt, int) and wt:
            self.last_world_tick_from_p40 = wt
        cid = obs.get("cid", "")
        logger.debug(
            "embodied observation: cid=%s world_tick=%s echo=%s",
            (cid[:12] if isinstance(cid, str) else "?"),
            wt, obs.get("echo", False),
        )

    # ── stats ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Snapshot для diagnostics push.

        Структура (для convention compliance — Хьюберт смотрит верхний
        уровень `embodied.*`):
          - connected, states_sent, observations_received, errors_total,
            pings_handled, latency_* — flat на верхнем уровне, скопированы
            из `EmbodiedWSClient.stats()`
          - ws — вложенный полный snapshot для granular debugging
            (URL, last_error, и т.д.)
          - observations_with_echo, last_world_tick_from_p40 — счётчики
            этого слоя.
        """
        ws_stats = self.ws.stats()
        out: dict = {
            "observations_with_echo": self.observations_with_echo,
            "last_world_tick_from_p40": self.last_world_tick_from_p40,
        }
        # Flatten основных полей на верхний уровень `embodied.*`.
        for key in (
            "connected",
            "states_sent",
            "observations_received",
            "errors_total",
            "pings_handled",
            "latency_samples",
            "latency_mean_ms",
            "latency_p50_ms",
            "latency_p95_ms",
            "latency_max_ms",
        ):
            if key in ws_stats:
                out[key] = ws_stats[key]
        # Полный ws-блок для debugging (URL, last_error и пр.).
        out["ws"] = ws_stats
        return out
