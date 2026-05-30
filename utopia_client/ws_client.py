"""WebSocket-канал клиент ↔ VPS ↔ P40.

В Phase F1 — только сетевой каркас:
  - persistent connection на wss://divisci.com/ws/colony/client?token=...
  - hello при подключении (имя колонии, версия)
  - ping раз в N секунд, pong от сервера
  - reconnect с экспоненциальной задержкой

Логика actions/observations будет в F3.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Optional

logger = logging.getLogger("utopia_client.ws")

PING_INTERVAL_SEC = 20.0
INITIAL_BACKOFF_SEC = 1.0
MAX_BACKOFF_SEC = 30.0
LOCAL_SAVE_INTERVAL_SEC = 60.0
# Phase G.3 (04.05.2026): пустой Мир. Если n_alive_owned=0 дольше grace и
# нет seed_pack/colony_reset в полёте — клиент шлёт respawn_owned_request
# через WS, P40 запускает _async_respawn (load → clone → genesis).
EMPTY_WORLD_GRACE_SEC = 120.0
EMPTY_WORLD_RETRY_SEC = 300.0
EMPTY_WORLD_CHECK_SEC = 30.0
# S6.0b-B (16.05.2026): период отправки `client_state_sync` (snapshot
# SFNN-правил на P40 → diskstore → возврат в seed_pack). 2 мин —
# компромисс между свежестью при reannounce и нагрузкой на БД.
STATE_SYNC_INTERVAL_SEC = 120.0
# Colony Ownership Migration §5.2 (28.05.2026): projection_batch — клиент
# шлёт публичные проекции owned Zodchiy P40 (P40 размещает в _world.creatures
# для физики/AOI/арбитража). Schema projection_batch_draft.md §3.
# 5 Hz (chem throttle из ТЗ Q4) = каждые 0.2с.
PROJECTION_BATCH_INTERVAL_SEC = 0.2
# Evolved-traits recovery (30.05.2026, Бендер): grace перед pull-safety-net
# traits_request. Даёт projection_batch реконсилиться у P40 и self_heal-push'у
# (ca3e3b2) долететь, прежде чем дёргать pull. ~8с > пара циклов obs_batch.
TRAITS_PULL_GRACE_SEC = 8.0
# Variant B leak fix (19.05.2026): cid считается мёртвым, если не появлялся
# в obs_batch STALE_CID_TICKS подряд. P40 World ~25 TPS — 1500 тиков ≈ 60с
# с запасом против транзитных гэпов obs_batch. GC запускается не чаще
# CID_GC_INTERVAL_TICKS ≈ 10с — дёшево, но успевает реагировать.
STALE_CID_TICKS = 1500
CID_GC_INTERVAL_TICKS = 250
# 0.10.9 (21.05.2026): backstop respawn-запроса, если obs_batch'и идут, а
# compute.organisms пуст (или все obs_per_cid orphan). Без него цикл
# disconnect/reconnect без re-seed (push_owned_seed re-announce filter,
# 1.4.83) оставляет cheef-PC в silent: handle_tick возвращает {} → P40
# никогда не получает actions_batch → silent fallback на brain. Порог
# 10 obs_batch ≈ 2с при 5 obs/s — достаточно, чтобы переждать stagger.
ORPHAN_OBS_RESPAWN_THRESHOLD = 10


class ColonyWSClient:
    """Держит WS к VPS в фоновом потоке."""

    def __init__(self, server: str, token: str, colony_name: str,
                 client_version: str, estimated_population: int = 0,
                 genesis_mode: str = "auto") -> None:
        # https://divisci.com -> wss://divisci.com/ws/colony/client
        if server.startswith("https://"):
            base = "wss://" + server[len("https://"):]
        elif server.startswith("http://"):
            base = "ws://" + server[len("http://"):]
        else:
            base = server
        self.url = f"{base.rstrip('/')}/ws/colony/client"
        self.token = token
        self.colony_name = colony_name
        self.client_version = client_version
        self.estimated_population = int(estimated_population)
        # Phase F.7.4: выбор пользователя при первом подключении.
        # 'fresh' — seed.norg only, 'donor' — fallback с чужими весами,
        # 'auto' — legacy. Прокидывается в hello + respawn_owned_request.
        gm = str(genesis_mode or "auto").lower()
        self.genesis_mode = gm if gm in ("auto", "fresh", "donor") else "auto"

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None
        # threading.Event — кросс-поточный флаг, выставляется немедленно в stop().
        # Защита от race: stop() вызван до инициализации _loop/_stop_event в _run().
        self._stop_flag = threading.Event()
        self._ws = None
        self.connected: bool = False
        self.last_pong_ts: float = 0.0
        self.last_error: str = ""
        # Личная статистика, обновляется из stats-сообщений P40
        self.n_alive_owned: int = 0
        self.last_stats_ts: float = 0.0
        self.last_world_tick: int = 0
        # Phase G.2: последний полученный world_meta (season/event), чтобы
        # логировать ТОЛЬКО переходы, а не каждый tick.
        self.last_world_meta: dict = {}
        # Phase G.3: момент последнего ненулевого n_alive_owned + момент
        # последнего отправленного respawn_owned_request (cooldown).
        self.last_owned_alive_ts: float = 0.0
        self._last_respawn_request_ts: float = 0.0
        # Server-объявленный режим из welcome (awaiting_seed_pack | normal | ...)
        self._welcome_mode: str = ""
        # Phase F3.0 echo-loop: один фоновый таск шлёт фейковый tick_summary.
        self._echo_task: Optional[asyncio.Task] = None
        self._echo_seq: int = 0
        self._echo_snapshots_received: int = 0
        self._echo_last_snap_ts_p40_ns: int = 0
        # Phase F3.1.b: локальная compute-колония, создаётся лениво при seed_start.
        self.compute = None  # type: Optional["LocalColonyCompute"]
        # Чанки seed копятся по (seed_id, cid). Метаданные — из seq=0.
        self._seed_buffers: dict[tuple[int, str], list[bytes]] = {}
        self._seed_meta: dict[tuple[int, str], dict] = {}
        self._seed_accepted: int = 0
        self._seed_failed: int = 0
        # F4: чанки seed.norg (один файл, нет cid). Складываются по seq до
        # получения seed_norg_complete, далее — write_seed_bytes.
        self._seed_norg_chunks: list[bytes] = []
        self._seed_norg_sha: str = ""
        self._seed_norg_total: int = 0
        # F4: если seed_chunk пришёл до seed_norg_complete — буферизуем
        # finalize до тех пор, пока локальный seed.norg не появится.
        # Список ключей (seed_id, cid), готовых к финализации.
        self._pending_finalize: list[tuple[int, str]] = []
        # Метрики obs/actions
        self._obs_batches_received: int = 0
        self._actions_batches_sent: int = 0
        # Phase F3.3.a: счётчик отправленных weights_dump.
        self._weights_dumps_sent: int = 0
        self._weights_requests_received: int = 0
        # Phase F3.3.b.2: cross-owner mate. Мать получает mate_request от P40,
        # делает кроссинговер локально и отвечает newborn.
        self._mate_requests_received: int = 0
        self._mate_newborns_sent: int = 0
        self._mate_rejects_sent: int = 0
        # Brain migration Etap 3.1 (11.05.2026): кеш child_org после кроссинговера.
        # Ключ — mother_cid; значение (child_org, ts). На первом obs_batch с новым
        # cid и parent_id==mother_cid ребёнок регистрируется в compute и
        # наследует predictor/высшие ткани через Y50. Старые записи (>120с)
        # выкашиваются на каждом обращении.
        self._pending_newborn_orgs: dict[str, tuple[object, float]] = {}
        self._newborn_attached: int = 0
        self._newborn_brain_inherited: int = 0
        # Фаза 3.3 obs migration (11.05.2026): ссылка на WorldStateCache
        # для shadow-сборки obs из локального кеша. Прокидывается main.py
        # сразу после _make_ws. None → shadow-build пропускается, поведение
        # как раньше (server obs используется напрямую).
        self.world_cache = None  # type: Optional[object]
        # Метрики shadow build (для diagnostics endpoint).
        self._client_obs_built: int = 0
        self._client_obs_skipped: int = 0   # cache не готов / meta нет
        self._client_obs_match: int = 0     # max |diff| < 1e-3
        self._client_obs_mismatch: int = 0  # max |diff| >= 1e-3
        self._client_obs_max_diff: float = 0.0  # running max |diff|
        # Per-slot аккумулятор: где именно расходится client vs server obs.
        # _slot_diff_sum[i] / _client_obs_built ~ среднее |diff| по слоту i.
        self._client_obs_slot_diff_sum = None  # type: Optional[object]  # np.array(64,)
        # Последний worst-slot пример (slot, client, server, lag_ticks).
        self._client_obs_last_worst: dict = {}
        # Phase 3.3 fix2: tick desync (cache_tick != server_world_tick) → skip.
        # Хранит инфо о последнем пропуске для диагностики.
        self._client_obs_last_tick_skip: dict = {}
        # Phase 3.3B: счётчик локальной сборки obs (когда сервер не прислал).
        # Помогает понять, сколько живой колонии работает через client builder.
        self._client_obs_local_built: int = 0
        # Variant B leak fix (19.05.2026): per-cid last world_tick seen в
        # obs_batch. P40 World удаляет мёртвых тихо — death-сообщения летят
        # только client → P40 (Phase F3.6). Без GC LocalColonyCompute копит
        # ghost'ов (наблюдали 2465 cid vs 10 live в Мире 19.05). Каждый
        # ghost держит organism + predictor + hebbian + SFNN-rules.
        self._cid_last_seen_tick: dict[str, int] = {}
        self._cid_gc_total: int = 0
        self._cid_gc_last_run_tick: int = 0
        # Issue #2 (30.05.2026): счётчик отправленных owned_bye cid (явный
        # despawn на P40 при локальном удалении — GC/смерть).
        self._owned_bye_sent: int = 0
        # 0.10.9 (21.05.2026): счётчик последовательных obs_batch'ей, у которых
        # ни один cid не зарегистрирован в compute.organisms. Триггерит
        # respawn_owned_request при достижении ORPHAN_OBS_RESPAWN_THRESHOLD.
        self._orphan_obs_streak: int = 0
        self._orphan_obs_respawns_sent: int = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._thread_main,
                                         name="utopia-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        # F.6.B (03.05.2026): graceful idle — отправить bye до закрытия WS.
        # VPS broker увидит close → emit client_disconnected → P40 заморозит
        # owned-особей. bye — явный сигнал «штатный idle, не crash».
        if self._loop and self.connected and self._ws is not None:
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self.send_bye(), self._loop)
                try:
                    fut.result(timeout=2.0)
                except Exception as e:
                    logger.warning("bye dispatch failed: %s", e)
            except RuntimeError:
                pass
        if self._loop and self._stop_event:
            try:
                self._loop.call_soon_threadsafe(self._stop_event.set)
            except RuntimeError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        # Phase F3.2.c: после join — поток compute больше не трогает; безопасно
        # сериализовать Hebbian-state особей в локальный кеш колонии.
        if self.compute is not None:
            try:
                from .seed_loader import colony_state_dir
                n = self.compute.save_all_states(
                    colony_state_dir(self.colony_name))
                logger.info("local-state saved: %d creatures", n)
            except Exception as e:
                logger.warning("save_all_states failed: %s", e)

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as e:
            logger.warning("ws thread crashed: %s", e)

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        # Если stop() успел сработать до этой точки — выходим сразу.
        if self._stop_flag.is_set():
            return
        backoff = INITIAL_BACKOFF_SEC
        while not self._stop_event.is_set():
            try:
                await self._session()
                backoff = INITIAL_BACKOFF_SEC
            except asyncio.CancelledError:
                return
            except Exception as e:
                self.connected = False
                self.last_error = f"{type(e).__name__}: {e}"
                logger.warning("ws error: %s", self.last_error)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                return
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, MAX_BACKOFF_SEC)

    async def _session(self) -> None:
        try:
            import websockets
        except ImportError:
            self.last_error = "websockets package not installed"
            await asyncio.sleep(30)
            return

        url = f"{self.url}?token={self.token}"
        logger.info("connecting %s", self.url)
        async with websockets.connect(url, max_size=512 * 1024,
                                       ping_interval=30, ping_timeout=120) as ws:
            self._ws = ws
            self.connected = True
            self.last_error = ""
            # Evolved-traits recovery: re-arm one-shot re-announce + pull-safety
            # net на новый коннект (welcome-handler шлёт после ingest snapshot).
            self._traits_announced_conn = False
            self._traits_pull_sent = False
            logger.info("connected")

            from .seed_loader import seed_sha256

            # A1: разведка локального seed-pack ДО hello. Если на диске
            # есть .pt от прошлой сессии — объявляем серверу через
            # local_save в hello + сразу шлём seed_pack_start/chunk*/complete.
            local_pack = self._scan_local_seed_pack()

            # Colony Ownership Migration §5.1 (28.05): n_local — сколько
            # owned Zodchiy у клиента в local persistence. P40 при n_local>0
            # НЕ запускает push_owned_seed, НЕ auto-respawn — ждёт
            # projection_batch. При n_local=0 — build_seed_pack как раньше.
            # Colony Ownership Migration §5.1: n_local = real count из
            # того же source что legacy _scan_local_seed_pack использует
            # (colony_state_dir, не отдельный states/ subdir — fix v0.11.1).
            # local_pack уже scanned выше.
            n_local = len(local_pack) if local_pack else 0
            # При n_local>0 client igнорирует incoming seed_start/chunk/
            # complete (defensive against legacy P40 push). Source of truth
            # — local persistence.
            self._reject_incoming_seeds = (n_local > 0)
            if self._reject_incoming_seeds:
                logger.info(
                    "client-authoritative: n_local=%d → reject incoming seeds",
                    n_local)

            hello_msg: dict = {
                "type": "hello",
                "colony_name": self.colony_name,
                "client_version": self.client_version,
                "estimated_population": self.estimated_population,
                "known_seed_hash": seed_sha256("wanderer"),
                "lineage": "wanderer",
                "genesis_mode": self.genesis_mode,
                "ts": int(time.time() * 1000),
                "n_local": n_local,
            }
            if local_pack:
                hello_msg["local_save"] = {
                    "available": True,
                    "n_creatures": len(local_pack),
                    "est_bytes": sum(len(p["weights"]) for p in local_pack),
                    "seed_revision": int(time.time()),
                }
            await ws.send(json.dumps(hello_msg))

            # Colony Ownership Migration §5.1/5.2 (29.05, fix 0.11.4):
            # client-authoritative restore. При n_local>0 client —
            # ЕДИНСТВЕННЫЙ хозяин: восстанавливает organisms в свой compute
            # из local .pt (E2 restore_colony_from_local), затем шлёт
            # projection_batch → P40 self-heal (f359f97) пересоздаёт owned.
            #
            # Раньше (баг): restore_colony_from_local был написан но НЕ
            # вызывался → compute пуст → нет biochem/projection/жизни. Client
            # пушил legacy seed_pack, но локально organisms не наполнялись.
            #
            # При n_local>0 — НЕ слать legacy _send_seed_pack (P40 self-heal
            # работает через projection_batch, не через seed pack). Только
            # restore локально.
            if self._reject_incoming_seeds:
                try:
                    if self.compute is None:
                        self._ensure_compute()
                    if self.compute is not None:
                        from .seed_loader import colony_state_dir
                        restored = self.compute.restore_colony_from_local(
                            colony_state_dir(self.colony_name))
                        logger.info(
                            "client-authoritative restore: %d organisms из local",
                            len(restored))
                except Exception as e:
                    logger.warning("restore_colony_from_local failed: %s", e)
            elif local_pack:
                # n_local==0 ветка не достигается (local_pack пуст → n_local=0),
                # но оставляем legacy push на случай edge-case формата.
                try:
                    await self._send_seed_pack(
                        ws, local_pack,
                        seed_revision=hello_msg["local_save"]["seed_revision"])
                except Exception as e:
                    logger.warning("send seed_pack failed: %s", e)

            ping_task = asyncio.create_task(self._ping_loop(ws))
            save_task = asyncio.create_task(self._save_loop())
            empty_task = asyncio.create_task(self._empty_world_loop())
            sync_task = asyncio.create_task(self._state_sync_loop(ws))
            # Colony Ownership Migration §5.2: periodic projection_batch.
            proj_task = asyncio.create_task(self._projection_batch_loop(ws))
            try:
                async for raw in ws:
                    try:
                        env = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(env, dict):
                        continue
                    msg = env.get("msg") if "msg" in env else env
                    if isinstance(msg, dict):
                        await self._handle(msg)
                    if self._stop_event.is_set():
                        break
            finally:
                ping_task.cancel()
                save_task.cancel()
                empty_task.cancel()
                sync_task.cancel()
                for t in (ping_task, save_task, empty_task, sync_task):
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
                self._ws = None
                self.connected = False

    async def send_bye(self) -> None:
        """F.6.B: graceful idle — отправить bye, далее WS закрывается штатно.

        Сервер на close → broker.client_disconnected → P40 freeze_personal
        owned-особей (не размножаются, неуязвимы для PvP, остаются в Мире).
        Hello при следующем подключении триггерит unfreeze_personal.
        """
        ws = self._ws
        if ws is None:
            return
        try:
            await ws.send(json.dumps({
                "type": "bye",
                "ts": int(time.time() * 1000),
            }))
            logger.info("bye sent (graceful idle)")
        except Exception as e:
            logger.warning("bye send failed: %s", e)

    async def _ping_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(PING_INTERVAL_SEC)
            try:
                await ws.send(json.dumps({"type": "ping",
                                          "ts": int(time.time() * 1000)}))
            except Exception:
                return

    async def _save_loop(self) -> None:
        """Периодическая запись Hebbian/predictor state в local cache.

        Без этого state живёт только в RAM до stop(). При crash без stop()
        теряется обучение. Когда P40 удалит свои `/data/colonies/*/members/*.pt`
        (план A6), local cache становится единственным backup'ом.
        """
        while True:
            await asyncio.sleep(LOCAL_SAVE_INTERVAL_SEC)
            if self.compute is None:
                continue
            try:
                from .seed_loader import colony_state_dir
                n = await asyncio.to_thread(
                    self.compute.save_all_states,
                    colony_state_dir(self.colony_name))
                logger.debug("periodic local-save: %d creatures", n)
            except Exception as e:
                logger.warning("periodic save_all_states failed: %s", e)

    async def _state_sync_loop(self, ws) -> None:
        """S6.0b-B2 (16.05.2026): periodic SFNN-state snapshot → P40.

        Раз в STATE_SYNC_INTERVAL_SEC клиент шлёт `client_state_sync` с
        текущими motor_sfnn_rule/higher_tissue_sfnn_rules и счётчиками
        per cid. Сервер кладёт blob на диск (sfnn_state_store) и подкладывает
        в seed_pack при следующем hello_recovery / admin_respawn.
        Это страхует от потери эволюционных коэффициентов при legitimate
        reset (для broker re-announce есть отдельный guard S6.0b-A).
        """
        while True:
            await asyncio.sleep(STATE_SYNC_INTERVAL_SEC)
            if self.compute is None:
                continue
            try:
                items = self.compute.collect_sfnn_state_sync_items()
            except Exception as e:
                logger.warning("collect_sfnn_state_sync_items failed: %s", e)
                continue
            if not items:
                continue
            try:
                await ws.send(json.dumps({
                    "type": "client_state_sync",
                    "items": items,
                }))
                logger.debug("client_state_sync sent: n=%d", len(items))
            except Exception as e:
                logger.warning("client_state_sync send failed: %s", e)
                return

    async def _projection_batch_loop(self, ws) -> None:
        """Colony Ownership Migration §5.2 (28.05.2026, Бендер): periodic
        projection_batch emit. 5 Hz (chem throttle Q4).

        Schema (projection_batch_draft.md §3 финал 28.05):
            {type: 'projection_batch', tick_client, ts_client_ms,
             creatures: [{cid, species_id, alive, frozen, action,
                          chem{7}, mental_break}, ...]}

        P40 при non-empty client n_local — ждёт эти batches вместо
        push_owned_seed (см. hello.n_local). Использует только для
        физики Мира, не authoritative.
        """
        self._projections_sent: int = getattr(self, "_projections_sent", 0)
        while True:
            await asyncio.sleep(PROJECTION_BATCH_INTERVAL_SEC)
            if self.compute is None:
                continue
            try:
                projections = self.compute.build_projection_batch()
            except Exception as e:
                logger.debug("build_projection_batch failed: %s", e)
                continue
            if not projections:
                continue
            try:
                await ws.send(json.dumps({
                    "type": "projection_batch",
                    "tick_client": int(self.last_world_tick),
                    "ts_client_ms": int(time.time() * 1000),
                    "creatures": projections,
                }))
                self._projections_sent += 1
            except Exception as e:
                logger.debug("projection_batch send failed: %s", e)
                return

    def _ingest_traits_snapshot(self, entries, reason: str = "snapshot") -> int:
        """Evolved-traits recovery: применить owned_traits_snapshot creatures
        в стор FILL-ONLY (client authoritative — baseline от P40 self-heal не
        затирает клиентский evolved). `entries` — list[{cid, traits{9}}].

        Источники: welcome.owned_traits_snapshot (inline) или top-level
        owned_traits_snapshot.creatures (self_heal/pull, ca3e3b2).
        Возвращает число заполненных trait-полей."""
        if not isinstance(entries, list) or self.compute is None \
                or not hasattr(self.compute, "ingest_owned_traits"):
            return 0
        filled = 0
        for ent in entries:
            if not isinstance(ent, dict):
                continue
            ecid = str(ent.get("cid") or "")
            etraits = ent.get("traits")
            if ecid and isinstance(etraits, dict):
                try:
                    filled += self.compute.ingest_owned_traits(
                        ecid, etraits, overwrite=False)
                except Exception as e:
                    logger.warning(
                        "owned_traits_snapshot ingest %s: %s", ecid, e)
        if filled:
            logger.info(
                "owned_traits_snapshot reason=%s: filled %d trait-fields",
                reason, filled)
        return filled

    async def _maybe_pull_traits(self) -> None:
        """Pull safety-net (ca3e3b2): один traits_request за коннект, если
        после grace у клиента есть owned organisms, но стор traits пуст
        (build_traits_announce_envelope is None). Grace даёт projection_batch
        реконсилиться и self_heal-push'у прийти; pull нужен только когда
        self_heal не сработал (P40 уже знает cid'ы). Idempotent per connect."""
        if getattr(self, "_traits_pull_sent", False):
            return
        await asyncio.sleep(TRAITS_PULL_GRACE_SEC)
        if getattr(self, "_traits_pull_sent", False):
            return
        compute = self.compute
        if compute is None or not getattr(compute, "organisms", None):
            return
        try:
            if compute.build_traits_announce_envelope() is not None:
                return  # стор уже наполнен (self_heal/welcome/.pt) — pull не нужен
        except Exception:
            return
        self._traits_pull_sent = True
        await self.request_traits_resync()

    async def request_traits_resync(self) -> bool:
        """Evolved-traits recovery (ca3e3b2): pull-путь. Шлёт traits_request →
        P40 отвечает owned_traits_snapshot reason='pull'. Резервный resync
        если self_heal-push разъехался / стор подозрительно пуст."""
        ws = self._ws
        if ws is None:
            return False
        try:
            await ws.send(json.dumps({"type": "traits_request"}))
            logger.info("traits_request sent (pull resync)")
            return True
        except Exception as e:
            logger.warning("traits_request send failed: %s", e)
            return False

    async def _maybe_send_traits_announce(self, force: bool = False) -> None:
        """Evolved-traits recovery (30.05.2026, Бендер): re-announce evolved
        body-traits для existing owned обратно на P40 по main ws (client→P40).

        `force=False` — one-shot per connect (guard `_traits_announced_conn`,
        ресет в connect-блоке): для welcome. `force=True` — игнор guard: для
        top-level owned_traits_snapshot (self_heal/pull), где стор только что
        наполнился и нужно отправить evolved поверх baseline даже если welcome
        уже выставил guard. P40 self_heal-push не повторяется (cid становится
        known) → петли нет.

        Канал — main ws. Конверт batch {type:'traits_announce',
        creatures:[{cid,traits}]}; P40 отвечает traits_ack. Идемпотентно
        (accept-over-baseline).
        """
        if not force and getattr(self, "_traits_announced_conn", False):
            return
        ws = self._ws
        compute = self.compute
        if ws is None or compute is None \
                or not hasattr(compute, "build_traits_announce_envelope"):
            return
        try:
            envelope = compute.build_traits_announce_envelope()
        except Exception as e:
            logger.warning("build_traits_announce_envelope failed: %s", e)
            return
        # Помечаем коннект обработанным даже при пустом сторе (нечего слать —
        # fresh colony), чтобы не пересобирать каждый welcome.
        self._traits_announced_conn = True
        if not envelope:
            return
        creatures = envelope.get("creatures", [])
        try:
            await ws.send(json.dumps(envelope))
        except Exception as e:
            logger.warning("traits_announce send failed: %s", e)
            return
        try:
            compute.mark_traits_announce_sent([c["cid"] for c in creatures])
        except Exception:
            pass
        logger.info("traits_announce sent: %d owned", len(creatures))

    async def _maybe_request_respawn(self, connect_ts: float) -> bool:
        """Phase G.3: одна проверка watchdog'а.

        Возвращает True если запрос отправлен, False — пропуск (любая из
        guard-проверок не прошла или send упал).
        """
        if not self.connected:
            return False
        if self.n_alive_owned > 0:
            return False
        # awaiting_seed_pack — сервер сам обещает прислать seed_pack
        # через _seed_pack_timeout_fallback. Не дублируем запрос.
        if self._welcome_mode == "awaiting_seed_pack":
            return False
        now = time.time()
        ref = self.last_owned_alive_ts or connect_ts
        since_alive = now - ref
        since_request = now - self._last_respawn_request_ts
        if since_alive < EMPTY_WORLD_GRACE_SEC:
            return False
        if (self._last_respawn_request_ts > 0
                and since_request < EMPTY_WORLD_RETRY_SEC):
            return False
        ws = self._ws
        if ws is None:
            return False
        try:
            await ws.send(json.dumps({
                "type": "respawn_owned_request",
                "colony_name": self.colony_name,
                "mode": self.genesis_mode,
                "n": 8,
                "ts": int(now * 1000),
            }))
        except Exception as e:
            logger.warning("respawn_owned_request send failed: %s", e)
            return False
        self._last_respawn_request_ts = now
        logger.info(
            "empty-world watchdog: respawn_owned_request sent "
            "(empty %.0fs, last_request %.0fs ago)",
            since_alive,
            since_request if self._last_respawn_request_ts else -1)
        return True

    async def _empty_world_loop(self) -> None:
        """Phase G.3 watchdog: респавн при долгом n_alive_owned=0.

        Если клиент остался без живых owned дольше EMPTY_WORLD_GRACE_SEC и
        Мир не объявил awaiting_seed_pack, шлёт `respawn_owned_request`
        через WS. P40 шедулит _async_respawn (load → clone → genesis),
        ответ идёт обычным потоком colony_reset → seed → obs_batch.
        Cooldown EMPTY_WORLD_RETRY_SEC между попытками.
        """
        # Стартовая отметка — момент запуска watchdog. Ноль значит «пока
        # ни одного живого не видели», старт grace-таймера = connect time.
        connect_ts = time.time()
        while True:
            await asyncio.sleep(EMPTY_WORLD_CHECK_SEC)
            await self._maybe_request_respawn(connect_ts)

    async def _handle(self, msg: dict) -> None:
        msg_type = msg.get("type", "")
        if msg_type == "welcome":
            n = msg.get("n_creatures")
            if isinstance(n, int):
                self.n_alive_owned = n
                if n > 0:
                    self.last_owned_alive_ts = time.time()
            wt = msg.get("world_tick")
            if isinstance(wt, int):
                self.last_world_tick = wt
            mode = str(msg.get("mode", "") or "")
            self._welcome_mode = mode
            # 0.10.9 (21.05.2026): после reconnect (broker keepalive re-announce
            # без colony_reset) world_tick может перескочить вперёд на N тиков,
            # пока ws_client ловит первый obs_batch. Если diff > STALE_CID_TICKS
            # — _gc_orphan_cids снесёт всех живых organisms. Промотаем grace на
            # текущий welcome.world_tick для cid'ов, уже хранящихся в compute.
            if (isinstance(wt, int) and wt > 0
                    and self.compute is not None):
                try:
                    for cid in list(self.compute.organisms.keys()):
                        self._cid_last_seen_tick[cid] = wt
                except Exception:
                    pass
            # Evolved-traits recovery (30.05.2026, Бендер): warm hello path —
            # welcome может нести inline owned_traits_snapshot: [{cid,traits{9}}]
            # (контракт Хьюберта f673741). FILL-ONLY ingest. Top-level
            # owned_traits_snapshot (self_heal/pull, ca3e3b2) — отдельный handler.
            self._ingest_traits_snapshot(
                msg.get("owned_traits_snapshot"), reason="welcome")
            logger.info(
                "welcome: world_tick=%s server_time=%s n_creatures=%s mode=%s",
                wt, msg.get("server_time"), n, mode or "normal")
            # Re-announce evolved traits обратно на P40 (one-shot per connect).
            # На warm reconnect organisms уже в compute (in-memory / restore_
            # colony_from_local в connect-блоке) → стор содержит evolved.
            try:
                await self._maybe_send_traits_announce()
            except Exception as e:
                logger.warning("traits_announce send failed: %s", e)
            # Pull safety-net: для client-authoritative welcome.snapshot пуст
            # (P40 owned=0 — ещё не знает cid'ов). self_heal-push (ca3e3b2)
            # покроет большинство, но если он не сработал (P40 уже знает cid'ы,
            # KNOWN, push'а нет), а стор пуст — дёрнем pull через grace.
            try:
                asyncio.create_task(self._maybe_pull_traits())
            except Exception as e:
                logger.debug("schedule traits pull failed: %s", e)
            return
        if msg_type == "owned_traits_snapshot":
            # Evolved-traits recovery (ca3e3b2, Хьюберт): P40 шлёт snapshot
            # отдельным top-level сообщением {type, reason, creatures}. reason:
            #   - "self_heal" — после projection_batch с unknown_cid>0 P40
            #     пересоздал owned в baseline и авто-пушит. КЛЮЧЕВОЙ путь для
            #     client-authoritative cheef (welcome.snapshot пуст, т.к. P40
            #     ещё не знал cid'ов — теперь узнал из projection_batch).
            #   - "pull" — ответ на наш traits_request (resync).
            # FILL-ONLY ingest (baseline не затирает evolved в сторе) → force
            # re-announce: guard _traits_announced_conn уже True после welcome,
            # но именно здесь стор наполнился → шлём evolved поверх baseline.
            reason = str(msg.get("reason", "") or "snapshot")
            self._ingest_traits_snapshot(msg.get("creatures"), reason=reason)
            try:
                await self._maybe_send_traits_announce(force=True)
            except Exception as e:
                logger.warning("traits_announce (snapshot) send failed: %s", e)
            return
        if msg_type == "traits_ack":
            # Evolved-traits recovery: P40 применил evolved поверх baseline
            # для applied_cids (accept-over-baseline). Снимаем pending.
            if self.compute is not None \
                    and hasattr(self.compute, "handle_traits_ack"):
                try:
                    self.compute.handle_traits_ack(msg)
                except Exception as e:
                    logger.warning("traits_ack handler error: %s", e)
            return
        if msg_type == "pong":
            self.last_pong_ts = time.time()
            return
        if msg_type == "ping_probe":
            # Phase F3.0: эхо для замера RTT P40↔клиент. Не подмешиваем
            # своё время — P40 считает RTT по своему монотонному часу.
            ws = self._ws
            if ws is not None:
                try:
                    await ws.send(json.dumps({
                        "type": "pong_probe",
                        "probe_id": msg.get("probe_id"),
                        "ts_p40_ns": msg.get("ts_p40_ns"),
                    }))
                except Exception:
                    pass
            return
        if msg_type == "seed_pack_ack":
            # A1: P40 ответил после restore_personal_from_pack или fallback.
            # restored>0 — наш local seed применён; иначе сервер ушёл в
            # fallback (load_personal/genesis) и ждать обычный seed_*.
            logger.info(
                "seed_pack_ack: restored=%s requested=%s failed=%s "
                "error=%s fallback=%s",
                msg.get("restored"), msg.get("requested"),
                msg.get("failed"), msg.get("error"),
                msg.get("fallback_path"))
            # Body Migration Phase 1 fix (27.05.2026): когда P40 успешно
            # восстановил cached seeds — fresh `seed_chunk` от него НЕ
            # последует, потому что seeds уже у нас локально. Это значит
            # `_ensure_compute()` через обычный seed-chunk path не вызовется,
            # `self.compute` останется None всю сессию. Последствия:
            #   - Hebbian-update/SFNN не работают (нет compute.organisms);
            #   - diagnostics push не отдаёт hebbian_per_tissue;
            #   - EmbodiedOrganism lazy-attach skip'ается → state_msgs=0
            #     на VPS-брокере.
            # Поэтому инициируем compute явно при restored>0. Если он уже
            # создан — `_ensure_compute` идемпотентен (early return на
            # `self.compute is not None`).
            try:
                if int(msg.get("restored", 0) or 0) > 0:
                    self._ensure_compute()
            except Exception as e:
                logger.warning(
                    "seed_pack_ack compute init failed: %s", e)
            return
        if msg_type == "stats":
            n = msg.get("n_alive_owned", 0)
            try:
                self.n_alive_owned = int(n)
            except (TypeError, ValueError):
                self.n_alive_owned = 0
            wt = msg.get("world_tick")
            if isinstance(wt, int):
                self.last_world_tick = wt
            now = time.time()
            self.last_stats_ts = now
            if self.n_alive_owned > 0:
                self.last_owned_alive_ts = now
            wm = msg.get("world_meta")
            if isinstance(wm, dict) and wm:
                old = self.last_world_meta
                old_season = str(old.get("season_name", "") or "")
                new_season = str(wm.get("season_name", "") or "")
                if old_season and new_season and old_season != new_season:
                    logger.info("season: %s → %s", old_season, new_season)
                old_event = str(old.get("active_event", "") or "")
                new_event = str(wm.get("active_event", "") or "")
                if old_event != new_event:
                    if new_event:
                        logger.info("event started: %s", new_event)
                    elif old_event:
                        logger.info("event ended: %s", old_event)
                self.last_world_meta = wm
                if self.compute is not None:
                    try:
                        setattr(self.compute, "world_meta", dict(wm))
                    except Exception:
                        pass
            logger.info("stats: n_alive_owned=%d world_tick=%s",
                        self.n_alive_owned, wt)
            return
        if msg_type == "respawn_owned_ack":
            logger.info(
                "respawn_owned_ack: colony=%s scheduled=%s",
                msg.get("colony_name"), msg.get("scheduled"))
            return
        if msg_type == "echo_request":
            await self._handle_echo_request(msg)
            return
        if msg_type == "world_snapshot":
            self._echo_snapshots_received += 1
            ts = msg.get("ts_p40_ns")
            if isinstance(ts, int):
                self._echo_last_snap_ts_p40_ns = ts
            logger.info("echo: world_snapshot seq=%s tick=%s",
                        msg.get("seq"), msg.get("world_tick"))
            return
        # ── colony_reset (P40 говорит: выкинь всех, новый seed идёт) ────
        if msg_type == "colony_reset":
            self._handle_colony_reset(msg)
            return
        # ── Phase F3.1.b/d: seed (chunked weights) ──────────────────────
        if msg_type == "seed_start":
            if getattr(self, "_reject_incoming_seeds", False):
                logger.warning(
                    "seed_start ignored (client-authoritative, n_local>0)")
                return
            self._handle_seed_start(msg)
            return
        if msg_type == "seed_chunk":
            if getattr(self, "_reject_incoming_seeds", False):
                return
            self._handle_seed_chunk(msg)
            return
        if msg_type == "seed_complete":
            if getattr(self, "_reject_incoming_seeds", False):
                logger.warning(
                    "seed_complete ignored (client-authoritative)")
                return
            await self._handle_seed_complete(msg)
            return
        # ── F4: seed.norg (центральный «днк-предок» от P40) ─────────────
        if msg_type == "seed_norg_start":
            self._handle_seed_norg_start(msg)
            return
        if msg_type == "seed_norg_chunk":
            self._handle_seed_norg_chunk(msg)
            return
        if msg_type == "seed_norg_complete":
            self._handle_seed_norg_complete(msg)
            return
        # ── Phase F3.1.b/c: obs_batch → handle_tick → actions_batch ─────
        if msg_type == "obs_batch":
            await self._handle_obs_batch(msg)
            return
        # ── Phase F3.3.a: pull-on-demand weights ────────────────────────
        if msg_type == "weights_request":
            await self._handle_weights_request(msg)
            return
        # ── Phase F3.3.b.2: cross-owner mate ────────────────────────────
        if msg_type == "mate_request":
            await self._handle_mate_request(msg)
            return
        # Прочее — F3 (force_death, world_event, newborn_ack — F3.4+)
        logger.debug("unhandled msg type=%s", msg_type)

    # ── Phase F3.1.b: seed handling ──────────────────────────────────────

    def _ensure_compute(self) -> bool:
        """Лениво создать LocalColonyCompute. False — если torch/neurocore не
        установлены."""
        if self.compute is not None:
            # S2.B (13.05.2026): обновляем ссылку на world_cache, если в
            # промежутке кеш мог быть пересоздан (idle→run).
            if (self.world_cache is not None
                    and getattr(self.compute, "world_cache", None)
                    is not self.world_cache):
                self.compute.world_cache = self.world_cache
            return True
        try:
            from .local_compute import LocalColonyCompute
            self.compute = LocalColonyCompute()
            # S2.B: пробрасываем world_cache в compute, чтобы
            # _compute_theory_of_mind мог дёрнуть tom_neighbors_view.
            if self.world_cache is not None:
                self.compute.world_cache = self.world_cache
            logger.info("compute initialized: device=%s", self.compute.device)
            return True
        except Exception as e:
            logger.warning("compute init failed: %s", e)
            return False

    def _handle_colony_reset(self, msg: dict) -> None:
        reason = msg.get("reason", "unknown")
        n_dropped = 0
        if self.compute is not None:
            n_dropped = self.compute.reset_all()
        self._seed_buffers.clear()
        self._seed_meta.clear()
        # Variant B (19.05.2026): reset_all() выбросил все organisms — стираем
        # last_seen, чтобы tracking начался с нуля при новом seed_pack.
        self._cid_last_seen_tick.clear()
        # Phase G.3: новый seed едет — снимаем awaiting/respawn-cooldown,
        # обновляем «последний раз alive», чтобы grace отсчитался заново.
        self._welcome_mode = ""
        self._last_respawn_request_ts = 0.0
        self.last_owned_alive_ts = time.time()
        logger.info("colony_reset reason=%s dropped=%d", reason, n_dropped)

    def _handle_seed_start(self, msg: dict) -> None:
        seed_id = int(msg.get("seed_id", 0))
        n = int(msg.get("n_creatures", 0))
        total = int(msg.get("total_bytes", 0))
        # Очищаем старые буферы под этот seed_id (на случай повторной рассылки).
        for key in [k for k in self._seed_buffers if k[0] == seed_id]:
            self._seed_buffers.pop(key, None)
            self._seed_meta.pop(key, None)
        self._seed_accepted = 0
        self._seed_failed = 0
        self._ensure_compute()
        logger.info("seed_start id=%d n=%d total_bytes=%d", seed_id, n, total)

    def _handle_seed_chunk(self, msg: dict) -> None:
        import base64
        seed_id = int(msg.get("seed_id", 0))
        cid = str(msg.get("cid", ""))
        seq = int(msg.get("seq", 0))
        last = bool(msg.get("last", False))
        payload_b64 = msg.get("payload_b64", "")
        if not cid or not payload_b64:
            logger.warning("seed_chunk: bad msg (cid=%s)", cid)
            return
        key = (seed_id, cid)
        try:
            chunk = base64.b64decode(payload_b64)
        except Exception as e:
            logger.warning("seed_chunk %s: b64 decode failed: %s", cid, e)
            self._seed_failed += 1
            return
        if seq == 0:
            meta = msg.get("meta")
            if isinstance(meta, dict):
                self._seed_meta[key] = meta
            self._seed_buffers[key] = []
        buf = self._seed_buffers.setdefault(key, [])
        buf.append(chunk)
        if last:
            # F4: если локального seed.norg ещё нет — отложим finalize до
            # seed_norg_complete (organism_from_weights читает архитектуру
            # из локального файла, иначе FileNotFoundError).
            from .seed_loader import seed_cached
            if not seed_cached("wanderer"):
                self._pending_finalize.append(key)
                logger.info(
                    "seed_chunk %s: deferred (waiting for seed.norg)", cid)
            else:
                self._finalize_seed_creature(key)

    def _finalize_seed_creature(self, key: tuple[int, str]) -> None:
        seed_id, cid = key
        chunks = self._seed_buffers.pop(key, [])
        meta = self._seed_meta.pop(key, {})
        if not chunks:
            logger.warning("finalize %s: no chunks", cid)
            self._seed_failed += 1
            return
        if self.compute is None and not self._ensure_compute():
            logger.warning("finalize %s: compute unavailable", cid)
            self._seed_failed += 1
            return
        # Phase F3.2.c: source-приоритет — local-cache (Hebbian-обученные) над
        # P40-seed. Битый local → fallback на P40-bytes (без потери особи).
        p40_bytes = b"".join(chunks)
        sources: list[tuple[str, bytes]] = []
        try:
            from .seed_loader import creature_state_path
            local_path = creature_state_path(self.colony_name, cid)
            if local_path.exists() and local_path.stat().st_size > 0:
                sources.append(("local-cache", local_path.read_bytes()))
        except Exception as e:
            logger.warning("local read %s failed: %s", cid, e)
        sources.append(("p40-seed", p40_bytes))

        from .seed_loader import organism_from_weights, seed_cache_path
        # Z7.i.d (16.05.2026): lineage из meta (P40 шлёт в seed_pack) с
        # fallback на WorldStateCache.creature_tom (Мир-снапшот). При
        # lineage="zodchiy" compute.add_creature развернёт 3 уникальные
        # ткани (cerebellum/amygdala/episodic). Дефолт "wanderer" —
        # клиентские особи (P40 owned) почти всегда Странники.
        lineage = str(meta.get("lineage", "") or "")
        if not lineage:
            try:
                wc = getattr(self.compute, "world_cache", None)
                if wc is not None:
                    tom_e = wc.creature_tom.get(cid)
                    if tom_e:
                        lineage = str(tom_e[0]) or ""
            except Exception:
                pass
        if not lineage:
            lineage = "wanderer"
        loaded = False
        for src, weights in sources:
            try:
                org, payload = organism_from_weights(
                    weights, seed_cache_path("wanderer"))
                self.compute.add_creature(
                    cid, org,
                    hebbian_enabled=True,
                    learning_rate=float(meta.get("learning_rate", 1e-4)),
                    trace_decay=float(meta.get("trace_decay", 0.9)),
                    lineage=lineage,
                )
                # Variant B (19.05.2026): grace-метка, чтобы свежий cid не
                # был сразу убран GC при world_tick > STALE_CID_TICKS. Если
                # cid живой — следующий obs_batch обновит last_seen.
                self._cid_last_seen_tick[cid] = self.last_world_tick
                self.compute.apply_inherited_state(cid, payload)
                # S6.0b-B (16.05.2026): persistent SFNN-state восстановление.
                # Сервер кладёт сохранённое правило клиента в payload['sfnn_state']
                # (через `client_state_sync` ← N тиков назад). После reset_all
                # / hello_recovery это единственный путь вернуть накопленную
                # эволюцию правил.
                sfnn_state = payload.get("sfnn_state") if isinstance(payload, dict) else None
                if sfnn_state and hasattr(self.compute, "restore_sfnn_state"):
                    try:
                        self.compute.restore_sfnn_state(cid, sfnn_state)
                    except Exception as e:
                        logger.warning(
                            "restore_sfnn_state cid=%s failed: %s", cid, e)
                # Evolved-traits recovery (30.05.2026, Бендер): owned-handoff
                # body-traits от P40 (контракт Хьюберта f673741). _creature_meta
                # кладёт 8 evolved int в meta["traits"] + diet_gene ОТДЕЛЬНО на
                # верхнем уровне meta. Мёржим в один 9-полевой dict → стор.
                # Без этого crossover для не-client-born особей читал бы median-
                # defaults. Local-cache (.pt) свои traits уже восстановил в
                # restore_persisted_state — здесь P40 закрывает остаток
                # (founders/elder/wanderer).
                if isinstance(meta, dict) and hasattr(self.compute, "ingest_owned_traits"):
                    handoff_traits = dict(meta.get("traits") or {})
                    if "diet_gene" in meta and "diet_gene" not in handoff_traits:
                        handoff_traits["diet_gene"] = meta["diet_gene"]
                    if handoff_traits:
                        try:
                            # FILL-ONLY: client authoritative — не затираем свой
                            # evolved (local-cache .pt мог восстановить раньше).
                            n_tr = self.compute.ingest_owned_traits(
                                cid, handoff_traits, overwrite=False)
                            if n_tr:
                                logger.info(
                                    "owned-handoff traits cid=%s n=%d", cid, n_tr)
                        except Exception as e:
                            logger.warning(
                                "ingest_owned_traits cid=%s failed: %s", cid, e)
                self._seed_accepted += 1
                logger.info("seed accepted cid=%s seed_id=%d bytes=%d source=%s",
                            cid, seed_id, len(weights), src)
                loaded = True
                break
            except Exception as e:
                logger.warning("finalize %s source=%s failed: %s", cid, src, e)
        if not loaded:
            self._seed_failed += 1
            return

        # A1: сохранить meta на диск рядом с .pt — чтобы при reconnect
        # клиент мог отправить её обратно в seed_pack_chunk[seq=0].
        # Без meta P40 при restore_personal_from_pack применит дефолты
        # (learning_rate=1e-4, trace_decay=0.9, diet_gene=0.5, random tile).
        try:
            from .seed_loader import colony_state_dir
            meta_path = colony_state_dir(self.colony_name) / f"{cid}.meta.json"
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            meta_path.write_text(
                json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.warning("save meta %s: %s", cid, e)

    # ── A1: seed_pack клиент→P40 ─────────────────────────────────────────
    # Cap: 10 особей по mtime DESC. Симметрия с MAX_SEED_PACK_CREATURES
    # на стороне сервера (server/colony_storage.py).
    MAX_SEED_PACK_CREATURES = 10
    SEED_PACK_CHUNK_SIZE = 240 * 1024  # 240 КиБ — как push_owned_seed P40→client

    def _scan_local_seed_pack(self) -> list[dict]:
        """A1: отобрать .pt от прошлой сессии для отправки в seed_pack.

        Читает colony_state_dir(name)/*.pt + соответствующий *.meta.json,
        сортирует по mtime DESC, берёт top-MAX_SEED_PACK_CREATURES.
        Возвращает [{"cid": str, "weights": bytes, "meta": dict}, ...].

        Пустой список — нет ничего на диске или ошибка чтения.
        """
        try:
            from .seed_loader import colony_state_dir
            d = colony_state_dir(self.colony_name)
            if not d.exists():
                return []
            pt_files = sorted(
                [p for p in d.glob("*.pt") if p.is_file() and p.stat().st_size > 0],
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )[:self.MAX_SEED_PACK_CREATURES]
            pack: list[dict] = []
            for pt in pt_files:
                cid = pt.stem
                try:
                    weights = pt.read_bytes()
                except Exception as e:
                    logger.warning("scan_local: read %s: %s", pt, e)
                    continue
                meta: dict = {}
                meta_path = d / f"{cid}.meta.json"
                if meta_path.exists():
                    try:
                        meta = json.loads(meta_path.read_text(encoding="utf-8"))
                        if not isinstance(meta, dict):
                            meta = {}
                    except Exception as e:
                        logger.warning("scan_local: meta %s: %s", cid, e)
                pack.append({"cid": cid, "weights": weights, "meta": meta})
            if pack:
                logger.info(
                    "local seed-pack found: n=%d total_bytes=%d (cap=%d)",
                    len(pack), sum(len(p["weights"]) for p in pack),
                    self.MAX_SEED_PACK_CREATURES)
            return pack
        except Exception as e:
            logger.warning("_scan_local_seed_pack failed: %s", e)
            return []

    async def _send_seed_pack(
        self, ws, pack: list[dict], *, seed_revision: int,
    ) -> None:
        """A1: отправить seed_pack_start/chunk*/complete после hello.

        Симметрия с push_owned_seed (P40→клиент): тот же chunk-size,
        тот же 5 мс throttle, base64-encoded payload. Сервер собирает
        в _handle(seed_pack_*), вызывает restore_personal_from_pack.
        """
        import base64
        total_bytes = sum(len(p["weights"]) for p in pack)
        await ws.send(json.dumps({
            "type": "seed_pack_start",
            "seed_revision": seed_revision,
            "n_creatures": len(pack),
            "total_bytes": total_bytes,
            "ts_client_ns": time.monotonic_ns(),
        }))
        sent_chunks = 0
        for entry in pack:
            cid = entry["cid"]
            weights: bytes = entry["weights"]
            meta: dict = entry["meta"] or {}
            chunks = [weights[i:i + self.SEED_PACK_CHUNK_SIZE]
                      for i in range(0, len(weights), self.SEED_PACK_CHUNK_SIZE)]
            if not chunks:
                chunks = [b""]
            total = len(chunks)
            for seq, blob in enumerate(chunks):
                msg: dict = {
                    "type": "seed_pack_chunk",
                    "seed_revision": seed_revision,
                    "cid": cid,
                    "seq": seq,
                    "total": total,
                    "payload_b64": base64.b64encode(blob).decode("ascii"),
                    "last": (seq == total - 1),
                }
                if seq == 0:
                    msg["meta"] = meta
                try:
                    await ws.send(json.dumps(msg))
                    sent_chunks += 1
                except Exception as e:
                    logger.warning(
                        "seed_pack_chunk send cid=%s seq=%d failed: %s",
                        cid, seq, e)
                    return
                await asyncio.sleep(0.005)
        await ws.send(json.dumps({
            "type": "seed_pack_complete",
            "seed_revision": seed_revision,
            "sent_creatures": len(pack),
            "sent_chunks": sent_chunks,
        }))
        logger.info(
            "seed_pack sent: n=%d chunks=%d bytes=%d revision=%d",
            len(pack), sent_chunks, total_bytes, seed_revision)

    async def _handle_seed_complete(self, msg: dict) -> None:
        seed_id = int(msg.get("seed_id", 0))
        ws = self._ws
        if ws is None:
            return
        ack = {
            "type": "seed_ack",
            "seed_id": seed_id,
            "accepted": self._seed_accepted,
            "failed": self._seed_failed,
            "ts": int(time.time() * 1000),
        }
        try:
            await ws.send(json.dumps(ack))
        except Exception as e:
            logger.warning("seed_ack send failed: %s", e)
        logger.info("seed_complete id=%d accepted=%d failed=%d",
                    seed_id, self._seed_accepted, self._seed_failed)

    # ── F4: seed.norg (центральный «днк-предок» от P40) ──────────────────

    def _handle_seed_norg_start(self, msg: dict) -> None:
        self._seed_norg_chunks = []
        self._seed_norg_sha = str(msg.get("sha256", "") or "")
        self._seed_norg_total = int(msg.get("total_bytes", 0) or 0)
        n_chunks = int(msg.get("n_chunks", 0) or 0)
        logger.info("seed_norg_start sha=%s bytes=%d chunks=%d",
                    self._seed_norg_sha[:12], self._seed_norg_total, n_chunks)

    def _handle_seed_norg_chunk(self, msg: dict) -> None:
        import base64
        payload_b64 = msg.get("payload_b64", "") or ""
        if not payload_b64:
            return
        try:
            self._seed_norg_chunks.append(base64.b64decode(payload_b64))
        except Exception as e:
            logger.warning("seed_norg_chunk b64 decode failed: %s", e)

    def _handle_seed_norg_complete(self, msg: dict) -> None:
        import hashlib
        from .seed_loader import write_seed_bytes
        data = b"".join(self._seed_norg_chunks)
        self._seed_norg_chunks = []
        expected = self._seed_norg_sha
        actual = hashlib.sha256(data).hexdigest() if data else ""
        if not data:
            logger.warning("seed_norg_complete: no data")
            return
        if expected and actual != expected:
            logger.warning("seed_norg_complete: sha mismatch expected=%s got=%s",
                           expected[:12], actual[:12])
            return
        try:
            path = write_seed_bytes(data, "wanderer")
        except Exception as e:
            logger.warning("seed_norg_complete: write failed: %s", e)
            return
        logger.info("seed_norg saved: %s sha=%s bytes=%d",
                    path, actual[:12], len(data))
        # Прокатываем отложенные finalize теперь, когда seed.norg на диске.
        pending = self._pending_finalize
        self._pending_finalize = []
        for key in pending:
            try:
                self._finalize_seed_creature(key)
            except Exception as e:
                logger.warning("deferred finalize %s failed: %s", key, e)

    # ── Brain migration Etap 3.1 — newborn attach + Y50 ─────────────────

    _NEWBORN_CACHE_TTL_SEC = 120.0

    def _attach_pending_newborns(self, creatures: list) -> None:
        """Etap 3.1: для каждой новой особи в obs_batch с известным parent_id —
        регистрирует child_org из кеша mate-pair кроссинговера и Y50-наследует
        мозг (predictor + S2.E/G/A/F) от родителя.

        Кеш заполняется в `_handle_mate_request` после успешного кроссинговера.
        Просроченные записи (>TTL) выкашиваются по дороге.
        """
        if self.compute is None or not self._pending_newborn_orgs:
            return
        # Сначала expire старых entries (>120s).
        now = time.time()
        stale = [
            k for k, (_, ts) in self._pending_newborn_orgs.items()
            if now - ts > self._NEWBORN_CACHE_TTL_SEC
        ]
        for k in stale:
            self._pending_newborn_orgs.pop(k, None)
        if not self._pending_newborn_orgs:
            return
        for c in creatures:
            cid = c.get("cid")
            parent_id = c.get("parent_id") or ""
            if not cid or not parent_id:
                continue
            cid_s = str(cid)
            if cid_s in self.compute.organisms:
                continue
            entry = self._pending_newborn_orgs.get(str(parent_id))
            if entry is None:
                continue
            child_org, _ts = entry
            # Унаследуем learning_rate/trace_decay от родителя если возможно.
            ctrl = self.compute.hebbian.get(str(parent_id))
            cfg = getattr(ctrl, "config", None) if ctrl is not None else None
            lr = float(getattr(cfg, "lr_reward", 1e-4)) if cfg else 1e-4
            decay = float(getattr(cfg, "eligibility_decay", 0.9)) if cfg else 0.9
            # Z7.i.d: lineage из snap'а (CreatureState.lineage). Дети
            # унаследуют lineage родителя; Z7.f hook на сервере уже мог
            # upgrade'нуть в "zodchiy", если parent.pending_upgrade_to_zodchiy
            # был выставлен. Читаем child's lineage напрямую из creature_tom.
            lineage = str(c.get("lineage", "") or "")
            if not lineage:
                try:
                    wc = getattr(self.compute, "world_cache", None)
                    if wc is not None:
                        tom_e = wc.creature_tom.get(cid_s)
                        if tom_e:
                            lineage = str(tom_e[0]) or ""
                except Exception:
                    pass
            if not lineage:
                lineage = "wanderer"
            try:
                self.compute.add_creature(
                    cid_s, child_org,
                    hebbian_enabled=True,
                    learning_rate=lr,
                    trace_decay=decay,
                    lineage=lineage,
                )
                # Variant B (19.05.2026): grace-метка для новорождённого.
                self._cid_last_seen_tick[cid_s] = self.last_world_tick
                self._newborn_attached += 1
            except Exception as e:
                logger.warning("attach newborn %s: %s", cid_s, e)
                continue
            try:
                ok = self.compute.inherit_brain_y50(str(parent_id), cid_s)
                if ok:
                    self._newborn_brain_inherited += 1
            except Exception as e:
                logger.warning("inherit_brain_y50 %s ← %s: %s",
                               cid_s, parent_id, e)
            # Body Migration Phase 2 этап 6 (27.05.2026, Бендер): gene
            # inheritance baseline-биохимии от матери. `add_creature` выше
            # установил `compute.biochem[cid_s] = make_default()` для
            # zodchiy (через Phase 2 этап 2 hook). Здесь заменяем на
            # inherited версию с σ=4.0 шумом по `inherit_baselines_*`.
            #
            # **Asexual fallback**: father biochem недоступен в текущей
            # mate_request schema (P40 шлёт только organism weights, не
            # baseline_*). Используем `inherit_baselines_asexual(child,
            # mother)`. Это semantic regression от ТЗ §2.4 sexual mean ±
            # noise — отдельный backlog item на schema bump (расширить
            # mate_request полем father_baseline_*).
            if lineage == "zodchiy":
                mother_bc = self.compute.biochem.get(str(parent_id))
                if mother_bc is not None:
                    try:
                        from .biochemistry import make_from_inheritance
                        self.compute.biochem[cid_s] = make_from_inheritance(
                            mother_bc, father=None)
                    except Exception as e:
                        logger.debug(
                            "biochem inherit %s ← %s: %s",
                            cid_s, parent_id, e)
            # Запись из кеша больше не нужна.
            self._pending_newborn_orgs.pop(str(parent_id), None)

    # ── Phase 3.3A: shadow obs из локального кеша (validation mode) ──────

    def _compare_shadow_obs(self, creatures: list, obs_per_cid: dict,
                              server_world_tick: int = 0) -> None:
        """Строит client obs из WorldStateCache и сравнивает с серверным.

        Не влияет на handle_tick (тот по-прежнему использует серверный obs).
        Логирует max-diff и счётчик match/mismatch в self._client_obs_*.
        Если кеш не bootstrap'нут или meta для cid отсутствует — increment
        skipped и идём дальше. Любой Exception — silent skip (не должен
        ронять основной поток).
        """
        cache = getattr(self, "world_cache", None)
        if cache is None or not getattr(cache, "is_bootstrapped", False):
            self._client_obs_skipped = getattr(self, "_client_obs_skipped", 0) + len(obs_per_cid)
            return
        try:
            import numpy as np
            from environment.observation_client import (
                ObsCreatureView, build_observation,
            )
        except Exception as e:
            logger.debug("shadow obs import failed: %s", e)
            self._client_obs_skipped += len(obs_per_cid)
            return
        # Сколько тиков кеш отстаёт от server_world_tick. Помогает понять
        # — расхождение от устаревшего snap'а, или от ошибки в builder'е.
        # Знаковая разница: кеш может уходить ВПЕРЁД obs_batch, если snap-поток
        # обогнал WS-очередь obs_batch'ей (происходит регулярно — snap пушится
        # ~500мс, obs_batch ~50мс, между ними нет тиковой синхронизации).
        cache_tick = int(getattr(cache, "last_tick", 0) or 0)
        snaps_applied = int(getattr(cache, "snaps_applied", 0) or 0)
        tick_diff = cache_tick - server_world_tick if server_world_tick else 0
        lag = max(0, server_world_tick - cache_tick) if server_world_tick else 0
        # Шейдов-сравнение валидно только при точно синхронизированных тиках.
        # Даже 1-тик lag даёт реальные расхождения в prey/pred slots 56-61
        # (хищник speed=3 за тик может перейти с N на S, prey_grad перевернётся).
        # Это не баг builder'а — это объективный сдвиг состояния Мира.
        # При tick_diff != 0 → инкрементируем skipped, выходим без сравнения.
        if server_world_tick and tick_diff != 0:
            self._client_obs_skipped += len(obs_per_cid)
            self._client_obs_last_tick_skip = {
                "cache_tick": cache_tick,
                "server_tick": int(server_world_tick),
                "tick_diff": int(tick_diff),
            }
            return
        if getattr(self, "_client_obs_slot_diff_sum", None) is None:
            try:
                self._client_obs_slot_diff_sum = np.zeros(64, dtype=np.float64)
            except Exception:
                self._client_obs_slot_diff_sum = None
        if not hasattr(self, "_client_obs_last_worst"):
            self._client_obs_last_worst = {}
        # creature_meta: {cid → (clan, sig_type)} — обновляется WorldFeedClient
        # из snap.creatures[]. Если для нашего cid ещё нет — пропускаем.
        meta = cache.creature_meta
        for c in creatures:
            cid = c.get("cid")
            if not cid:
                continue
            cid_s = str(cid)
            server_obs = obs_per_cid.get(cid_s)
            if server_obs is None:
                continue
            if cid_s not in meta:
                self._client_obs_skipped += 1
                continue
            try:
                cv = ObsCreatureView(
                    row=int(c.get("row", 0)),
                    col=int(c.get("col", 0)),
                    energy=float(c.get("energy", 0.0) or 0.0),
                    steps_taken=int(c.get("steps_taken", 0)),
                )
                view = cache.obs_world_view(self_cid=cid_s)
                client_obs = build_observation(cv, view)
                self._client_obs_built += 1
                slot_diff = np.abs(client_obs - server_obs)
                diff = float(slot_diff.max())
                if self._client_obs_slot_diff_sum is not None:
                    try:
                        self._client_obs_slot_diff_sum += slot_diff
                    except Exception:
                        pass
                if diff > self._client_obs_max_diff:
                    self._client_obs_max_diff = diff
                if diff < 1e-3:
                    self._client_obs_match += 1
                else:
                    self._client_obs_mismatch += 1
                    worst = int(slot_diff.argmax())
                    self._client_obs_last_worst = {
                        "slot": worst,
                        "client": float(client_obs[worst]),
                        "server": float(server_obs[worst]),
                        "max_diff": diff,
                        "lag_ticks": int(lag),
                        "cache_tick": cache_tick,
                        "server_tick": int(server_world_tick),
                        "snaps_applied": snaps_applied,
                    }
                    if self._client_obs_mismatch <= 5 or \
                            self._client_obs_mismatch % 100 == 0:
                        logger.info(
                            "shadow_obs mismatch cid=%s max=%.4f slot=%d "
                            "client=%.4f server=%.4f lag=%d",
                            cid_s, diff, worst,
                            float(client_obs[worst]),
                            float(server_obs[worst]),
                            int(lag),
                        )
            except Exception as e:
                logger.debug("shadow obs build %s: %s", cid_s, e)
                self._client_obs_skipped += 1

    # ── Phase F3.1.b/c: obs_batch → actions_batch ────────────────────────

    def _collect_obs_batch(
        self, creatures: list
    ) -> tuple[dict, dict, dict]:
        """Phase 3.3B: собирает obs/events/intero по cid.

        Если сервер прислал obs → используем (backward-compat для старых
        серверов). Если obs нет → строим локально из `world_cache`. После
        стабилизации серверный путь удалим. Stale до cache_tick (~3 тика
        lag) — допустимо, slow-moving особи это не ломает.
        """
        import numpy as np
        obs_per_cid: dict = {}
        events_per_cid: dict = {}
        intero_per_cid: dict = {}
        cache = getattr(self, "world_cache", None)
        local_builder_ready = bool(
            cache is not None and getattr(cache, "is_bootstrapped", False))
        _obs_view_cls = None
        _build_obs_fn = None
        if local_builder_ready:
            try:
                from environment.observation_client import (
                    ObsCreatureView as _ObsCreatureView,
                    build_observation as _build_observation,
                )
                _obs_view_cls = _ObsCreatureView
                _build_obs_fn = _build_observation
            except Exception as e:
                logger.debug("obs builder import failed: %s", e)
                local_builder_ready = False
        for c in creatures:
            cid = c.get("cid")
            if not cid:
                continue
            cid_s = str(cid)
            obs = c.get("obs")
            obs_arr = None
            if obs is not None:
                try:
                    obs_arr = np.asarray(obs, dtype=np.float32)
                except Exception as e:
                    logger.debug("obs parse %s: %s", cid_s, e)
                    obs_arr = None
            if obs_arr is None and local_builder_ready:
                # Сервер не шлёт obs (Phase 3.3B) → строим локально.
                # Если в кеше для cid нет meta — особь только что появилась
                # в obs_batch до прихода snap'а с creatures-дельтой.
                # Пропускаем, получит STAY на этом тике.
                try:
                    cv = _obs_view_cls(
                        row=int(c.get("row", 0)),
                        col=int(c.get("col", 0)),
                        energy=float(c.get("energy", 0.0) or 0.0),
                        steps_taken=int(c.get("steps_taken", 0)),
                    )
                    view = cache.obs_world_view(self_cid=cid_s)
                    obs_arr = _build_obs_fn(cv, view)
                    self._client_obs_local_built = (
                        getattr(self, "_client_obs_local_built", 0) + 1)
                except Exception as e:
                    logger.debug("local obs build %s: %s", cid_s, e)
                    obs_arr = None
            if obs_arr is None:
                continue
            obs_per_cid[cid_s] = obs_arr
            # Phase F3.2.a/b: события прошлого тика — для Hebbian R3 reward.
            # Поля могут отсутствовать у старых P40 — компонуем с дефолтами.
            events_per_cid[cid_s] = {
                "ate": bool(c.get("ate", False)),
                "killed": bool(c.get("killed", False)),
                "damage_taken": float(c.get("damage_taken", 0.0) or 0.0),
                "delta_energy": float(c.get("delta_energy", 0.0) or 0.0),
            }
            # Brain migration (10.05.2026): intero_7 для S2.F insula forward.
            # Старые серверы поле не шлют → fallback пустой.
            intero = c.get("intero")
            if intero is not None:
                try:
                    intero_per_cid[cid_s] = np.asarray(
                        intero, dtype=np.float32)
                except Exception:
                    pass
        return obs_per_cid, events_per_cid, intero_per_cid

    async def _handle_obs_batch(self, msg: dict) -> None:
        creatures = msg.get("creatures") or []
        world_tick = int(msg.get("world_tick", 0))
        ts_echo = msg.get("ts_p40_ns")
        self._obs_batches_received += 1
        if self.compute is None or not creatures:
            return
        # Brain migration Etap 3.1: до handle_tick — привязать новорождённых.
        # P40 шлёт parent_id для каждой особи. Если cid новый, и мы кешировали
        # child_org после mate-pair кроссинговера, регистрируем + Y50.
        self._attach_pending_newborns(creatures)
        obs_per_cid, events_per_cid, intero_per_cid = self._collect_obs_batch(
            creatures)
        if not obs_per_cid:
            return
        # Фаза 3.3A obs migration (11.05.2026): shadow-сборка obs из локального
        # кеша. Поведение не меняется — forward по-прежнему по серверному obs.
        # Сравниваем slot-by-slot, метрики идут в diagnostics(). После
        # подтверждения совпадения в live (3.3B) клиент переключается на свой
        # obs и сервер перестаёт его слать для owned.
        self._compare_shadow_obs(creatures, obs_per_cid,
                                   server_world_tick=world_tick)
        # Variant B (19.05.2026): фиксируем last_seen и запускаем GC ghost'ов.
        # world_tick==0 бывает только до welcome — на этой стадии compute
        # пуст и GC всё равно не нужен.
        if world_tick > 0:
            self.last_world_tick = world_tick
            for cid in obs_per_cid:
                self._cid_last_seen_tick[cid] = world_tick
            gced = self._gc_orphan_cids(world_tick)
            if gced:
                await self._send_owned_bye(gced)
        # 0.10.9 (21.05.2026): orphan-obs backstop. Если ни один cid из
        # obs_per_cid не зарегистрирован в compute.organisms — handle_tick
        # вернёт {} → ws.send actions_batch не сработает → P40 пометит
        # silent → brain fallback. Это и был cheef-PC залип 21.05 после
        # broker keepalive re-announce без re-seed (push фильтр 1.4.83).
        # Запрашиваем respawn у P40, чтобы тот заново прислал seed_pack.
        try:
            organisms = self.compute.organisms
            known = sum(1 for cid in obs_per_cid if cid in organisms)
            if known == 0:
                self._orphan_obs_streak += 1
                if (self._orphan_obs_streak
                        >= ORPHAN_OBS_RESPAWN_THRESHOLD):
                    self._orphan_obs_streak = 0
                    self._orphan_obs_respawns_sent += 1
                    asyncio.create_task(
                        self._request_orphan_respawn(world_tick))
            else:
                self._orphan_obs_streak = 0
        except Exception:
            self._orphan_obs_streak = 0
        # Body Migration Phase 2 (27.05.2026, Бендер): sync server-side
        # biochem deps в client biochem state перед handle_tick. Без этого
        # decay_step считает по stale полям (всегда energy=100, hydration=100,
        # infected=False) и cortisol-from-hunger / cortisol-from-thirst /
        # histamine-from-infection не работают. Использует .get(default)
        # — legacy server без поля не ломает.
        biochem_dict = getattr(self.compute, "biochem", None)
        if biochem_dict:
            for _c in creatures:
                _cid = _c.get("cid")
                _bc = biochem_dict.get(_cid) if _cid else None
                if _bc is None:
                    continue
                if "energy" in _c:
                    try:
                        _bc.energy = float(_c["energy"])
                    except (TypeError, ValueError):
                        pass
                if "hydration" in _c:
                    try:
                        _bc.hydration = float(_c["hydration"])
                    except (TypeError, ValueError):
                        pass
                if "infected" in _c:
                    _bc.infected = bool(_c["infected"])
                if "infection_severity" in _c:
                    try:
                        _bc.infection_severity = float(_c["infection_severity"])
                    except (TypeError, ValueError):
                        pass
                if "pair_bond_strength" in _c:
                    try:
                        _bc.pair_bond_strength = float(_c["pair_bond_strength"])
                    except (TypeError, ValueError):
                        pass
                if "last_social_tick" in _c:
                    try:
                        _bc.last_social_tick = int(_c["last_social_tick"])
                    except (TypeError, ValueError):
                        pass
                # Heal cosmetic backlog (memory project-roadmap-phase):
                # freshly seeded zodchiy starts с last_social_tick=0 → loner →
                # catatonic → frozen → не еят → death. Set к current world_tick
                # при первом obs_batch чтобы избежать false loner от рождения.
                if int(getattr(_bc, "last_social_tick", 0)) == 0:
                    try:
                        _bc.last_social_tick = int(world_tick)
                    except (TypeError, ValueError):
                        pass
        try:
            actions = self.compute.handle_tick(obs_per_cid,
                                                events_per_cid=events_per_cid,
                                                intero_per_cid=intero_per_cid,
                                                world_tick=world_tick)
        except Exception as e:
            logger.warning("handle_tick failed: %s", e)
            return
        ws = self._ws
        if ws is None or not actions:
            return
        # Phase emas pushback (03.05.2026): подмешиваем phase_emas per-creature,
        # чтобы сервер видел Phase 1/2/6 метрики для owned. Поле опциональное —
        # старые серверы игнорируют, новые применяют через _apply_phase_emas.
        creatures_out: list = []
        for cid, a in actions.items():
            entry: dict = {
                "cid": cid,
                "action": int(a["action"]),
                "target_id": a.get("target_id"),
            }
            try:
                emas = self.compute.get_phase_emas(cid) if self.compute else None
            except Exception:
                emas = None
            if emas:
                entry["phase_emas"] = emas
            creatures_out.append(entry)
        out = {
            "type": "actions_batch",
            "world_tick": world_tick,
            "ts_p40_ns_echo": ts_echo,
            "creatures": creatures_out,
        }
        try:
            await ws.send(json.dumps(out))
            self._actions_batches_sent += 1
        except Exception as e:
            logger.warning("actions_batch send failed: %s", e)

    async def _request_orphan_respawn(self, world_tick: int) -> None:
        """0.10.9 (21.05.2026): запросить у P40 повторную рассылку seed_pack.

        Триггерится в `_handle_obs_batch`, когда подряд ORPHAN_OBS_RESPAWN_THRESHOLD
        тиков ни один cid из obs_per_cid не оказался в `compute.organisms`.
        Используем тот же envelope, что и watchdog (`respawn_owned_request`):
        P40 шедулит `_async_respawn` (load_personal / genesis_personal) и
        присылает обычный seed_pack + colony_reset. Side-effect: cooldown через
        `_last_respawn_request_ts`, чтобы не штурмовать P40 при затяжной
        проблеме.
        """
        ws = self._ws
        if ws is None:
            return
        now = time.time()
        if (self._last_respawn_request_ts > 0
                and now - self._last_respawn_request_ts
                < EMPTY_WORLD_RETRY_SEC):
            return
        try:
            await ws.send(json.dumps({
                "type": "respawn_owned_request",
                "colony_name": self.colony_name,
                "mode": self.genesis_mode,
                "n": 8,
                "ts": int(now * 1000),
                "reason": "orphan_obs",
            }))
        except Exception as e:
            logger.warning("orphan respawn send failed: %s", e)
            return
        self._last_respawn_request_ts = now
        logger.warning(
            "orphan-obs backstop: respawn_owned_request sent "
            "(streak=%d, world_tick=%d, sent_total=%d)",
            ORPHAN_OBS_RESPAWN_THRESHOLD, world_tick,
            self._orphan_obs_respawns_sent)

    # ── Variant B (19.05.2026): GC ghost'ов в LocalColonyCompute ─────────

    def _gc_orphan_cids(self, world_tick: int) -> list[str]:
        """Чистим LocalColonyCompute от cid'ов, которых давно нет в obs_batch.

        P40 удаляет умерших из World молча (Phase F3.6 шлёт death только
        client → P40, не наоборот). Без GC `self.compute.organisms` и
        параллельные structures (predictor / hebbian / motor_sfnn_rule /
        higher_tissue_sfnn_*) растут до OOM. На cheef-PC 19.05 наблюдали
        2465 cid vs 10 live → ghost-leak ~85МБ предикторов.

        Запускается не чаще CID_GC_INTERVAL_TICKS, чистит cid с
        last_seen < world_tick - STALE_CID_TICKS либо отсутствующий
        в _cid_last_seen_tick совсем (значит не появлялся ни разу).

        Возвращает список удалённых cid — вызывающий шлёт по ним `owned_bye`
        (issue #2: явный despawn на P40 вместо ожидания 24ч freeze-sticky cap).
        """
        compute = self.compute
        if compute is None:
            return []
        if world_tick - self._cid_gc_last_run_tick < CID_GC_INTERVAL_TICKS:
            return []
        self._cid_gc_last_run_tick = world_tick
        threshold = world_tick - STALE_CID_TICKS
        if threshold <= 0:
            # World моложе grace-периода — рано GC, новорождённые могли
            # ещё не попасть в obs_batch.
            return []
        orphans: list[str] = []
        for cid in list(compute.organisms.keys()):
            last = self._cid_last_seen_tick.get(cid)
            if last is None or last < threshold:
                orphans.append(cid)
        if not orphans:
            return []
        for cid in orphans:
            try:
                compute.remove_creature(cid)
            except Exception as e:
                logger.warning("remove_creature %s failed: %s", cid, e)
            self._cid_last_seen_tick.pop(cid, None)
        self._cid_gc_total += len(orphans)
        logger.info(
            "cid GC: removed %d orphan(s), live=%d, total_gc=%d, threshold=%d",
            len(orphans), len(compute.organisms),
            self._cid_gc_total, threshold)
        return orphans

    async def _send_owned_bye(self, cids: list[str]) -> None:
        """Issue #2 (30.05.2026, Бендер): явный despawn owned cid на P40.

        Когда client локально выкинул cid (GC orphan, смерть battle/mate-fail,
        regen) — шлём `owned_bye {cids}`, чтобы P40 despawn'нул проекцию сразу,
        а не держал её frozen-sticky до 24ч safety cap. Counterpart к handler
        `_handle_owned_bye` в colony_pusher (neurocore 5512521).
        """
        ws = self._ws
        if ws is None or not cids:
            return
        try:
            await ws.send(json.dumps({
                "type": "owned_bye",
                "cids": list(cids),
                "ts": int(time.time() * 1000),
            }))
            self._owned_bye_sent += len(cids)
            logger.info("owned_bye sent: %d cid(s)", len(cids))
        except Exception as e:
            logger.warning("owned_bye send failed: %s", e)

    # ── Phase F3.3.a: weights_request → weights_dump ─────────────────────

    async def _handle_weights_request(self, msg: dict) -> None:
        """P40 запросил актуальные веса конкретной нашей особи.

        Сериализуем `compute.save_state(cid)` через torch.save → b64 → отдаём
        одним сообщением. Чанкование оставлено как долг — добавим, если
        замер blob+envelope превысит 512 КБ ws-лимит.
        """
        import base64
        import io
        cid = str(msg.get("cid", ""))
        request_id = msg.get("request_id")
        ws = self._ws
        if ws is None:
            return
        self._weights_requests_received += 1

        async def _reject(reason: str) -> None:
            try:
                await ws.send(json.dumps({
                    "type": "weights_dump",
                    "cid": cid, "request_id": request_id,
                    "ok": False, "reason": reason,
                }))
            except Exception as e:
                logger.warning("weights_dump reject send failed: %s", e)

        if not cid or self.compute is None:
            await _reject("no_compute")
            return
        payload = self.compute.save_state(cid)
        if payload is None:
            await _reject("unknown_cid")
            return
        try:
            import torch
            buf = io.BytesIO()
            torch.save(payload, buf)
            blob = buf.getvalue()
        except Exception as e:
            logger.warning("weights_dump serialize %s failed: %s", cid, e)
            await _reject(f"serialize_error: {type(e).__name__}")
            return
        try:
            await ws.send(json.dumps({
                "type": "weights_dump",
                "cid": cid, "request_id": request_id,
                "ok": True,
                "blob_b64": base64.b64encode(blob).decode("ascii"),
            }))
            self._weights_dumps_sent += 1
        except Exception as e:
            logger.warning("weights_dump send failed: %s", e)

    # ── Phase F3.3.b.2: mate_request → newborn ──────────────────────────

    async def _handle_mate_request(self, msg: dict) -> None:
        """P40 запросил у нас (мать-владельца) собрать ребёнка кроссинговером.

        Шаги: загружаем отца из присланных bytes (через тот же путь что и
        seed-finalize), берём свою мать из compute.organisms, deepcopy → reset
        → apply_crossover_inheritance → сериализуем child_blob → отвечаем.
        """
        import base64
        import copy as _copy
        request_id = msg.get("request_id")
        mother_cid = str(msg.get("mother_cid", ""))
        father_cid = str(msg.get("father_cid", ""))
        sigma_scale = float(msg.get("sigma_scale", 1.0) or 1.0)
        father_blob_b64 = msg.get("father_blob_b64", "")
        ws = self._ws
        if ws is None:
            return
        self._mate_requests_received += 1

        async def _reject(reason: str) -> None:
            self._mate_rejects_sent += 1
            try:
                await ws.send(json.dumps({
                    "type": "newborn",
                    "request_id": request_id,
                    "mother_cid": mother_cid,
                    "father_cid": father_cid,
                    "ok": False, "reason": reason,
                }))
            except Exception as e:
                logger.warning("newborn reject send failed: %s", e)

        if not mother_cid or self.compute is None:
            await _reject("no_compute")
            return
        mother_org = self.compute.organisms.get(mother_cid)
        if mother_org is None:
            await _reject("unknown_mother")
            return
        try:
            father_blob = base64.b64decode(father_blob_b64)
        except Exception as e:
            logger.warning("mate_request bad b64 from P40: %s", e)
            await _reject("bad_father_blob")
            return
        try:
            from .seed_loader import organism_from_weights, seed_cache_path
            father_org, _ = organism_from_weights(
                father_blob, seed_cache_path("wanderer"))
        except Exception as e:
            logger.warning("mate_request father load failed: %s", e)
            await _reject(f"father_load_error: {type(e).__name__}")
            return
        try:
            from .crossover import apply_crossover_inheritance, serialize_organism_blob
            child_org = _copy.deepcopy(mother_org)
            if hasattr(child_org, "reset_states"):
                child_org.reset_states()
            apply_crossover_inheritance(child_org, mother_org, father_org,
                                         sigma_scale=sigma_scale)
            child_blob = serialize_organism_blob(child_org)
        except Exception as e:
            logger.warning("mate_request crossover failed: %s", e)
            await _reject(f"crossover_error: {type(e).__name__}")
            return
        # Brain migration Etap 3.1: ребёнка пока неизвестно как зовут (P40
        # создаст CreatureState и пришлёт cid через obs_batch с parent_id).
        # Запомним child_org, чтобы при первом obs зарегистрировать в compute
        # и Y50-наследовать predictor + 4 высшие ткани от матери.
        try:
            self._pending_newborn_orgs[mother_cid] = (child_org, time.time())
        except Exception as e:
            logger.debug("pending_newborn cache failed: %s", e)
        try:
            await ws.send(json.dumps({
                "type": "newborn",
                "request_id": request_id,
                "mother_cid": mother_cid,
                "father_cid": father_cid,
                "ok": True,
                "child_blob_b64": base64.b64encode(child_blob).decode("ascii"),
            }))
            self._mate_newborns_sent += 1
        except Exception as e:
            logger.warning("newborn send failed: %s", e)

    async def _handle_echo_request(self, msg: dict) -> None:
        action = msg.get("action", "")
        if action == "stop":
            await self._stop_echo()
            logger.info("echo: stop")
            return
        if action == "start":
            duration = int(msg.get("duration_sec", 60) or 60)
            interval_ms = int(msg.get("summary_interval_ms", 1000) or 1000)
            fake_n = int(msg.get("fake_n", 50) or 50)
            await self._stop_echo()
            self._echo_seq = 0
            self._echo_task = asyncio.create_task(
                self._echo_send_loop(duration, interval_ms, fake_n))
            logger.info("echo: start dur=%ds interval=%dms n=%d",
                        duration, interval_ms, fake_n)

    async def _stop_echo(self) -> None:
        task = self._echo_task
        self._echo_task = None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

    async def _echo_send_loop(self, duration_sec: int, interval_ms: int,
                              fake_n: int) -> None:
        """Phase F3.0: шлёт фейковый tick_summary раз в interval_ms.

        Каждый summary — fake_n записей по ~80 байт (energy/age/pos/action/score).
        В ts_p40_ns_echo возвращаем последний полученный ts из world_snapshot —
        P40 считает round-trip latency.
        """
        import random
        deadline = time.monotonic() + duration_sec
        interval = max(0.05, interval_ms / 1000.0)
        try:
            while time.monotonic() < deadline:
                ws = self._ws
                if ws is None:
                    return
                self._echo_seq += 1
                summaries = [
                    {
                        "cid": f"c{i:04d}",
                        "e": round(random.random(), 3),
                        "a": random.randint(0, 1000),
                        "x": random.randint(0, 31),
                        "y": random.randint(0, 31),
                        "act": random.randint(0, 7),
                        "score": round(random.random(), 3),
                    }
                    for i in range(fake_n)
                ]
                payload = {
                    "type": "tick_summary",
                    "world_tick": self.last_world_tick,
                    "client_seq": self._echo_seq,
                    "ts_p40_ns_echo": self._echo_last_snap_ts_p40_ns,
                    "summaries": summaries,
                }
                try:
                    await ws.send(json.dumps(payload, separators=(",", ":")))
                except Exception as e:
                    logger.warning("echo send failed: %s", e)
                    return
                try:
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    return
        finally:
            logger.info("echo: send loop finished seq=%d", self._echo_seq)
