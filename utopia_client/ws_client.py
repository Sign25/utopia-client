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
# Perf (0.11.17, Бендер): min-интервал между handle_tick'ами. handle_tick
# GIL-bound (N орг × 20 тканей Python+torch); даже offload'нутый в to_thread он
# держит GIL. При catch-up бёрсте obs после reconnect тики идут back-to-back →
# GIL ~100% → ws-event-loop голодает → keepalive ping timeout. Cap ~6.7 Hz
# (>5 Hz P40-throttle, steady-state не трогает) рубит бёрсты, оставляя GIL-
# передышку event-loop'у. Дроп лишних obs безопасен (P40 reuse last action).
MIN_TICK_INTERVAL_SEC = 0.15
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
                        # Carrying-capacity cap (31.05.2026): restore сбросил
                        # .pt сверх ёмкости → owned_bye на P40, чтобы despawn'нул
                        # их проекции (иначе осиротеют → фантомное /stats).
                        cull = list(getattr(self.compute, "_cull_bye_cids", []) or [])
                        if cull:
                            await self._send_owned_bye(cull)
                            self.compute._cull_bye_cids = []
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
                proj_task.cancel()
                for t in (ping_task, save_task, empty_task, sync_task,
                          proj_task):
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
                # Elite snapshot (02.06.2026, Фрай): снимок здоровых обученных
                # мозгов в elite-слот (переживает вымирание) → recovery поднимает
                # оттуда. snapshot_elite гейтит на здоровье (>=min_alive живых).
                # Single-organism pivot (ТЗ e3cc81b, Фрай ОК#1): порог 4 — это
                # колониальное допущение; для n=1 Адама len(alive)=1<4 → elite
                # молча не снимался → страховка-на-вымирание мертва. Под флагом
                # min_alive=1 (адаптация порога, НЕ гейт — durability-positive).
                _min_alive = 1 if getattr(
                    self.compute, "_single_organism", False) else 4
                await asyncio.to_thread(
                    self.compute.snapshot_elite,
                    colony_state_dir(self.colony_name) / "elite",
                    _min_alive)
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
        # Client-authoritative НЕ шлёт legacy respawn_owned_request (31.05.2026,
        # Бендер): источник десинка — legacy _async_respawn создавал клоны с
        # НОВЫМИ cid'ами, не совпадающими с client-набором → P40/client desync,
        # rates не шли. В client-authoritative организмы приходят из local .pt
        # + projection-model self-heal + reproduction, НЕ из legacy respawn.
        # Пустая client-authoritative колония восстанавливается genesis'ом
        # (n_local=0 ветка) или репродукцией, не legacy. Для остальных user'ов
        # (n_local=0 на старте) respawn остаётся штатным.
        if getattr(self, "_reject_incoming_seeds", False):
            # Client-authoritative recovery (02.06.2026, Фрай): вымирание online
            # → НЕ legacy-respawn (десинк клонов), а re-genesis из ELITE-слота
            # (обученные мозги) → обучение продолжается, не с untrained-нуля.
            await self._maybe_recover_from_elite(connect_ts)
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

    async def _maybe_recover_from_elite(self, connect_ts: float) -> bool:
        """Client-authoritative recovery (02.06.2026, Фрай): вымирание online →
        re-genesis из elite-слота (обученные мозги, переживают вымирание). P40
        auto_respawn отключён 28.05 (десинк клонов) → клиент восстанавливает
        сам, НЕ ждёт reconnect. Cooldown EMPTY_WORLD_RETRY_SEC. Grace —
        отличить транзитный obs-гэп от настоящего вымирания."""
        if self.compute is None or self.n_alive_owned > 0:
            return False
        now = time.time()
        ref = self.last_owned_alive_ts or connect_ts
        if now - ref < EMPTY_WORLD_GRACE_SEC:
            return False
        if (getattr(self, "_last_elite_recover_ts", 0.0) > 0
                and now - self._last_elite_recover_ts < EMPTY_WORLD_RETRY_SEC):
            return False
        # Подтверждаем РЕАЛЬНОЕ вымирание (compute пуст или все мертвы).
        try:
            alive = sum(1 for cid in list(self.compute.organisms.keys())
                        if cid not in self.compute._dead_cids)
        except Exception:
            alive = 0
        if alive > 0:
            return False
        try:
            from .seed_loader import colony_state_dir
            elite_dir = colony_state_dir(self.colony_name) / "elite"
            restored = await asyncio.to_thread(
                self.compute.restore_colony_from_local, elite_dir)
            self._last_elite_recover_ts = now
            if restored:
                # Воскрешённые cid'ы могли остаться в _dead_cids → снимаем,
                # иначе projection пометит alive=False сразу.
                for cid in restored:
                    self.compute._dead_cids.discard(cid)
                self.last_owned_alive_ts = now  # сброс grace
                self.n_alive_owned = len(restored)
                logger.info(
                    "ELITE RECOVERY: re-genesis %d обученных мозгов из elite "
                    "(вымирание %.0fs) — обучение продолжается", len(restored),
                    now - ref)
                return True
            logger.warning(
                "ELITE RECOVERY: elite-слот пуст (%s) — нечего поднимать",
                elite_dir)
        except Exception as e:
            logger.warning("ELITE RECOVERY failed: %s", e)
        return False

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

    @staticmethod
    def _apply_weather_to_obs(obs_arr, temperature):
        """§10.9 ПОГОДА v0.1 (контракт Хьюберт/Фрай 09.06): temperature ∈[-1,1] →
        obs[35] (Adam-only слот, был steps_taken/5000 — не-сенсорный счётчик
        возраста; steps_taken жив в payload-поле, инфа не теряется). obs Адама
        строит КЛИЕНТ (owned skip_obs) → temp инжектится ЗДЕСЬ; server-патч
        obs[35] owned не достигает. Поле Adam-only (Хьюберт шлёт только Адаму) →
        presence = гейт. None → no-op (до деплоя weather.py). Predictor target
        авто-включает temp@35 (obs[:64] фикс) → prediction-давление. v0.1
        perception-only (без world-эффекта; respawn-возмущение — v0.2). Мутирует
        obs_arr на месте, возвращает его."""
        if temperature is None:
            return obs_arr
        try:
            obs_arr[35] = float(temperature)
        except (IndexError, TypeError, ValueError):
            pass
        return obs_arr

    @staticmethod
    def _apply_rhythm_to_obs(obs_arr, rhythm):
        """Ритм-аффорданс (Фрай 14.06): 4 циклических фазовых канала → obs[68:72]
        (STATE_DIM-хвост [64:80]: P40 zeros, designated internal, client-free).
        rhythm = dict {day_phase_sin/cos, year_phase_sin/cos} ∈[-1,1] (контракт
        Хьюберт WORLD_ADAM_TIME_PHASE_OBS; Адаму отд. полем — skip_obs, как
        temperature@35). None/не-dict → no-op (dormant, флаг OFF). obs Адама может
        быть уже, чем 72 (local-builder) → паддим до 80 (контракт np[80]), чтобы
        ритм-канал существовал при флаге ON; [64:68] не используется (self4
        строится client-side). daytime_fraction — диагностика, НЕ встраивается.
        Мутирует/возвращает obs_arr (паддинг возвращает НОВЫЙ массив)."""
        if not isinstance(rhythm, dict) or obs_arr is None:
            return obs_arr
        import numpy as np
        try:
            if obs_arr.shape[0] < 72:
                _pad = np.zeros(80, dtype=obs_arr.dtype)
                _pad[:obs_arr.shape[0]] = obs_arr
                obs_arr = _pad
            obs_arr[68] = float(rhythm.get("day_phase_sin", 0.0) or 0.0)
            obs_arr[69] = float(rhythm.get("day_phase_cos", 0.0) or 0.0)
            obs_arr[70] = float(rhythm.get("year_phase_sin", 0.0) or 0.0)
            obs_arr[71] = float(rhythm.get("year_phase_cos", 0.0) or 0.0)
        except (IndexError, TypeError, ValueError):
            pass
        return obs_arr

    @staticmethod
    def _apply_social_to_obs(obs_arr, social):
        """social_signals этап A (Фрай 16.06): 4 направленных tribe-канала → obs[72:76]
        (STATE_DIM-хвост [64:80]: P40 zeros, designated internal, client-free).
        social = dict {food_ns, food_ew, danger_ns, danger_ew} ∈[-1,1] (контракт
        Хьюберт WORLD_ADAM_TRIBE_SIGNALS; payload-ключ tribe_signals, Адаму отд.
        полем — skip_obs, как temperature@35). None/не-dict → no-op (dormant, флаг
        OFF). Паддим до 80 если obs уже <76 (контракт np[80]). Зеркалит
        _apply_rhythm_to_obs. Мутирует/возвращает obs_arr (паддинг → НОВЫЙ массив)."""
        if not isinstance(social, dict) or obs_arr is None:
            return obs_arr
        import numpy as np
        try:
            if obs_arr.shape[0] < 76:
                _pad = np.zeros(80, dtype=obs_arr.dtype)
                _pad[:obs_arr.shape[0]] = obs_arr
                obs_arr = _pad
            obs_arr[72] = float(social.get("food_ns", 0.0) or 0.0)
            obs_arr[73] = float(social.get("food_ew", 0.0) or 0.0)
            obs_arr[74] = float(social.get("danger_ns", 0.0) or 0.0)
            obs_arr[75] = float(social.get("danger_ew", 0.0) or 0.0)
        except (IndexError, TypeError, ValueError):
            pass
        return obs_arr

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
        rates_per_cid: dict = {}
        # Newborn-инстинкт (01.06.2026, Фрай/Хьюберт): on_flora + carried_food.
        # P40 шлёт authoritative on_flora/flora_kind/carried_food (deploy 9f8d99d)
        # — убирает desync (client-угадывание по stale cache.flora+creature_pos →
        # p40_ate=0). Предпочитаем P40, fallback на кэш (переходный период).
        on_flora_per_cid: dict = {}
        carried_food_per_cid: dict = {}
        nearest_flora_per_cid: dict = {}  # Хьюберт 05.06: {dr,dc,dist,kind}|None
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
        # Eat income (01.06.2026): flora-позиции для _flora_income — РАЗ на батч
        # (не на cid). cache.flora = set[(row,col,kind)] → dict[(r,c)→kind].
        flora_pos: dict = {}
        if cache is not None:
            try:
                flora_pos = {(int(r), int(cc)): int(k)
                             for (r, cc, k) in cache.flora}
            except Exception:
                flora_pos = {}
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
            # §10.9 ПОГОДА v0.1: temperature → obs[35]. Хьюберт шлёт ВЛОЖЕННО
            # c["weather"]["temperature"] (+ патчит server-obs[35]). Читаем оба
            # места (nested + flat fallback). Идемпотентно: Адам на server-obs
            # (temp@35 уже есть) → пишем то же; client-built → заполняем. None →
            # no-op. WEATHER_DIAG ниже логирует obs[35] для валидации.
            _wx = c.get("weather")
            _temp = (_wx.get("temperature") if isinstance(_wx, dict) else None)
            if _temp is None:
                _temp = c.get("temperature")
            obs_arr = self._apply_weather_to_obs(obs_arr, _temp)
            # РИТМ (Фрай 14.06): циклическое время → obs[68:72] (STATE_DIM-хвост,
            # client-free). Хьюберт шлёт под WORLD_ADAM_TIME_PHASE_OBS отд. полями
            # (skip_obs, как temperature). Контейнер: nested c["time_phase"] |
            # flat-поля на c (читаем оба — контракт контейнера подтвердить с
            # Хьюбертом при активации по RHYTHM_DIAG). None → no-op (dormant).
            # GATE: client_flag rhythm (compute._rhythm_enabled) — независимый
            # client-rollback, парный к server-флагу. OFF → НЕ инжектим → obs[68:72]=0
            # → RHYTHM_DIAG честен (diag≠0 ⟺ predictor видит ритм). Флипать синхронно.
            _rh = None
            if getattr(self.compute, "_rhythm_enabled", False):
                # nested-контейнер приоритетно, иначе собрать из flat-полей
                _rh_nested = c.get("time_phase")
                if isinstance(_rh_nested, dict):
                    _rh = _rh_nested
                elif any(k in c for k in ("day_phase_sin", "day_phase_cos",
                                          "year_phase_sin", "year_phase_cos")):
                    _rh = {
                        "day_phase_sin": c.get("day_phase_sin", 0.0),
                        "day_phase_cos": c.get("day_phase_cos", 0.0),
                        "year_phase_sin": c.get("year_phase_sin", 0.0),
                        "year_phase_cos": c.get("year_phase_cos", 0.0),
                    }
            obs_arr = self._apply_rhythm_to_obs(obs_arr, _rh)
            obs_per_cid[cid_s] = obs_arr
            if _rh is not None:
                self._rhythm_diag_n = getattr(self, "_rhythm_diag_n", 0) + 1
                if self._rhythm_diag_n % 600 == 1:
                    try:
                        logger.info(
                            "RHYTHM_DIAG cid=%s obs[68:72]=[%.3f,%.3f,%.3f,%.3f] "
                            "daytime_frac=%s src=%s", cid_s,
                            float(obs_arr[68]), float(obs_arr[69]),
                            float(obs_arr[70]), float(obs_arr[71]),
                            c.get("daytime_fraction"),
                            "time_phase" if isinstance(c.get("time_phase"), dict)
                            else "flat")
                    except Exception:
                        pass
            # social_signals этап A (Фрай 16.06): tribe-радар Старших → obs[72:76]
            # (STATE_DIM-хвост, client-free). Хьюберт шлёт payload-ключом tribe_signal
            # {food_ns,food_ew,danger_ns,danger_ew} ∈[-1,1] под WORLD_ADAM_TRIBE_SIGNALS
            # (отд. полем — skip_obs, как temperature). GATE: client_flag social_signals
            # (compute._social_enabled) — независимый rollback, парный к server-флагу +
            # WORLD_ELDER_PROACTIVE_DANGER. OFF → НЕ инжектим → obs[72:76]=0 →
            # math-equivalent довходу 72. Флипать СИНХРОННО (joint-go Хьюберта+Фрая).
            _soc = None
            if getattr(self.compute, "_social_enabled", False):
                # tribe_signal (sing.) = канон сервера a12f4cf; tribe_signals (plur.)
                # — fallback на ранний контракт. Затем flat-поля. Зеркало weather-nested.
                _soc_nested = c.get("tribe_signal")
                if not isinstance(_soc_nested, dict):
                    _soc_nested = c.get("tribe_signals")
                if isinstance(_soc_nested, dict):
                    _soc = _soc_nested
                elif any(k in c for k in ("food_ns", "food_ew",
                                          "danger_ns", "danger_ew")):
                    _soc = {
                        "food_ns": c.get("food_ns", 0.0),
                        "food_ew": c.get("food_ew", 0.0),
                        "danger_ns": c.get("danger_ns", 0.0),
                        "danger_ew": c.get("danger_ew", 0.0),
                    }
            if _soc is not None:
                obs_arr = self._apply_social_to_obs(obs_arr, _soc)
                obs_per_cid[cid_s] = obs_arr
                self._social_diag_n = getattr(self, "_social_diag_n", 0) + 1
                if self._social_diag_n % 600 == 1:
                    try:
                        logger.info(
                            "SOCIAL_DIAG cid=%s obs[72:76]=[%.3f,%.3f,%.3f,%.3f] "
                            "src=%s", cid_s, float(obs_arr[72]), float(obs_arr[73]),
                            float(obs_arr[74]), float(obs_arr[75]),
                            "tribe_signal" if isinstance(c.get("tribe_signal"), dict)
                            else "tribe_signals" if isinstance(c.get("tribe_signals"), dict)
                            else "flat")
                    except Exception:
                        pass
            if _temp is not None:
                self._weather_diag_n = getattr(self, "_weather_diag_n", 0) + 1
                if self._weather_diag_n % 600 == 1:
                    try:
                        logger.info("WEATHER_DIAG cid=%s obs[35]=%.4f temp=%.4f "
                                    "src=%s server_obs=%s", cid_s,
                                    float(obs_arr[35]), float(_temp),
                                    "weather" if isinstance(_wx, dict) else "flat",
                                    c.get("obs") is not None)
                    except Exception:
                        pass
            # Phase F3.2.a/b: события прошлого тика — для Hebbian R3 reward.
            # Поля могут отсутствовать у старых P40 — компонуем с дефолтами.
            events_per_cid[cid_s] = {
                "ate": bool(c.get("ate", False)),
                "killed": bool(c.get("killed", False)),
                "damage_taken": float(c.get("damage_taken", 0.0) or 0.0),
                "delta_energy": float(c.get("delta_energy", 0.0) or 0.0),
                # PHASE 2.5m (Хьюберт 9f6e9e7): kill-аккумуляторы (read-and-reset,
                # паттерн damage_acc 0.13.15) — НЕ теряются при throttle 3-5 Hz, в
                # отличие от per-tick killed/delta_energy. kill_energy_acc=Σ энергии
                # убийств (55 medium / 21 prey), kill_count_acc=число. Энергия УЖЕ
                # применена через delta_energy — здесь ТОЛЬКО meat-GC ось + счётчик.
                "kill_energy_acc": float(c.get("kill_energy_acc", 0.0) or 0.0),
                "kill_count_acc": int(c.get("kill_count_acc", 0) or 0),
                # PHASE B eating (#6, eating.md): прогресс многотикового поедания —
                # eating_progress(0..1)/target_kind/remaining/total. Рефлекс-floor
                # (Phase A) уже держит «ест→продолжать»; B потребляет progress для
                # awareness + near-complete-commit (не бросать ради upgrade).
                "eating_progress": float(c.get("eating_progress", 0.0) or 0.0),
                "eating_target_kind": c.get("eating_target_kind"),
                "eating_remaining_ticks": int(c.get("eating_remaining_ticks", 0) or 0),
                "eating_total_ticks": int(c.get("eating_total_ticks", 0) or 0),
                # PHASE C (eating.md): труп от kill. on_corpse — Адам НА тайле трупа
                # → corpse-EAT рефлекс (тот же детерм. floor, что on_flora).
                "on_corpse": bool(c.get("on_corpse", False)),
            }
            # PHASE B confirm: подтвердить эмит obs #6 Хьюбертом (1/50, rate-limited).
            if "eating_progress" in c:
                self._eatprog_n = getattr(self, "_eatprog_n", 0) + 1
                if self._eatprog_n % 50 == 0:
                    logger.info("EAT_PROGRESS cid=%s progress=%.2f kind=%s remain=%s/%s",
                                cid_s, float(c.get("eating_progress", 0.0) or 0.0),
                                c.get("eating_target_kind"),
                                c.get("eating_remaining_ticks"),
                                c.get("eating_total_ticks"))
            # Infection contact (01.06.2026, Фрай): P40 детектит контакт
            # больной↔здоровый (физика пространства) → events.infection_contact.
            # Прокидываем в event → _apply_biochem_events бутстрапит infection
            # (infected=True, severity=0.05); прогресс/death ведёт клиент.
            # Формат (Хьюберт): список [{from_cid, severity_hint}]. Прокидываем
            # как есть → _apply_biochem_events берёт max severity_hint.
            _ic = c.get("infection_contact")
            if _ic:
                events_per_cid[cid_s]["infection_contact"] = _ic
            # §3 death_suppressed (Хьюберт 2b0f3a2): P40 chokepoint suppress'нул
            # death-вектор (PvP/age/...) → events.death_suppressed=[{reason}].
            # Второй вход в paralysis (energy-независим, Фрай). Клиент применяет
            # paralysis своей механикой (handle_tick → _enter_paralysis).
            _ds = c.get("death_suppressed")
            if _ds:
                events_per_cid[cid_s]["death_suppressed"] = _ds
            # Hydration income (31.05→01.06.2026, Шеф): питьё — из бесконечного
            # террейна (WATER-тайл ничего не расходует, арбитраж P40 не нужен, в
            # отличие от еды-сущности). Клиент владеет всем водным контуром:
            # если P40 прислал delta_hydration — берём его; иначе начисляем сами
            # при нахождении на/рядом с водой (_near_water). Переиспользует
            # готовый путь _apply_biochem_events (clamp до max_hydration).
            # Client владеет водным доходом (01.06.2026, Шеф). P40 ШЛЁТ
            # delta_hydration=0 для owned (их метаболизм не симулируется
            # server-side) → нельзя гейтить income на отсутствие ключа (был
            # баг: client-income в else к `if delta_hydration in c` → мёртв с
            # 0.11.30, P40-ключ всегда перебивал). Теперь приоритет: ненулевой
            # P40 > client-террейн > нулевой P40 (последний сохраняем для
            # активации hydration-оси в _apply_biochem_events).
            _p40_dh = (float(c.get("delta_hydration", 0.0) or 0.0)
                       if "delta_hydration" in c else None)
            _rc = self._resolve_pos(cid_s, c)
            # on_flora: P40 authoritative (9f8d99d) > client cache.flora fallback.
            _p40_onf = c.get("on_flora")
            if _p40_onf is not None:
                on_flora_per_cid[cid_s] = bool(_p40_onf)
            else:
                on_flora_per_cid[cid_s] = bool(
                    _rc is not None and (_rc[0], _rc[1]) in flora_pos)
            # nearest_flora точный нав-сигнал (Хьюберт 05.06): {dr,dc,dist,kind}|None
            # (Elder argmin). Прайор: dist==0→GATHER, dist>0→точный шаг, None→smell.
            _p40_nf = c.get("nearest_flora")
            # PHASE 2 (Хьюберт 39ba52b): nearest_medium_prey — отдельный нав-сигнал
            # на среднюю дичь (55, труднее мелкой). Может прийти БЕЗ флоры рядом →
            # держим контейнер даже при nearest_flora=None (kind=None → discrimination
            # не путаем; нав читает ["medium_prey"] по паттерну ["edible"]).
            _p40_mp = c.get("nearest_medium_prey")
            # PHASE C: nearest_corpse {dr,dc,dist,energy_remaining} — нав к трупу (как
            # medium_prey). Держим контейнер даже при nearest_flora=None.
            _p40_corpse = c.get("nearest_corpse")
            # PREDATOR-HUNT (Фрай 14.06): nearest_predator {dr,dc,dist,hp,max_hp,hp_ratio,
            # attackable,kind,fauna_id} — добивание РАНЕНОГО хищника (как medium_prey/corpse).
            _p40_pred = c.get("nearest_predator")
            if (_p40_nf is not None or _p40_mp is not None
                    or _p40_corpse is not None or _p40_pred is not None):
                _ent = dict(_p40_nf) if _p40_nf is not None else {
                    "dr": 0, "dc": 0, "dist": None, "kind": None}
                # Phase 1 feeding-ladder (Хьюберт d972ea7, Adam-only): nearest_EDIBLE
                # для НАВИГАЦИИ (мимо обесцененной травы); legacy nearest_flora +
                # obs[63] kind остаются для discrimination (Фрай-инвариант). Кладём
                # edible-цель в ["edible"] — obs[62-63] читают legacy, нав читает edible.
                _p40_ne = c.get("nearest_edible_flora")
                if _p40_ne is not None:
                    _ent["edible"] = _p40_ne
                if _p40_mp is not None:
                    _ent["medium_prey"] = _p40_mp
                if _p40_corpse is not None:
                    _ent["corpse"] = _p40_corpse
                if _p40_pred is not None:
                    _ent["predator_hunt"] = _p40_pred
                nearest_flora_per_cid[cid_s] = _ent
            # carried_food: P40 authoritative (physics на P40) — если шлёт.
            _p40_cf = c.get("carried_food")
            if _p40_cf is not None:
                try:
                    carried_food_per_cid[cid_s] = int(_p40_cf)
                except (TypeError, ValueError):
                    pass
            # NAV_VIS (01.06.2026, Фрай): видят ли флору в радиусе зрения (vr) +
            # on_flora-rate. Низкий sees_rate → «пустыни»/мало флоры; высокий
            # sees, низкий onf → видят, но не доходят (навигация). Аккумулируем.
            if not hasattr(self, "_nav_vis"):
                self._nav_vis = {"cids": 0, "sees": 0, "onf": 0, "eat": 0,
                                 "batches": 0}
            _nv = self._nav_vis
            _nv["cids"] += 1
            if on_flora_per_cid[cid_s]:
                _nv["onf"] += 1
            # /stats Блок 4 active_eat_rate: доля тиков с эмитом EAT(14)/GATHER(13)
            # (compute.last_action) — «учится ли активно добывать» (Track 2).
            if self.compute is not None:
                _la = self.compute._stat_last_action.get(cid_s, -1) \
                    if hasattr(self.compute, "_stat_last_action") else -1
                if _la in (13, 14):
                    _nv["eat"] = _nv.get("eat", 0) + 1
            if _rc is not None:
                _vr = int(float((self.compute.traits.get(cid_s) or {}).get(
                    "vision_radius", 7) or 7)) if self.compute else 7
                if self._flora_in_radius(_rc[0], _rc[1], flora_pos, _vr):
                    _nv["sees"] += 1
                # ARRIVAL-диагностика (Фрай 05.06): локальная flora-плотность +
                # достижимость — развести (a) distribution (локально пусто) vs
                # (b) granularity/pathing (рядом есть, не ландится). Rate-limit.
                self._flora_diag_n = getattr(self, "_flora_diag_n", 0) + 1
                if self._flora_diag_n % 50 == 0:
                    _n1, _n2, _n3, _nd, _wp = self._local_flora_diag(
                        _rc[0], _rc[1], flora_pos)
                    logger.info(
                        "FLORA_LOCAL_DIAG cid=%s pos=(%d,%d) n_r1=%d n_r2=%d "
                        "n_r3=%d nearest=%d water_path=%d global=%d",
                        cid_s, _rc[0], _rc[1], _n1, _n2, _n3, _nd, _wp,
                        len(flora_pos))
            _near = (_rc is not None and self._near_water(_rc[0], _rc[1]))
            if _p40_dh is not None and _p40_dh != 0.0:
                events_per_cid[cid_s]["delta_hydration"] = _p40_dh
            elif _near:
                events_per_cid[cid_s]["delta_hydration"] = self._WATER_RESTORE
            elif _p40_dh is not None:
                events_per_cid[cid_s]["delta_hydration"] = _p40_dh  # 0 → ось вкл
            # Eat income client-side (01.06.2026, Шеф/Хьюберт): симметрия с водой.
            # P40 шлёт delta_energy>0 только когда owned ТОЧНО на flora-тайле
            # (passive_eating), но зодчие ходят случайно → редко попадают → голод
            # за 13мин. Кредитуем близость к flora (радиус 1), если P40 прислал 0
            # (приоритет ненулевого P40 — без двойного начисления on-flora).
            if events_per_cid[cid_s]["delta_energy"] == 0.0 and _rc is not None:
                # Faithful passive_flora_eating (server world.py:3174-3259):
                # traits vision/diet/eff из стора (дефолты server: vision=7,
                # diet=0.5, eff=10). Заменяет прежний fixed _flora_income —
                # чистый порт серверной формулы, без заплаток.
                _tr = ((self.compute.traits.get(cid_s) or {})
                       if self.compute is not None else {})
                _fe = self._passive_flora_income(
                    _rc[0], _rc[1], flora_pos,
                    float(_tr.get("vision_radius", 7) or 7),
                    float(_tr.get("diet_gene", 0.5) or 0.5),
                    float(_tr.get("efficiency", 10) or 10))
                if _fe > 0.0:
                    events_per_cid[cid_s]["delta_energy"] = _fe
            # Brain migration (10.05.2026): intero_7 для S2.F insula forward.
            # Старые серверы поле не шлют → fallback пустой.
            intero = c.get("intero")
            if intero is not None:
                try:
                    intero_per_cid[cid_s] = np.asarray(
                        intero, dtype=np.float32)
                except Exception:
                    pass
            # Body Migration метаболизм (31.05.2026, контракт Хьюберт): P40 шлёт
            # effective per-tick rates для owned-zodchiy. Client интегрирует
            # (energy/hydration/telomere) в handle_tick → _apply_metabolism.
            # Per-sec контракт (01.06.2026, Хьюберт): P40 переименовывает
            # *_now → *_per_sec (energy/сек). Нормализуем в *_per_sec, предпочитая
            # новые ключи, с fallback на *_now — НЕТ окна rate=0 в переходный
            # период (когда P40 уже шлёт *_per_sec, а handler ждал *_now).
            _has_ps = any(k in c for k in (
                "step_cost_per_sec", "telomere_decay_per_sec",
                "thirst_per_sec"))
            # §3.5 АСИНХРОННЫЕ ТЕМПЫ (Фрай 06.06): per-tick rate от Хьюберта
            # (= per_sec / world_TPS, server знает TPS) → метаболизм Адама на
            # ЕГО тиковой шкале, не wall-dt. _apply_metabolism применит per-apply.
            _has_tick = "step_cost_per_tick" in c
            if _has_ps or _has_tick or any(k in c for k in (
                    "step_cost_now", "telomere_decay_now", "thirst_now")):
                rates_per_cid[cid_s] = {
                    "step_cost_per_sec": float(c.get(
                        "step_cost_per_sec", c.get("step_cost_now", 0.0)) or 0.0),
                    "telomere_decay_per_sec": float(c.get(
                        "telomere_decay_per_sec",
                        c.get("telomere_decay_now", 0.0)) or 0.0),
                    "thirst_per_sec": float(c.get(
                        "thirst_per_sec", c.get("thirst_now", 0.0)) or 0.0),
                    # Режим: P40 уже шлёт *_per_sec → wall-clock dt-интеграция;
                    # только *_now (legacy) → dt=1 (per-apply, без регресса/over-
                    # drain). Авто-апгрейд когда Хьюберт завершит rename.
                    "_per_sec": _has_ps,
                }
                if _has_tick:
                    rates_per_cid[cid_s]["step_cost_per_tick"] = float(
                        c.get("step_cost_per_tick", 0.0) or 0.0)
                    rates_per_cid[cid_s]["thirst_per_tick"] = float(
                        c.get("thirst_per_tick", 0.0) or 0.0)
                    rates_per_cid[cid_s]["telomere_decay_per_tick"] = float(
                        c.get("telomere_decay_per_tick", 0.0) or 0.0)
                    # DAMAGE-канал (Фрай 07.06): predator damage_per_tick →
                    # energy per-client-tick (§3.5). 0 когда хищник не бьёт.
                    rates_per_cid[cid_s]["damage_per_tick"] = float(
                        c.get("damage_per_tick", 0.0) or 0.0)
                    # BMR (Шеф 12.06, Phase 2.5h): базовый метаболизм — energy
                    # тратится ВСЕГДА (STAY/EAT/GATHER тоже), не только при движении.
                    # = step_cost × φ⁻² (Хьюберт 96c5c8f). Применяется безусловно в
                    # _apply_metabolism (отдельно от move-gated step_cost).
                    rates_per_cid[cid_s]["basal_drain_per_tick"] = float(
                        c.get("basal_drain_per_tick", 0.0) or 0.0)
                rates_per_cid[cid_s]["basal_drain_per_sec"] = float(
                    c.get("basal_drain_per_sec", 0.0) or 0.0)
        # NAV_VIS периодический лог (раз в ~300 батчей ≈ 60с при 5Гц).
        if hasattr(self, "_nav_vis"):
            self._nav_vis["batches"] += 1
            if self._nav_vis["batches"] >= 300:
                _nv = self._nav_vis
                _cc = max(1, _nv["cids"])
                logger.info(
                    "NAV_VIS batches=300 sees_flora_rate=%.3f onf_rate=%.3f "
                    "active_eat_rate=%.3f flora_total=%d cid_samples=%d",
                    _nv["sees"] / _cc, _nv["onf"] / _cc,
                    _nv.get("eat", 0) / _cc, len(flora_pos), _nv["cids"])
                # /stats Блок 4: стопка foraging-долей в compute → owner-diagnostics.
                if self.compute is not None:
                    self.compute._stat_foraging = {
                        "onf_rate": round(_nv["onf"] / _cc, 3),
                        "sees_flora_rate": round(_nv["sees"] / _cc, 3),
                        "active_eat_rate": round(_nv.get("eat", 0) / _cc, 3),
                    }
                self._nav_vis = {"cids": 0, "sees": 0, "onf": 0, "eat": 0,
                                 "batches": 0}
        return (obs_per_cid, events_per_cid, intero_per_cid, rates_per_cid,
                on_flora_per_cid, carried_food_per_cid, nearest_flora_per_cid)

    def _tile_water(self, row: int, col: int) -> bool:
        """Один тайл = WATER? (для reachability-диагностики пути к флоре)."""
        wc = getattr(self, "world_cache", None)
        if wc is None:
            return False
        cfg = getattr(wc, "config", None)
        terrain = getattr(wc, "terrain", b"")
        if cfg is None or not terrain:
            return False
        size = int(getattr(cfg, "size", 0) or 0)
        if size <= 0 or len(terrain) < size * size:
            return False
        r, c = int(row), int(col)
        if 0 <= r < size and 0 <= c < size:
            return terrain[r * size + c] == self._WATER_TILE
        return False

    def _local_flora_diag(self, row: int, col: int, flora_pos: dict):
        """ARRIVAL-диагностика (Фрай 05.06): локальная flora-плотность вокруг
        Адама (n в манхэттен-радиусе 1/2/3) + ближайшая дистанция + вода на
        straight-line пути к ближайшей флоре. Разводит (a) distribution vs
        (b) granularity/pathing. Возвращает (n1, n2, n3, nearest, water_path)."""
        if not flora_pos:
            return (0, 0, 0, -1, 0)
        n1 = n2 = n3 = 0
        nearest = 10 ** 9
        nf = None
        for (fr, fc) in flora_pos:
            d = abs(fr - row) + abs(fc - col)
            if d <= 1:
                n1 += 1
            if d <= 2:
                n2 += 1
            if d <= 3:
                n3 += 1
            if d < nearest:
                nearest = d
                nf = (fr, fc)
        water_path = 0
        if nf is not None and 0 < nearest <= 8:
            steps = max(abs(nf[0] - row), abs(nf[1] - col)) or 1
            for i in range(1, steps + 1):
                rr = row + int(round((nf[0] - row) * i / steps))
                cc = col + int(round((nf[1] - col) * i / steps))
                if self._tile_water(rr, cc):
                    water_path += 1
        return (n1, n2, n3, (nearest if nf is not None else -1), water_path)

    @staticmethod
    def _flora_in_radius(row: int, col: int, flora_pos: dict, vr: int) -> bool:
        """Есть ли флора в манхэттен-радиусе vr от (row,col). Ранний выход на
        первой найденной. Без torus-wrap (диагностика, край-эффект мал)."""
        if not flora_pos or vr <= 0:
            return False
        for dr in range(-vr, vr + 1):
            rem = vr - abs(dr)
            for dc in range(-rem, rem + 1):
                if (row + dr, col + dc) in flora_pos:
                    return True
        return False

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
        (obs_per_cid, events_per_cid, intero_per_cid, rates_per_cid,
         on_flora_per_cid, carried_food_per_cid, nearest_flora_per_cid) = \
            self._collect_obs_batch(creatures)
        if not obs_per_cid:
            return
        self._on_flora_per_cid = on_flora_per_cid
        self._nearest_flora_per_cid = nearest_flora_per_cid
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
            # Grace для свежевосстановленных (restore) cid'ов (31.05.2026,
            # Бендер): их НЕТ в obs (P40 после purge/restart их не знает), но
            # они в compute.organisms. Без grace GC видит last_seen=None →
            # сносит НЕМЕДЛЕННО, ДО того как projection_batch установит их на
            # P40 (self-heal) → дедлок (это и выкосило 16 restored). setdefault
            # = только новым (уже отслеживаемые сохраняют своё last_seen) →
            # переживут STALE-окно, пока projection self-heal'нет → obs пойдут.
            if self.compute is not None:
                try:
                    for cid in list(self.compute.organisms.keys()):
                        self._cid_last_seen_tick.setdefault(cid, world_tick)
                except Exception:
                    pass
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
                # Liveness fix (30.05.2026, Бендер): obs_batch с known owned
                # creatures = особи живы и наблюдаются P40. Раньше n_alive_owned/
                # last_owned_alive_ts обновлялись ТОЛЬКО из welcome/stats —
                # у client-authoritative cheef они часто =0 (welcome
                # n_creatures=0 до projection-реконсиляции; stats редкие/0). В
                # итоге empty-world watchdog ложно слал respawn_owned_request
                # каждые ~grace сек → P40 _async_respawn → reconnect → broker
                # supersede старой сессии (1001) → "no close frame" → петля,
                # n_alive не устаканивается. obs для owned — авторитетный
                # signal "колония НЕ пуста"; stats/welcome поправят при приходе.
                self.n_alive_owned = known
                self.last_owned_alive_ts = time.time()
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
        # Offload handle_tick + actions-build на thread-executor (0.11.16,
        # Бендер): N орг × 20 тканей forward синхронно блокировали ws-event-loop
        # → ws-read голодал → keepalive ping timeout (broker→client queue MAX
        # при 142 орг). torch forward releases GIL → event-loop успевает на
        # ping/read во время тика. Guard _tick_in_progress: если предыдущий тик
        # ещё идёт (handle_tick медленнее obs-rate), пропускаем heavy-обработку
        # этого batch — liveness/GC/last_seen уже обновлены выше, P40 reuse'ит
        # последний action на пропущенных тиках. Дроп stale > pile-up.
        if getattr(self, "_tick_in_progress", False):
            return
        # Rate-limit (0.11.17): min-интервал между тиками — GIL-передышка
        # event-loop'у, рубит catch-up бёрсты obs после reconnect (прямая
        # причина keepalive при 145 орг). Steady-state 5 Hz (P40 throttle)
        # ниже cap'а → не затрагивается.
        _now = time.monotonic()
        if _now - getattr(self, "_last_tick_done_ts", 0.0) < MIN_TICK_INTERVAL_SEC:
            return
        self._tick_in_progress = True
        try:
            creatures_out = await asyncio.to_thread(
                self._run_tick_and_build,
                obs_per_cid, events_per_cid, intero_per_cid, world_tick,
                rates_per_cid, on_flora_per_cid, carried_food_per_cid,
                nearest_flora_per_cid)
        except Exception as e:
            logger.warning("handle_tick failed: %s", e)
            return
        finally:
            self._tick_in_progress = False
            self._last_tick_done_ts = time.monotonic()
        ws = self._ws
        if ws is None or not creatures_out:
            return
        # Water-seek рефлекс (31.05.2026, Хьюберт+Бендер): zodchiy НЕ навигируют
        # к воде (obs несёт food-градиент, не water) → drink_sum=0 → дегидратация
        # выкашивала колонию. При low-hydration (<30% max) override action к
        # ближайшей воде (terrain из world_cache). После → ambient water_restore
        # на P40 → delta_hydration>0 → ось воды заработает (тогда вернём смерть).
        # КОЛОНИАЛЬНЫЕ seek-рефлексы (Фрай 06.06): water/food-seek перезаписывают
        # computed action на move-к-воде/ягоде ПОСЛЕ handle_tick, до отправки.
        # Корень всей саги: Адам вечно голоден (energy~1<450) → food-seek клобберил
        # КАЖДОЕ его действие (STAY/GATHER/прайор/мотор) на move-к-ягоде → P40
        # видел только move, STAY/park никогда не доходили. Рефлексы — для zodchiy
        # (не навигируют). У single-organism Адама СВОЯ выученная политика
        # (прайор+мотор+park) → ОТКЛЮЧАЕМ рефлексы, ведёт он сам. ПАРНО с Хьюбертом
        # (он гейтит анти-осц off для owned — иначе STAY переопределится на сервере).
        _single = (bool(getattr(self.compute, "_single_organism", False))
                   if self.compute is not None else False)
        if not _single:
            try:
                self._apply_water_seek(creatures, creatures_out)
            except Exception as e:
                logger.warning("water-seek override failed: %r", e)  # DIAG 0.11.32
            # Berry/fruit-seek (01.06.2026, Хьюберт): ПОСЛЕ water (вода приоритет) —
            # голодных ведём к высокоценной флоре (grass не кормит, net −7.85/сек).
            try:
                self._apply_food_seek(creatures, creatures_out)
            except Exception as e:
                logger.warning("food-seek override failed: %r", e)
        else:
            # single-organism (Фрай 08.06, аффорданс Гидратация): food-seek OFF
            # (флору ведёт arrival-commit прайор+мотор), но WATER-seek ВКЛ — вода
            # КРИТИЧНА-к-выживанию И НЕОТКРЫВАЕМА (obs БЕЗ water-градиента → Адам не
            # может выучить нав к воде) → рефлекс-зачаток ОПРАВДАН доктриной
            # «минимум врождённого». Не клобберит как food: фаирит ТОЛЬКО при жажде
            # (hydration<30, _apply_water_seek), не каждый тик. Зеркало masked-фуража:
            # вода была подпёрта water-halo, Адам пить активно не умел.
            try:
                self._apply_water_seek(creatures, creatures_out)
            except Exception as e:
                logger.warning("water-seek override failed: %r", e)
            self._seek_gate_n = getattr(self, "_seek_gate_n", 0) + 1
            if self._seek_gate_n % 200 == 1:
                logger.info("SEEK_GATE single_organism → food-seek OFF, "
                            "WATER-seek ON (Гидратация, Фрай 08.06)")
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
        # F5 skill-growth re-announce (01.06.2026, Фрай) — THROTTLED 0.11.48:
        # без throttle спамил ~1/сек (3038/час!) → WS-сатурация → keepalive-
        # таймауты → срыв obs/кормёжки → энергоголод → вымирание (ночь 31.05).
        # Батчим: re-announce раз в ~30с (traits меняются медленно, P40-
        # пропагация может подождать; client-authoritative — traits в store).
        try:
            if getattr(self.compute, "_skill_changed_cids", None):
                _now = time.monotonic()
                if _now - getattr(self, "_last_skill_announce_ts", 0.0) >= 30.0:
                    self._last_skill_announce_ts = _now
                    self.compute._skill_changed_cids.clear()
                    await self._maybe_send_traits_announce(force=True)
        except Exception as e:
            logger.debug("skill traits re-announce failed: %s", e)

    # Water-seek рефлекс (31.05.2026). Tile.WATER=1 (environment/world.py).
    _WATER_TILE = 1
    _WATER_SEEK_HYDRATION = 30.0   # < 30/100 max → искать воду (бинарный режим)
    # life_critical bypass (Фрай/Хьюберт 11.06): hydration ≤ crit → water-seek
    # move помечается life_critical=true → P40 ОБХОДИТ §3-force-STAY (узко,
    # scope is_adam, default false). Размыкает дедлок: бессмертный Адам ползёт
    # к воде даже в §3-параличе (energy≤0 от хищник-дрейна) → recovery hyd→energy.
    _HYDRATION_CRITICAL = 15.0     # ≤ → water-move life-critical (bypass §3-STAY)
    _ACTION_FLEE = 10              # FLEE — survival-escape от угрозы (тоже bypass)
    _FLEE_BOOST_ADR = 55.0         # Fib порог adrenaline для flee-boost=1 (band-ручка Фрая)
    _FLEE_BURST_ADR = 75.0         # прямой контакт (camp, adr~80) → burst +2 (Шеф-гибрид camp-break)
    _MOVE_ACTIONS = frozenset({0, 1, 2, 3, 10})  # локомоция (MOVE кардинальные + FLEE)
    _HUNT_ACTIONS = frozenset({0, 1, 2, 3, 5})    # hunt-move: MOVE к добыче + ATTACK
    _ENERGY_CRITICAL = 89.0       # ≤ → hunt life_critical (голод-bypass §3, зеркало hyd≤15)
    #                              → persist=true (Хьюберт c8c2af8): сервер дед-реконит
    #                              последний MOVE каждый world-тик пока клиент молчит
    #                              (мой client-tick медленнее world → без persist P40
    #                              force-STAY'ит через 30т=2с → Адам застывает/прыгает).
    # §3.2 (Фрай 09.06.2026): φ-onset градуального felt-drive. 0.382·100=38.2
    # (нативная hydration-шкала [0,100], НЕ squashed insula slot[1]). hydration<
    # onset → felt>0; felt=(onset−hyd)/onset∈[0,1] масштабирует приоритет
    # рефлекса A (duty-cycle). Тюнится по выживанию. Заменяет бинарный 30% под
    # флагом compute._felt_thirst_drive_enabled.
    _THIRST_ONSET = 38.2
    # §3.2 concave felt-кривая (Фрай 09.06, первое окно): линейный felt в mid-зоне
    # слаб (hyd~30: felt~0.15 → редкий override → коррекция поздно → провал до 0 +
    # berserk). felt^φ⁻¹ (concave, степень 0.618) бустит mid (0.15→0.30), держит
    # φ-onset (0 при onset, 1 при hyd=0), монотонна. Строгое улучшение в нужную
    # сторону: ранняя коррекция жажды.
    _THIRST_FELT_POWER = 0.6180339887498949   # φ⁻¹
    _WATER_SEEK_RADIUS = 8         # тайлов вокруг — дальше random walk
    # Berry/fruit-seek (01.06.2026, Хьюберт audit): grass net −7.85/сек (НЕ
    # кормит), berry +18/сек. Голодные Зодчие должны идти к высокоценной флоре,
    # не пастись на траве. flora_kind: 1=GRASS,2=BERRY,3-6=FRUIT (φ× сытнее).
    _BERRY_FRUIT_KINDS = frozenset({2, 3, 4, 5, 6})
    _FOOD_SEEK_ENERGY = 450.0      # energy < порог → искать berry/fruit
    _FOOD_SEEK_RADIUS = 12         # тайлов вокруг (vision-ish)
    # Water income client-side (01.06.2026, Шеф): на/рядом с WATER → доход
    # hydration. φ⁷≈29.03 = prey_kill_energy (mirror world.py:551).
    _WATER_RESTORE = 1.618033988749895 ** 7
    # Eat income client-side (01.06.2026, Шеф/Хьюберт): на/рядом с flora →
    # доход energy. grass=φ⁻²≈0.382, berry=grass·φ³≈1.618 (mirror world.py:311/
    # 516). FloraType: GRASS=1, BERRY=2. P40 шлёт delta_energy только когда
    # owned ТОЧНО на flora (passive_eating), но зодчие ходят случайно (только
    # move-действия, Хьюберт) → почти не попадают → голод за 13мин. Клиент
    # кредитует близость (как воду), приоритет ненулевого P40.
    _GRASS_ENERGY = 1.618033988749895 ** -2        # φ⁻² ≈ 0.382 (GRASS=1)
    _BERRY_ENERGY = 1.618033988749895              # φ ≈ 1.618 (BERRY=2)
    _TREE_FRUIT_ENERGY = 1.618033988749895 ** 3    # φ³ ≈ 4.236 (FRUIT=3-6)
    _FRUIT_KINDS = frozenset({3, 4, 5, 6})         # FloraType FRUIT_APPLE..DATE

    def _water_seek_action(self, row: int, col: int):
        """Направление (0=N,1=S,2=E,3=W) к ближайшему WATER-тайлу в радиусе,
        или None если воды рядом нет / нет terrain. terrain из world_cache
        (bytes size×size, row-major). Greedy: ближайший по Manhattan."""
        wc = getattr(self, "world_cache", None)
        if wc is None:
            return None
        cfg = getattr(wc, "config", None)
        terrain = getattr(wc, "terrain", b"")
        if cfg is None or not terrain:
            return None
        size = int(getattr(cfg, "size", 0) or 0)
        if size <= 0 or len(terrain) < size * size:
            return None
        r0, c0 = int(row), int(col)
        best = None
        best_d = 10**9
        # ПРОГРЕССИВНОЕ расширение радиуса (Фрай 08.06, робастность): база 8 →
        # 16→32→64→128→карта, стоп на ПЕРВОМ найденном кольце (ближайшая вода).
        # Иначе при воде дальше базового радиуса возвращали None → слепой explore
        # (мог идти в противоположную сторону) → дегидратация в водо-скудном месте.
        # Скан только при жажде (вызов из _apply_water_seek), стоимость ограничена.
        _base = self._WATER_SEEK_RADIUS
        for R in (_base, _base * 2, _base * 4, _base * 8, _base * 16, size):
            best = None
            best_d = 10**9
            for r in range(max(0, r0 - R), min(size, r0 + R + 1)):
                base = r * size
                for c in range(max(0, c0 - R), min(size, c0 + R + 1)):
                    if terrain[base + c] == self._WATER_TILE:
                        d = abs(r - r0) + abs(c - c0)
                        if 0 < d < best_d:
                            best_d = d
                            best = (r, c)
            if best is not None:
                break
        if best is None:
            return None
        dr = best[0] - r0
        dc = best[1] - c0
        if abs(dr) >= abs(dc):
            return 0 if dr < 0 else 1   # N / S
        return 2 if dc > 0 else 3       # E / W

    def _food_seek_action(self, row: int, col: int, berry_pos):
        """Направление (0=N,1=S,2=E,3=W) к ближайшему berry/fruit-тайлу в радиусе.
        None если УЖЕ на нём (→ не оверрайдим, инстинкт GATHER/EAT) или нет рядом.
        berry_pos — set[(row,col)] высокоценной флоры (kind 2-6)."""
        if not berry_pos:
            return None
        r0, c0 = int(row), int(col)
        if (r0, c0) in berry_pos:
            return None  # на берри/фрукте → инстинкт ест, не перебиваем
        best = None
        best_d = 10 ** 9
        R = self._FOOD_SEEK_RADIUS
        for dr in range(-R, R + 1):
            rem = R - abs(dr)
            for dc in range(-rem, rem + 1):
                if (r0 + dr, c0 + dc) in berry_pos:
                    d = abs(dr) + abs(dc)
                    if 0 < d < best_d:
                        best_d = d
                        best = (r0 + dr, c0 + dc)
        if best is None:
            return None
        dr = best[0] - r0
        dc = best[1] - c0
        if abs(dr) >= abs(dc):
            return 0 if dr < 0 else 1   # N / S
        return 2 if dc > 0 else 3       # E / W

    def _apply_food_seek(self, creatures, creatures_out) -> None:
        """Override move к ближайшему berry/fruit для голодных (energy<порог),
        кто НЕ на берри/фрукте. Хьюберт audit: grass не кормит (net −7.85/сек),
        berry +18/сек → вести к высокоценной флоре. Зеркалит water-seek; вода
        (жажда) применяется РАНЬШЕ → приоритет, food-seek жаждущих не трогает."""
        compute = self.compute
        if compute is None:
            return
        biochem = getattr(compute, "biochem", None)
        cache = getattr(self, "world_cache", None)
        if not biochem or cache is None:
            return
        try:
            _flora = list(cache.flora)
            berry_pos = {(int(r), int(c)) for (r, c, k) in _flora
                         if int(k) in self._BERRY_FRUIT_KINDS}
        except Exception:
            return
        # Диагностика распределения типов флоры (даже если berry нет) — чтобы
        # отличить «нет berry рядом (тундра)» от «kind в кэше битый».
        self._food_seek_diag = getattr(self, "_food_seek_diag", 0) + 1
        if self._food_seek_diag >= 25:   # ~раз в 1-2 мин при медленном тике
            self._food_seek_diag = 0
            _grass = sum(1 for (_r, _c, k) in _flora if int(k) == 1)
            _berry = sum(1 for (_r, _c, k) in _flora if int(k) == 2)
            _fruit = sum(1 for (_r, _c, k) in _flora if int(k) in (3, 4, 5, 6))
            _other = len(_flora) - _grass - _berry - _fruit
            logger.info(
                "FLORA_KINDS total=%d grass=%d berry=%d fruit=%d other=%d "
                "berry_fruit_pos=%d", len(_flora), _grass, _berry, _fruit,
                _other, len(berry_pos))
        if not berry_pos:
            return
        pos = {}
        for c in creatures:
            cid = c.get("cid")
            if cid is None:
                continue
            rc = self._resolve_pos(cid, c)
            if rc is not None:
                pos[str(cid)] = rc
        n_seek = 0
        n_hungry = 0
        for entry in creatures_out:
            cid = entry.get("cid")
            bc = biochem.get(cid) if cid else None
            if bc is None:
                continue
            if float(getattr(bc, "energy", 1e9)) >= self._FOOD_SEEK_ENERGY:
                continue
            # жаждущий → water-seek в приоритете (уже оверрайднул), не трогаем
            if float(getattr(bc, "hydration", 100.0)) < self._WATER_SEEK_HYDRATION:
                continue
            # YIELD-TO-EAT (Фрай 16.06): уже на съедобной флоре (серверный
            # on_flora → compute._on_food → eat-reflex владеет EAT) → food-seek
            # НЕ перебивает. Иначе client berry_pos рассинхронен с server
            # on_flora → food-seek перетирал легитимный EAT на MOVE-к-ягоде →
            # укус не стартовал (EAT_PROGRESS kind=None) → голод среди ягод в
            # режиме energy<450 (где food-seek активен). Тот же сигнал, что
            # eat-reflex → десинк снят одним guard'ом.
            if getattr(compute, "_on_food", {}).get(cid):
                continue
            n_hungry += 1
            rc = pos.get(cid)
            if rc is None or rc[0] is None:
                continue
            fd = self._food_seek_action(rc[0], rc[1], berry_pos)
            if fd is not None:
                entry["action"] = int(fd)
                n_seek += 1
        if n_seek or n_hungry:
            self._food_seek_overrides = getattr(
                self, "_food_seek_overrides", 0) + n_seek
            logger.debug("FOOD_SEEK hungry=%d seek_override=%d berry_pos=%d",
                         n_hungry, n_seek, len(berry_pos))

    def _near_water(self, row: int, col: int) -> bool:
        """Организм на ИЛИ рядом (5 клеток, радиус 1: self + 4 соседа) с
        WATER-тайлом → может пить (mirror world.py:3551). terrain из
        world_cache (bytes size×size). Границы клампим (без torus — на проде
        вода в центре, edge-эффект пренебрежим)."""
        wc = getattr(self, "world_cache", None)
        if wc is None:
            return False
        cfg = getattr(wc, "config", None)
        terrain = getattr(wc, "terrain", b"")
        if cfg is None or not terrain:
            return False
        size = int(getattr(cfg, "size", 0) or 0)
        if size <= 0 or len(terrain) < size * size:
            return False
        r0, c0 = int(row), int(col)
        for dr, dc in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = r0 + dr, c0 + dc
            if 0 <= r < size and 0 <= c < size:
                if terrain[r * size + c] == self._WATER_TILE:
                    return True
        return False

    def _resolve_pos(self, cid, c=None):
        """(row, col) организма. Источник истины — world_cache.creature_pos
        (dict[cid → (x,y)=(col,row)], world_cache.py:384, заполняется из
        snap.creatures). obs-словари obs_batch row/col НЕ содержат — это и была
        причина drink_sum=0: позиция читалась оттуда → дефолт (0,0)=PLAIN, ни
        water-seek, ни income не срабатывали. Fallback на c — если задан и есть."""
        wc = getattr(self, "world_cache", None)
        if wc is not None:
            try:
                p = wc.creature_pos.get(str(cid))
                if p is not None:
                    return int(p[1]), int(p[0])  # (row=y, col=x)
            except Exception:
                pass
        if c is not None and "row" in c and "col" in c:
            try:
                return int(c["row"]), int(c["col"])
            except (TypeError, ValueError):
                pass
        return None

    def _passive_flora_income(self, row, col, flora_pos,
                              vision: float, diet: float, eff: float) -> float:
        """Faithful порт server passive_flora_eating (world.py:3174-3259).
        owned тикаются на клиенте (P40 phase-out) → клиент кормит по серверной
        формуле, без заплаток. Vision-scan (current + vision>=8: 4 orth +
        vision>=11: 4 diag), eat ОДНУ flora/тик:
          gain = flora_energy(kind) × (1-diet_gene) × kleiber(eff) × vision_bonus
        flora_energy: GRASS=φ⁻², BERRY=φ, FRUIT=φ³. kleiber=(eff/10)^(1/φ).
        vision_bonus=1+(vision-3)×0.02. memory-бонус не портирован (memory_tiles
        не трекаются на client)."""
        if not flora_pos:
            return 0.0
        r0, c0 = int(row), int(col)
        cells = [(r0, c0)]
        if vision >= 8:
            cells += [(r0 - 1, c0), (r0 + 1, c0), (r0, c0 - 1), (r0, c0 + 1)]
        if vision >= 11:
            cells += [(r0 - 1, c0 - 1), (r0 - 1, c0 + 1),
                      (r0 + 1, c0 - 1), (r0 + 1, c0 + 1)]
        kleiber = (max(1.0, eff) / 10.0) ** (1.0 / 1.618033988749895)
        vbonus = 1.0 + (vision - 3.0) * 0.02
        for (r, c) in cells:
            kind = flora_pos.get((r, c))
            if kind is None:
                continue
            if kind in self._FRUIT_KINDS:
                base = self._TREE_FRUIT_ENERGY
            elif kind == 2:
                base = self._BERRY_ENERGY
            else:
                base = self._GRASS_ENERGY
            return base * (1.0 - diet) * kleiber * vbonus  # одна flora/тик
        return 0.0

    def _flee_speed_boost(self, cid) -> int:
        """Predator v0.1 Часть 2 (Фрай/Хьюберт 11.06): доп. flee-шаги ∝ adrenaline.
        ADDITIVE-модель P40 (Хьюберт 9f92495): FLEE base = flee_distance=3 (= хищник,
        паритет), extra_steps = boost (+0..2 клетки, collision-aware). Жалоба Шефа
        «режим супермена» снята additive-моделью (boost больше НЕ множит → телепорта
        нет). Шеф: «ускорение лишь незначительное» → magnitude КАП на 1 (+1 клетка =
        4 vs хищник 3, минимальный отрыв):
          adr ≥_FLEE_BOOST_ADR (Fib 55) → 1 (минимальный отрыв, хищник близко)
          иначе                          → 0 (паритет — бежит вровень с хищником)
        CAMP-BREAK burst (Шеф-гибрид 11.06): при ПРЯМОМ контакте (хищник camp'ит
        вплотную, pred_prox~1.0 → adr~80 ≥ _FLEE_BURST_ADR=75) разрешаем КОРОТКИЙ
        рывок +2 (=5 клеток vs хищник 3) — разорвать camp. БРИФ: adrenaline спадает
        когда хищник отстал → boost→1→0 → назад к паритету (honor Шеф «незначительно»:
        burst = камп-брейкер, не sustained-преимущество; gap открыт, паритет держит).
          adr ≥75 (прямой контакт/camp) → 2 (рывок разорвать)
          adr ≥55 (близкий хищник)       → 1 (минимальный отрыв)
          иначе                           → 0 (паритет)
        Порог/onset/magnitude — band-ручки Фрая (switch-timing = learnable ось)."""
        compute = self.compute
        if compute is None:
            return 0
        bc = getattr(compute, "biochem", {}).get(cid)
        if bc is None:
            return 0
        adr = float(getattr(bc, "adrenaline", 0.0))
        # Шеф 12.06: рывок КАП на +1 (убрал +2 burst). С новой базой Адама=2
        # (Хьюберт server-side) boost+1 → 3 клетки > хищник 2.24 → отрыв/camp-break;
        # база 2 < хищник → нужен boost для побега. Добыча <2 → Адам ловит + рывком.
        if adr >= self._FLEE_BOOST_ADR:
            return 1
        return 0

    def _apply_water_seek(self, creatures, creatures_out) -> None:
        """Override action на move-к-воде для жаждущих (hydration<30%). zodchiy
        не учатся искать воду (obs без water-градиента) → дегидратация. Рефлекс:
        thirsty → шаг к ближайшей воде → доход (_near_water в _collect_obs_batch).
        Позиция из creature_pos (world_cache), НЕ из obs-словаря."""
        compute = self.compute
        if compute is None:
            return
        biochem = getattr(compute, "biochem", None)
        if not biochem:
            return
        pos = {}
        for c in creatures:
            cid = c.get("cid")
            if cid is None:
                continue
            rc = self._resolve_pos(cid, c)
            if rc is not None:
                pos[str(cid)] = rc  # (row, col)
        n_seek = 0
        n_thirsty = 0
        n_near = 0
        # ROBUSTNESS (Фрай 08.06): зеркало 4-way escape для воды. Трекаем pos
        # жаждущего; нет прогресса к воде (pos застрял) ИЛИ воды нет в радиусе →
        # ИССЛЕДУЙ ротацией 4 направлений → сменит pos/найдёт воду. Иначе хрупкость:
        # вода недостижима (за препятствием) / далеко (>радиус) → пин/random.
        if not hasattr(self, "_water_seek_pos"):
            self._water_seek_pos = {}
        if not hasattr(self, "_water_seek_stuck"):
            self._water_seek_stuck = {}
        # §3.2 felt-drive: duty-cycle аккумулятор приоритета рефлекса A per-cid.
        if not hasattr(self, "_thirst_accum"):
            self._thirst_accum = {}
        _felt_on = (self.compute is not None
                    and getattr(self.compute, "_felt_thirst_drive_enabled", False))
        for entry in creatures_out:
            cid = entry.get("cid")
            bc = biochem.get(cid) if cid else None
            if bc is None:
                continue
            hyd = float(getattr(bc, "hydration", 100.0))
            # felt ∈ [0,1] — сила «чувства жажды» (thirst-афферент). Бинарный
            # режим: felt=1.0 при hydration<30 (детерминированная компульсия, как
            # было). Градуальный (§3.2): onset φ=38.2, felt=(onset−hyd)/onset →
            # жажднее ⇒ сильнее тяга. Не жаждет → пропуск (+ сброс аккумулятора).
            if _felt_on:
                if hyd >= self._THIRST_ONSET:
                    self._thirst_accum.pop(cid, None)
                    continue
                _lin = max(0.0, min(1.0, (self._THIRST_ONSET - hyd) / self._THIRST_ONSET))
                felt = _lin ** self._THIRST_FELT_POWER   # concave φ⁻¹: бустит mid
            else:
                if hyd >= self._WATER_SEEK_HYDRATION:
                    continue
                felt = 1.0
            n_thirsty += 1
            rc = pos.get(cid)
            if rc is None or rc[0] is None:
                continue
            if self._near_water(rc[0], rc[1]):
                # У воды (радиус 1) → пьёт (income _WATER_RESTORE φ⁷). НЕ оверрайдим
                # (его политика; income восстановит hydration быстро). Стук-сброс.
                n_near += 1
                self._water_seek_stuck[cid] = 0
                self._water_seek_pos[cid] = rc
                self._thirst_accum.pop(cid, None)   # §3.2 пьёт → сброс duty-cycle
                continue
            # Не у воды + жажда → веди к воде. Стук-детект (зеркало arrival-escape):
            if self._water_seek_pos.get(cid) == rc:
                self._water_seek_stuck[cid] = self._water_seek_stuck.get(cid, 0) + 1
            else:
                self._water_seek_stuck[cid] = 0
            self._water_seek_pos[cid] = rc
            _sn = self._water_seek_stuck.get(cid, 0)
            wd = self._water_seek_action(rc[0], rc[1])
            if wd is None or _sn >= 13:
                # воды нет в радиусе ИЛИ пинуется (недостижима) → ИССЛЕДУЙ:
                # ротация 4 кардинальных (sn//5)%4 → любое открытое сменит pos.
                wd = [0, 2, 1, 3][(_sn // 5) % 4]   # N, E, S, W
            # §3.2 градуальный duty-cycle (детерминированный, без RNG): копим
            # felt; шаг-к-воде только при накоплении ≥1 (rate=felt/тик). felt=1
            # ⇒ каждый тик (бинарная компульсия). felt мал ⇒ редкий нудж — мозг
            # большую часть тиков свободен. «Жажднее → сильнее тяга», ОДИН контур.
            if felt < 1.0:
                _acc = self._thirst_accum.get(cid, 0.0) + felt
                if _acc < 1.0:
                    self._thirst_accum[cid] = _acc
                    continue   # этот тик felt не перебивает действие мозга
                self._thirst_accum[cid] = _acc - 1.0
            entry["action"] = int(wd)
            # life_critical (Фрай/Хьюберт 11.06): при критичной гидратации
            # water-move помечается → P40 обходит §3-force-STAY → Адам ползёт к
            # воде в параличе → recovery. Явно перезаписываем (мог быть FLEE-флаг).
            entry["life_critical"] = bool(hyd <= self._HYDRATION_CRITICAL)
            n_seek += 1
        if n_seek:
            self._water_seek_overrides = getattr(
                self, "_water_seek_overrides", 0) + n_seek
        # Диагностика (01.06.2026): раз в ~600 вызовов — видимость работы
        # water-seek/income (поз/жаждущие/у воды/override). Подтверждает фикс
        # позиции на проде (creature_pos vs пустой obs-словарь).
        self._ws_diag_ticks = getattr(self, "_ws_diag_ticks", 0) + 1
        if self._ws_diag_ticks >= 200:
            self._ws_diag_ticks = 0
            logger.info(
                "WATER_SEEK_DIAG creatures=%d with_pos=%d thirsty=%d "
                "near_water=%d seek_override=%d", len(creatures_out), len(pos),
                n_thirsty, n_near, n_seek)

    def _run_tick_and_build(self, obs_per_cid, events_per_cid,
                            intero_per_cid, world_tick, rates_per_cid=None,
                            on_flora_per_cid=None, carried_food_per_cid=None,
                            nearest_flora_per_cid=None):
        """Offload-worker (0.11.16): handle_tick + сборка creatures_out с
        phase_emas. Запускается в asyncio.to_thread — torch-forward'ы N орг не
        блокируют ws-event-loop. Весь compute-доступ изолирован здесь; на
        event-loop одновременно может идти build_projection_batch (read-mostly,
        list-снапшоты в 0.11.14 защищают от dict-race; stale-read безвреден).
        Phase emas (03.05.2026): per-creature Phase 1/2/6 метрики для owned."""
        if self.compute is None:
            return None
        # Ритм-ось (Путь 2, Фрай 15.06): is_night → compute (client-authoritative из
        # world_cache) для метрики neg_dark_loss. Глобальный (мир один), Адам owned.
        _wc = getattr(self, "world_cache", None)
        if _wc is not None:
            try:
                self.compute._world_is_night = bool(_wc.is_night)
            except Exception:
                pass
        actions = self.compute.handle_tick(
            obs_per_cid, events_per_cid=events_per_cid,
            intero_per_cid=intero_per_cid, world_tick=world_tick,
            rates_per_cid=rates_per_cid, on_flora_per_cid=on_flora_per_cid,
            carried_food_per_cid=carried_food_per_cid,
            nearest_flora_per_cid=nearest_flora_per_cid)
        if not actions:
            return None
        # persist-opt-in scope (Хьюберт c8c2af8): только Адам (single-organism).
        # Его мозг-клиент тикает медленнее world-loop → между эмитами P40 без
        # persist force-STAY'ит (30т) → заморозка. persist на локомоции → дед-
        # реконинг → feed плотный как у фауны (мелкие collision-aware шаги).
        _single = (bool(getattr(self.compute, "_single_organism", False))
                   if self.compute is not None else False)
        creatures_out: list = []
        for cid, a in actions.items():
            act = int(a["action"])
            entry: dict = {
                "cid": cid,
                "action": act,
                "target_id": a.get("target_id"),
            }
            # persist=true на локомоции Адама → сервер продолжает MOVE/FLEE каждый
            # world-тик пока следующий client-tick не передумает (75т=5с safety-STAY).
            if _single and act in self._MOVE_ACTIONS:
                entry["persist"] = True
            # PHASE 2 POUNCE (Фрай 12.06): в упор к средней дичи (dist≤_POUNCE_DIST,
            # голоден+способен — флаг ставит obs-loop) → +1 рывок на кардинальном
            # MOVE = короткий burst, нагоняет добычу до контакта (НЕ непрерывный
            # буст: Адам база=добыча база, нагон за счёт усталости + этого прыжка).
            # FLEE (10) свой speed_boost ниже — pounce только на 0-3.
            if (_single and act in (0, 1, 2, 3)
                    and getattr(self.compute, "_hunt_pounce", {}).get(cid)):
                entry["speed_boost"] = 1
            # life_critical (Фрай/Хьюберт 11.06): FLEE = survival-escape от угрозы
            # → P40 обходит §3-force-STAY (бежать от хищника даже в параличе).
            # water-seek-флаг навешивается позже в _apply_water_seek (по hydration).
            if act == self._ACTION_FLEE:
                entry["life_critical"] = True
                # speed_boost (Фрай/Хьюберт 11.06 predator v0.1 Часть 2): доп.
                # шаги flee ∝ adrenaline. Калибровка Фрая: побег при СИЛЬНОЙ
                # реакции (adr высок=хищник близко=хорошая vigilance), паритет
                # при слабой → evasion-ткани имеют смысл расти. start soft (max 3).
                entry["speed_boost"] = self._flee_speed_boost(cid)
            # HUNT-WHEN-STARVING = life_critical (Фрай hunting.md, ДО grass-cut):
            # зеркало water-when-thirsty (hyd≤15). Голод критичен (energy≤89) +
            # добыча достижима (obs[58]>0.15) + hunt-action (MOVE/ATTACK) → обходит
            # §3-force-STAY. Иначе −трава→голод→§3→в параличе НЕ охотится→голод-
            # капкан (тот же класс, что вода/хищник). Только Адам с hunting ON.
            elif (_single and act in self._HUNT_ACTIONS
                  and getattr(self.compute, "_hunting_enabled", False)):
                _bc = (getattr(self.compute, "biochem", {}) or {}).get(cid)
                _e = float(getattr(_bc, "energy", 999.0)) if _bc is not None else 999.0
                if _e <= self._ENERGY_CRITICAL:
                    _obs = (obs_per_cid or {}).get(cid)
                    try:
                        _pp = float(_obs[58]) if (_obs is not None and len(_obs) > 58) else 0.0
                    except Exception:
                        _pp = 0.0
                    # мелкая (obs[58]) ИЛИ §3-contact-флаг (средняя dist≤1, не в
                    # obs[58]) → life_critical: сервер пропустит ATTACK сквозь §3.
                    _contact = bool(getattr(self.compute, "_hunt_contact", {}).get(cid))
                    if _pp > 0.15 or _contact:    # добыча достижима / вплотную
                        entry["life_critical"] = True
            try:
                emas = self.compute.get_phase_emas(cid)
            except Exception:
                emas = None
            if emas:
                entry["phase_emas"] = emas
            creatures_out.append(entry)
        return creatures_out

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
        # Client-authoritative НЕ шлёт legacy respawn (31.05.2026): orphan-obs
        # для client-authoritative = stale P40-проекции (десинк), не повод
        # legacy-respawn'ить (создаёт клоны с новыми cid → углубляет десинк).
        if getattr(self, "_reject_incoming_seeds", False):
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

        # Single-organism pivot (01.06.2026, ТЗ e3cc81b §1): под флагом
        # репродукция отключена. P40 в single-режиме не должен слать
        # mate_request (контракт Б2 с Хьюбертом), но если прислал — вежливо
        # отклоняем, не собирая ребёнка. Код кроссинговера ниже сохранён.
        if self.compute is not None and getattr(
                self.compute, "_single_organism", False):
            await _reject("single_organism")
            return

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
