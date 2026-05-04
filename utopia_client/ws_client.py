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
                                       ping_interval=20, ping_timeout=20) as ws:
            self._ws = ws
            self.connected = True
            self.last_error = ""
            logger.info("connected")

            from .seed_loader import seed_sha256

            # A1: разведка локального seed-pack ДО hello. Если на диске
            # есть .pt от прошлой сессии — объявляем серверу через
            # local_save в hello + сразу шлём seed_pack_start/chunk*/complete.
            local_pack = self._scan_local_seed_pack()

            hello_msg: dict = {
                "type": "hello",
                "colony_name": self.colony_name,
                "client_version": self.client_version,
                "estimated_population": self.estimated_population,
                "known_seed_hash": seed_sha256(),
                "genesis_mode": self.genesis_mode,
                "ts": int(time.time() * 1000),
            }
            if local_pack:
                hello_msg["local_save"] = {
                    "available": True,
                    "n_creatures": len(local_pack),
                    "est_bytes": sum(len(p["weights"]) for p in local_pack),
                    "seed_revision": int(time.time()),
                }
            await ws.send(json.dumps(hello_msg))

            if local_pack:
                try:
                    await self._send_seed_pack(
                        ws, local_pack,
                        seed_revision=hello_msg["local_save"]["seed_revision"])
                except Exception as e:
                    logger.warning("send seed_pack failed: %s", e)

            ping_task = asyncio.create_task(self._ping_loop(ws))
            save_task = asyncio.create_task(self._save_loop())
            empty_task = asyncio.create_task(self._empty_world_loop())
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
                for t in (ping_task, save_task, empty_task):
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
                "n": 5,
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
            logger.info(
                "welcome: world_tick=%s server_time=%s n_creatures=%s mode=%s",
                wt, msg.get("server_time"), n, mode or "normal")
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
            self._handle_seed_start(msg)
            return
        if msg_type == "seed_chunk":
            self._handle_seed_chunk(msg)
            return
        if msg_type == "seed_complete":
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
            return True
        try:
            from .local_compute import LocalColonyCompute
            self.compute = LocalColonyCompute()
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
            if not seed_cached():
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
        loaded = False
        for src, weights in sources:
            try:
                org, payload = organism_from_weights(weights, seed_cache_path())
                self.compute.add_creature(
                    cid, org,
                    hebbian_enabled=True,
                    learning_rate=float(meta.get("learning_rate", 1e-4)),
                    trace_decay=float(meta.get("trace_decay", 0.9)),
                )
                self.compute.apply_inherited_state(cid, payload)
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
            path = write_seed_bytes(data)
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

    # ── Phase F3.1.b/c: obs_batch → actions_batch ────────────────────────

    async def _handle_obs_batch(self, msg: dict) -> None:
        import numpy as np
        creatures = msg.get("creatures") or []
        world_tick = int(msg.get("world_tick", 0))
        ts_echo = msg.get("ts_p40_ns")
        self._obs_batches_received += 1
        if self.compute is None or not creatures:
            return
        obs_per_cid: dict = {}
        events_per_cid: dict = {}
        for c in creatures:
            cid = c.get("cid")
            obs = c.get("obs")
            if not cid or obs is None:
                continue
            try:
                obs_per_cid[str(cid)] = np.asarray(obs, dtype=np.float32)
            except Exception as e:
                logger.debug("obs parse %s: %s", cid, e)
                continue
            # Phase F3.2.a/b: события прошлого тика — для Hebbian R3 reward.
            # Поля могут отсутствовать у старых P40 — компонуем с дефолтами.
            events_per_cid[str(cid)] = {
                "ate": bool(c.get("ate", False)),
                "killed": bool(c.get("killed", False)),
                "damage_taken": float(c.get("damage_taken", 0.0) or 0.0),
                "delta_energy": float(c.get("delta_energy", 0.0) or 0.0),
            }
        if not obs_per_cid:
            return
        try:
            actions = self.compute.handle_tick(obs_per_cid,
                                                events_per_cid=events_per_cid)
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
            father_org, _ = organism_from_weights(father_blob, seed_cache_path())
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
