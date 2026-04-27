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


class ColonyWSClient:
    """Держит WS к VPS в фоновом потоке."""

    def __init__(self, server: str, token: str, colony_name: str,
                 client_version: str, estimated_population: int = 0) -> None:
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
        # Метрики obs/actions
        self._obs_batches_received: int = 0
        self._actions_batches_sent: int = 0
        # Phase F3.3.a: счётчик отправленных weights_dump.
        self._weights_dumps_sent: int = 0
        self._weights_requests_received: int = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._thread_main,
                                         name="utopia-ws", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
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

            await ws.send(json.dumps({
                "type": "hello",
                "colony_name": self.colony_name,
                "client_version": self.client_version,
                "estimated_population": self.estimated_population,
                "ts": int(time.time() * 1000),
            }))

            ping_task = asyncio.create_task(self._ping_loop(ws))
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
                try:
                    await ping_task
                except (asyncio.CancelledError, Exception):
                    pass
                self._ws = None
                self.connected = False

    async def _ping_loop(self, ws) -> None:
        while True:
            await asyncio.sleep(PING_INTERVAL_SEC)
            try:
                await ws.send(json.dumps({"type": "ping",
                                          "ts": int(time.time() * 1000)}))
            except Exception:
                return

    async def _handle(self, msg: dict) -> None:
        msg_type = msg.get("type", "")
        if msg_type == "welcome":
            n = msg.get("n_creatures")
            if isinstance(n, int):
                self.n_alive_owned = n
            wt = msg.get("world_tick")
            if isinstance(wt, int):
                self.last_world_tick = wt
            logger.info("welcome: world_tick=%s server_time=%s n_creatures=%s",
                        wt, msg.get("server_time"), n)
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
        if msg_type == "stats":
            n = msg.get("n_alive_owned", 0)
            try:
                self.n_alive_owned = int(n)
            except (TypeError, ValueError):
                self.n_alive_owned = 0
            wt = msg.get("world_tick")
            if isinstance(wt, int):
                self.last_world_tick = wt
            self.last_stats_ts = time.time()
            logger.info("stats: n_alive_owned=%d world_tick=%s",
                        self.n_alive_owned, wt)
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
        # ── Phase F3.1.b/c: obs_batch → handle_tick → actions_batch ─────
        if msg_type == "obs_batch":
            await self._handle_obs_batch(msg)
            return
        # ── Phase F3.3.a: pull-on-demand weights ────────────────────────
        if msg_type == "weights_request":
            await self._handle_weights_request(msg)
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
        out = {
            "type": "actions_batch",
            "world_tick": world_tick,
            "ts_p40_ns_echo": ts_echo,
            "creatures": [
                {"cid": cid, "action": int(a["action"]),
                 "target_id": a.get("target_id")}
                for cid, a in actions.items()
            ],
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
        одним сообщением. Чанкование оставлено на будущее (для тихоходки
        ~7-15 КБ raw, b64 + envelope < 50 КБ — fits 512 KB ws-лимит).
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
