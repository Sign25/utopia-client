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
        # Прочее — F3 (actions/observations)
        logger.debug("unhandled msg type=%s", msg_type)

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
