"""WebSocket-клиент для Embodied API (Body Migration Этап 3a Phase 1).

Подключается к VPS брокеру `/ws/embodied/client?token=<colony_push_token>`.
Шлёт msgpack `embodied/state`, принимает `embodied/observation` от P40
(через broker), фильтрует server keepalive `ping` (отвечает `pong`).
Reconnect с exponential backoff.

Phase 1: skeleton + echo roundtrip latency. Phase 2+: интеграция с
биохимией, тканями, NEAT (см. `docs/tasks/tz_body_migration.md`).

Транспортный контракт (ТЗ v1.3 §3.1, embodied_broker.py):
  - `send_bytes(msgpack.packb(payload))` — broker оборачивает в
    envelope `{type:"embodied_state", from:user_id, data:<raw>}` сам.
  - Incoming `bytes` → msgpack.unpackb:
      * `{"type":"ping"}` — server keepalive, отвечаем `{"type":"pong"}`
      * иначе — observation dict от P40 echo handler
        (`{world_tick, echo, cid, pos, protocol_version}`).
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections import deque
from typing import Callable, Optional

import msgpack
import websockets

logger = logging.getLogger("utopia_client.embodied")

DEFAULT_URL_PATH = "/ws/embodied/client"
WS_MAX_SIZE = 256 * 1024
WS_OPEN_TIMEOUT_SEC = 10.0
LATENCY_WINDOW = 500  # последних N latencies для p50/p95/max
RECONNECT_INITIAL_SEC = 1.0
RECONNECT_MAX_SEC = 30.0


class EmbodiedWSClient:
    """Клиент `wss://divisci.com/ws/embodied/client` с msgpack-транспортом.

    Запускается в фоновом потоке (asyncio внутри), пользовательский код
    дёргает `start()`, `send_state(dict)`, `stop()` синхронно. Все
    тяжёлые вещи (network I/O, reconnect, ping/pong) — в собственном
    event loop'е.

    Latency tracking:
      - `send_state({"cid": ...})` запоминает `time.monotonic()`
      - При приходе observation с тем же `cid` (P40 echo) — diff
        записывается в `_latencies_ms`. `stats()` возвращает mean/p50/p95.

    Reconnect:
      - Exponential backoff 1s → 2s → 4s → ... → 30s
      - При успешном connect — backoff сбрасывается на 1s.
    """

    def __init__(
        self,
        server: str,
        token: str,
        on_observation: Optional[Callable[[dict], None]] = None,
        *,
        url_path: str = DEFAULT_URL_PATH,
    ) -> None:
        host = (server.replace("https://", "").replace("http://", "")
                .rstrip("/"))
        self.url = f"wss://{host}{url_path}?token={token}"
        self.on_observation = on_observation
        self._stop_flag = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws = None
        # stats
        self.connected: bool = False
        self.states_sent: int = 0
        self.observations_received: int = 0
        self.pings_handled: int = 0
        self.errors_total: int = 0
        self.last_error: str = ""
        # latency tracking — round-trip от send_state(cid) до echo с тем же cid
        self._latencies_ms: deque[float] = deque(maxlen=LATENCY_WINDOW)
        self._sent_at: dict[str, float] = {}  # cid → monotonic ts

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._thread_main, name="embodied-ws", daemon=True,
        )
        self._thread.start()
        logger.info("EmbodiedWSClient started url=%s",
                    self.url.split('?')[0])

    def stop(self) -> None:
        self._stop_flag.set()
        if self._loop is not None and self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._close_ws(), self._loop)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        logger.info("EmbodiedWSClient stopped")

    # ── send (sync API) ─────────────────────────────────────────────────

    def send_state(self, payload: dict) -> bool:
        """Async-safe: ставит msg в очередь через run_coroutine_threadsafe.

        True — отправка запланирована, False — клиент не connected или
        loop не работает. Реальная send-операция произойдёт в event loop'е.
        """
        if not self.connected or self._loop is None:
            return False
        try:
            asyncio.run_coroutine_threadsafe(
                self._send_state_async(payload), self._loop)
            return True
        except Exception as e:
            logger.debug("embodied send_state error: %s", e)
            return False

    # ── internal: thread + event loop ───────────────────────────────────

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run())
        finally:
            try:
                self._loop.close()
            except Exception:
                pass
            self._loop = None

    async def _run(self) -> None:
        backoff = RECONNECT_INITIAL_SEC
        while not self._stop_flag.is_set():
            try:
                async with websockets.connect(
                    self.url,
                    max_size=WS_MAX_SIZE,
                    ping_interval=None,  # server-driven ping в broker
                    open_timeout=WS_OPEN_TIMEOUT_SEC,
                ) as ws:
                    self._ws = ws
                    self.connected = True
                    backoff = RECONNECT_INITIAL_SEC
                    logger.info("embodied connected")
                    async for raw in ws:
                        await self._handle_msg(raw)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.connected = False
                self.last_error = str(e)
                self.errors_total += 1
                if self._stop_flag.is_set():
                    break
                logger.warning("embodied disconnect: %s (backoff %.0fs)",
                               e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_MAX_SEC)
        self.connected = False
        self._ws = None

    async def _send_state_async(self, payload: dict) -> None:
        if self._ws is None:
            return
        cid = payload.get("cid")
        if isinstance(cid, str) and cid:
            self._sent_at[cid] = time.monotonic()
        try:
            await self._ws.send(msgpack.packb(payload))
            self.states_sent += 1
        except Exception as e:
            logger.debug("embodied send error: %s", e)
            self.errors_total += 1

    async def _close_ws(self) -> None:
        if self._ws is not None:
            try:
                await self._ws.close(1000, "bye")
            except Exception:
                pass

    async def _handle_msg(self, raw) -> None:
        if not isinstance(raw, (bytes, bytearray)):
            return
        try:
            msg = msgpack.unpackb(raw, raw=False)
        except Exception as e:
            logger.debug("embodied unpack error: %s", e)
            self.errors_total += 1
            return
        # 1) Server keepalive ping → respond pong (см. embodied_broker.py
        # _ping_loop: server шлёт {"type":"ping"} каждые 30с).
        if isinstance(msg, dict) and msg.get("type") == "ping":
            try:
                if self._ws is not None:
                    await self._ws.send(msgpack.packb({"type": "pong"}))
                self.pings_handled += 1
            except Exception as e:
                logger.debug("embodied pong send error: %s", e)
            return
        # 2) Observation от P40 echo
        if isinstance(msg, dict):
            self.observations_received += 1
            obs_cid = msg.get("cid")
            if isinstance(obs_cid, str) and obs_cid:
                sent_ts = self._sent_at.pop(obs_cid, None)
                if sent_ts is not None:
                    lat_ms = (time.monotonic() - sent_ts) * 1000.0
                    self._latencies_ms.append(lat_ms)
            if self.on_observation is not None:
                try:
                    self.on_observation(msg)
                except Exception as e:
                    logger.debug("embodied on_observation error: %s", e)

    # ── stats ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        lats = list(self._latencies_ms)
        out = {
            "url": self.url.split('?')[0],
            "connected": self.connected,
            "states_sent": self.states_sent,
            "observations_received": self.observations_received,
            "pings_handled": self.pings_handled,
            "errors_total": self.errors_total,
            "last_error": self.last_error,
            "latency_samples": len(lats),
        }
        if lats:
            srt = sorted(lats)
            out["latency_mean_ms"] = round(sum(srt) / len(srt), 1)
            out["latency_p50_ms"] = round(srt[len(srt) // 2], 1)
            p95_idx = min(int(len(srt) * 0.95), len(srt) - 1)
            out["latency_p95_ms"] = round(srt[p95_idx], 1)
            out["latency_max_ms"] = round(srt[-1], 1)
        return out
