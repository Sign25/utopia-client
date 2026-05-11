"""WS-подписчик на /ws/feed VPS-broker'а — раздаёт snap Мира всем клиентам.

Фаза 2 переноса obs на клиент: один общий snap (~65 КБ JSON) рассылается
VPS-broker'ом всем подписчикам с частотой публикации P40 (5-15 Гц).

Клиент:
  1. при старте — HTTP bootstrap() (terrain + biomes + конфиг)
  2. подключается к wss://divisci.com/ws/feed (public, без токена)
  3. на каждый `{type: snapshot, payload: <snap>}` зовёт cache.apply_snap(payload)

Архитектурно эквивалент ColonyWSClient: thread + asyncio + websockets,
экспоненциальный backoff при обрывах. На P40 snap собирается фоновым
таском `_visual_snapshot_cache_loop`, оттуда `world_pusher` шлёт в
`wss://divisci.com/ws/world`, broker раздаёт всем `/ws/feed`-подписчикам.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Optional

from .world_cache import WorldStateCache

logger = logging.getLogger("utopia_client.world_feed")

INITIAL_BACKOFF_SEC = 1.0
MAX_BACKOFF_SEC = 30.0


class WorldFeedClient:
    """Подписчик /ws/feed: на каждый snap зовёт WorldStateCache.apply_snap."""

    def __init__(self, server: str, cache: WorldStateCache) -> None:
        if server.startswith("https://"):
            base = "wss://" + server[len("https://"):]
        elif server.startswith("http://"):
            base = "ws://" + server[len("http://"):]
        else:
            base = server
        self.url = f"{base.rstrip('/')}/ws/feed"
        self.cache = cache

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._stop_flag = threading.Event()
        # Метрики
        self.connected: bool = False
        self.last_error: str = ""
        self.snapshots_received: int = 0

    # ─────────────────────────── lifecycle ───────────────────────────────

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_flag.clear()
        self._thread = threading.Thread(
            target=self._thread_main, daemon=True, name="world-feed")
        self._thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        loop = self._loop
        ev = self._stop_event
        if loop is not None and ev is not None:
            try:
                loop.call_soon_threadsafe(ev.set)
            except RuntimeError:
                pass
        t = self._thread
        if t is not None:
            t.join(timeout=5.0)
        self._thread = None
        self._loop = None
        self._stop_event = None

    def stats(self) -> dict:
        return {
            "url": self.url,
            "connected": self.connected,
            "snapshots_received": self.snapshots_received,
            "last_error": self.last_error,
            **self.cache.stats(),
        }

    # ─────────────────────────── async core ──────────────────────────────

    def _thread_main(self) -> None:
        try:
            asyncio.run(self._run())
        except Exception as e:
            logger.warning("world feed thread crashed: %s", e)

    async def _run(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        if self._stop_flag.is_set():
            return

        await self._bootstrap_loop()

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
                logger.warning("world feed error: %s", self.last_error)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                return
            except asyncio.TimeoutError:
                pass
            backoff = min(backoff * 2, MAX_BACKOFF_SEC)

    async def _bootstrap_loop(self) -> None:
        """Однократный HTTP bootstrap. Повтор при ошибках до успеха или stop()."""
        if self.cache.is_bootstrapped:
            return
        backoff = INITIAL_BACKOFF_SEC
        while not self._stop_event.is_set() and not self.cache.is_bootstrapped:
            try:
                await asyncio.to_thread(self.cache.bootstrap)
                return
            except Exception as e:
                self.last_error = f"bootstrap: {type(e).__name__}: {e}"
                logger.warning(
                    "world bootstrap failed: %s, retry in %.1fs",
                    self.last_error, backoff)
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

        logger.info("world feed connecting to %s", self.url)
        async with websockets.connect(
                self.url, max_size=2 * 1024 * 1024,
                ping_interval=20, ping_timeout=20) as ws:
            self.connected = True
            self.last_error = ""
            logger.info("world feed connected")
            try:
                async for raw in ws:
                    if self._stop_event.is_set():
                        break
                    self._consume_message(raw)
            finally:
                self.connected = False

    # ─────────────────────────── message handler ─────────────────────────

    def _consume_message(self, raw: str | bytes) -> bool:
        """Разобрать один WS-frame и при `type=snapshot` скормить в cache.

        Возвращает True, если кадр был распознан и применён. Все ошибки
        парсинга/apply поглощаются (записываются в last_apply_error на cache).
        """
        try:
            env = json.loads(raw)
        except (TypeError, ValueError, json.JSONDecodeError):
            return False
        if not isinstance(env, dict):
            return False
        if env.get("type") != "snapshot":
            return False
        payload = env.get("payload")
        if not isinstance(payload, dict):
            return False
        try:
            applied = self.cache.apply_snap(payload)
        except Exception as e:
            logger.warning("apply_snap raised: %s", e)
            return False
        if applied:
            self.snapshots_received += 1
        return applied
