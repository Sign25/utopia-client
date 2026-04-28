"""Ring-buffer logger handler + общая настройка логирования.

Идея: помимо stderr пишем последние N строк в память, чтобы main-петля
могла периодически слать их на VPS (POST /api/colony/log_tail).
Параллельно — RotatingFileHandler в ~/.utopia-client/client.log на случай,
если ringbuffer не помог (краш до push).
"""

from __future__ import annotations

import logging
import logging.handlers
from collections import deque
from pathlib import Path
from typing import Deque

from .config import config_dir

_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


class RingBufferHandler(logging.Handler):
    """Кольцевой буфер последних N форматированных строк лога."""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self.capacity = capacity
        self._buf: Deque[str] = deque(maxlen=capacity)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._buf.append(self.format(record))
        except Exception:
            self.handleError(record)

    def tail(self, n: int = 200) -> list[str]:
        if n <= 0 or not self._buf:
            return []
        if n >= len(self._buf):
            return list(self._buf)
        return list(self._buf)[-n:]


_RING: RingBufferHandler | None = None


def setup_logging(level: int = logging.INFO,
                  ring_capacity: int = 500,
                  log_file_bytes: int = 1_000_000,
                  log_file_backups: int = 3) -> RingBufferHandler:
    """Идемпотентная настройка root-логгера: console + file + ring.

    Возвращает RingBufferHandler — main-петля использует его tail() для пуша.
    """
    global _RING
    root = logging.getLogger()
    root.setLevel(level)

    fmt = logging.Formatter(_FORMAT)

    # Console (stderr) — оставляем как было.
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
               for h in root.handlers):
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        root.addHandler(ch)

    # File — RotatingFileHandler.
    log_path = config_dir() / "client.log"
    if not any(isinstance(h, logging.handlers.RotatingFileHandler)
               and Path(getattr(h, "baseFilename", "")) == log_path
               for h in root.handlers):
        try:
            fh = logging.handlers.RotatingFileHandler(
                log_path, maxBytes=log_file_bytes,
                backupCount=log_file_backups, encoding="utf-8",
            )
            fh.setFormatter(fmt)
            root.addHandler(fh)
        except Exception as e:
            root.warning("file logging disabled: %s", e)

    # Ring buffer (один на процесс).
    if _RING is None:
        _RING = RingBufferHandler(capacity=ring_capacity)
        _RING.setFormatter(fmt)
        root.addHandler(_RING)

    return _RING


def get_ring() -> RingBufferHandler | None:
    return _RING
