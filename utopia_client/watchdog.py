"""Внутренний watchdog: kill основного процесса, если heartbeat-файл протух.

NSSM перезапускает только упавший процесс. Если main жив, но WS-петля
залипла (GIL-thrash, deadlock, бесконечный wait) — heartbeat.txt
перестаёт обновляться, но процесс не умирает. Этот watchdog ловит такие
случаи: читает heartbeat-файл раз в 60с, если timestamp старше 5 мин —
taskkill main, NSSM поднимает заново.

Запускается ВТОРЫМ NSSM-сервисом (UtopiaClientWatchdog) параллельно
основному. Сам себя watchdog не сторожит — если он упал, NSSM
перезапустит его без последствий для main.
"""

from __future__ import annotations

import logging
import os
import pathlib
import subprocess
import sys
import tempfile
import time

STALE_THRESHOLD_SEC = 300  # 5 минут без heartbeat → процесс считается зависшим
CHECK_INTERVAL_SEC = 60

logger = logging.getLogger("utopia_watchdog")


def _state_dir() -> pathlib.Path:
    base = os.environ.get("APPDATA") or os.environ.get("XDG_STATE_HOME") \
        or tempfile.gettempdir()
    return pathlib.Path(base) / "utopia-client"


def _kill_pid(pid: int) -> bool:
    if sys.platform == "win32":
        try:
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True, check=False, timeout=10)
            return True
        except Exception as e:
            logger.warning("taskkill failed: %s", e)
            return False
    else:
        try:
            os.kill(pid, 9)
            return True
        except Exception as e:
            logger.warning("os.kill failed: %s", e)
            return False


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    state = _state_dir()
    hb_path = state / "heartbeat.txt"
    pid_path = state / "main.pid"
    logger.info("watchdog started, watching %s (stale > %ds → kill)",
                hb_path, STALE_THRESHOLD_SEC)

    while True:
        time.sleep(CHECK_INTERVAL_SEC)
        if not hb_path.exists():
            continue
        try:
            ts = float(hb_path.read_text().strip())
        except Exception:
            continue
        age = time.time() - ts
        if age <= STALE_THRESHOLD_SEC:
            continue
        if not pid_path.exists():
            logger.warning("heartbeat stale (%.0fs) but no pid file", age)
            continue
        try:
            pid = int(pid_path.read_text().strip())
        except Exception as e:
            logger.warning("cannot read pid: %s", e)
            continue
        logger.warning("heartbeat stale %.0fs (>%ds), killing pid=%d",
                       age, STALE_THRESHOLD_SEC, pid)
        if _kill_pid(pid):
            # Дать NSSM время заметить смерть и поднять заново.
            time.sleep(15)


if __name__ == "__main__":
    sys.exit(main() or 0)
