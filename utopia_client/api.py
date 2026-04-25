"""HTTP-клиент для VPS (divisci.com)."""

from __future__ import annotations

import logging
import requests

logger = logging.getLogger("utopia_client.api")


class UtopiaAPI:
    def __init__(self, server: str, token: str, timeout: float = 15.0) -> None:
        self.server = server.rstrip("/")
        self.token = token
        self.timeout = timeout

    def push_stats(
        self,
        name: str,
        n_alive: int,
        best_fitness: float,
        generation: int,
        world_tick: int,
        extra: dict | None = None,
    ) -> bool:
        url = f"{self.server}/api/colony/stats"
        payload = {
            "name": name,
            "n_alive": int(n_alive),
            "best_fitness": float(best_fitness),
            "generation": int(generation),
            "world_tick": int(world_tick),
            "extra": extra or {},
        }
        try:
            r = requests.post(
                url, json=payload,
                headers={"X-Push-Token": self.token},
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return True
            logger.warning("push_stats HTTP %d: %s", r.status_code, r.text[:200])
            return False
        except Exception as e:
            logger.warning("push_stats error: %s", e)
            return False

    def fetch_command(self) -> dict | None:
        """Опросить желаемое состояние от VPS. None при ошибке."""
        url = f"{self.server}/api/colony/command"
        try:
            r = requests.get(
                url,
                headers={"X-Push-Token": self.token},
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return r.json()
            logger.warning("fetch_command HTTP %d: %s", r.status_code, r.text[:200])
            return None
        except Exception as e:
            logger.warning("fetch_command error: %s", e)
            return None

    def push_benchmark(self, result: dict) -> bool:
        url = f"{self.server}/api/colony/benchmark"
        try:
            r = requests.post(
                url, json=result,
                headers={"X-Push-Token": self.token},
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return True
            logger.warning("push_benchmark HTTP %d: %s", r.status_code, r.text[:200])
            return False
        except Exception as e:
            logger.warning("push_benchmark error: %s", e)
            return False

    def fetch_seed(self, dest_path: str) -> bool:
        url = f"{self.server}/api/seed"
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in r.iter_content(8192):
                        f.write(chunk)
            return True
        except Exception as e:
            logger.warning("fetch_seed error: %s", e)
            return False
