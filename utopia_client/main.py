"""Точка входа клиента: heartbeat + команды CLI.

Пока минимум — heartbeat (POST stats каждые N секунд). Локального Мира
здесь нет; это будет в следующих фазах. Heartbeat нужен, чтобы клиент
появился в личном кабинете на divisci.com со статусом «live».
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from . import __version__
from .api import UtopiaAPI
from .benchmark import estimate_population, run_full
from .config import DEFAULT_SERVER, get_or_prompt, load_config, save_config

HEARTBEAT_SEC = 30.0

logger = logging.getLogger("utopia_client")


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )


def _ensure_config() -> dict:
    cfg = load_config()
    if not cfg.get("server"):
        cfg["server"] = DEFAULT_SERVER
        save_config(cfg)
    cfg["token"] = get_or_prompt(
        "token", "Push-токен (страница Кабинет на divisci.com)", secret=True,
    )
    cfg["name"] = get_or_prompt(
        "name", "Имя колонии (короткое, например home-3060ti)",
    )
    return cfg


def cmd_config(args: argparse.Namespace) -> int:
    cfg = load_config()
    if not cfg:
        print("Конфиг пуст. Запусти `run` или `benchmark` чтобы создать.")
        return 1
    safe = dict(cfg)
    if "token" in safe:
        safe["token"] = safe["token"][:6] + "…(скрыт)"
    print(json.dumps(safe, indent=2, ensure_ascii=False))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    cfg = _ensure_config()
    print("Замер производительности…")
    result = run_full()
    n = estimate_population(result)
    result["estimated_population"] = n
    print(json.dumps(result, indent=2, ensure_ascii=False))
    cfg["benchmark"] = result
    save_config(cfg)
    print(f"\nРекомендуемая популяция: {n} особей.")
    print("Сохранено в конфиг.")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _ensure_config()
    api = UtopiaAPI(cfg["server"], cfg["token"])
    name = cfg["name"]
    logger.info("starting heartbeat as colony=%s server=%s", name, cfg["server"])
    bench = cfg.get("benchmark", {})
    extra_static = {
        "client_version": __version__,
        "estimated_population": bench.get("estimated_population", 0),
        "cpu_gflops": bench.get("cpu_gflops", 0.0),
        "gpu": bench.get("gpu", {"available": False}),
    }
    tick = 0
    while True:
        ok = api.push_stats(
            name=name,
            n_alive=0,
            best_fitness=0.0,
            generation=0,
            world_tick=tick,
            extra={**extra_static, "status": "idle"},
        )
        logger.info("heartbeat tick=%d ok=%s", tick, ok)
        tick += 1
        time.sleep(HEARTBEAT_SEC)


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    p = argparse.ArgumentParser(prog="utopia-client",
                                description="Клиент распределённой эволюции NeuroCore")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="Запустить heartbeat")
    sub.add_parser("benchmark", help="Замерить ПК и оценить популяцию")
    sub.add_parser("config", help="Показать текущий конфиг")
    args = p.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "benchmark":
        return cmd_benchmark(args)
    if args.cmd == "config":
        return cmd_config(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
