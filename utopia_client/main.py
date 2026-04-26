"""Точка входа клиента: daemon-петля + CLI.

В режиме `run` клиент опрашивает VPS: что делать (idle/benchmark/run).
- idle      — heartbeat n_alive=0, status=idle
- benchmark — замер CPU/RAM/GPU, push результат, ждём команду
- run       — heartbeat status=running (локальной симуляции пока нет, Phase D)

Команда меняется через UI на divisci.com → /me → кнопки.
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
from .ws_client import ColonyWSClient

HEARTBEAT_SEC = 30.0
COMMAND_POLL_SEC = 10.0

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


def cmd_set_token(args: argparse.Namespace) -> int:
    cfg = load_config()
    if not cfg.get("server"):
        cfg["server"] = DEFAULT_SERVER
    cfg["token"] = args.token.strip()
    save_config(cfg)
    print("Токен сохранён.")
    return 0


def cmd_set_name(args: argparse.Namespace) -> int:
    cfg = load_config()
    if not cfg.get("server"):
        cfg["server"] = DEFAULT_SERVER
    cfg["name"] = args.name.strip()
    save_config(cfg)
    print(f"Имя колонии: {cfg['name']}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    cfg = _ensure_config()
    print("Замер производительности…")
    result = _run_benchmark()
    print(json.dumps(result, indent=2, ensure_ascii=False))
    cfg["benchmark"] = result
    save_config(cfg)
    print(f"\nРекомендуемая популяция: {result['estimated_population']} особей.")
    api = UtopiaAPI(cfg["server"], cfg["token"])
    if api.push_benchmark(result):
        print("Результат отправлен в Утопию.")
    return 0


def _run_benchmark() -> dict:
    result = run_full()
    result["estimated_population"] = estimate_population(result)
    result["client_version"] = __version__
    result["measured_at"] = int(time.time())
    return result


def _heartbeat(api: UtopiaAPI, name: str, status: str, tick: int,
               bench: dict, n_alive: int = 0) -> bool:
    extra = {
        "client_version": __version__,
        "estimated_population": bench.get("estimated_population", 0),
        "cpu_gflops": bench.get("cpu_gflops", 0.0),
        "gpu": bench.get("gpu", {"available": False}),
        "status": status,
    }
    return api.push_stats(
        name=name,
        n_alive=n_alive,
        best_fitness=0.0,
        generation=0,
        world_tick=tick,
        extra=extra,
    )


def cmd_run(args: argparse.Namespace) -> int:
    """Daemon-петля: опрашивает команду от VPS и реагирует."""
    cfg = _ensure_config()
    api = UtopiaAPI(cfg["server"], cfg["token"])
    name = cfg["name"]
    logger.info("starting daemon as colony=%s server=%s", name, cfg["server"])

    bench_initial = cfg.get("benchmark", {})
    ws = ColonyWSClient(
        cfg["server"], cfg["token"], name, __version__,
        estimated_population=bench_initial.get("estimated_population", 0),
    )
    ws.start()

    tick = 0
    last_heartbeat = 0.0
    last_poll = 0.0
    current_state = "idle"
    bench = cfg.get("benchmark", {})

    try:
        while True:
            now = time.monotonic()

            # Опрос команды
            if now - last_poll >= COMMAND_POLL_SEC:
                cmd = api.fetch_command()
                last_poll = now
                if cmd:
                    desired = cmd.get("state", "idle")
                    if desired != current_state:
                        logger.info("command: %s -> %s", current_state, desired)
                    # benchmark — выполняем сразу, состояние сбрасывает VPS на idle
                    if desired == "benchmark":
                        logger.info("running benchmark…")
                        result = _run_benchmark()
                        cfg["benchmark"] = result
                        save_config(cfg)
                        bench = result
                        if api.push_benchmark(result):
                            logger.info("benchmark pushed: pop=%d gflops=%.2f",
                                        result["estimated_population"],
                                        result["cpu_gflops"])
                        current_state = "idle"
                    else:
                        current_state = desired

            # Heartbeat
            if now - last_heartbeat >= HEARTBEAT_SEC:
                n_alive = ws.n_alive_owned
                ok = _heartbeat(api, name, current_state, tick, bench, n_alive)
                logger.info("heartbeat tick=%d state=%s ws=%s n_alive=%d ok=%s",
                            tick, current_state, ws.connected, n_alive, ok)
                tick += 1
                last_heartbeat = now

            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("interrupted, stopping…")
    finally:
        ws.stop()
    return 0


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    print(f"=== Utopia Client v{__version__} ===")
    p = argparse.ArgumentParser(prog="utopia-client",
                                description="Клиент распределённой эволюции NeuroCore")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run", help="Запустить daemon (опрос команды + heartbeat)")
    sub.add_parser("benchmark", help="Замерить ПК и оценить популяцию")
    sub.add_parser("config", help="Показать текущий конфиг")
    p_token = sub.add_parser("set-token", help="Записать push-токен в конфиг")
    p_token.add_argument("token", help="Push-токен из Кабинета на divisci.com")
    p_name = sub.add_parser("set-name", help="Записать имя колонии в конфиг")
    p_name.add_argument("name", help="Короткое имя колонии (например home-3060ti)")
    args = p.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "benchmark":
        return cmd_benchmark(args)
    if args.cmd == "config":
        return cmd_config(args)
    if args.cmd == "set-token":
        return cmd_set_token(args)
    if args.cmd == "set-name":
        return cmd_set_name(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
