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
from .log_buffer import get_ring, setup_logging
from .ws_client import ColonyWSClient

HEARTBEAT_SEC = 30.0
COMMAND_POLL_SEC = 10.0
LOGPUSH_SEC = 30.0
LOGPUSH_LINES = 200
DIAGNOSTICS_PUSH_SEC = 30.0
SELFUPDATE_CHECK_SEC = 60.0

logger = logging.getLogger("utopia_client")


def _setup_logging() -> None:
    setup_logging(level=logging.INFO)


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
    if not cfg.get("genesis_mode"):
        cfg["genesis_mode"] = _prompt_genesis_mode(cfg)
        save_config(cfg)
    return cfg


def _prompt_genesis_mode(cfg: dict) -> str:
    """Phase F.7.4: при первом подключении спросить, как создать колонию.

    Опрашивает /api/world/genepool/info (без auth), показывает CLI-меню:
      [1] С генофонда (потомки активных колоний Мира)
      [2] С чистого листа (свежие веса из seed.norg)
    Если donor_pool_size=0 — опция 1 недоступна, авто-выбор 'fresh'.
    Сохранённое значение прокидывается в hello через ColonyWSClient.

    v0.9.18: flush=True, try/except EOFError, логируем конечный выбор —
    embedded Python на Windows иногда теряет prompt без явного flush;
    при non-TTY (запуск из сервиса) input() кидает EOFError → 'fresh'.
    """
    api = UtopiaAPI(cfg["server"], cfg.get("token", ""))
    info = api.get_genepool_info() or {}
    donor_available = bool(info.get("donor_available"))
    pool_size = int(info.get("donor_pool_size", 0) or 0)
    avg_score = float(info.get("donor_avg_score", 0.0) or 0.0)
    world_pop = int(info.get("world_population", 0) or 0)

    print(flush=True)
    print("=== Создание колонии ===", flush=True)
    print(f"Мир: {world_pop} особей онлайн, "
          f"донорский пул: {pool_size} ({avg_score:.0f} ср. score)",
          flush=True)
    print(flush=True)
    if donor_available:
        print(" [1] С генофонда — потомки активных колоний (быстрый старт)",
              flush=True)
    else:
        print(" [1] С генофонда — недоступно (Мир пока пустой)", flush=True)
    print(" [2] С чистого листа — свежие веса из seed.norg", flush=True)
    print(flush=True)
    if not donor_available:
        print("Авто-выбор: с чистого листа.", flush=True)
        logger.info("genesis_mode: fresh (donor pool empty)")
        return "fresh"
    while True:
        try:
            sys.stdout.flush()
            choice = input("Выбор [1/2, по умолчанию 1]: ").strip()
        except EOFError:
            print("Нет интерактивного ввода — авто-выбор: с генофонда.",
                  flush=True)
            logger.info("genesis_mode: donor (EOFError fallback)")
            return "donor"
        if choice in ("", "1"):
            print("Выбрано: с генофонда.", flush=True)
            logger.info("genesis_mode: donor")
            return "donor"
        if choice == "2":
            print("Выбрано: с чистого листа.", flush=True)
            logger.info("genesis_mode: fresh")
            return "fresh"
        print("Введите 1 или 2.", flush=True)


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


def cmd_set_genesis_mode(args: argparse.Namespace) -> int:
    mode = args.mode.strip().lower()
    if mode not in ("donor", "fresh"):
        print(f"Неизвестный режим: {mode!r}. Допустимо: donor, fresh.",
              flush=True)
        return 2
    cfg = load_config()
    if not cfg.get("server"):
        cfg["server"] = DEFAULT_SERVER
    cfg["genesis_mode"] = mode
    save_config(cfg)
    print(f"genesis_mode = {mode}. Перезапусти клиент, чтобы применить.",
          flush=True)
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


def _make_ws(cfg: dict, name: str) -> ColonyWSClient:
    """F.6.B: фабрика WS-клиента. Создаём заново на каждый idle→run, чтобы
    после bye+close был чистый hello (не зависим от состояния прошлой
    сессии: _stop_flag, _seed_buffers, compute и т.п.)."""
    bench = cfg.get("benchmark", {})
    return ColonyWSClient(
        cfg["server"], cfg["token"], name, __version__,
        estimated_population=bench.get("estimated_population", 0),
        genesis_mode=cfg.get("genesis_mode", "auto"),
    )


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


def _try_self_update(api: UtopiaAPI) -> bool:
    """Если /api/client/info вернул новую версию — скачать zip, распаковать
    поверх установочной директории и сделать execv. Возвращает True если
    обновление прошло (этот процесс уже не вернётся; код ниже не выполнится).
    """
    info = api.get_client_info()
    if not info:
        return False
    remote_ver = str(info.get("version") or "")
    if not remote_ver or remote_ver == __version__:
        return False
    # Сравнение лексикографическое подходит для семвера X.Y.Z
    try:
        cur = tuple(int(p) for p in __version__.split("."))
        nxt = tuple(int(p) for p in remote_ver.split("."))
    except Exception:
        return False
    if nxt <= cur:
        return False
    logger.info("self-update: %s → %s, downloading…", __version__, remote_ver)
    import os
    import tempfile
    import zipfile
    from pathlib import Path
    fd, tmp_path = tempfile.mkstemp(prefix="utopia_client_", suffix=".zip")
    os.close(fd)
    try:
        if not api.download_client_zip(tmp_path):
            return False
        # Установочный каталог = parent(parent(__file__)) для main.py.
        pkg_dir = Path(__file__).resolve().parent  # …/utopia_client
        install_dir = pkg_dir.parent              # …/(install_root)
        with zipfile.ZipFile(tmp_path) as z:
            z.extractall(install_dir)
        logger.info("self-update: extracted to %s, restarting via execv",
                    install_dir)
    except Exception as e:
        logger.warning("self-update failed: %s", e)
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    # Перезапуск. На Windows os.execv не цитирует argv — путь с пробелом
    # (напр. "C:\Users\Mr. Krabs\...\python.exe") ломает командную строку,
    # интерпретатор стартует от мусорного префикса до первого пробела.
    # Поэтому на Windows используем subprocess.Popen + sys.exit (всё равно
    # exec там реализован как spawn-and-exit), на POSIX — os.execv.
    argv = [sys.executable, "-m", "utopia_client.main"] + sys.argv[1:]
    try:
        if os.name == "nt":
            import subprocess
            subprocess.Popen(argv, close_fds=False)
            sys.exit(0)
        else:
            os.execv(sys.executable, argv)
    except Exception as e:
        logger.error("restart failed: %s — exit(75) для перезапуска снаружи", e)
        sys.exit(75)
    return True  # недостижимо


def cmd_run(args: argparse.Namespace) -> int:
    """Daemon-петля: опрашивает команду от VPS и реагирует."""
    cfg = _ensure_config()
    api = UtopiaAPI(cfg["server"], cfg["token"])
    name = cfg["name"]
    logger.info("starting daemon as colony=%s server=%s", name, cfg["server"])

    # F.6.B: WS подключается только в state=run. В idle WS не открыт →
    # P40 видит client_disconnected → freeze_personal (owned-особи живут
    # в Мире, не размножаются). При idle→run hello триггерит unfreeze.
    ws: ColonyWSClient | None = None

    last_heartbeat = 0.0
    last_poll = 0.0
    last_logpush = 0.0
    last_diag_push = 0.0
    last_selfupdate_check = 0.0
    current_state = "idle"
    bench = cfg.get("benchmark", {})
    ring = get_ring()

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
                        # benchmark не должен оставлять WS открытым.
                        if ws is not None:
                            ws.stop()
                            ws = None
                            logger.info("ws stopped (benchmark→idle)")
                        current_state = "idle"
                    else:
                        # F.6.B: переходы run↔idle переключают WS.
                        if desired == "run" and ws is None:
                            ws = _make_ws(cfg, name)
                            ws.start()
                            logger.info("ws started (idle→run)")
                        elif desired == "idle" and ws is not None:
                            ws.stop()  # отправит bye → close
                            ws = None
                            logger.info("ws stopped with bye (run→idle)")
                        current_state = desired

            # Heartbeat
            if now - last_heartbeat >= HEARTBEAT_SEC:
                n_alive = ws.n_alive_owned if ws is not None else 0
                world_tick = ws.last_world_tick if ws is not None else 0
                ws_connected = ws.connected if ws is not None else False
                ok = _heartbeat(api, name, current_state, world_tick, bench, n_alive)
                logger.info("heartbeat world_tick=%d state=%s ws=%s n_alive=%d ok=%s",
                            world_tick, current_state, ws_connected, n_alive, ok)
                last_heartbeat = now

            # Push tail логов в VPS (admin потом достаёт через cabinet_log.sh)
            if ring is not None and now - last_logpush >= LOGPUSH_SEC:
                try:
                    api.push_log_tail(ring.tail(LOGPUSH_LINES))
                except Exception as e:
                    logger.debug("log_tail push skipped: %s", e)
                last_logpush = now

            # Phase 1/2/6 diagnostics push (only when compute is alive).
            if (ws is not None and ws.compute is not None
                    and now - last_diag_push >= DIAGNOSTICS_PUSH_SEC):
                try:
                    diag = ws.compute.diagnostics()
                    diag["world_tick"] = ws.last_world_tick
                    diag["dump"] = ws.compute._dump_state()
                    api.push_diagnostics(name, diag)
                except Exception as e:
                    logger.debug("diagnostics push skipped: %s", e)
                last_diag_push = now

            # Self-update проверка: дёргаем /api/client/info, при новой версии
            # скачиваем zip, распаковываем, execv. Этот вызов может не вернуться.
            if now - last_selfupdate_check >= SELFUPDATE_CHECK_SEC:
                last_selfupdate_check = now
                try:
                    if _try_self_update(api):
                        return 0  # недостижимо после execv
                except Exception as e:
                    logger.warning("self-update check error: %s", e)

            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("interrupted, stopping…")
    finally:
        if ws is not None:
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
    sub.add_parser("bench-gpu", help="Phase F3.0: forward StemCell на CUDA, ms/tps")
    p_token = sub.add_parser("set-token", help="Записать push-токен в конфиг")
    p_token.add_argument("token", help="Push-токен из Кабинета на divisci.com")
    p_name = sub.add_parser("set-name", help="Записать имя колонии в конфиг")
    p_name.add_argument("name", help="Короткое имя колонии (например home-3060ti)")
    p_gm = sub.add_parser("set-genesis-mode",
                          help="Сменить режим создания колонии (donor/fresh)")
    p_gm.add_argument("mode", choices=["donor", "fresh"],
                      help="donor — старт от обученных доноров; "
                           "fresh — с чистого листа")
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
    if args.cmd == "set-genesis-mode":
        return cmd_set_genesis_mode(args)
    if args.cmd == "bench-gpu":
        from .gpu_bench import cmd_bench_gpu
        return cmd_bench_gpu(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
