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
import os
import pathlib
import sys
import tempfile
import time

from . import __version__
from .api import UtopiaAPI
from .benchmark import estimate_population, run_full
from .config import (
    DEFAULT_P40_URL,
    DEFAULT_SERVER,
    get_or_prompt,
    load_config,
    save_config,
)
from .log_buffer import get_ring, setup_logging
from .ws_client import ColonyWSClient
from .world_cache import WorldStateCache
from .world_feed_client import WorldFeedClient
# Body Migration Этап 3a Phase 1 (Бендер, 27.05.2026): embodied API skeleton.
# Импортируется лениво при включении флага cfg["embodied_enabled"] чтобы
# не тянуть msgpack/websockets на startup при выключенном фиче-флаге.

HEARTBEAT_SEC = 30.0
COMMAND_POLL_SEC = 10.0
LOGPUSH_SEC = 30.0
LOGPUSH_LINES = 200
DIAGNOSTICS_PUSH_SEC = 30.0
SELFUPDATE_CHECK_SEC = 60.0

logger = logging.getLogger("utopia_client")

# Uptime процесса utopia-client — отправляется в ColonyDiagnostics для
# UI StatsPage (карточка «Клиент: Xч Ym»).
_PROCESS_STARTED_AT: float = time.monotonic()


def _state_dir() -> pathlib.Path:
    """Каталог для local heartbeat/pid (watchdog читает оттуда)."""
    base = os.environ.get("APPDATA") or os.environ.get("XDG_STATE_HOME") \
        or tempfile.gettempdir()
    path = pathlib.Path(base) / "utopia-client"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return path


def _write_heartbeat_file() -> None:
    """Local heartbeat-маркер для watchdog.py. Stale > 5 мин → процесс kill."""
    try:
        (_state_dir() / "heartbeat.txt").write_text(str(time.time()))
    except Exception:
        pass


def _write_pid_file() -> None:
    """Pid основного процесса — watchdog знает кого killать."""
    try:
        (_state_dir() / "main.pid").write_text(str(os.getpid()))
    except Exception:
        pass


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
    # Имя колонии теперь живёт в кабинете на divisci.com (profile.name).
    # Дёргаем VPS через push-токен; кешируем в config.json для офлайна.
    name = _fetch_colony_name(cfg)
    if name and name != cfg.get("name"):
        if cfg.get("name"):
            logger.info("colony name: %s → %s (из кабинета)", cfg["name"], name)
        cfg["name"] = name
        save_config(cfg)
    elif not cfg.get("name"):
        # VPS недоступен и нет кеша — отказываем, без имени работать нельзя.
        raise SystemExit(
            "Не удалось получить имя колонии: VPS недоступен, локальный кеш "
            "пуст. Открой Кабинет на divisci.com, заполни поле «имя», "
            "потом запусти клиент снова."
        )
    if not cfg.get("genesis_mode"):
        cfg["genesis_mode"] = _prompt_genesis_mode(cfg)
        save_config(cfg)
    return cfg


def _fetch_colony_name(cfg: dict) -> str:
    """Запросить colony_name (и user_id) из кабинета на VPS. '' при недоступности.

    Side-effect: при успехе пишет `user_id` в cfg (нужен для Z7.i.b LAN-вызова
    P40 endpoint `/api/world/upgrade_lineage_to_zodchiy`, у которого нет VPS-auth
    и который требует явный uuid пользователя).
    """
    try:
        api = UtopiaAPI(cfg["server"], cfg.get("token", ""))
        ident = api.get_identity() or {}
        name = str(ident.get("colony_name", "")).strip()
        uid = str(ident.get("user_id", "")).strip()
        if uid and cfg.get("user_id") != uid:
            cfg["user_id"] = uid
            save_config(cfg)
        return name
    except Exception as e:
        logger.warning("colony name fetch failed: %s", e)
        return ""


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


def cmd_tag_adam(args: argparse.Namespace) -> int:
    """Single-organism pivot этап 1: пометить cid как Адама на P40 (LAN).

    Preconditions (иначе 400/404): cid уже спроецирован как owned-zodchiy в
    Мире (клиент в run + single_organism, projection_batch self-heal'ил cid).
    Дёргать ПОСЛЕ того, как heartbeat показал n_alive>=1 для этого cid.
    """
    cfg = load_config()
    if not cfg.get("server"):
        print("Конфиг пуст — сначала запусти клиент (run).", flush=True)
        return 1
    p40_url = str(os.environ.get("UTOPIA_P40_URL")
                  or cfg.get("p40_url") or DEFAULT_P40_URL)
    api = UtopiaAPI(cfg["server"], cfg.get("token", ""))
    print(f"tag_adam cid={args.cid} → {p40_url}/api/world/adam/tag", flush=True)
    resp = api.tag_adam(p40_url, args.cid)
    if resp is None:
        print("tag_adam: НЕ удалось (см. лог). Проверь, что cid спроецирован "
              "как owned-zodchiy в Мире (n_alive>=1).", flush=True)
        return 1
    print(json.dumps(resp, indent=2, ensure_ascii=False), flush=True)
    print(f"✓ Адам помечен: adam_cid={resp.get('adam_cid')} "
          f"passive_flora_eating={resp.get('passive_flora_eating')}", flush=True)
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
               bench: dict, n_alive: int = 0,
               colony_summary: Optional[dict] = None,
               creature_stats: Optional[dict] = None) -> bool:
    extra = {
        "client_version": __version__,
        "estimated_population": bench.get("estimated_population", 0),
        "cpu_gflops": bench.get("cpu_gflops", 0.0),
        "gpu": bench.get("gpu", {"available": False}),
        "status": status,
    }
    # colony_summary (01.06.2026, UI /stats): агрегат выживания/эволюции/
    # обучения → public_meta.extra.colony_summary → useStatsData.
    if colony_summary:
        extra["colony_summary"] = colony_summary
    # creature_stats (01.06.2026, UI): per-creature client-only поля (species_id/
    # topo/inst) keyed by cid — тем же надёжным каналом (diag-push ловит 502).
    if creature_stats:
        extra["creature_stats"] = creature_stats
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
        # Чистим __pycache__ — иначе Python может подгрузить старый .pyc,
        # пока mtime новых .py отстаёт или совпадает с .pyc (Windows).
        import shutil
        for cache_dir in pkg_dir.rglob("__pycache__"):
            try:
                shutil.rmtree(cache_dir, ignore_errors=True)
            except Exception:
                pass
        logger.info("self-update: extracted to %s (pycache cleared), restarting via execv",
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


def _try_neurocore_update(api: UtopiaAPI) -> bool:
    """Если sha256 neurocore-зеркала на VPS отличается от установленного —
    переустановить пакет и перезапустить процесс. Возвращает True, если
    обновление прошло (этот процесс уже не вернётся).

    Локальный sha256 хранится в `<install_dir>/neurocore_sha256.txt`.
    Если файла нет — это либо первая установка, либо обновление со старого
    клиента; в обоих случаях переустанавливаем, чтобы гарантировать, что
    локальный neurocore соответствует тому, что отдаёт зеркало.
    """
    info = api.get_neurocore_info()
    if not info:
        return False
    remote_sha = str(info.get("sha256") or "")
    if not remote_sha:
        return False

    import os
    import subprocess
    from pathlib import Path

    pkg_dir = Path(__file__).resolve().parent
    install_dir = pkg_dir.parent
    hash_file = install_dir / "neurocore_sha256.txt"

    local_sha = None
    if hash_file.exists():
        try:
            local_sha = hash_file.read_text().strip()
        except Exception:
            local_sha = None

    if local_sha == remote_sha:
        return False

    logger.info("neurocore-update: %s → %s, reinstalling…",
                (local_sha or "(none)")[:8], remote_sha[:8])
    url = f"{api.server}/api/client/neurocore.zip"
    cmd = [sys.executable, "-m", "pip", "install",
           "--force-reinstall", "--no-deps", url]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=300)
    except Exception as e:
        logger.warning("neurocore-update pip error: %s", e)
        return False
    if result.returncode != 0:
        logger.warning("neurocore-update pip failed: %s",
                       (result.stderr or "")[-500:])
        return False
    try:
        hash_file.write_text(remote_sha)
    except Exception as e:
        logger.warning("neurocore-update: cannot persist hash: %s", e)
    logger.info("neurocore-update: installed, restarting via execv")

    argv = [sys.executable, "-m", "utopia_client.main"] + sys.argv[1:]
    try:
        if os.name == "nt":
            subprocess.Popen(argv, close_fds=False)
            sys.exit(0)
        else:
            os.execv(sys.executable, argv)
    except Exception as e:
        logger.error("neurocore-update restart failed: %s — exit(75)", e)
        sys.exit(75)
    return True  # недостижимо


def cmd_run(args: argparse.Namespace) -> int:
    """Daemon-петля: опрашивает команду от VPS и реагирует."""
    cfg = _ensure_config()
    api = UtopiaAPI(cfg["server"], cfg["token"])
    name = cfg["name"]
    _write_pid_file()
    _write_heartbeat_file()
    logger.info("starting daemon as colony=%s server=%s", name, cfg["server"])

    # Prefetch wanderer-сида (17 тканей) до подъёма WS. Без этого
    # ws_client отвергает каждый seed_chunk → "deferred (waiting for
    # seed.norg)" и owned-особи никогда не оживают.
    try:
        from .seed_loader import ensure_seed, seed_cached
        if not seed_cached("wanderer"):
            ensure_seed(api, lineage="wanderer")
    except Exception as e:
        logger.warning("wanderer seed prefetch failed: %s", e)

    # F.6.B: WS подключается только в state=run. В idle WS не открыт →
    # P40 видит client_disconnected → freeze_personal (owned-особи живут
    # в Мире, не размножаются). При idle→run hello триггерит unfreeze.
    ws: ColonyWSClient | None = None
    # Фаза 2: WS-подписчик snap'а Мира (broker /ws/feed) + локальный кеш.
    # Поднимается вместе с ws в run, гасится в idle. Кеш переживает обрывы.
    world_cache: WorldStateCache | None = None
    world_feed: WorldFeedClient | None = None
    # Body Migration Этап 3a Phase 1 (27.05.2026): embodied API skeleton.
    # off-by-default через `cfg["embodied_enabled"]`. Поднимается вместе
    # с ws в idle→run, гасится при run→idle / benchmark / restart.
    # Phase 1 — только echo roundtrip latency, без интеграции с биохимией.
    embodied_ws = None  # type: ignore[var-annotated]
    embodied_org = None  # type: ignore[var-annotated]

    # Phase 4 этап F (28.05.2026): local-only reproduction loop state.
    # Gate'нут через `cfg["local_repro_enabled"]` (default False).
    # Активация — координированный rolling restart: Хьюберт ставит
    # WorldConfig.client_owns_reproduction=True на P40 + Шеф deploys
    # client с этим флагом True одновременно. До этого момента detect
    # loop спит — P40 продолжает инициировать mate_request для owned
    # Zodchiy (Phase 4 R1 mitigation — vision §3.2 cross-owner ban).
    last_repro_check_tick = 0
    REPRO_CHECK_INTERVAL_TICKS = 100  # ~3.3 сек @ 30 TPS

    last_heartbeat = 0.0
    last_poll = 0.0
    last_logpush = 0.0
    last_diag_push = 0.0
    last_selfupdate_check = 0.0
    current_state = "idle"
    # SFNN S3.activate: применённый флаг sfnn_higher из flags. None — ещё не
    # видели команду, не трогаем дефолт compute. Сравниваем с поллом.
    applied_sfnn_higher: bool | None = None
    # SFNN S3.activate-motor: то же для motor_policy ветки.
    applied_sfnn_motor: bool | None = None
    # SFNN S6.9: то же для 10 базовых тканей.
    applied_sfnn_basic: bool | None = None
    # Single-organism pivot (ТЗ e3cc81b): применённый флаг single_organism.
    applied_single_organism: bool | None = None
    # Track 2 / направление (б) (Фрай 02.06.2026): применённый флаг insula_temp
    # (insula-стресс → temperature-модуляция). Через client_flags → мгновенный
    # on/off без деплоя: запуск по go Фрая, tripwire-откат при disruption.
    applied_insula_temp: bool | None = None
    # §3.2 (Фрай 09.06.2026): felt-thirst gradual drive. Через client_flags →
    # мгновенный on/off без рестарта; false = kill-switch (откат к бинарному
    # 30% water-seek), backoff если выживание падает.
    applied_felt_thirst_drive: bool | None = None
    # §10.8 (Фрай 09.06.2026): рост ТКАНЕЙ (узлами). client_flags → мгновенный
    # on/off; дефолт OFF (dormant). Растит узел после насыщения связей.
    applied_tissue_growth: bool | None = None
    applied_tissue_graduation: bool | None = None
    applied_behavioral_probe: str | None = None
    applied_behavioral_gc: bool | None = None
    # Ступень 2 (Фрай 03.06.2026): motor renorm growth-cap. Через client_flags
    # (числовой) → мгновенная рекалибровка renorm-супрессора без рестарта.
    applied_motor_renorm_cap: float | None = None
    # Ступень 2 (a): motor Oja-scale (ослабление Oja-стабилизатора для теста).
    applied_motor_oja_scale: float | None = None
    # Инстинкт-развязка (Фрай 03.06): сила food/prey/predator-направления,
    # развязанная от bias_scale (прекондишн навыка). Tune via client_flags.
    applied_instinct_dir_strength: float | None = None
    # Голос мотора (Фрай 03.06 curriculum): множитель motor_delta под
    # single_organism. Фаза 1 убавить / Фаза 2 fade-up (тест SFNN-модулятора).
    applied_motor_voice: float | None = None
    # De-saturation tanh-головы motor (Фрай 04.06): T-override (выбить из залипшего
    # экстремума + держать отзывчивой) + lr_scale (анти-saddle-flip).
    applied_motor_temp: float | None = None
    applied_motor_lr_scale: float | None = None
    # output_proj-specific развязка Oja/renorm (Фрай 04.06, верифиц. dw_radial≈1).
    applied_motor_oja_out: float | None = None
    applied_motor_renorm_cap_out: float | None = None
    # Policy-gradient на output_proj (Фрай 04.06, rule-upgrade): разучить колонию.
    applied_motor_pg: float | None = None
    applied_motor_pg_lr: float | None = None
    # Медленный batch-REINFORCE канал (Фрай 05.06, порт WorldTrainer, MIGRATION GAP).
    applied_motor_slow: float | None = None
    applied_motor_park: float | None = None
    applied_motor_stayf: float | None = None
    applied_damage_factor: float | None = None
    # Reward-баланс forage/hunt (Фрай 04.06): серверный энергобаланс вместо
    # плоских равных +5/+5 (корень бистабильности мотора).
    applied_reward_balance: float | None = None
    applied_reward_weights: tuple | None = None
    # Glucose→energy конверсия (Фрай 04.06 экономика): rate, плотная еда→net-positive.
    applied_glucose_energy_rate: float | None = None
    # Z7.i.b (Zodchiy): последнее значение `lineage_upgrade_pending` из
    # client_flags. Триггер edge-detect: False/None → True вызывает
    # P40 Z7.g endpoint (один раз на rising edge). VPS-flag сейчас не
    # сбрасывается клиентом; чтобы повторить апгрейд, кабинет должен
    # сначала вернуть флаг в false, потом снова в true.
    applied_lineage_upgrade: bool | None = None
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
                    # Remote restart: ack команду (VPS сбросит в idle), погасить
                    # WS, sys.exit(0). systemd-сервис перезапустит процесс,
                    # после старта desired_colony_state будет уже idle.
                    if desired == "restart":
                        logger.info("remote restart requested — acking + exiting")
                        api.ack_command()
                        if ws is not None:
                            ws.stop()
                        if world_feed is not None:
                            world_feed.stop()
                        if embodied_ws is not None:
                            embodied_ws.stop()
                        return 0
                    # Принудительный self-update: ack + проверка _try_self_update
                    # и _try_neurocore_update немедленно (вне ритма 60с). При
                    # успехе execv не вернётся; иначе остаёмся в текущем
                    # current_state (на след. poll увидим уже idle от VPS).
                    if desired == "update_now":
                        logger.info("remote update_now requested — acking + check")
                        api.ack_command()
                        try:
                            if _try_self_update(api):
                                return 0  # недостижимо после execv
                            if _try_neurocore_update(api):
                                return 0  # недостижимо после execv
                            logger.info("update_now: уже на свежей версии")
                        except Exception as e:
                            logger.warning("update_now check error: %s", e)
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
                        if world_feed is not None:
                            world_feed.stop()
                            world_feed = None
                            world_cache = None
                            logger.info("world feed stopped (benchmark→idle)")
                        if embodied_ws is not None:
                            embodied_ws.stop()
                            embodied_ws = None
                            embodied_org = None
                            logger.info("embodied stopped (benchmark→idle)")
                        current_state = "idle"
                    else:
                        # F.6.B: переходы run↔idle переключают WS.
                        if desired == "run" and ws is None:
                            ws = _make_ws(cfg, name)
                            ws.start()
                            logger.info("ws started (idle→run)")
                            # Фаза 2: snap-кеш Мира.
                            world_cache = WorldStateCache(base_url=cfg["server"])
                            world_feed = WorldFeedClient(
                                server=cfg["server"], cache=world_cache)
                            world_feed.start()
                            # Фаза 3.3A obs migration: ws видит кеш для
                            # shadow-сборки obs (валидация vs server obs).
                            ws.world_cache = world_cache
                            logger.info("world feed started (idle→run)")
                            # Body Migration Phase 1: optional embodied WS.
                            # Lazy import чтобы msgpack/websockets не тянулись
                            # на старте при выключенном флаге.
                            if cfg.get("embodied_enabled", False):
                                try:
                                    from .embodied_ws import EmbodiedWSClient
                                    from .embodied import EmbodiedOrganism

                                    # Phase 4 F: dispatcher для msg.type
                                    # routing. Phase 1 echo обработывается
                                    # как раньше (legacy), newborn_announce_ack
                                    # forwards в compute.
                                    def _embodied_dispatcher(msg: dict) -> None:
                                        if not isinstance(msg, dict):
                                            return
                                        mtype = msg.get("type", "")
                                        if mtype == "newborn_announce_ack":
                                            if ws is not None and ws.compute is not None:
                                                try:
                                                    ws.compute.handle_newborn_announce_ack(msg)
                                                except Exception as e:
                                                    logger.warning(
                                                        "newborn_announce_ack handler error: %s", e)
                                        # else: Phase 1 echo / legacy obs —
                                        # latency stats уже считаются в
                                        # _handle_msg перед callback.

                                    embodied_ws = EmbodiedWSClient(
                                        cfg["server"], cfg["token"],
                                        on_observation=_embodied_dispatcher,
                                    )
                                    embodied_ws.start()
                                    if ws.compute is not None:
                                        embodied_org = EmbodiedOrganism(
                                            ws.compute, embodied_ws)
                                    logger.info("embodied started (idle→run)")
                                except Exception as e:
                                    logger.warning(
                                        "embodied init failed: %s", e)
                                    embodied_ws = None
                                    embodied_org = None
                        elif desired == "idle" and ws is not None:
                            ws.stop()  # отправит bye → close
                            ws = None
                            logger.info("ws stopped with bye (run→idle)")
                            if world_feed is not None:
                                world_feed.stop()
                                world_feed = None
                                world_cache = None
                                logger.info("world feed stopped (run→idle)")
                            if embodied_ws is not None:
                                embodied_ws.stop()
                                embodied_ws = None
                                embodied_org = None
                                logger.info("embodied stopped (run→idle)")
                        current_state = desired

                    # SFNN S3.activate: применить sfnn_higher из flags.
                    # Применяем независимо от desired state (флаг может
                    # переключаться когда колония уже в run).
                    flags = cmd.get("flags") or {}
                    target_sfnn = bool(flags.get("sfnn_higher", False))
                    if target_sfnn != applied_sfnn_higher and ws is not None \
                            and ws.compute is not None:
                        try:
                            n = ws.compute.set_higher_sfnn(target_sfnn)
                            applied_sfnn_higher = target_sfnn
                            logger.info("sfnn_higher → %s (%d organisms)",
                                        target_sfnn, n)
                        except Exception as e:
                            logger.warning("set_higher_sfnn failed: %s", e)
                    target_motor = bool(flags.get("sfnn_motor", False))
                    if target_motor != applied_sfnn_motor and ws is not None \
                            and ws.compute is not None:
                        try:
                            n = ws.compute.set_motor_sfnn(target_motor)
                            applied_sfnn_motor = target_motor
                            logger.info("sfnn_motor → %s (%d organisms)",
                                        target_motor, n)
                        except Exception as e:
                            logger.warning("set_motor_sfnn failed: %s", e)
                    target_basic = bool(flags.get("sfnn_basic", False))
                    if target_basic != applied_sfnn_basic and ws is not None \
                            and ws.compute is not None:
                        try:
                            n = ws.compute.set_basic_sfnn(target_basic)
                            applied_sfnn_basic = target_basic
                            logger.info("sfnn_basic → %s (%d organisms)",
                                        target_basic, n)
                        except Exception as e:
                            logger.warning("set_basic_sfnn failed: %s", e)

                    # Single-organism pivot (01.06.2026, ТЗ e3cc81b): флаг
                    # гейтит колониальные механики (репродукция/speciation).
                    # Применяем независимо от desired state — флаг может
                    # переключиться, когда колония уже в run.
                    target_single = bool(flags.get("single_organism", False))
                    if target_single != applied_single_organism \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_single_organism(target_single)
                            applied_single_organism = target_single
                            logger.info("single_organism → %s", target_single)
                        except Exception as e:
                            logger.warning("set_single_organism failed: %s", e)

                    # Track 2 / направление (б): insula-стресс → temperature-
                    # модуляция. Мгновенный on/off без деплоя (запуск по go
                    # Фрая; tripwire-откат при деградации foraging). edge-detect
                    # как single_organism — применяем независимо от desired state.
                    target_insula_temp = bool(flags.get("insula_temp", False))
                    if target_insula_temp != applied_insula_temp \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_insula_temp(target_insula_temp)
                            applied_insula_temp = target_insula_temp
                            logger.info("insula_temp → %s", target_insula_temp)
                        except Exception as e:
                            logger.warning("set_insula_temp failed: %s", e)

                    # §3.2: felt-thirst gradual drive. Мгновенный on/off без
                    # деплоя (запуск по go Фрая; kill-switch=false → бинарный
                    # 30%; backoff при падении выживания). edge-detect.
                    target_felt_thirst = bool(flags.get("felt_thirst_drive", False))
                    if target_felt_thirst != applied_felt_thirst_drive \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_felt_thirst_drive(target_felt_thirst)
                            applied_felt_thirst_drive = target_felt_thirst
                            logger.info("felt_thirst_drive → %s", target_felt_thirst)
                        except Exception as e:
                            logger.warning("set_felt_thirst_drive failed: %s", e)

                    # §10.8: рост ТКАНЕЙ (узлами). Мгновенный on/off; дефолт OFF
                    # (dormant). Растит узел после насыщения связей. edge-detect.
                    target_tissue_growth = bool(flags.get("tissue_growth", False))
                    if target_tissue_growth != applied_tissue_growth \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_tissue_growth(target_tissue_growth)
                            applied_tissue_growth = target_tissue_growth
                            logger.info("tissue_growth → %s", target_tissue_growth)
                        except Exception as e:
                            logger.warning("set_tissue_growth failed: %s", e)

                    # Stage 1 GRADUATION (Фрай 10.06): durable-сайдкар → граф-узел
                    # через §3-контур. Мгновенный on/off; OFF = revert узлов в
                    # сайдкары. Дефолт OFF (dormant). edge-detect.
                    target_tissue_grad = bool(flags.get("tissue_graduation", False))
                    if target_tissue_grad != applied_tissue_graduation \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_tissue_graduation(target_tissue_grad)
                            applied_tissue_graduation = target_tissue_grad
                            logger.info("tissue_graduation → %s", target_tissue_grad)
                        except Exception as e:
                            logger.warning("set_tissue_graduation failed: %s", e)

                    # §10.3 Stage 3 (Фрай go 10.06): behavioral-GC — парная
                    # ревизия graduated-узлов по самочувствию. Дефолт OFF
                    # (dormant). Мгновенный on/off; OFF=abort+restore edge. edge-detect.
                    target_beh_gc = bool(flags.get("behavioral_gc", False))
                    if target_beh_gc != applied_behavioral_gc \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_behavioral_gc(target_beh_gc)
                            applied_behavioral_gc = target_beh_gc
                            logger.info("behavioral_gc → %s", target_beh_gc)
                        except Exception as e:
                            logger.warning("set_behavioral_gc failed: %s", e)

                    # §10.3 Step-1 (Фрай 10.06): behavioral-probe — ablate
                    # graduated-ткань (строковый флаг = роль, ""=снять) для
                    # замера сигнала по измерениям. Мгновенно, обратимо. edge-detect.
                    target_probe = str(flags.get("behavioral_probe", "") or "")
                    if target_probe != applied_behavioral_probe \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_behavioral_probe(target_probe)
                            applied_behavioral_probe = target_probe
                            logger.info("behavioral_probe → %r", target_probe)
                        except Exception as e:
                            logger.warning("set_behavioral_probe failed: %s", e)

                    # Ступень 2 (full-world): motor renorm growth-cap (числовой
                    # флаг). Мгновенная рекалибровка renorm-супрессора без рестарта.
                    target_renorm_cap = float(flags.get("motor_renorm_cap", 1.0))
                    if target_renorm_cap != applied_motor_renorm_cap \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_motor_renorm_cap(target_renorm_cap)
                            applied_motor_renorm_cap = target_renorm_cap
                            logger.info("motor_renorm_cap → %.2f", target_renorm_cap)
                        except Exception as e:
                            logger.warning("set_motor_renorm_cap failed: %s", e)

                    # Ступень 2 (a): motor Oja-scale (числовой флаг). Ослабление
                    # Oja-стабилизатора для теста «свободная магнитуда → flip↓?».
                    target_oja_scale = float(flags.get("motor_oja_scale", 1.0))
                    if target_oja_scale != applied_motor_oja_scale \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_motor_oja_scale(target_oja_scale)
                            applied_motor_oja_scale = target_oja_scale
                            logger.info("motor_oja_scale → %.2f", target_oja_scale)
                        except Exception as e:
                            logger.warning("set_motor_oja_scale failed: %s", e)

                    # Инстинкт-развязка: сила food/prey/predator-направления
                    # (Фрай — прекондишн навыка, развязан от bias_scale).
                    target_instinct = float(flags.get("instinct_dir_strength", 0.0))
                    if target_instinct != applied_instinct_dir_strength \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_instinct_dir_strength(target_instinct)
                            applied_instinct_dir_strength = target_instinct
                            logger.info("instinct_dir_strength → %.2f", target_instinct)
                        except Exception as e:
                            logger.warning("set_instinct_dir_strength failed: %s", e)

                    # Голос мотора (Фрай curriculum): Фаза 1/2 SFNN-модулятор-тест.
                    target_voice = float(flags.get("motor_voice", 1.0))
                    if target_voice != applied_motor_voice \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_motor_voice(target_voice)
                            applied_motor_voice = target_voice
                            logger.info("motor_voice → %.2f", target_voice)
                        except Exception as e:
                            logger.warning("set_motor_voice failed: %s", e)

                    # De-saturation tanh-головы motor (Фрай 04.06): T-override.
                    # >1 делит pre-tanh → залипшие ±0.99 дельты в отзывчивый
                    # диапазон → градиент работает → REINFORCE выбивает из экстремума.
                    target_motor_temp = float(flags.get("motor_temp", 0.0))
                    if target_motor_temp != applied_motor_temp \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_motor_temp(target_motor_temp)
                            applied_motor_temp = target_motor_temp
                            logger.info("motor_temp → %.2f", target_motor_temp)
                        except Exception as e:
                            logger.warning("set_motor_temp failed: %s", e)

                    # Anti-saddle-flip (Фрай 04.06): множитель eta REINFORCE.
                    target_motor_lr = float(flags.get("motor_lr_scale", 1.0))
                    if target_motor_lr != applied_motor_lr_scale \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_motor_lr_scale(target_motor_lr)
                            applied_motor_lr_scale = target_motor_lr
                            logger.info("motor_lr_scale → %.3f", target_motor_lr)
                        except Exception as e:
                            logger.warning("set_motor_lr_scale failed: %s", e)

                    # output_proj-specific Oja-развязка (Фрай 04.06): первично снять
                    # лок (Oja свампит reward-направление на policy-выходе).
                    target_oja_out = float(flags.get("motor_oja_out", 1.0))
                    if target_oja_out != applied_motor_oja_out \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_motor_oja_out(target_oja_out)
                            applied_motor_oja_out = target_oja_out
                            logger.info("motor_oja_out → %.2f", target_oja_out)
                        except Exception as e:
                            logger.warning("set_motor_oja_out failed: %s", e)
                    target_rcap_out = float(flags.get("motor_renorm_cap_out", 1.0))
                    if target_rcap_out != applied_motor_renorm_cap_out \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_motor_renorm_cap_out(target_rcap_out)
                            applied_motor_renorm_cap_out = target_rcap_out
                            logger.info("motor_renorm_cap_out → %.2f", target_rcap_out)
                        except Exception as e:
                            logger.warning("set_motor_renorm_cap_out failed: %s", e)

                    # Policy-gradient на output_proj (Фрай 04.06): разучить
                    # вестигиальную колониальную политику + выучить forage.
                    target_pg = float(flags.get("motor_pg", 0.0))
                    if target_pg != applied_motor_pg \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_motor_pg(target_pg)
                            applied_motor_pg = target_pg
                            logger.info("motor_pg → %.1f", target_pg)
                        except Exception as e:
                            logger.warning("set_motor_pg failed: %s", e)
                    target_pg_lr = float(flags.get("motor_pg_lr", 1.618))
                    if target_pg_lr != applied_motor_pg_lr \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_motor_pg_lr(target_pg_lr)
                            applied_motor_pg_lr = target_pg_lr
                            logger.info("motor_pg_lr → %.3f", target_pg_lr)
                        except Exception as e:
                            logger.warning("set_motor_pg_lr failed: %s", e)

                    # Медленный batch-REINFORCE канал (Фрай 05.06, порт WorldTrainer).
                    target_slow = float(flags.get("motor_slow", 0.0))
                    if target_slow != applied_motor_slow \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_motor_slow(target_slow)
                            applied_motor_slow = target_slow
                            logger.info("motor_slow → %.1f", target_slow)
                        except Exception as e:
                            logger.warning("set_motor_slow failed: %s", e)

                    # Изолирующий тест override-мотора (Фрай 06.06): on-flora →
                    # STAY безусловно (паркуем). Обратимо. Обычно с motor_voice=0.
                    target_park = float(flags.get("motor_park_test", 0.0))
                    if target_park != applied_motor_park \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_motor_park_test(target_park)
                            applied_motor_park = target_park
                            logger.info("motor_park_test → %.1f", target_park)
                        except Exception as e:
                            logger.warning("set_motor_park_test failed: %s", e)

                    # STAY-исполнение контроль (Фрай 06.06): безусловный STAY,
                    # тест honored-ли STAY на P40. Обратимо.
                    target_stayf = float(flags.get("motor_stay_force", 0.0))
                    if target_stayf != applied_motor_stayf \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_motor_stay_force(target_stayf)
                            applied_motor_stayf = target_stayf
                            logger.info("motor_stay_force → %.1f", target_stayf)
                        except Exception as e:
                            logger.warning("set_motor_stay_force failed: %s", e)

                    # DAMAGE-канал калибровка (Фрай 07.06): множитель урона
                    # хищника к energy. Мал→расти. 0=off.
                    target_dmg = float(flags.get("damage_factor", 0.0))
                    if target_dmg != applied_damage_factor \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_damage_factor(target_dmg)
                            applied_damage_factor = target_dmg
                            logger.info("damage_factor → %.3f", target_dmg)
                        except Exception as e:
                            logger.warning("set_damage_factor failed: %s", e)

                    # Reward-баланс forage/hunt (Фрай 04.06): серверный
                    # энергобаланс вместо плоских равных +5/+5 (корень
                    # бистабильности мотора). on + 3 веса.
                    target_rbal = float(flags.get("reward_balance", 0.0))
                    if target_rbal != applied_reward_balance \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_reward_balance(target_rbal)
                            applied_reward_balance = target_rbal
                            logger.info("reward_balance → %.1f", target_rbal)
                        except Exception as e:
                            logger.warning("set_reward_balance failed: %s", e)
                    target_rw = (flags.get("reward_forage_w"),
                                 flags.get("reward_kill_w"),
                                 flags.get("reward_risk_w"))
                    if target_rw != applied_reward_weights \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_reward_weights(*target_rw)
                            applied_reward_weights = target_rw
                            logger.info("reward_weights → %s", target_rw)
                        except Exception as e:
                            logger.warning("set_reward_weights failed: %s", e)

                    # Glucose→energy конверсия (Фрай экономика): плотная еда→viable.
                    target_ger = float(flags.get("glucose_energy_rate", 0.0))
                    if target_ger != applied_glucose_energy_rate \
                            and ws is not None and ws.compute is not None:
                        try:
                            ws.compute.set_glucose_energy_rate(target_ger)
                            applied_glucose_energy_rate = target_ger
                            logger.info("glucose_energy_rate → %.4f", target_ger)
                        except Exception as e:
                            logger.warning("set_glucose_energy_rate failed: %s", e)

                    # Z7.i.b/c (Zodchiy, 16.05.2026): на rising edge флага
                    # lineage_upgrade_pending делаем ДВА действия:
                    #   (Z7.i.b) — дёргаем P40 Z7.g endpoint по LAN.
                    #             Серверный консьюм Z7.f при следующей
                    #             репродукции сделает потомка серверного
                    #             Странника Зодчим. Сейчас почти всегда
                    #             no-op (у Шефа все Странники клиентские).
                    #   (Z7.i.c) — клиентский Genome-флип
                    #             `lineage_upgrade_to_zodchiy=True` на всех
                    #             owned organisms. Pure-Z7.c гейтит по
                    #             lineage="wanderer" → не-Странников
                    #             сбросит без апгрейда.
                    target_upgrade = bool(
                        flags.get("lineage_upgrade_pending", False))
                    if target_upgrade and applied_lineage_upgrade is not True:
                        uid = str(cfg.get("user_id", "") or "").strip()
                        if not uid:
                            logger.warning(
                                "lineage_upgrade_pending=True but user_id "
                                "missing in config — skip P40 call")
                        else:
                            p40_url = str(
                                os.environ.get("UTOPIA_P40_URL")
                                or cfg.get("p40_url")
                                or DEFAULT_P40_URL)
                            resp = api.trigger_p40_lineage_upgrade(p40_url, uid)
                            if resp is not None:
                                logger.info(
                                    "lineage_upgrade_pending → P40 ok: "
                                    "creature=%s energy=%.1f",
                                    resp.get("creature_id"),
                                    float(resp.get("energy", 0.0)))
                            else:
                                logger.info(
                                    "lineage_upgrade_pending → P40 no-op "
                                    "(unreachable / no server wanderer)")
                        # Z7.i.c: client-side Genome-флип независимо от
                        # успеха P40-call'а (у Шефа все Странники
                        # клиентские — это primary path).
                        if ws is not None and ws.compute is not None:
                            try:
                                n_flipped = ws.compute \
                                    .set_lineage_upgrade_pending(True)
                                logger.info(
                                    "lineage_upgrade_pending → client "
                                    "Genome-flip on %d organisms", n_flipped)
                            except Exception as e:
                                logger.warning(
                                    "set_lineage_upgrade_pending(True) "
                                    "failed: %s", e)
                        applied_lineage_upgrade = True
                    elif not target_upgrade and applied_lineage_upgrade:
                        # Falling edge: кабинет очистил флаг, сбрасываем
                        # память, чтобы следующий True → снова триггернул.
                        # Заодно гасим клиентский Genome-флаг — если pure-Z7.c
                        # не успел consume (нет репродукции между rising
                        # и falling edge), флаг должен исчезнуть.
                        if ws is not None and ws.compute is not None:
                            try:
                                ws.compute.set_lineage_upgrade_pending(False)
                            except Exception as e:
                                logger.warning(
                                    "set_lineage_upgrade_pending(False) "
                                    "failed: %s", e)
                        applied_lineage_upgrade = False

            # Body Migration Phase 1: lazy-attach + emit embodied/state.
            # Fix 27.05.2026: при idle→run `ws.compute` ещё None (compute
            # создаётся внутри ColonyWSClient после seed_complete от P40).
            # Поэтому attach делаем здесь — когда compute уже доступен.
            # throttle внутри emit_alive_owned гарантирует period_sec.
            if (embodied_ws is not None and embodied_org is None
                    and ws is not None and ws.compute is not None):
                try:
                    from .embodied import EmbodiedOrganism
                    embodied_org = EmbodiedOrganism(ws.compute, embodied_ws)
                    logger.info("embodied organism attached to compute "
                                "(n_organisms=%d)", len(ws.compute.organisms))
                except Exception as e:
                    logger.warning("embodied attach failed: %s", e)
            if embodied_org is not None and ws is not None:
                try:
                    embodied_org.emit_alive_owned(
                        world_tick=ws.last_world_tick)
                except Exception as e:
                    logger.debug("embodied emit error: %s", e)

            # Phase 4 этап F (28.05.2026): local-only reproduction loop.
            # Periodic detect mate-pairs среди own colony → emit
            # newborn_announce → ack handler в _embodied_dispatcher
            # выше.
            #
            # Gate'нут через `local_repro_enabled` (default False).
            # ВКЛЮЧАЕТСЯ только синхронно с P40-side toggle
            # `WorldConfig.client_owns_reproduction=True` — иначе
            # дублирование (P40 шлёт mate_request И client детектит
            # pair → два newborn). См. R1 в design doc
            # docs/phase4_local_reproduction_flow.md.
            if (cfg.get("local_repro_enabled", False)
                    and ws is not None and ws.compute is not None
                    and embodied_ws is not None and embodied_ws.connected
                    and ws.last_world_tick - last_repro_check_tick
                        >= REPRO_CHECK_INTERVAL_TICKS):
                try:
                    born = ws.compute.detect_and_emit_mate_pairs(
                        world_tick=int(ws.last_world_tick),
                        embodied_client=embodied_ws,
                    )
                    if born:
                        logger.info(
                            "local repro: emitted %d newborns at tick %d",
                            len(born), ws.last_world_tick)
                    last_repro_check_tick = int(ws.last_world_tick)
                except Exception as e:
                    logger.warning("local repro detect failed: %s", e)
                    last_repro_check_tick = int(ws.last_world_tick)

            # Heartbeat
            if now - last_heartbeat >= HEARTBEAT_SEC:
                n_alive = ws.n_alive_owned if ws is not None else 0
                world_tick = ws.last_world_tick if ws is not None else 0
                ws_connected = ws.connected if ws is not None else False
                _summary = None
                _cstats = None
                if ws is not None and ws.compute is not None:
                    try:
                        _summary = ws.compute.build_colony_summary()
                        _cstats = ws.compute.build_creature_stats()
                    except Exception as e:
                        logger.debug("build colony/creature stats failed: %s", e)
                ok = _heartbeat(api, name, current_state, world_tick, bench,
                                n_alive, _summary, _cstats)
                feed_snaps = world_feed.snapshots_received if world_feed else 0
                feed_conn = world_feed.connected if world_feed else False
                logger.info(
                    "heartbeat world_tick=%d state=%s ws=%s n_alive=%d "
                    "feed=%s snaps=%d ok=%s",
                    world_tick, current_state, ws_connected, n_alive,
                    feed_conn, feed_snaps, ok)
                _write_heartbeat_file()
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
                    # 16.05.2026: uptime процесса клиента (сек) для UI.
                    diag["client_uptime_sec"] = int(
                        time.monotonic() - _PROCESS_STARTED_AT)
                    # Фаза 3.3A: метрики client-built obs vs server obs.
                    shadow = {
                        "built": ws._client_obs_built,
                        "skipped": ws._client_obs_skipped,
                        "match": ws._client_obs_match,
                        "mismatch": ws._client_obs_mismatch,
                        "max_diff": ws._client_obs_max_diff,
                        "last_worst": dict(ws._client_obs_last_worst or {}),
                        # Phase 3.3 fix2: tick desync — основная причина
                        # большинства skip'ов в live. Помогает отличить
                        # «cache не готов» от «cache ушёл вперёд obs_batch».
                        "last_tick_skip": dict(
                            getattr(ws, "_client_obs_last_tick_skip", {}) or {}),
                        # Phase 3.3B: сколько раз клиент построил obs локально
                        # (когда сервер не прислал obs для owned).
                        "local_built": getattr(
                            ws, "_client_obs_local_built", 0),
                    }
                    # Размеры кеша мира: помогает понять, есть ли вообще
                    # flora/fauna/signals у клиента (если 0 — apply_snap не
                    # дошёл до полного кадра).
                    cache = getattr(ws, "world_cache", None)
                    if cache is not None:
                        try:
                            shadow["cache"] = {
                                "flora": len(cache.flora),
                                "fauna": len(cache.fauna),
                                "signals": len(cache.signals),
                                "creature_pos": len(cache.creature_pos),
                                "meta": len(cache.creature_meta),
                                "last_tick": cache.last_tick,
                                "snaps_applied": cache.snaps_applied,
                                "full_frames": cache.full_frames_received,
                            }
                        except Exception:
                            pass
                    # Топ-3 слота по среднему |diff|. Помогает понять, где
                    # client_obs расходится с server_obs (terrain? signals?
                    # ally_positions?). Без этого 100% mismatch неотличим от
                    # «одно поле ошиблось на всём obs».
                    slot_sum = ws._client_obs_slot_diff_sum
                    built = max(1, ws._client_obs_built)
                    if slot_sum is not None:
                        try:
                            mean = slot_sum / float(built)
                            order = mean.argsort()[::-1][:3]
                            shadow["top_slots"] = [
                                {"slot": int(i), "mean_diff": float(mean[i])}
                                for i in order
                            ]
                        except Exception:
                            pass
                    diag["shadow_obs"] = shadow
                    # Variant B leak fix (19.05.2026): метрики GC ghost'ов.
                    # tracked — сколько cid под наблюдением, total_removed —
                    # сколько уже выкосили. Если total_removed растёт но
                    # n_alive не падает — GC работает, утечка реальна.
                    diag["cid_gc"] = {
                        "total_removed": getattr(ws, "_cid_gc_total", 0),
                        "tracked": len(
                            getattr(ws, "_cid_last_seen_tick", {}) or {}),
                        "last_run_tick": getattr(
                            ws, "_cid_gc_last_run_tick", 0),
                    }
                    # 0.10.9 (21.05.2026): orphan-obs backstop observability.
                    # streak — текущая последовательность obs_batch'ей без
                    # совпадения, respawns_sent — сколько раз пришлось дёрнуть
                    # P40 за seed_pack.
                    diag["orphan_obs"] = {
                        "streak": getattr(ws, "_orphan_obs_streak", 0),
                        "respawns_sent": getattr(
                            ws, "_orphan_obs_respawns_sent", 0),
                    }
                    # Body Migration Phase 1: stats embodied клиента
                    # (connected, states_sent, observations_received,
                    # latency mean/p50/p95). При выключенном флаге блока нет.
                    if embodied_org is not None:
                        try:
                            diag["embodied"] = embodied_org.stats()
                        except Exception as e:
                            logger.debug("embodied stats error: %s", e)
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
                    # Neurocore-зависимость обновляется отдельно от клиента —
                    # при изменении sha256 на зеркале переустанавливаем пакет.
                    if _try_neurocore_update(api):
                        return 0  # недостижимо после execv
                except Exception as e:
                    logger.warning("self-update check error: %s", e)

            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("interrupted, stopping…")
    finally:
        if ws is not None:
            ws.stop()
        if world_feed is not None:
            world_feed.stop()
        if embodied_ws is not None:
            embodied_ws.stop()
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
    p_gm = sub.add_parser("set-genesis-mode",
                          help="Сменить режим создания колонии (donor/fresh)")
    p_gm.add_argument("mode", choices=["donor", "fresh"],
                      help="donor — старт от обученных доноров; "
                           "fresh — с чистого листа")
    p_tag = sub.add_parser("tag-adam",
                           help="Single-organism: пометить cid как Адама на P40")
    p_tag.add_argument("cid", help="cid спроецированного owned-zodchiy")
    args = p.parse_args(argv)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "benchmark":
        return cmd_benchmark(args)
    if args.cmd == "config":
        return cmd_config(args)
    if args.cmd == "set-token":
        return cmd_set_token(args)
    if args.cmd == "set-genesis-mode":
        return cmd_set_genesis_mode(args)
    if args.cmd == "tag-adam":
        return cmd_tag_adam(args)
    if args.cmd == "bench-gpu":
        from .gpu_bench import cmd_bench_gpu
        return cmd_bench_gpu(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
