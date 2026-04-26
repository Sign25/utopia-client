"""Загрузка стартового seed.norg с VPS (Phase F3.4).

Все клиенты получают **одинаковый** стартовый организм с `/api/seed`
Утопии. Локально кешируется в ~/.utopia-client/seed.norg.

Использование:
    from utopia_client.seed_loader import ensure_seed, load_founders
    path = ensure_seed(api)            # скачать если нет
    organisms = load_founders(path, n=5)  # 5 одинаковых founder
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .api import UtopiaAPI

logger = logging.getLogger("utopia_client.seed")

# Кеш seed.norg рядом с конфигом клиента.
_DEFAULT_SEED_PATH = Path(
    os.getenv("UTOPIA_SEED_PATH", str(Path.home() / ".utopia-client" / "seed.norg"))
)


def seed_cache_path() -> Path:
    return _DEFAULT_SEED_PATH


def seed_cached() -> bool:
    return _DEFAULT_SEED_PATH.exists() and _DEFAULT_SEED_PATH.stat().st_size > 0


def ensure_seed(api: "UtopiaAPI", *, force: bool = False) -> Optional[Path]:
    """Скачать seed.norg в локальный кеш (если ещё нет / force=True).

    Возвращает путь к локальному файлу, либо None если скачать не удалось.
    """
    path = _DEFAULT_SEED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    if seed_cached() and not force:
        logger.info("seed cached at %s (%d bytes)", path, path.stat().st_size)
        return path
    logger.info("fetching seed.norg from VPS → %s", path)
    if not api.fetch_seed(str(path)):
        logger.warning("seed fetch failed")
        return None
    logger.info("seed downloaded: %d bytes", path.stat().st_size)
    return path


def load_founders(seed_path: Path, n: int):
    """Распаковать seed в N независимых founder с уникальными id, идентичными
    весами (deepcopy). Импорт neurocore — ленивый, чтобы CLI без compute-ядра
    (например, только benchmark) не требовал neurocore."""
    from storage.norg import load_norg, unpack_composite

    data = load_norg(seed_path)
    founders = []
    for _ in range(max(1, n)):
        org, _meta = unpack_composite(data)
        org.id = f"seed_{uuid.uuid4().hex[:12]}"
        founders.append(org)
    return founders
