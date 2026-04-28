"""Загрузка стартового seed.norg с VPS (Phase F3.4).

Все клиенты получают **одинаковый** стартовый организм с `/api/seed`
Утопии. Локально кешируется в ~/.utopia-client/seed.norg.

Использование:
    from utopia_client.seed_loader import ensure_seed, load_founders
    path = ensure_seed(api)            # скачать если нет
    organisms = load_founders(path, n=5)  # 5 одинаковых founder
"""

from __future__ import annotations

import hashlib
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


def seed_sha256() -> str:
    """SHA256 локального seed.norg (hex). Пустая строка если файла нет."""
    if not seed_cached():
        return ""
    try:
        return hashlib.sha256(_DEFAULT_SEED_PATH.read_bytes()).hexdigest()
    except Exception as e:
        logger.warning("seed_sha256 failed: %s", e)
        return ""


def write_seed_bytes(data: bytes) -> Path:
    """Атомарно записать сырые байты в локальный seed.norg.

    Используется WS-каналом seed_norg_complete. Запись через временный файл
    + os.replace, чтобы не оставить наполовину-записанный seed при крэше.
    """
    path = _DEFAULT_SEED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return path


# Phase F3.2.c: локальный кеш Hebbian-state особей колонии.
def colony_state_dir(colony_name: str) -> Path:
    """Каталог локальных state-файлов особей колонии (Hebbian + tissues + selector).

    Override через env `UTOPIA_COLONIES_DIR`. Имя колонии = subdir.
    """
    base = Path(os.getenv("UTOPIA_COLONIES_DIR",
                          str(Path.home() / ".utopia-client" / "colonies")))
    return base / colony_name


def creature_state_path(colony_name: str, cid: str) -> Path:
    return colony_state_dir(colony_name) / f"{cid}.pt"


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


def organism_from_weights(weights_bytes: bytes, seed_path: Path):
    """Phase F3.1.b/d: создать CompositeOrganism с обученными весами от P40.

    Скелет (архитектура tissues) — из локального seed.norg. Веса tissues —
    накатываются из переданных bytes (формат `_save_member_pt` на P40:
    `{"tissues_state_dict": {tid: state_dict}, "hebbian": ..., "selector": ...,
    "predictor": ..., ...}`).

    Возвращает (organism, payload_dict). Payload отдаётся клиенту, чтобы
    отдельно накатить hebbian/selector/predictor через
    LocalColonyCompute.apply_inherited_state.

    При несовпадении tid между seed-архитектурой и payload — соответствующий
    tissue остаётся со seed-весами (warning).
    """
    import io
    import torch

    payload = torch.load(io.BytesIO(weights_bytes), map_location="cpu",
                         weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"weights payload must be dict, got {type(payload)}")

    org = load_founders(seed_path, 1)[0]
    if "tissues_state_dict" not in payload:
        raise ValueError("payload missing 'tissues_state_dict'")
    tsd = payload["tissues_state_dict"]
    if not isinstance(tsd, dict) or not tsd:
        raise ValueError("tissues_state_dict must be non-empty dict")

    for tid, sd in tsd.items():
        if tid not in org.tissues:
            logger.warning("organism_from_weights: tid=%s not in seed (skip)", tid)
            continue
        try:
            org.tissues[tid].load_state_dict(sd)
        except Exception as e:
            logger.warning("organism_from_weights: tid=%s load failed: %s", tid, e)

    return org, payload
