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
# Два варианта: elder (10 тканей nexus) и wanderer (17 тканей).
_DEFAULT_SEED_PATH = Path(
    os.getenv("UTOPIA_SEED_PATH", str(Path.home() / ".utopia-client" / "seed.norg"))
)
_WANDERER_SEED_PATH = Path(
    os.getenv("UTOPIA_WANDERER_SEED_PATH",
              str(Path.home() / ".utopia-client" / "wanderer.norg"))
)


def _path_for(lineage: str) -> Path:
    """Кеш-путь по lineage. 'wanderer' → wanderer.norg, иначе seed.norg."""
    return _WANDERER_SEED_PATH if lineage == "wanderer" else _DEFAULT_SEED_PATH


def seed_cache_path(lineage: str = "elder") -> Path:
    return _path_for(lineage)


def seed_cached(lineage: str = "elder") -> bool:
    p = _path_for(lineage)
    return p.exists() and p.stat().st_size > 0


def seed_sha256(lineage: str = "elder") -> str:
    """SHA256 локального seed-файла (hex). Пустая строка если файла нет."""
    if not seed_cached(lineage):
        return ""
    try:
        return hashlib.sha256(_path_for(lineage).read_bytes()).hexdigest()
    except Exception as e:
        logger.warning("seed_sha256(%s) failed: %s", lineage, e)
        return ""


def write_seed_bytes(data: bytes, lineage: str = "elder") -> Path:
    """Атомарно записать сырые байты в локальный seed-файл по lineage.

    Используется WS-каналом seed_norg_complete. Запись через временный файл
    + os.replace, чтобы не оставить наполовину-записанный seed при крэше.
    """
    path = _path_for(lineage)
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


def ensure_seed(api: "UtopiaAPI", *, force: bool = False,
                lineage: str = "elder") -> Optional[Path]:
    """Скачать seed-файл в локальный кеш (если ещё нет / force=True).

    `lineage`:
      - "elder" (default) → /api/seed (world.norg, 10 тканей nexus).
      - "wanderer"        → /api/seed/wanderer (17 тканей).

    Возвращает путь к локальному файлу, либо None если скачать не удалось.
    """
    path = _path_for(lineage)
    path.parent.mkdir(parents=True, exist_ok=True)
    if seed_cached(lineage) and not force:
        logger.info("seed[%s] cached at %s (%d bytes)",
                    lineage, path, path.stat().st_size)
        return path
    logger.info("fetching seed[%s] from VPS → %s", lineage, path)
    if lineage == "wanderer":
        ok = api.fetch_wanderer_seed(str(path))
    else:
        ok = api.fetch_seed(str(path))
    if not ok:
        logger.warning("seed[%s] fetch failed", lineage)
        return None
    logger.info("seed[%s] downloaded: %d bytes",
                lineage, path.stat().st_size)
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

    # Новый формат (P40 ≥ 30.04.2026): tissues_by_role — стабильный ключ.
    # Legacy: tissues_state_dict (tid-keyed) — теряется при rebuild seed.
    if "tissues_by_role" in payload:
        tbr = payload["tissues_by_role"]
        if not isinstance(tbr, dict) or not tbr:
            raise ValueError("tissues_by_role must be non-empty dict")
        role_to_tid = {
            getattr(t, "role", ""): tid for tid, t in org.tissues.items()
        }
        for role, sd in tbr.items():
            tid = role_to_tid.get(role)
            if tid is None:
                logger.warning(
                    "organism_from_weights: role=%s not in seed (skip)", role)
                continue
            try:
                org.tissues[tid].load_state_dict(sd)
            except Exception as e:
                logger.warning(
                    "organism_from_weights: role=%s load failed: %s", role, e)
    elif "tissues_state_dict" in payload:
        tsd = payload["tissues_state_dict"]
        if not isinstance(tsd, dict) or not tsd:
            raise ValueError("tissues_state_dict must be non-empty dict")
        for tid, sd in tsd.items():
            if tid not in org.tissues:
                logger.warning(
                    "organism_from_weights: tid=%s not in seed (skip)", tid)
                continue
            try:
                org.tissues[tid].load_state_dict(sd)
            except Exception as e:
                logger.warning(
                    "organism_from_weights: tid=%s load failed: %s", tid, e)
    else:
        raise ValueError(
            "payload missing 'tissues_by_role' or 'tissues_state_dict'")

    return org, payload
