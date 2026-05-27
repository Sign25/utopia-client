"""Phase 4 Z2/Z3: episodic memory persistence через client restart.

ТЗ Body Migration v1.3 §4 Phase 4 acceptance #2:
  > Episodic memory persistence через client restart

Episodic — это higher tissue (data_dim=64, τ=233, long_ema=0.6) живущая
в `LocalColonyCompute.episodic[cid]`. Создаётся в `add_creature` для
lineage="zodchiy" и теряется при restart (in-memory only).

Этот модуль — **pure helpers** для save/load на диск. Wired в
`local_compute.add_creature` (load при наличии файла) и
`local_compute.remove_creature` (save если организм жив).

**Layout:**
  $config_dir/colonies/memory/<cid>.pt — torch.save({
      "version": 1,
      "cid": str,
      "episodic_state_dict": dict,
      "last_episodic_recall": Tensor | None,
      "ts_saved": float,
  })

**Determinism:** torch.save / torch.load — bit-exact. Load с
`weights_only=True` (safety от ACE).
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("utopia_client.memory_store")

MEMORY_FORMAT_VERSION = 1


def memory_dir(base: Optional[Path] = None) -> Path:
    """Папка где лежат `<cid>.pt` файлы episodic memory.

    Default — `config_dir() / colonies / memory`. `base` параметр для
    тестов (override через monkeypatch).
    """
    if base is None:
        from .config import colonies_dir
        base = colonies_dir()
    d = Path(base) / "memory"
    d.mkdir(parents=True, exist_ok=True)
    return d


def memory_file(cid: str, base: Optional[Path] = None) -> Path:
    """Путь к файлу для конкретного cid."""
    # Защита от path-injection (cid обычно uuid, но всё же)
    safe = "".join(c for c in str(cid) if c.isalnum() or c in "-_")
    if not safe:
        raise ValueError(f"invalid cid for memory file: {cid!r}")
    return memory_dir(base) / f"{safe}.pt"


def save_memory_state(
    cid: str,
    episodic_tissue: Any,
    last_episodic_recall: Any = None,
    *,
    base: Optional[Path] = None,
) -> Optional[Path]:
    """Сохранить episodic state одного организма.

    Args:
        cid: creature_id
        episodic_tissue: ткань `episodic` (тот же объект что
            `LocalColonyCompute.episodic[cid]`). Если None — save skipped.
        last_episodic_recall: последний recall vector (Tensor [64]) или None.
        base: override директории для тестов.

    Returns:
        Path к записанному файлу или None если skipped (нет ткани).

    Errors:
        Не пробрасывает torch.save исключения наружу — logger.warning и
        возвращает None. Persistence не должна валить runtime.
    """
    if episodic_tissue is None:
        return None
    try:
        import torch
    except ImportError:
        logger.warning("save_memory_state: torch not available, skipping cid=%s",
                       cid)
        return None
    try:
        state = episodic_tissue.state_dict()
    except Exception as e:
        logger.warning("save_memory_state %s: state_dict failed: %s", cid, e)
        return None

    recall_cpu = None
    if last_episodic_recall is not None:
        try:
            recall_cpu = last_episodic_recall.detach().cpu()
        except Exception:
            recall_cpu = None

    payload = {
        "version": MEMORY_FORMAT_VERSION,
        "cid": str(cid),
        "episodic_state_dict": state,
        "last_episodic_recall": recall_cpu,
        "ts_saved": time.time(),
    }
    path = memory_file(cid, base=base)
    try:
        torch.save(payload, path)
        logger.debug("save_memory_state %s → %s", cid, path)
        return path
    except Exception as e:
        logger.warning("save_memory_state %s torch.save failed: %s", cid, e)
        return None


def load_memory_state(
    cid: str,
    *,
    base: Optional[Path] = None,
) -> Optional[dict]:
    """Загрузить episodic state для cid. None если файла нет или ошибка.

    Returns dict с ключами `episodic_state_dict`, `last_episodic_recall`,
    `version`, `cid`, `ts_saved`. weights_only=True для safety.
    """
    path = memory_file(cid, base=base)
    if not path.exists():
        return None
    try:
        import torch
    except ImportError:
        logger.warning("load_memory_state: torch not available, cid=%s", cid)
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        # weights_only=True может отказать на старых форматах — лог и skip
        logger.warning("load_memory_state %s read failed (%s): %s",
                       cid, path, e)
        return None

    # Sanity: version check
    ver = payload.get("version") if isinstance(payload, dict) else None
    if ver != MEMORY_FORMAT_VERSION:
        logger.warning("load_memory_state %s: version mismatch %s != %s — skip",
                       cid, ver, MEMORY_FORMAT_VERSION)
        return None
    return payload


def apply_memory_state_to_tissue(
    payload: dict,
    episodic_tissue: Any,
) -> bool:
    """Применить загруженный state к episodic ткани. Returns True если OK."""
    if not isinstance(payload, dict) or episodic_tissue is None:
        return False
    state = payload.get("episodic_state_dict")
    if not isinstance(state, dict):
        return False
    try:
        episodic_tissue.load_state_dict(state)
        return True
    except Exception as e:
        logger.warning("apply_memory_state load_state_dict failed: %s", e)
        return False


def delete_memory_file(cid: str, *, base: Optional[Path] = None) -> bool:
    """Удалить файл памяти (для cleanup мёртвых организмов)."""
    path = memory_file(cid, base=base)
    if path.exists():
        try:
            path.unlink()
            return True
        except Exception as e:
            logger.warning("delete_memory_file %s failed: %s", cid, e)
    return False


def list_saved_cids(*, base: Optional[Path] = None) -> list[str]:
    """Список всех cid'ов для которых есть saved memory (для diag)."""
    d = memory_dir(base)
    return sorted(p.stem for p in d.glob("*.pt") if p.is_file())
