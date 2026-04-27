"""Phase F3.3.b.2: половой кроссинговер на клиенте (Variant B).

Локальная копия `_apply_crossover_inheritance` из P40 (server/routes_world.py).
Используется при обработке `mate_request` — мать собирает ребёнка из своего
организма и присланных весов отца.

Контракт идентичен P40-версии: per-tissue 50/50 + Y50-шум на weight-параметрах,
bias/1D копируются 1:1, асимметричные пресеты обрабатываются graceful.
"""
from __future__ import annotations

import logging
import random
from typing import Optional

import torch

logger = logging.getLogger(__name__)

NOISE_SCALE_DEFAULT = 0.0902  # 1/φ⁵ — тот же масштаб что у Y50.


def apply_crossover_inheritance(child_org, mother_org, father_org,
                                noise_scale: float = NOISE_SCALE_DEFAULT,
                                sigma_scale: float = 1.0,
                                rng: Optional[random.Random] = None) -> dict:
    """Per-tissue 50/50: монетка → мать или отец, далее Y50-шум на weight.

    Если ткань есть только у одного родителя — берём от того, у кого есть.
    Bias и 1D-параметры копируются 1:1.

    Возвращает dict {tissue_id → "mother"|"father"} для метрики/лога.
    """
    rng = rng or random
    effective_scale = noise_scale * sigma_scale
    chosen: dict[str, str] = {}

    def _tissues(org):
        return getattr(org, "tissues", None) or {}

    m_tissues = _tissues(mother_org)
    f_tissues = _tissues(father_org)
    c_tissues = _tissues(child_org)

    with torch.no_grad():
        for tid, c_tissue in c_tissues.items():
            m_t = m_tissues.get(tid)
            f_t = f_tissues.get(tid)
            if m_t is None and f_t is None:
                continue
            if m_t is None:
                src, src_label = f_t, "father"
            elif f_t is None:
                src, src_label = m_t, "mother"
            else:
                if rng.random() < 0.5:
                    src, src_label = m_t, "mother"
                else:
                    src, src_label = f_t, "father"
            chosen[tid] = src_label

            src_cells = getattr(src, "cells", {})
            child_cells = getattr(c_tissue, "cells", {})
            for cid, src_cell in src_cells.items():
                child_cell = child_cells.get(cid)
                if child_cell is None:
                    continue
                src_params = dict(src_cell.named_parameters())
                for pname, cp in child_cell.named_parameters():
                    pp = src_params.get(pname)
                    if pp is None or pp.shape != cp.shape:
                        continue
                    if pp.dim() >= 2 and "weight" in pname:
                        std = max(float(pp.data.std().item()), 1e-6)
                        noise = torch.randn_like(pp.data) * effective_scale * std
                        cp.data.copy_(0.5 * pp.data + 0.5 * noise)
                    else:
                        cp.data.copy_(pp.data)
    return chosen


def serialize_organism_blob(org) -> bytes:
    """Сериализовать tissues_state_dict в формате, который P40 ждёт от newborn.

    Формат тот же что и у `LocalColonyCompute.save_state` — словарь с ключом
    `tissues_state_dict`. Hebbian/selector ребёнка не передаём — они будут
    созданы заново на сервере (`_make_hebbian` от offspring CreatureState).
    """
    import io
    payload = {
        "tissues_state_dict": {
            tid: t.state_dict() for tid, t in org.tissues.items()
        }
    }
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()
