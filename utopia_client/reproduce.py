"""Reproduce-гибрид (Phase F3.5 + Phase 4 Z2 NEAT) — клиентская сторона.

Когда LocalColonyCompute видит, что организм A выбрал REPRODUCE-action,
вызывается `build_reproduce_envelope(A_cid, A.organism)`:
  1. снять state_dict тканей родителя
  2. наложить gaussian-шум σ=Y50 (1/φ⁵≈0.0902) — point mutations (L1)
  3. **Phase 4:** опционально мутировать `tissue_topology_genes` через
     `mutate_topology_genes` — Z2 NEAT мутации межтканевого графа (L3t)
  4. сжать `torch.save(...)` через zstd, упаковать в base64
  5. envelope `{type: "reproduce", parent_cid, child_weights_b64}`

P40 валидирует, наследует физические traits через `_inherit_traits` и
регистрирует newborn в Мире → отвечает `newborn_ack {parent_cid, child_cid}`.
Клиент по ack добавляет новый организм в LocalColonyCompute с теми же
мутированными весами.

Архитектурные мутации (n_embd Fibonacci, ablation_mask flip) — на стороне P40,
требуют world-context (clan size, niche carnivore quota).

**Phase 4 (27.05.2026):** NEAT межтканевые мутации (add/remove/change_type/
weight rebges) переезжают на клиент через re-use `core.tissue_topology` из
neurocore[client] package — bit-exact identical с server-side выполнением.
Schema envelope расширяется опциональным `tissue_topology_genes` полем
(P40 уже принимает его в register_newborn).
"""

from __future__ import annotations

import base64
import io
import logging
import random
from typing import Iterable, Optional

logger = logging.getLogger("utopia_client.reproduce")

# Y50 σ = 1/φ⁵ ≈ 0.0902 — стандартное отклонение наследования весов.
_PHI = (1 + 5 ** 0.5) / 2
DEFAULT_SIGMA = _PHI ** -5  # ≈ 0.09017

# Phase 4 NEAT topology mutation rates (docs/zodchiy.md §1.4 = server defaults).
DEFAULT_P_TOPO_ADD: float = 0.02
DEFAULT_P_TOPO_REMOVE: float = 0.01
DEFAULT_P_TOPO_CHANGE_TYPE: float = 0.005
DEFAULT_P_TOPO_WEIGHT: float = 0.05
DEFAULT_TOPO_WEIGHT_SIGMA: float = 0.1


def _extract_tissues_state_dict(organism) -> dict:
    """{tid: state_dict} всех тканей CompositeOrganism. Legacy."""
    return {tid: t.state_dict() for tid, t in organism.tissues.items()}


def _extract_tissues_by_role(organism) -> dict:
    """{role: state_dict} — стабильный ключ между сессиями (P40 ≥ 30.04.2026).

    tissue_id = uuid.uuid4()[:8] нестабилен после rebuild seed, поэтому
    P40 теперь ожидает role-keyed формат (sensory, motor, brain, ...).
    """
    out: dict = {}
    for tid, t in organism.tissues.items():
        role = getattr(t, "role", "") or f"_unknown_{tid}"
        out[role] = t.state_dict()
    return out


def mutate_state_dict(tissues_sd: dict, *, sigma: float = DEFAULT_SIGMA,
                      generator=None) -> dict:
    """Y50: child_W = parent_W + σ · std(parent_W) · noise."""
    import torch

    out: dict = {}
    for tid, sd in tissues_sd.items():
        new_sd: dict = {}
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                std = float(v.detach().float().std().item())
                if std > 0:
                    noise = torch.randn(v.shape, generator=generator,
                                        device=v.device, dtype=v.dtype)
                    new_sd[k] = v + sigma * std * noise
                else:
                    new_sd[k] = v.clone()
            else:
                new_sd[k] = v.clone() if isinstance(v, torch.Tensor) else v
        out[tid] = new_sd
    return out


def pack_zstd_b64(payload: dict) -> str:
    """torch.save → zstd → base64 (UTF-safe строка для JSON-envelope)."""
    import torch
    import zstandard as zstd

    buf = io.BytesIO()
    torch.save(payload, buf)
    raw = buf.getvalue()
    cctx = zstd.ZstdCompressor(level=3)
    compressed = cctx.compress(raw)
    return base64.b64encode(compressed).decode("ascii")


def unpack_zstd_b64(b64: str):
    """base64 → zstd → torch.load(payload).

    weights_only=True — payload содержит только tensors/dicts/strs (state_dict),
    защита от ACE если канал к P40 будет скомпрометирован.
    """
    import torch
    import zstandard as zstd

    compressed = base64.b64decode(b64)
    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(compressed)
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def build_reproduce_envelope(parent_cid: str, organism, *,
                              sigma: float = DEFAULT_SIGMA,
                              brain_state_dicts: dict | None = None,
                              brain_emas: dict | None = None) -> dict:
    """Собрать envelope для отправки на P40.

    Возвращает dict готовый к json.dumps:
        {type: "reproduce", parent_cid, child_weights_b64, sigma}

    `child_weights_b64` — zstd-base64 от
        {"tissues_by_role": {role: state_dict},
         "brain": {predictor|dopamine|imagination|planner|insula: state_dict},
         "brain_emas": {predictor_loss_ema|intrinsic_ema|...: float}}

    Brain migration Etap 3.2 (11.05.2026): мозг (predictor + S2.E/G/A/F)
    мутируется на клиенте (Y50 σ=1/φ⁵) и отправляется на P40. Сервер
    `register_newborn` грузит готовые мутированные веса в ColonyMember
    без дополнительной Y50 (она уже применена здесь). Если
    `brain_state_dicts is None` — envelope шлёт только тело (legacy).
    """
    parent_sd = _extract_tissues_by_role(organism)
    child_sd = mutate_state_dict(parent_sd, sigma=sigma)
    payload: dict = {"tissues_by_role": child_sd}
    if brain_state_dicts:
        # Y50: те же правила что для тела, но per-tissue ключи
        # (predictor, dopamine, imagination, planner, insula).
        brain_mut = mutate_state_dict(brain_state_dicts, sigma=sigma)
        payload["brain"] = brain_mut
    if brain_emas:
        # EMA-агрегаты — float, не мутируются. Стартовый baseline для ребёнка.
        payload["brain_emas"] = {str(k): float(v) for k, v in brain_emas.items()}
    return {
        "type": "reproduce",
        "parent_cid": str(parent_cid),
        "child_weights_b64": pack_zstd_b64(payload),
        "sigma": float(sigma),
    }


def apply_state_dict(organism, tissues_sd: dict) -> int:
    """Загрузить state_dict в ткани organism. Поддерживает оба формата:
    role-keyed (новый, P40 ≥ 30.04.2026) и tid-keyed (legacy).
    """
    role_to_tid = {
        getattr(t, "role", ""): tid for tid, t in organism.tissues.items()
    }
    n = 0
    for key, sd in tissues_sd.items():
        # Сначала пробуем role-key, затем legacy tid-key.
        tid = role_to_tid.get(key, key if key in organism.tissues else None)
        if tid is None:
            logger.warning("apply_state_dict: tissue %s missing in organism", key)
            continue
        try:
            organism.tissues[tid].load_state_dict(sd)
            n += 1
        except Exception as e:
            logger.warning("apply_state_dict: load %s failed: %s", key, e)
    return n


# ────────────────────────────────────────────────────────────────────
# Phase 4 Z2 NEAT topology mutations (re-use neurocore[client])
# ────────────────────────────────────────────────────────────────────

def _default_zodchiy_available_roles() -> list[str]:
    """Доступные роли для NEAT-мутаций (TISSUE_ROLES_ZODCHIY минус motor_policy).

    motor_policy — sidecar (per-organism MLP вне tissue graph), не
    участвует в межтканевых рёбрах. Соответствует server-side
    `core.evolution.mutate_tissue_topology_for_organism`.
    """
    from core.constants import TISSUE_ROLES_ZODCHIY
    return [r for r in TISSUE_ROLES_ZODCHIY if r != "motor_policy"]


def mutate_topology_genes(
    parent_genes_dict: list[dict],
    *,
    available_roles: Optional[Iterable[str]] = None,
    p_add: float = DEFAULT_P_TOPO_ADD,
    p_remove: float = DEFAULT_P_TOPO_REMOVE,
    p_change_type: float = DEFAULT_P_TOPO_CHANGE_TYPE,
    p_weight: float = DEFAULT_P_TOPO_WEIGHT,
    weight_sigma: float = DEFAULT_TOPO_WEIGHT_SIGMA,
    rng: Optional[random.Random] = None,
) -> list[dict]:
    """Phase 4 NEAT мутации `tissue_topology_genes` для reproduce envelope.

    Принимает gene list как dict'ы (serialized form, как в seed_pack
    payload и colony_storage из P40), применяет 4 базовые мутации
    через `core.tissue_topology.mutate_tissue_topology` (re-use из
    neurocore[client]), возвращает mutated list dict'ов готовый для
    envelope.

    **Determinism:** при том же `rng` (`random.Random(seed)`) — bit-exact
    тот же результат. Используется для math equivalence test.

    **Innovation tracker:** разделяется глобальным
    `TISSUE_INNOVATION_TRACKER` (singleton в `core.tissue_topology`) —
    инновации синхронизируются между client/server при условии что
    обе стороны видят те же edges (через envelope round-trip).

    Args:
        parent_genes_dict: список dict'ов формата `TissueConnectionGene.to_dict()`
        available_roles: роли допустимые для add-мутации (default: Zodchiy без
            motor_policy)
        p_add / p_remove / p_change_type / p_weight: probabilities — defaults
            те же что server (docs/zodchiy.md §1.4)
        weight_sigma: σ для weight mutation
        rng: `random.Random` instance для воспроизводимости (None — global)

    Returns:
        Новый список dict'ов с применёнными мутациями. Не мутирует input.
    """
    from core.tissue_topology import (
        TissueConnectionGene,
        TISSUE_INNOVATION_TRACKER,
        mutate_tissue_topology,
    )

    if available_roles is None:
        available_roles = _default_zodchiy_available_roles()

    # Parse input dicts → gene objects (effectively deepcopy через сериализацию)
    genes: list = [
        TissueConnectionGene.from_dict(d) for d in (parent_genes_dict or [])
    ]
    mutate_tissue_topology(
        genes,
        available_roles=list(available_roles),
        p_add=p_add,
        p_remove=p_remove,
        p_change_type=p_change_type,
        p_weight=p_weight,
        weight_sigma=weight_sigma,
        rng=rng,
        tracker=TISSUE_INNOVATION_TRACKER,
    )
    return [g.to_dict() for g in genes]


def crossover_topology_genes(
    dominant_genes_dict: list[dict],
    recessive_genes_dict: list[dict],
    *,
    rng: Optional[random.Random] = None,
) -> list[dict]:
    """Phase 4 NEAT кроссинговер межтканевых рёбер по innovation_id.

    Re-use из `core.tissue_topology.crossover_tissue_topology`:
      - Matching (innovation у обоих): 50/50 копия
      - Disjoint/excess: только от dominant

    Для применения в mate-pair flow: после parent-selection
    (по fitness) — собирается список genes для child.

    Returns:
        Новый список dict'ов. Input не мутируется.
    """
    from core.tissue_topology import (
        TissueConnectionGene,
        crossover_tissue_topology,
    )

    dom = [TissueConnectionGene.from_dict(d) for d in (dominant_genes_dict or [])]
    rec = [TissueConnectionGene.from_dict(d) for d in (recessive_genes_dict or [])]
    out = crossover_tissue_topology(dom, rec, rng=rng)
    return [g.to_dict() for g in out]
