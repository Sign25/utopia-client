"""Phase 4 этап D — client-side speciation (Z6.c).

ТЗ Body Migration v1.3 §4 Phase 4 acceptance #4:
  > Speciation тригерится локально (centroid threshold)

Q3 DECIDED (vision §3.7): виды клиент-локальны. Каждый клиент держит
свой SpeciesRegistry — никакой кросс-клиентной дедупликации.

Re-use: `core.tissue_speciation` из neurocore[client] — bit-exact same
math что server. Этот модуль — тонкий wrapper с persistence на диск.

**Layout:**
  $colonies_dir/species_registry.json

**Determinism:** species centroid = topology snapshot founder'а
(historical constant). При assign:
  - Empty registry: create founder species
  - Match within threshold: return matching species_id
  - No match: create new species (mutation jump)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger("utopia_client.speciation")

DEFAULT_THRESHOLD = 1.5  # из core.tissue_speciation.DEFAULT_THRESHOLD


def speciation_registry_path(base: Optional[Path] = None) -> Path:
    """Путь к JSON registry. Default — $colonies_dir/species_registry.json."""
    if base is None:
        from .config import colonies_dir
        base = colonies_dir()
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    return base / "species_registry.json"


def load_or_create_registry(base: Optional[Path] = None):
    """Загрузить registry с диска или вернуть новый пустой.

    Returns `SpeciesRegistry` из core.tissue_speciation.
    """
    from core.tissue_speciation import load_registry
    path = speciation_registry_path(base)
    reg = load_registry(path)
    logger.info("speciation registry: %s (%d species)",
                path, len(reg.all()))
    return reg


def save_registry(registry, base: Optional[Path] = None) -> Path:
    """Atomic write registry на диск."""
    from core.tissue_speciation import save_registry_atomic
    path = speciation_registry_path(base)
    save_registry_atomic(registry, path)
    return path


def assign_species(
    registry,
    topology_genes_dict: list[dict],
    *,
    tick: int = 0,
    founder_cid: Optional[str] = None,
    threshold: float = DEFAULT_THRESHOLD,
    c_excess: float = 1.0,
    c_disjoint: float = 1.0,
    c_weight: float = 0.4,
) -> tuple[int, bool]:
    """Назначить species_id для организма с данной topology.

    Args:
        registry: SpeciesRegistry instance (mutated in-place)
        topology_genes_dict: list of TissueConnectionGene.to_dict()
        tick: world tick для birth_tick / extinct_tick (если новый species)
        founder_cid: cid организма-основателя (для трекинга)
        threshold: max distance чтобы считаться «свой вид»
        c_excess / c_disjoint / c_weight: NEAT distance coefficients

    Returns:
        (species_id, is_new_species)
        - is_new_species=True если создан новый species
        - False если найден match
    """
    from core.tissue_topology import TissueConnectionGene

    topology = [
        TissueConnectionGene.from_dict(d) for d in (topology_genes_dict or [])
    ]

    # Empty registry — founder
    if registry.is_empty():
        s = registry.create(
            centroid=topology,
            birth_tick=int(tick),
            parent_species_id=None,
            founder_cid=founder_cid,
        )
        logger.info("speciation: founder species_id=%d for cid=%s",
                    s.species_id, founder_cid)
        return (s.species_id, True)

    # Best match search
    best = registry.find_best_match(
        topology,
        threshold=threshold,
        c_excess=c_excess,
        c_disjoint=c_disjoint,
        c_weight=c_weight,
    )
    if best is not None:
        if best.extinct:
            # Convergent revival — снять extinct
            registry.revive(best.species_id)
            logger.info("speciation: revive species_id=%d for cid=%s",
                        best.species_id, founder_cid)
        return (best.species_id, False)

    # No match → mutation jump → new species
    # parent_species_id — мы не знаем родителя topology, оставляем None
    # (P40 сторона имеет full lineage tracking; client не обязан).
    s = registry.create(
        centroid=topology,
        birth_tick=int(tick),
        parent_species_id=None,
        founder_cid=founder_cid,
    )
    logger.info("speciation: new species_id=%d for cid=%s (mutation jump)",
                s.species_id, founder_cid)
    return (s.species_id, True)


def get_species_summary(registry) -> dict:
    """Снимок для diagnostics push.

    Returns:
        {"n_species": int, "n_active": int, "n_extinct": int,
         "max_species_id": int}
    """
    all_species = registry.all()
    active = [s for s in all_species if not s.extinct]
    extinct = [s for s in all_species if s.extinct]
    return {
        "n_species": len(all_species),
        "n_active": len(active),
        "n_extinct": len(extinct),
        "max_species_id": registry.max_species_id,
    }
