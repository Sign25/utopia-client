"""Phase 4 этап D — speciation client-side tests.

Pure-функции wrap'нутые поверх core.tissue_speciation. Тестируем:
  - empty registry → founder species_id=0
  - match within threshold → existing species_id
  - distance > threshold → new species (mutation jump)
  - extinct revival
  - persistence: save → reload → same species_ids
  - summary diagnostics
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("core.tissue_speciation")
pytest.importorskip("core.tissue_topology")


def _gene_dict(innov: int, src: str, tgt: str, weight: float = 1.0,
               enabled: bool = True) -> dict:
    return {
        "innovation": innov,
        "source_role": src,
        "target_role": tgt,
        "source_port": "output",
        "target_port": "input",
        "conn_type": "direct",
        "weight": weight,
        "enabled": enabled,
    }


# ────────────────────────────────────────────────────────────────────
# Empty / founder
# ────────────────────────────────────────────────────────────────────

def test_empty_registry_creates_founder():
    from utopia_client.speciation import assign_species, load_or_create_registry

    reg = load_or_create_registry.__wrapped__ if hasattr(
        load_or_create_registry, "__wrapped__"
    ) else None
    # Easier: create empty registry directly
    from core.tissue_speciation import SpeciesRegistry
    reg = SpeciesRegistry()

    genes = [_gene_dict(0, "brain", "motor")]
    sid, is_new = assign_species(reg, genes, tick=10, founder_cid="cid-a")
    assert sid == 0
    assert is_new is True
    assert reg.has(0)
    s = reg.get(0)
    assert s.founder_cid == "cid-a"
    assert s.birth_tick == 10


def test_empty_genes_founder_still_assigned():
    """Founder с пустым topology — тоже species_id=0."""
    from core.tissue_speciation import SpeciesRegistry
    from utopia_client.speciation import assign_species

    reg = SpeciesRegistry()
    sid, is_new = assign_species(reg, [], tick=0, founder_cid="adam")
    assert sid == 0
    assert is_new is True


# ────────────────────────────────────────────────────────────────────
# Match / new species
# ────────────────────────────────────────────────────────────────────

def test_identical_topology_matches_same_species():
    """Same topology → same species (distance 0 ≤ threshold)."""
    from core.tissue_speciation import SpeciesRegistry
    from utopia_client.speciation import assign_species

    reg = SpeciesRegistry()
    genes = [_gene_dict(0, "brain", "motor")]
    sid_a, _ = assign_species(reg, genes, founder_cid="a")
    sid_b, is_new = assign_species(reg, genes, founder_cid="b")
    assert sid_a == sid_b == 0
    assert is_new is False


def test_far_topology_creates_new_species():
    """Большой distance > threshold → new species."""
    from core.tissue_speciation import SpeciesRegistry
    from utopia_client.speciation import assign_species

    reg = SpeciesRegistry()
    g_a = [_gene_dict(0, "brain", "motor")]
    sid_a, _ = assign_species(reg, g_a, founder_cid="a")
    # Полностью disjoint topology — высокий distance
    g_b = [
        _gene_dict(50, "sensory", "brain"),
        _gene_dict(51, "brain", "muscle"),
        _gene_dict(52, "motor", "sensory"),
    ]
    sid_b, is_new = assign_species(reg, g_b, threshold=0.5, founder_cid="b")
    assert sid_b != sid_a
    assert is_new is True


def test_close_topology_uses_lower_threshold():
    """С низким threshold — даже близкая topology → new species."""
    from core.tissue_speciation import SpeciesRegistry
    from utopia_client.speciation import assign_species

    reg = SpeciesRegistry()
    g_a = [
        _gene_dict(0, "brain", "motor", weight=1.0),
        _gene_dict(1, "sensory", "brain", weight=1.0),
    ]
    sid_a, _ = assign_species(reg, g_a, founder_cid="a")
    # Weight diff = 1.0 → c_weight=0.4 contribution ~0.4
    g_b = [
        _gene_dict(0, "brain", "motor", weight=2.0),
        _gene_dict(1, "sensory", "brain", weight=2.0),
    ]
    sid_b, is_new = assign_species(
        reg, g_b, threshold=0.01, founder_cid="b")
    assert is_new is True
    assert sid_b != sid_a


# ────────────────────────────────────────────────────────────────────
# Extinct revival
# ────────────────────────────────────────────────────────────────────

def test_extinct_species_revived_on_match():
    """Mark extinct → match identical topology → revive."""
    from core.tissue_speciation import SpeciesRegistry
    from utopia_client.speciation import assign_species

    reg = SpeciesRegistry()
    genes = [_gene_dict(0, "brain", "motor")]
    sid_a, _ = assign_species(reg, genes, founder_cid="a")
    reg.mark_extinct(sid_a, tick=100)
    assert reg.get(sid_a).extinct is True

    sid_b, is_new = assign_species(reg, genes, founder_cid="b")
    assert sid_b == sid_a
    assert is_new is False
    assert reg.get(sid_a).extinct is False, "revival не сработал"


# ────────────────────────────────────────────────────────────────────
# Persistence
# ────────────────────────────────────────────────────────────────────

def test_save_then_load_preserves_species(tmp_path, monkeypatch):
    from utopia_client import config as cli_config
    monkeypatch.setattr(cli_config, "colonies_dir",
                        lambda: tmp_path / "colonies")

    from core.tissue_speciation import SpeciesRegistry
    from utopia_client.speciation import (
        assign_species, save_registry, load_or_create_registry,
    )

    # Создаём 3 species
    reg_a = SpeciesRegistry()
    assign_species(reg_a, [_gene_dict(0, "brain", "motor")],
                   tick=1, founder_cid="cid-1")
    assign_species(reg_a, [_gene_dict(99, "sensory", "muscle"),
                            _gene_dict(100, "muscle", "brain")],
                   threshold=0.01, tick=2, founder_cid="cid-2")
    assign_species(reg_a, [_gene_dict(200, "motor", "sensory")],
                   threshold=0.01, tick=3, founder_cid="cid-3")
    assert len(reg_a.all()) == 3

    save_registry(reg_a)

    # Reload
    reg_b = load_or_create_registry()
    assert len(reg_b.all()) == 3
    # Same species_ids
    ids_a = sorted(s.species_id for s in reg_a.all())
    ids_b = sorted(s.species_id for s in reg_b.all())
    assert ids_a == ids_b


def test_load_missing_returns_empty(tmp_path, monkeypatch):
    """Если файла нет — fresh empty registry без crash."""
    from utopia_client import config as cli_config
    monkeypatch.setattr(cli_config, "colonies_dir",
                        lambda: tmp_path / "colonies")

    from utopia_client.speciation import load_or_create_registry
    reg = load_or_create_registry()
    assert reg.is_empty()


# ────────────────────────────────────────────────────────────────────
# Summary diagnostics
# ────────────────────────────────────────────────────────────────────

def test_summary_empty_registry():
    from core.tissue_speciation import SpeciesRegistry
    from utopia_client.speciation import get_species_summary

    reg = SpeciesRegistry()
    s = get_species_summary(reg)
    assert s["n_species"] == 0
    assert s["n_active"] == 0
    assert s["n_extinct"] == 0
    assert s["max_species_id"] == -1


def test_summary_with_active_and_extinct():
    from core.tissue_speciation import SpeciesRegistry
    from utopia_client.speciation import assign_species, get_species_summary

    reg = SpeciesRegistry()
    sid_a, _ = assign_species(reg, [_gene_dict(0, "brain", "motor")],
                              founder_cid="a")
    sid_b, _ = assign_species(reg, [_gene_dict(99, "sensory", "muscle")],
                              threshold=0.01, founder_cid="b")
    reg.mark_extinct(sid_a, tick=10)

    s = get_species_summary(reg)
    assert s["n_species"] == 2
    assert s["n_active"] == 1
    assert s["n_extinct"] == 1
    assert s["max_species_id"] == 1
