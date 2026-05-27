"""Phase 4 Z2 NEAT topology mutations — client-side tests.

Проверяем что `mutate_topology_genes` / `crossover_topology_genes`:
  1. определены и importable
  2. правильно использует re-use из `core.tissue_topology` (neurocore[client])
  3. detrminism: same rng seed → bit-exact same result
  4. signature compatibility: dict-in / dict-out (envelope format)
  5. respects mutation rate p=0 (no-op)
  6. crossover NEAT rules (matching + disjoint)
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Skip if neurocore[client] не доступен (CI без deps)
pytest.importorskip("core.tissue_topology")
pytest.importorskip("core.constants")


def _make_gene_dict(innovation: int, src: str, tgt: str,
                    conn_type: str = "direct", weight: float = 1.0,
                    enabled: bool = True) -> dict:
    """Helper: собрать gene dict в формате `TissueConnectionGene.to_dict()`."""
    return {
        "innovation": innovation,
        "source_role": src,
        "target_role": tgt,
        "source_port": "output",
        "target_port": "input",
        "conn_type": conn_type,
        "weight": weight,
        "enabled": enabled,
    }


# ────────────────────────────────────────────────────────────────────
# Imports + signature
# ────────────────────────────────────────────────────────────────────

def test_mutate_topology_genes_importable():
    from utopia_client.reproduce import mutate_topology_genes
    assert callable(mutate_topology_genes)


def test_crossover_topology_genes_importable():
    from utopia_client.reproduce import crossover_topology_genes
    assert callable(crossover_topology_genes)


def test_default_zodchiy_roles_excludes_motor_policy():
    from utopia_client.reproduce import _default_zodchiy_available_roles
    roles = _default_zodchiy_available_roles()
    assert "motor_policy" not in roles
    assert len(roles) > 0
    # Должно быть несколько zodchiy-tissue ролей (brain, motor, sensory, etc.)
    expected_some = {"brain", "motor", "sensory"}
    assert expected_some.issubset(set(roles)), \
        f"missing expected roles: {expected_some - set(roles)}"


# ────────────────────────────────────────────────────────────────────
# Mutation behavior
# ────────────────────────────────────────────────────────────────────

def test_mutate_topology_empty_genes_no_crash():
    """Пустой список + дефолтные rates — не должен падать."""
    from utopia_client.reproduce import mutate_topology_genes
    result = mutate_topology_genes([], rng=random.Random(0))
    assert isinstance(result, list)
    # При p_add=0.02 на rng(0) может или не может появиться edge


def test_mutate_topology_p_zero_noop():
    """При всех p=0.0 — мутации не происходят (identity, кроме deepcopy)."""
    from utopia_client.reproduce import mutate_topology_genes

    parent = [
        _make_gene_dict(0, "brain", "motor", weight=0.5),
        _make_gene_dict(1, "sensory", "brain", weight=0.7),
    ]
    result = mutate_topology_genes(
        parent,
        p_add=0.0, p_remove=0.0, p_change_type=0.0, p_weight=0.0,
        rng=random.Random(42),
    )
    assert len(result) == 2
    # Сравниваем по innovation + role + weight + enabled
    for r, p in zip(result, parent):
        assert r["innovation"] == p["innovation"]
        assert r["source_role"] == p["source_role"]
        assert r["target_role"] == p["target_role"]
        assert r["weight"] == p["weight"]
        assert r["enabled"] == p["enabled"]


def test_mutate_topology_does_not_mutate_input():
    """Input list не должен меняться (защита от случайных in-place изменений)."""
    from utopia_client.reproduce import mutate_topology_genes

    parent = [_make_gene_dict(0, "brain", "motor", weight=0.5)]
    parent_copy = [dict(g) for g in parent]
    _ = mutate_topology_genes(parent, rng=random.Random(0))
    assert parent == parent_copy, "input dict list был изменён in-place"


def test_mutate_topology_determinism_same_seed():
    """Same rng seed → bit-exact same result."""
    from utopia_client.reproduce import mutate_topology_genes

    parent = [
        _make_gene_dict(0, "brain", "motor", weight=0.5),
        _make_gene_dict(1, "sensory", "brain", weight=0.7),
        _make_gene_dict(2, "motor", "muscle", weight=1.0, conn_type="modulating"),
    ]
    # high mutation rates чтобы гарантированно были изменения
    high = dict(p_add=1.0, p_remove=1.0, p_change_type=1.0, p_weight=1.0)

    r1 = mutate_topology_genes(parent, rng=random.Random(7), **high)
    r2 = mutate_topology_genes(parent, rng=random.Random(7), **high)
    assert r1 == r2, f"determinism violated:\n  r1={r1}\n  r2={r2}"


def test_mutate_topology_determinism_different_seeds_diverge():
    """Different seeds → divergent (negative test)."""
    from utopia_client.reproduce import mutate_topology_genes

    parent = [_make_gene_dict(0, "brain", "motor", weight=0.5)]
    high = dict(p_add=1.0, p_remove=0.0, p_change_type=0.0, p_weight=0.5)

    r1 = mutate_topology_genes(parent, rng=random.Random(1), **high)
    r2 = mutate_topology_genes(parent, rng=random.Random(2), **high)
    # Хотя бы одно поле должно отличаться (или длина списков)
    assert r1 != r2, "different seeds gave bit-identical result"


def test_mutate_topology_weight_clamped_to_range():
    """weight clipped в [-2.0, 2.0]."""
    from utopia_client.reproduce import mutate_topology_genes

    parent = [_make_gene_dict(0, "brain", "motor", weight=1.95)]
    result = mutate_topology_genes(
        parent,
        p_add=0.0, p_remove=0.0, p_change_type=0.0, p_weight=1.0,
        weight_sigma=2.0,  # огромный sigma → weight push к лимиту
        rng=random.Random(0),
    )
    assert -2.0 <= result[0]["weight"] <= 2.0


def test_mutate_topology_remove_marks_disabled():
    """remove-мутация ставит enabled=False, не удаляет ген (NEAT historical marker)."""
    from utopia_client.reproduce import mutate_topology_genes

    parent = [_make_gene_dict(0, "brain", "motor", enabled=True)]
    result = mutate_topology_genes(
        parent,
        p_add=0.0, p_remove=1.0, p_change_type=0.0, p_weight=0.0,
        rng=random.Random(0),
    )
    assert len(result) == 1, "ген должен остаться (NEAT marker)"
    assert result[0]["enabled"] is False


# ────────────────────────────────────────────────────────────────────
# Crossover behavior
# ────────────────────────────────────────────────────────────────────

def test_crossover_topology_empty_returns_empty():
    from utopia_client.reproduce import crossover_topology_genes
    result = crossover_topology_genes([], [], rng=random.Random(0))
    assert result == []


def test_crossover_topology_disjoint_from_dominant_only():
    """Disjoint genes (только у одного родителя) — берутся только от dominant."""
    from utopia_client.reproduce import crossover_topology_genes

    dom = [
        _make_gene_dict(0, "brain", "motor"),
        _make_gene_dict(1, "sensory", "brain"),  # disjoint
    ]
    rec = [
        _make_gene_dict(0, "brain", "motor"),
        _make_gene_dict(99, "motor", "muscle"),  # disjoint (от recessive)
    ]
    result = crossover_topology_genes(dom, rec, rng=random.Random(0))
    innovations = {g["innovation"] for g in result}
    # disjoint от dominant (1) должен быть; disjoint от recessive (99) — нет
    assert 1 in innovations
    assert 99 not in innovations
    assert 0 in innovations  # matching


def test_crossover_topology_determinism_same_seed():
    """Same rng → same result (NEAT 50/50 matching gene selection)."""
    from utopia_client.reproduce import crossover_topology_genes

    dom = [_make_gene_dict(0, "brain", "motor", weight=0.5)]
    rec = [_make_gene_dict(0, "brain", "motor", weight=0.9)]

    r1 = crossover_topology_genes(dom, rec, rng=random.Random(13))
    r2 = crossover_topology_genes(dom, rec, rng=random.Random(13))
    assert r1 == r2


def test_crossover_topology_does_not_mutate_inputs():
    from utopia_client.reproduce import crossover_topology_genes

    dom = [_make_gene_dict(0, "brain", "motor", weight=0.5)]
    rec = [_make_gene_dict(0, "brain", "motor", weight=0.9)]
    dom_copy = [dict(g) for g in dom]
    rec_copy = [dict(g) for g in rec]

    _ = crossover_topology_genes(dom, rec, rng=random.Random(0))
    assert dom == dom_copy
    assert rec == rec_copy


# ────────────────────────────────────────────────────────────────────
# Round-trip: dict ↔ gene объект
# ────────────────────────────────────────────────────────────────────

def test_dict_roundtrip_preserves_all_fields():
    """to_dict → from_dict → to_dict — все поля сохраняются."""
    from core.tissue_topology import TissueConnectionGene

    original = _make_gene_dict(
        42, "brain", "motor",
        conn_type="modulating",
        weight=-0.5,
        enabled=False,
    )
    gene = TissueConnectionGene.from_dict(original)
    roundtrip = gene.to_dict()
    assert roundtrip == original
