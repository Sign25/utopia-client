"""Phase 4 этап C: integration mutate_topology в build_reproduce_envelope.

Тесты:
  1. mutate_topology=False (default) — backward compat, payload без поля
  2. mutate_topology=True + empty genes — skip без crash
  3. mutate_topology=True + non-empty genes — payload contains
     tissue_topology_genes (mutated через topology_rng)
  4. Determinism: same topology_rng → same envelope payload
  5. Pack/unpack roundtrip — genes сохраняются через zstd+b64
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")
pytest.importorskip("core.tissue_topology")


# ────────────────────────────────────────────────────────────────────
# Fake organism (минимальный для тестов envelope)
# ────────────────────────────────────────────────────────────────────

class _FakeTissue:
    """Tissue с state_dict() для extract_tissues_by_role."""

    def __init__(self, role: str):
        import torch
        self.role = role
        self._sd = {
            "weight": torch.randn(4, 4),
            "bias": torch.zeros(4),
        }

    def state_dict(self) -> dict:
        return self._sd


class _FakeOrganism:
    """CompositeOrganism-mock с tissues и tissue_topology_genes."""

    def __init__(self, genes=None):
        self.tissues: dict[str, _FakeTissue] = {
            "t-brain": _FakeTissue("brain"),
            "t-motor": _FakeTissue("motor"),
        }
        self.tissue_topology_genes = genes or []


def _make_gene():
    """Создаёт TissueConnectionGene для теста."""
    from core.tissue_topology import TissueConnectionGene
    from core.connection import ConnectionType
    return TissueConnectionGene(
        innovation=0,
        source_role="brain",
        target_role="motor",
        conn_type=ConnectionType.DIRECT,
        weight=0.5,
        enabled=True,
    )


# ────────────────────────────────────────────────────────────────────
# mutate_topology=False (default)
# ────────────────────────────────────────────────────────────────────

def test_envelope_default_no_topology_field():
    """Default mutate_topology=False — payload не содержит genes."""
    from utopia_client.reproduce import (
        build_reproduce_envelope, unpack_zstd_b64,
    )
    org = _FakeOrganism(genes=[_make_gene()])  # есть genes, но flag False
    env = build_reproduce_envelope("parent-1", org)
    payload = unpack_zstd_b64(env["child_weights_b64"])
    assert "tissue_topology_genes" not in payload


def test_envelope_default_backward_compat():
    """Default envelope содержит legacy ключи (tissues_by_role)."""
    from utopia_client.reproduce import (
        build_reproduce_envelope, unpack_zstd_b64,
    )
    org = _FakeOrganism()
    env = build_reproduce_envelope("p-1", org)
    assert env["type"] == "reproduce"
    assert env["parent_cid"] == "p-1"
    payload = unpack_zstd_b64(env["child_weights_b64"])
    assert "tissues_by_role" in payload


# ────────────────────────────────────────────────────────────────────
# mutate_topology=True
# ────────────────────────────────────────────────────────────────────

def test_envelope_topology_opt_in_empty_genes_skip():
    """mutate_topology=True но genes=[] — поле не добавляется (нет смысла)."""
    from utopia_client.reproduce import (
        build_reproduce_envelope, unpack_zstd_b64,
    )
    org = _FakeOrganism(genes=[])
    env = build_reproduce_envelope("p-1", org, mutate_topology=True)
    payload = unpack_zstd_b64(env["child_weights_b64"])
    assert "tissue_topology_genes" not in payload


def test_envelope_topology_opt_in_with_genes():
    """mutate_topology=True + genes — payload содержит mutated list."""
    from utopia_client.reproduce import (
        build_reproduce_envelope, unpack_zstd_b64,
    )
    g = _make_gene()
    org = _FakeOrganism(genes=[g])
    env = build_reproduce_envelope(
        "p-1", org,
        mutate_topology=True,
        topology_rng=random.Random(42),
    )
    payload = unpack_zstd_b64(env["child_weights_b64"])
    assert "tissue_topology_genes" in payload
    assert isinstance(payload["tissue_topology_genes"], list)
    assert len(payload["tissue_topology_genes"]) >= 1
    # Каждый gene — dict с innovation
    g0 = payload["tissue_topology_genes"][0]
    assert "innovation" in g0
    assert "source_role" in g0
    assert "target_role" in g0


def test_envelope_topology_determinism():
    """Same topology_rng seed → identical envelope payloads."""
    from utopia_client.reproduce import (
        build_reproduce_envelope, unpack_zstd_b64,
    )
    g = _make_gene()

    # Создаём два независимых organism (важно — у нас shared _make_gene
    # возвращает new instance каждый раз)
    org_a = _FakeOrganism(genes=[_make_gene()])
    org_b = _FakeOrganism(genes=[_make_gene()])

    env_a = build_reproduce_envelope(
        "p-1", org_a, mutate_topology=True, topology_rng=random.Random(7))
    env_b = build_reproduce_envelope(
        "p-1", org_b, mutate_topology=True, topology_rng=random.Random(7))

    payload_a = unpack_zstd_b64(env_a["child_weights_b64"])
    payload_b = unpack_zstd_b64(env_b["child_weights_b64"])

    # Mutated genes должны быть identical (same rng → bit-exact)
    assert payload_a["tissue_topology_genes"] == \
        payload_b["tissue_topology_genes"]


def test_envelope_topology_no_field_no_crash():
    """Organism без tissue_topology_genes field — нет crash."""
    from utopia_client.reproduce import (
        build_reproduce_envelope, unpack_zstd_b64,
    )

    class _BareOrganism:
        def __init__(self):
            self.tissues = {"t-brain": _FakeTissue("brain")}
            # БЕЗ tissue_topology_genes

    env = build_reproduce_envelope("p-1", _BareOrganism(), mutate_topology=True)
    payload = unpack_zstd_b64(env["child_weights_b64"])
    assert "tissue_topology_genes" not in payload


def test_envelope_topology_roundtrip_preserves_genes():
    """Pack/unpack через zstd+b64 сохраняет genes без потерь."""
    from utopia_client.reproduce import (
        build_reproduce_envelope, unpack_zstd_b64,
    )
    # high p_add чтобы гарантированно вырастить >1 gene
    from utopia_client.reproduce import mutate_topology_genes
    g = _make_gene()
    org = _FakeOrganism(genes=[g])
    # Сохраняем ожидаемый mutation result отдельно (same seed)
    expected = mutate_topology_genes([g.to_dict()], rng=random.Random(99))

    env = build_reproduce_envelope(
        "p-1", org, mutate_topology=True, topology_rng=random.Random(99))
    payload = unpack_zstd_b64(env["child_weights_b64"])
    assert payload["tissue_topology_genes"] == expected
