"""Z2.b межтканевой NEAT-overlay на клиенте (01.06.2026, Фрай).

Зодчий перестраивает граф тканей через `tissue_topology_genes` (не только
веса). Проверяем проброс генов через жизненный цикл:
  - save_state сериализует genes (resume/elite не теряет divergence);
  - restore_persisted_state восстанавливает genes + re-apply overlay;
  - mate-flow зовёт crossover_org_topology_for_zodchiy (gated на zodchiy);
  - motor_policy исключён из available_roles (sidecar-policy, не топология).

Speciation оживёт сама: assign_species читает непустые genes →
topology_distance → реальные виды по архитектуре графа.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
# `core` приходит из установленного neurocore[client] в проде. Для локального
# dev-прогона подкладываем исходники NeuroCore рядом (как делает re-use ядра).
_NEUROCORE = _ROOT.parent / "NeuroCore"
if _NEUROCORE.exists() and str(_NEUROCORE) not in sys.path:
    sys.path.insert(0, str(_NEUROCORE))

pytest.importorskip("torch")
pytest.importorskip("core")  # skip если neurocore-ядро недоступно

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402


class _FakeOrg:
    """Минимальный organism: только поля, которые трогает persist/overlay."""

    def __init__(self, genes=None):
        self.tissues = {}          # роли нет → overlay genes игнорирует (sidecar)
        self.connections = []      # apply_topology_overlay_to_org: list(...)
        self._cached_levels = None
        self.tissue_topology_genes = list(genes or [])


def _gene(innovation=1, src="sensory", dst="brain", weight=0.7):
    from core.tissue_topology import TissueConnectionGene
    return TissueConnectionGene(
        innovation=innovation, source_role=src, target_role=dst, weight=weight)


# ── save_state сериализует genes ──────────────────────────────────────

def test_save_state_serializes_genes():
    c = LocalColonyCompute(device="cpu")
    g = _gene(innovation=7, src="brain", dst="motor", weight=1.3)
    c.organisms["c1"] = _FakeOrg(genes=[g])
    payload = c.save_state("c1")
    assert payload is not None
    assert "tissue_topology_genes" in payload
    assert payload["tissue_topology_genes"] == [g.to_dict()]


def test_save_state_omits_empty_genes():
    c = LocalColonyCompute(device="cpu")
    c.organisms["c1"] = _FakeOrg(genes=[])
    payload = c.save_state("c1")
    assert payload is not None
    assert "tissue_topology_genes" not in payload  # пустые не пишем


# ── restore_persisted_state восстанавливает genes + overlay ───────────

def test_restore_persisted_state_restores_genes():
    c = LocalColonyCompute(device="cpu")
    org = _FakeOrg(genes=[])          # стартует без генов (founder)
    c.organisms["c1"] = org
    g = _gene(innovation=9, src="sensory", dst="planner")
    c.restore_persisted_state("c1", {"tissue_topology_genes": [g.to_dict()]})
    assert len(org.tissue_topology_genes) == 1
    assert org.tissue_topology_genes[0].innovation == 9
    assert org.tissue_topology_genes[0].source_role == "sensory"
    assert org.tissue_topology_genes[0].target_role == "planner"


def test_persist_roundtrip_genes_preserved():
    # save → load: genes идентичны (divergence переживает restart).
    c1 = LocalColonyCompute(device="cpu")
    g = _gene(innovation=42, src="brain", dst="amygdala", weight=-0.5)
    c1.organisms["c1"] = _FakeOrg(genes=[g])
    payload = c1.save_state("c1")

    c2 = LocalColonyCompute(device="cpu")
    org2 = _FakeOrg(genes=[])
    c2.organisms["c1"] = org2
    c2.restore_persisted_state("c1", payload)
    assert [x.to_dict() for x in org2.tissue_topology_genes] == [g.to_dict()]


# ── available_roles: motor_policy исключён ────────────────────────────

def test_default_zodchiy_roles_excludes_motor_policy():
    from utopia_client.reproduce import _default_zodchiy_available_roles
    roles = _default_zodchiy_available_roles()
    assert roles, "available_roles не должен быть пустым"
    assert "motor_policy" not in roles  # sidecar policy, не часть графа


# ── crossover_org_topology_for_zodchiy: gated на lineage ──────────────

def test_topology_crossover_noop_for_non_zodchiy():
    from core.tissue_topology import crossover_org_topology_for_zodchiy
    from utopia_client.reproduce import _default_zodchiy_available_roles
    child, mom, dad = _FakeOrg(), _FakeOrg(), _FakeOrg()
    applied = crossover_org_topology_for_zodchiy(
        child, mom, dad, lineage="elder",
        available_roles=_default_zodchiy_available_roles(),
        rng=random.Random(0))
    assert applied is False  # Старший фикс. граф — overlay не для него


def test_topology_crossover_applies_for_zodchiy():
    from core.tissue_topology import crossover_org_topology_for_zodchiy
    from utopia_client.reproduce import _default_zodchiy_available_roles
    child, mom, dad = _FakeOrg(), _FakeOrg(), _FakeOrg()
    applied = crossover_org_topology_for_zodchiy(
        child, mom, dad, lineage="zodchiy",
        available_roles=_default_zodchiy_available_roles(),
        rng=random.Random(0))
    assert applied is True
    assert isinstance(child.tissue_topology_genes, list)
