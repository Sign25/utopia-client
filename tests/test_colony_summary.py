"""colony_summary + per-creature stats-поля для UI /stats (01.06.2026).

Агрегат выживания/эволюции/обучения (extra.colony_summary) и плоские
per-creature поля (gen/topo/age/inst/food) в projection_batch.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_NEUROCORE = _ROOT.parent / "NeuroCore"  # core для test_projection (локальный dev)
if _NEUROCORE.exists() and str(_NEUROCORE) not in sys.path:
    sys.path.insert(0, str(_NEUROCORE))

pytest.importorskip("torch")

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402


def _c():
    return LocalColonyCompute(device="cpu")


def _org(generation=0, genes=None, telomere=1.0):
    return types.SimpleNamespace(
        generation=generation,
        tissue_topology_genes=list(genes or []),
        telomere=telomere,
    )


# ── build_colony_summary ──────────────────────────────────────────────

def test_summary_empty_shape():
    c = _c()
    s = c.build_colony_summary()
    assert s["n_alive"] == 0 and s["n_species_alive"] == 0
    # структура контракта на месте
    assert set(s["energy"]) == {"income", "cost", "net", "ratio"}
    assert set(s["newborn"]) == {"natural", "bootstrap", "instinct_active"}
    assert set(s["deaths"]) == {"starvation", "telomere", "infection"}
    assert isinstance(s["history"], list)


def test_summary_counts_alive_species_eff():
    c = _c()
    c.organisms["a"] = _org()
    c.organisms["b"] = _org()
    c.organisms["d"] = _org()
    c._dead_cids.add("d")               # мёртвый не считается
    c.species_id.update({"a": 1, "b": 2, "d": 2})
    c.traits.update({"a": {"efficiency": 8}, "b": {"efficiency": 12}})
    s = c.build_colony_summary()
    assert s["n_alive"] == 2
    assert s["n_species_alive"] == 2    # виды 1 и 2 среди живых
    assert s["eff_mean"] == 10.0        # (8+12)/2


def test_summary_newborn_and_deaths():
    c = _c()
    c.organisms["n"] = _org()
    c._birth_tick["n"] = 0              # активный инстинкт → instinct_active
    c._n_natural_newborn = 3
    c._n_bootstrap_rejuv = 5
    c._deaths_by_cause["starvation"] = 7
    s = c.build_colony_summary()
    assert s["newborn"] == {"natural": 3, "bootstrap": 5, "instinct_active": 1}
    assert s["deaths"]["starvation"] == 7


def test_summary_energy_from_last_window():
    c = _c()
    c._last_window = {"inc": 12.0, "cost": 100.0, "net": -88.0, "ratio": 0.12}
    s = c.build_colony_summary()
    assert s["energy"]["income"] == 12.0
    assert s["energy"]["ratio"] == 0.12


# ── per-creature stats-поля (diagnostics.creatures → фронт) ───────────

def test_per_creature_has_stats_fields():
    c = _c()
    from core.tissue_topology import TissueConnectionGene  # noqa
    g = TissueConnectionGene(innovation=1, source_role="sensory",
                             target_role="brain")
    c.organisms["x"] = _org(generation=4, genes=[g])
    c.species_id["x"] = 3
    c._last_world_tick = 250
    c._birth_tick["x"] = 0             # age=250 → inst=0.5
    c._carried_food["x"] = 2
    rows = c._per_creature_stats()
    p = next(r for r in rows if r["cid"] == "x")
    assert p["species_id"] == 3
    assert p["gen"] == 4
    assert p["topo"] == 1              # одно ребро-ген
    assert p["age"] == 250
    assert abs(p["inst"] - 0.5) < 1e-6
    assert p["food"] == 2


def test_per_creature_null_age_for_untracked():
    c = _c()
    c.organisms["y"] = _org(generation=1)
    rows = c._per_creature_stats()
    p = next(r for r in rows if r["cid"] == "y")
    assert p["gen"] == 1 and p["topo"] == 0
    assert p["age"] is None and p["inst"] is None   # не newborn → None


# ── build_creature_stats (heartbeat-канал для UI) ─────────────────────

def test_creature_stats_compact_keyed_by_cid():
    from core.tissue_topology import TissueConnectionGene  # noqa
    g = TissueConnectionGene(innovation=1, source_role="sensory",
                             target_role="brain")
    c = _c()
    c.organisms["x"] = _org(genes=[g])
    c.species_id["x"] = 5
    c._last_world_tick = 100
    c._birth_tick["x"] = 0                # age=100 → inst=0.8
    cs = c.build_creature_stats()
    assert set(cs.keys()) == {"x"}
    assert cs["x"]["species_id"] == 5
    assert cs["x"]["topo"] == 1
    assert abs(cs["x"]["inst"] - 0.8) < 1e-6


def test_creature_stats_skips_dead_and_null_inst():
    c = _c()
    c.organisms["a"] = _org()             # не newborn → inst None
    c.organisms["d"] = _org()
    c._dead_cids.add("d")
    cs = c.build_creature_stats()
    assert "d" not in cs                  # мёртвого нет
    assert cs["a"]["inst"] is None        # не newborn
