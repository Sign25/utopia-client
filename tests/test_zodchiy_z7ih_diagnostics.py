"""Z7.i.h — Зодчий-сигналы в colony_diagnostics (17.05.2026).

Закрывает observability-шов: после Z7.i.f клиент знает snapshot-нормы
3 Зодчий-тканей per-cid (`last_cerebellum_delta`/`last_amygdala_valence`/
`last_episodic_recall`), но `diagnostics()` их не агрегировал и не
прокидывал per-creature в `_per_creature_stats`. Z7.i.h.A добавляет:
  - colony-level блок `zodchiy_sidecar` (n_active, 3 avg-метрики)
  - per-creature поля `client_cerebellum_delta_norm`,
    `client_amygdala_valence`, `client_episodic_recall_norm`

VPS-сторону (Z7.i.h.B) и Кабинет (Z7.i.h.C) шлёт отдельные правки.

Покрытие:
  Z7ih.A.1   empty colony → zodchiy_sidecar block есть, n_active=0
  Z7ih.A.2   wanderer only → n_active=0 (snapshot store'ы пусты)
  Z7ih.A.3   single zodchiy after forward → n_active=1, нормы>0
  Z7ih.A.4   mixed wanderer + zodchiy → n_active=1, аггрегаты корректны
  Z7ih.A.5   per-creature: wanderer имеет 0.0 поля
  Z7ih.A.6   per-creature: zodchiy after forward — поля>0
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("storage.norg")
pytest.importorskip("core.workbench")


@pytest.fixture
def seed_file(tmp_path, monkeypatch):
    seed_path = tmp_path / "seed.norg"
    monkeypatch.setenv("WORLD_SEED_PATH", str(seed_path))
    import importlib
    from environment import seed_loader as ns_loader
    importlib.reload(ns_loader)
    ns_loader.ensure_seed(preset="nexus")
    monkeypatch.setenv("UTOPIA_SEED_PATH", str(tmp_path / "client_seed.norg"))
    monkeypatch.setenv("UTOPIA_WANDERER_SEED_PATH",
                       str(tmp_path / "client_seed.norg"))
    monkeypatch.setenv("UTOPIA_COLONIES_DIR", str(tmp_path / "colonies"))
    client_seed = tmp_path / "client_seed.norg"
    client_seed.write_bytes(seed_path.read_bytes())
    from utopia_client import seed_loader as cli_loader
    importlib.reload(cli_loader)
    return client_seed


def _make_compute_with(seed_file, n: int = 1):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    orgs = load_founders(seed_file, n)
    return compute, orgs


def _obs(compute):
    import torch
    return torch.randn(1, 64, dtype=torch.float32, device=compute.device)


# ── Z7ih.A.1 — empty colony ──────────────────────────────────────────


def test_diagnostics_empty_has_zodchiy_block(seed_file):
    """n=0 ветка содержит zodchiy_sidecar блок с нулями."""
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    d = compute.diagnostics()
    assert "zodchiy_sidecar" in d
    block = d["zodchiy_sidecar"]
    assert block["n_active"] == 0
    assert block["cerebellum_delta_norm_avg"] == 0.0
    assert block["amygdala_valence_avg"] == 0.0
    assert block["episodic_recall_norm_avg"] == 0.0


# ── Z7ih.A.2 — wanderer only ─────────────────────────────────────────


def test_diagnostics_wanderer_only(seed_file):
    """Wanderer (default) → n_active=0, аггрегаты нули."""
    compute, orgs = _make_compute_with(seed_file, 1)
    compute.add_creature("w1", orgs[0])  # default lineage="wanderer"
    compute._compute_higher_tissues("w1", _obs(compute))
    d = compute.diagnostics()
    block = d["zodchiy_sidecar"]
    assert block["n_active"] == 0
    assert block["cerebellum_delta_norm_avg"] == 0.0


# ── Z7ih.A.3 — single zodchiy ────────────────────────────────────────


def test_diagnostics_zodchiy_aggregates(seed_file):
    """Zodchiy после forward → n_active=1, нормы>=0, valence∈[-1,1]."""
    compute, orgs = _make_compute_with(seed_file, 1)
    compute.add_creature("z1", orgs[0], lineage="zodchiy")
    compute._compute_higher_tissues("z1", _obs(compute))
    d = compute.diagnostics()
    block = d["zodchiy_sidecar"]
    assert block["n_active"] == 1
    assert block["cerebellum_delta_norm_avg"] >= 0.0
    assert -1.0 <= block["amygdala_valence_avg"] <= 1.0
    assert block["episodic_recall_norm_avg"] >= 0.0
    # Совпадение с L2 last_cerebellum_delta — average единственной нормы.
    cer_tensor = compute.last_cerebellum_delta["z1"]
    expected = float(cer_tensor.norm().item())
    assert math.isclose(
        block["cerebellum_delta_norm_avg"], round(expected, 4),
        rel_tol=1e-3, abs_tol=1e-3)


# ── Z7ih.A.4 — mixed wanderer + zodchiy ──────────────────────────────


def test_diagnostics_mixed_lineages(seed_file):
    """1 wanderer + 1 zodchiy → n_active=1 (только Зодчий несёт snapshot)."""
    compute, orgs = _make_compute_with(seed_file, 2)
    compute.add_creature("w1", orgs[0])
    compute.add_creature("z1", orgs[1], lineage="zodchiy")
    compute._compute_higher_tissues("w1", _obs(compute))
    compute._compute_higher_tissues("z1", _obs(compute))
    d = compute.diagnostics()
    block = d["zodchiy_sidecar"]
    assert block["n_active"] == 1


# ── Z7ih.A.5 — per-creature wanderer ─────────────────────────────────


def test_per_creature_wanderer_zero_fields(seed_file):
    """Wanderer в _per_creature_stats — 3 Зодчий-поля = 0.0."""
    compute, orgs = _make_compute_with(seed_file, 1)
    compute.add_creature("w1", orgs[0])
    compute._compute_higher_tissues("w1", _obs(compute))
    d = compute.diagnostics()
    creatures = d.get("creatures", [])
    assert len(creatures) == 1
    c = creatures[0]
    assert c["client_cerebellum_delta_norm"] == 0.0
    assert c["client_amygdala_valence"] == 0.0
    assert c["client_episodic_recall_norm"] == 0.0


# ── Z7ih.A.6 — per-creature zodchiy ──────────────────────────────────


def test_per_creature_zodchiy_populated(seed_file):
    """Zodchiy после forward — все 3 поля заполнены, нормы>=0."""
    compute, orgs = _make_compute_with(seed_file, 1)
    compute.add_creature("z1", orgs[0], lineage="zodchiy")
    compute._compute_higher_tissues("z1", _obs(compute))
    d = compute.diagnostics()
    creatures = d.get("creatures", [])
    assert len(creatures) == 1
    c = creatures[0]
    assert c["cid"] == "z1"
    assert c["client_cerebellum_delta_norm"] >= 0.0
    assert -1.0 <= c["client_amygdala_valence"] <= 1.0
    assert c["client_episodic_recall_norm"] >= 0.0
    # Хотя бы одно поле должно быть строго > 0 (forward даёт ненулевой
    # сигнал на случайном obs).
    assert (c["client_cerebellum_delta_norm"] > 0.0
            or c["client_episodic_recall_norm"] > 0.0), \
           "ожидаем хотя бы одну ненулевую норму после Зодчий forward"
