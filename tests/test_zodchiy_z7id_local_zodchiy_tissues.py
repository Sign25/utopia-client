"""Z7.i.d/Z7.i.e (Зодчий, 16.05.2026) — клиентский plumbing уникальных тканей.

Что проверяем:
  Z7.i.d (plumbing):
  - `add_creature(lineage="zodchiy")` создаёт 3 sidecar-ткани
    (cerebellum/amygdala/episodic).
  - `add_creature(lineage="wanderer")` (default) НЕ создаёт ни одной из
    Зодчий-тканей — обратная совместимость не нарушена.
  - SFNNRule.for_role() инициализируется в `higher_tissue_sfnn_rule` для
    каждой из трёх тканей (Z7.i.e: unified storage, не `zodchiy_extra_*`).
  - `remove_creature` чистит все три store'а + sfnn_rule.
  - `reset_all` обнуляет step-counter и зачищает sfnn_rule по всем тканям.
  - SFNNRule.for_role() даёт ROLE_DEFAULTS значения (τ/R3/TD/algorithm).

  Z7.i.e (forward + apply-step):
  - `_compute_higher_tissues` после Zodchiy add_creature пишет
    last_cerebellum_delta/last_amygdala_valence/last_episodic_recall.
  - Forward-hooks `higher_tissue_sfnn_acts` для 3 тканей заполняются
    после forward (≥1 synapse_type на ткань).
  - `_higher_tissue_sfnn_update_step` инкрементит
    higher_tissue_sfnn_steps[t] и накапливает trace в
    higher_tissue_sfnn_trace[t][cid].
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
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


def _make_compute(seed_file):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    return compute, org


def test_wanderer_default_has_no_zodchiy_tissues(seed_file):
    """Default lineage="wanderer" → ни одной Зодчий-ткани."""
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org)  # default
    assert compute.cerebellum.get("c1") is None
    assert compute.amygdala.get("c1") is None
    assert compute.episodic.get("c1") is None


def test_zodchiy_lineage_creates_three_tissues(seed_file):
    """lineage="zodchiy" → cerebellum/amygdala/episodic созданы."""
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    assert compute.cerebellum.get("c1") is not None
    assert compute.amygdala.get("c1") is not None
    assert compute.episodic.get("c1") is not None


def test_zodchiy_sfnn_rule_populated(seed_file):
    """Z7.i.e: SFNNRule.for_role() в `higher_tissue_sfnn_rule` (unified)."""
    from utopia_client.local_compute import _ZODCHIY_EXTRA_TISSUES
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    for t in _ZODCHIY_EXTRA_TISSUES:
        assert "c1" in compute.higher_tissue_sfnn_rule[t]
        rule = compute.higher_tissue_sfnn_rule[t]["c1"]
        assert hasattr(rule, "coeffs") and rule.coeffs
        any_syn = next(iter(rule.coeffs.values()))
        for attr in ("A", "B", "C", "D", "eta"):
            assert hasattr(any_syn, attr), f"{t}: SFNNSynapseCoeffs missing {attr}"
        for attr in ("tau", "td_coupling"):
            assert hasattr(rule, attr), f"{t}: SFNNRule missing {attr}"


def test_zodchiy_sfnn_rule_not_created_for_wanderer(seed_file):
    """Wanderer не получает Zodchiy-sfnn-rule (но dict-ключи tissues живут)."""
    from utopia_client.local_compute import _ZODCHIY_EXTRA_TISSUES
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org)
    for t in _ZODCHIY_EXTRA_TISSUES:
        # Хранилище инициализировано ({}) в __init__, но cid в нём нет.
        assert "c1" not in compute.higher_tissue_sfnn_rule[t]


def test_remove_creature_clears_zodchiy_state(seed_file):
    """remove_creature вычищает 3 sidecar + sfnn_rule по всем тканям."""
    from utopia_client.local_compute import _ZODCHIY_EXTRA_TISSUES
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    compute.remove_creature("c1")
    assert compute.cerebellum.get("c1") is None
    assert compute.amygdala.get("c1") is None
    assert compute.episodic.get("c1") is None
    for t in _ZODCHIY_EXTRA_TISSUES:
        assert "c1" not in compute.higher_tissue_sfnn_rule[t]


def test_reset_all_clears_zodchiy_counters(seed_file):
    """reset_all обнуляет step-counter Зодчий-тканей в unified-storage."""
    from utopia_client.local_compute import _ZODCHIY_EXTRA_TISSUES
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    for t in _ZODCHIY_EXTRA_TISSUES:
        compute.higher_tissue_sfnn_steps[t] = 42
    compute.reset_all()
    for t in _ZODCHIY_EXTRA_TISSUES:
        assert compute.higher_tissue_sfnn_steps[t] == 0
        # rule очищается только через remove_creature (reset_all chain).
        assert compute.higher_tissue_sfnn_rule[t] == {}


def test_zodchiy_extra_tissues_const_is_three(seed_file):
    """Хранители-constant: 3 ткани, имена точны (исключает опечатки)."""
    from utopia_client.local_compute import _ZODCHIY_EXTRA_TISSUES
    assert _ZODCHIY_EXTRA_TISSUES == ("cerebellum", "amygdala", "episodic")


def test_zodchiy_sfnn_rule_uses_for_role_defaults(seed_file):
    """Z7.i.d.1: SFNNRule.for_role применён, R3/TD ненулевые для всех 3-х."""
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    cer = compute.higher_tissue_sfnn_rule["cerebellum"]["c1"]
    assert abs(cer.tau - 21.0) < 1e-6
    assert abs(cer.r_imm_weight - 0.7) < 1e-6
    assert abs(cer.td_coupling - 1.0) < 1e-6
    assert cer.algorithm == "reward_output"
    amy = compute.higher_tissue_sfnn_rule["amygdala"]["c1"]
    assert abs(amy.tau - 233.0) < 1e-6
    assert abs(amy.r_med_weight - 0.6) < 1e-6
    assert abs(amy.td_coupling - 1.0) < 1e-6
    assert amy.algorithm == "reward_output"
    epi = compute.higher_tissue_sfnn_rule["episodic"]["c1"]
    assert abs(epi.tau - 233.0) < 1e-6
    assert abs(epi.r_long_weight - 0.6) < 1e-6
    assert abs(epi.td_coupling - 0.0) < 1e-6
    assert epi.algorithm == "oja_input"


# ── Z7.i.e: forward + apply-step ──────────────────────────────────────────


def _make_obs_tensor(compute):
    import torch
    return torch.randn(1, 64, dtype=torch.float32, device=compute.device)


def test_zodchiy_forward_populates_last_snapshots(seed_file):
    """Z7.i.e: после _compute_higher_tissues snapshots должны заполниться."""
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    obs = _make_obs_tensor(compute)
    compute._compute_higher_tissues("c1", obs)
    # cerebellum: torch.Tensor [16]
    cer = compute.last_cerebellum_delta.get("c1")
    assert cer is not None and cer.numel() == 16
    # amygdala: float в [-1, 1]
    amy = compute.last_amygdala_valence.get("c1")
    assert amy is not None and -1.0 <= float(amy) <= 1.0
    # episodic: torch.Tensor [64]
    epi = compute.last_episodic_recall.get("c1")
    assert epi is not None and epi.numel() == 64


def test_zodchiy_forward_noop_for_wanderer(seed_file):
    """Wanderer: forward не пишет в zodchiy-snapshot store'ы."""
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org)  # default lineage
    obs = _make_obs_tensor(compute)
    compute._compute_higher_tissues("c1", obs)
    assert compute.last_cerebellum_delta.get("c1") is None
    assert compute.last_amygdala_valence.get("c1") is None
    assert compute.last_episodic_recall.get("c1") is None


def test_zodchiy_forward_hooks_populate_acts(seed_file):
    """Z7.i.e: forward-hooks заполняют higher_tissue_sfnn_acts для 3 тканей."""
    from utopia_client.local_compute import _ZODCHIY_EXTRA_TISSUES
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    obs = _make_obs_tensor(compute)
    compute._compute_higher_tissues("c1", obs)
    for t in _ZODCHIY_EXTRA_TISSUES:
        acts = compute.higher_tissue_sfnn_acts.get(t, {}).get("c1")
        assert acts is not None, f"{t}: acts dict not created"
        # Хотя бы один synapse-тип должен быть заполнен после forward.
        assert len(acts) > 0, f"{t}: no synapse activations captured"


def test_zodchiy_apply_step_increments_counters(seed_file):
    """Z7.i.e: после forward + apply-step счётчики растут, trace накапливается."""
    from utopia_client.local_compute import _ZODCHIY_EXTRA_TISSUES
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    obs = _make_obs_tensor(compute)
    # Forward (заполняет acts через hooks).
    compute._compute_higher_tissues("c1", obs)
    # Прямой вызов apply-step (вне handle_tick'а, чтобы не зависеть от
    # mate-step pipeline). r_imm_eff=0.1 для нетривиальной формулы.
    for t in _ZODCHIY_EXTRA_TISSUES:
        before = compute.higher_tissue_sfnn_steps[t]
        compute._higher_tissue_sfnn_update_step(
            t, "c1", dopa_td_mult=1.0,
            r_imm_eff=0.1, r_med_eff=0.0, r_long_eff=0.0)
        after = compute.higher_tissue_sfnn_steps[t]
        assert after == before + 1, f"{t}: counter did not increment"
        # Trace должен быть создан хотя бы для одного synapse.
        trace = compute.higher_tissue_sfnn_trace[t].get("c1")
        assert trace is not None and len(trace) > 0, \
            f"{t}: trace not populated"


def test_zodchiy_apply_step_noop_without_acts(seed_file):
    """Z7.i.e: без forward (no acts) apply-step делает early return."""
    from utopia_client.local_compute import _ZODCHIY_EXTRA_TISSUES
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    # БЕЗ forward — acts пусты.
    for t in _ZODCHIY_EXTRA_TISSUES:
        before = compute.higher_tissue_sfnn_steps[t]
        compute._higher_tissue_sfnn_update_step(
            t, "c1", dopa_td_mult=1.0,
            r_imm_eff=0.0, r_med_eff=0.0, r_long_eff=0.0)
        after = compute.higher_tissue_sfnn_steps[t]
        assert after == before, f"{t}: step ran without acts (should no-op)"


def test_zodchiy_remove_clears_hooks_and_trace(seed_file):
    """Z7.i.e: remove_creature снимает hooks и стирает trace."""
    from utopia_client.local_compute import _ZODCHIY_EXTRA_TISSUES
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    obs = _make_obs_tensor(compute)
    compute._compute_higher_tissues("c1", obs)
    compute._higher_tissue_sfnn_update_step(
        "cerebellum", "c1", r_imm_eff=0.1)
    compute.remove_creature("c1")
    for t in _ZODCHIY_EXTRA_TISSUES:
        assert "c1" not in compute.higher_tissue_sfnn_acts.get(t, {})
        assert "c1" not in compute.higher_tissue_sfnn_hook_handles.get(t, {})
        assert "c1" not in compute.higher_tissue_sfnn_trace.get(t, {})
        assert "c1" not in compute.higher_tissue_sfnn_row_norms.get(t, {})
