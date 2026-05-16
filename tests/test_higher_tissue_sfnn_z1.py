"""Z1 (Зодчий, 16.05.2026) — унифицированный SFNN apply-step для 7 высших.

Что проверяем:
  - `_higher_tissue_sfnn_update_step` использует ту же S6.5-формулу, что
    `_basic_sfnn_update_step` (τ-eligibility trace + R3-eff + td_coupling).
  - `higher_tissue_sfnn_trace` инициализирован для всех 7 высших тканей
    и накапливает trace при apply-step.
  - lifecycle: после `remove_creature`/`reset_all` trace очищается.
  - apply-step инкрементит `higher_tissue_sfnn_steps`.
  - r_imm_eff с активным w_imm меняет ΔW; trace растёт с decay·prev + new.
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
torch = pytest.importorskip("torch")


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


def _make_compute_with_creature(seed_file, cid="c1"):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature(cid, org, hebbian_enabled=True)
    # Один тик с событием — чтобы forward higher-тканей произошёл и acts
    # были захвачены hooks'ами.
    rng = np.random.default_rng(42)
    compute.handle_tick(
        {cid: rng.normal(size=80).astype(np.float32)},
        events_per_cid={cid: {"ate": 0, "killed": 0,
                                "damage_taken": 0, "delta_energy": 0.0}},
    )
    return compute, org


def test_trace_store_keys_for_all_higher_tissues(seed_file):
    """`higher_tissue_sfnn_trace` инициализирован для всех 7 тканей."""
    from utopia_client.local_compute import _HIGHER_SFNN_TISSUES
    compute, _ = _make_compute_with_creature(seed_file)
    for t in _HIGHER_SFNN_TISSUES:
        assert t in compute.higher_tissue_sfnn_trace


def test_unified_apply_changes_weights_and_steps(seed_file):
    """S6.5-формула меняет веса хотя бы у одной высшей + steps растут."""
    from utopia_client.local_compute import _HIGHER_SFNN_TISSUES
    compute, _ = _make_compute_with_creature(seed_file)
    snapshots: dict[str, torch.Tensor] = {}
    for t in _HIGHER_SFNN_TISSUES:
        tissue = getattr(compute, t, {}).get("c1")
        if tissue is None:
            continue
        for pname, p in tissue.named_parameters():
            if pname.endswith(".weight") and p.dim() == 2:
                snapshots[t] = p.data.clone()
                break
    steps_before = dict(compute.higher_tissue_sfnn_steps)
    for t in _HIGHER_SFNN_TISSUES:
        compute._higher_tissue_sfnn_update_step(
            t, "c1",
            dopa_td_mult=1.2, r_imm_eff=0.3, r_med_eff=0.0, r_long_eff=0.0,
        )
    changed = []
    for t, before in snapshots.items():
        tissue = getattr(compute, t, {}).get("c1")
        if tissue is None:
            continue
        for pname, p in tissue.named_parameters():
            if pname.endswith(".weight") and p.dim() == 2:
                if not torch.equal(before, p.data):
                    changed.append(t)
                break
    assert changed, "ни одна высшая ткань не изменила веса"
    incremented = [t for t in _HIGHER_SFNN_TISSUES
                   if compute.higher_tissue_sfnn_steps[t] > steps_before[t]]
    assert incremented, "higher_tissue_sfnn_steps не выросли"


def test_trace_accumulates_across_steps(seed_file):
    """После двух apply-step trace ткани с активным forward растёт (decay·prev + new)."""
    from utopia_client.local_compute import _HIGHER_SFNN_TISSUES
    compute, _ = _make_compute_with_creature(seed_file)
    for t in _HIGHER_SFNN_TISSUES:
        compute._higher_tissue_sfnn_update_step(t, "c1", r_imm_eff=0.1)
    role_with_trace = None
    synapse_with_trace = None
    for t in _HIGHER_SFNN_TISSUES:
        per_cid = compute.higher_tissue_sfnn_trace.get(t, {}).get("c1")
        if per_cid:
            role_with_trace = t
            synapse_with_trace = next(iter(per_cid.keys()))
            break
    assert role_with_trace is not None, "trace не создан ни для одной ткани"
    t1 = compute.higher_tissue_sfnn_trace[role_with_trace]["c1"][
        synapse_with_trace].clone()
    compute._higher_tissue_sfnn_update_step(
        role_with_trace, "c1", r_imm_eff=0.1)
    t2 = compute.higher_tissue_sfnn_trace[role_with_trace]["c1"][
        synapse_with_trace]
    assert not torch.equal(t1, t2)


def test_remove_creature_clears_trace(seed_file):
    """`remove_creature` чистит higher_tissue_sfnn_trace[cid]."""
    from utopia_client.local_compute import _HIGHER_SFNN_TISSUES
    compute, _ = _make_compute_with_creature(seed_file)
    for t in _HIGHER_SFNN_TISSUES:
        compute._higher_tissue_sfnn_update_step(t, "c1", r_imm_eff=0.1)
    has_trace = any(
        compute.higher_tissue_sfnn_trace.get(t, {}).get("c1")
        for t in _HIGHER_SFNN_TISSUES)
    assert has_trace, "ни одного trace не создалось"
    compute.remove_creature("c1")
    for t in _HIGHER_SFNN_TISSUES:
        assert "c1" not in compute.higher_tissue_sfnn_trace.get(t, {})


def test_reset_all_clears_trace(seed_file):
    """`reset_all` обнуляет higher_tissue_sfnn_trace по всем тканям."""
    from utopia_client.local_compute import _HIGHER_SFNN_TISSUES
    compute, _ = _make_compute_with_creature(seed_file)
    for t in _HIGHER_SFNN_TISSUES:
        compute._higher_tissue_sfnn_update_step(t, "c1", r_imm_eff=0.1)
    compute.reset_all()
    for t in _HIGHER_SFNN_TISSUES:
        assert compute.higher_tissue_sfnn_trace[t] == {}


def test_r_imm_weight_modulates_dw(seed_file):
    """При rule.r_imm_weight>0 r_imm_eff>0 меняет ΔW vs r_imm_eff=0."""
    from utopia_client.local_compute import _HIGHER_SFNN_TISSUES
    compute, _ = _make_compute_with_creature(seed_file)
    # Найдём ткань с активным forward (acts заполнены).
    target_role = None
    for t in _HIGHER_SFNN_TISSUES:
        acts = compute.higher_tissue_sfnn_acts.get(t, {}).get("c1")
        if acts:
            target_role = t
            break
    assert target_role is not None, "ни одной активной высшей ткани"
    # Активируем w_imm у правила (стартует с 0 в ROLE_DEFAULTS).
    rule = compute.higher_tissue_sfnn_rule[target_role]["c1"]
    rule.r_imm_weight = 0.5
    # Снимок весов.
    tissue = compute.__dict__[target_role]["c1"]
    w_before = {n: p.data.clone()
                 for n, p in tissue.named_parameters() if p.dim() == 2}
    # apply с r_imm_eff=0.0.
    compute._higher_tissue_sfnn_update_step(
        target_role, "c1", r_imm_eff=0.0)
    w_zero = {n: p.data.clone()
              for n, p in tissue.named_parameters() if p.dim() == 2}
    # Откат + apply с r_imm_eff=1.0.
    for n, p in tissue.named_parameters():
        if p.dim() == 2:
            p.data.copy_(w_before[n])
    # Очистим trace, иначе вторая ветка получит унаследованную историю.
    compute.higher_tissue_sfnn_trace[target_role]["c1"] = {}
    compute._higher_tissue_sfnn_update_step(
        target_role, "c1", r_imm_eff=1.0)
    w_one = {n: p.data.clone()
              for n, p in tissue.named_parameters() if p.dim() == 2}
    diff_any = any(not torch.equal(w_zero[n], w_one[n]) for n in w_zero)
    assert diff_any, "r_imm_weight>0 не модулирует ΔW"
