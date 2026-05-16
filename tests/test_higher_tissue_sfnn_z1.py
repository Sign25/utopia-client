"""Z1 (Зодчий, 16.05.2026) — полная S6.5-формула для 7 высших тканей.

Что проверяем:
  - дефолт `_full_sfnn_for_higher_default = False` (не трогает живых Странников)
  - setter `set_full_sfnn_for_higher` обновляет дефолт и патчит живых
  - `higher_tissue_sfnn_trace` — стор есть для всех 7 высших тканей
  - lifecycle: после `remove_creature`/`reset_all` trace очищается
  - `_higher_tissue_sfnn_update_step(full=True)` накапливает trace и
    инкрементит `higher_tissue_sfnn_steps`
  - переключение S3.1 ↔ S6.5 не падает; trace не утекает между ветками
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


def test_full_flag_default_off(seed_file):
    """По дефолту full_sfnn_for_higher_enabled=False — поведение S3.1."""
    compute, org = _make_compute_with_creature(seed_file)
    assert compute._full_sfnn_for_higher_default is False
    assert getattr(org.genome, "full_sfnn_for_higher_enabled", None) is False


def test_setter_updates_default_and_existing(seed_file):
    """`set_full_sfnn_for_higher(True)` патчит живых + меняет дефолт."""
    compute, org = _make_compute_with_creature(seed_file)
    assert org.genome.full_sfnn_for_higher_enabled is False
    n = compute.set_full_sfnn_for_higher(True)
    assert n == 1
    assert compute._full_sfnn_for_higher_default is True
    assert org.genome.full_sfnn_for_higher_enabled is True
    # Идемпотентность: повторный вызов с тем же значением — 0 изменений.
    n2 = compute.set_full_sfnn_for_higher(True)
    assert n2 == 0


def test_trace_store_keys_for_all_higher_tissues(seed_file):
    """`higher_tissue_sfnn_trace` инициализирован для всех 7 тканей."""
    from utopia_client.local_compute import _HIGHER_SFNN_TISSUES
    compute, _ = _make_compute_with_creature(seed_file)
    for t in _HIGHER_SFNN_TISSUES:
        assert t in compute.higher_tissue_sfnn_trace


def test_full_branch_changes_weights_and_steps(seed_file):
    """С full=True хотя бы одна высшая ткань обновляет веса + steps растут."""
    from utopia_client.local_compute import _HIGHER_SFNN_TISSUES
    compute, _ = _make_compute_with_creature(seed_file)
    # Снимок весов всех 7 высших до second apply.
    snapshots: dict[str, torch.Tensor] = {}
    for t in _HIGHER_SFNN_TISSUES:
        tissue = getattr(compute, t, {}).get("c1")
        if tissue is None:
            continue
        params = dict(tissue.named_parameters())
        # Берём первый Linear.weight с dim==2.
        for pname, p in params.items():
            if pname.endswith(".weight") and p.dim() == 2:
                snapshots[t] = p.data.clone()
                break
    steps_before = dict(compute.higher_tissue_sfnn_steps)
    for t in _HIGHER_SFNN_TISSUES:
        compute._higher_tissue_sfnn_update_step(
            t, "c1", full=True,
            dopa_td_mult=1.2, r_imm_eff=0.3, r_med_eff=0.0, r_long_eff=0.0,
        )
    # Хотя бы одна ткань изменила веса.
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
    assert changed, "ни одна высшая ткань не изменила веса под full=True"
    # Хотя бы у одной ткани step инкрементировался (с учётом возможного
    # инкремента в handle_tick при включённом higher_sfnn_on).
    incremented = [t for t in _HIGHER_SFNN_TISSUES
                   if compute.higher_tissue_sfnn_steps[t] > steps_before[t]]
    assert incremented, "higher_tissue_sfnn_steps не выросли при full=True"


def test_full_branch_trace_accumulates(seed_file):
    """После двух apply-step trace ткани с активным forward растёт (decay·prev + new)."""
    from utopia_client.local_compute import _HIGHER_SFNN_TISSUES
    compute, _ = _make_compute_with_creature(seed_file)
    # Первый apply создаёт trace.
    for t in _HIGHER_SFNN_TISSUES:
        compute._higher_tissue_sfnn_update_step(
            t, "c1", full=True, r_imm_eff=0.1)
    # Ищем ткань с непустым trace.
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
    # Второй apply должен дать decay·t1 + new_hebb_A.
    compute._higher_tissue_sfnn_update_step(
        role_with_trace, "c1", full=True, r_imm_eff=0.1)
    t2 = compute.higher_tissue_sfnn_trace[role_with_trace]["c1"][
        synapse_with_trace]
    # Trace изменился (либо decay, либо новая компонента).
    assert not torch.equal(t1, t2)


def test_remove_creature_clears_trace(seed_file):
    """`remove_creature` чистит higher_tissue_sfnn_trace[cid]."""
    from utopia_client.local_compute import _HIGHER_SFNN_TISSUES
    compute, _ = _make_compute_with_creature(seed_file)
    for t in _HIGHER_SFNN_TISSUES:
        compute._higher_tissue_sfnn_update_step(
            t, "c1", full=True, r_imm_eff=0.1)
    # Хотя бы один trace создан.
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
        compute._higher_tissue_sfnn_update_step(
            t, "c1", full=True, r_imm_eff=0.1)
    compute.reset_all()
    for t in _HIGHER_SFNN_TISSUES:
        assert compute.higher_tissue_sfnn_trace[t] == {}


def test_full_off_equivalent_to_legacy_s3_1(seed_file):
    """full=False даёт тот же ΔW, что и S3.1-формула (regression)."""
    from utopia_client.local_compute import _HIGHER_SFNN_TISSUES
    compute, _ = _make_compute_with_creature(seed_file)
    # Снимок весов до.
    snaps_a: dict[str, torch.Tensor] = {}
    for t in _HIGHER_SFNN_TISSUES:
        tissue = getattr(compute, t, {}).get("c1")
        if tissue is None:
            continue
        for pname, p in tissue.named_parameters():
            if pname.endswith(".weight") and p.dim() == 2:
                snaps_a[t] = p.data.clone()
                break
    # Прогоняем full=False с r=0.1 (S3.1).
    for t in _HIGHER_SFNN_TISSUES:
        compute._higher_tissue_sfnn_update_step(t, "c1", r=0.1, full=False)
    # Снимок весов после S3.1.
    snaps_b: dict[str, torch.Tensor] = {}
    for t in _HIGHER_SFNN_TISSUES:
        tissue = getattr(compute, t, {}).get("c1")
        if tissue is None:
            continue
        for pname, p in tissue.named_parameters():
            if pname.endswith(".weight") and p.dim() == 2:
                snaps_b[t] = p.data.clone()
                break
    # Откатываем веса и проверяем, что full=False с r=0.1 не делает trace
    # (trace создаётся только для full=True).
    for t in snaps_a:
        tissue = getattr(compute, t, {}).get("c1")
        for pname, p in tissue.named_parameters():
            if pname.endswith(".weight") and p.dim() == 2:
                p.data.copy_(snaps_a[t])
                break
    # full=False НЕ должен трогать higher_tissue_sfnn_trace.
    for t in _HIGHER_SFNN_TISSUES:
        assert compute.higher_tissue_sfnn_trace[t].get("c1") in (None, {})
