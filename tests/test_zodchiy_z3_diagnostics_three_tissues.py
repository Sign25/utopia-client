"""Z3.B (Зодчий, 17.05.2026) — диагностика per-tissue для 3 Зодчий-тканей.

Что проверяем:
  - `diagnostics()` (n=0 stub): `higher_sfnn` содержит ключи всех 7 высших +
    3 Зодчий-тканей со {steps_total, eta_avg, A_avg}.
  - `diagnostics()` после `add_creature(lineage="zodchiy")` + forward +
    apply-step: `higher_sfnn["cerebellum"]["steps_total"] > 0` и
    `eta_avg > 0`, `A_avg > 0` (правило из ROLE_DEFAULTS через for_role).
  - Для `lineage="wanderer"` cid'а: ключи присутствуют, но steps=0,
    eta=0.0, A=0.0 (правило не создано в higher_tissue_sfnn_rule[t]).
  - `higher_sfnn["enabled_pct"]` корректно учитывает только колонию
    (1.0 если все живые имеют флаг, 0.0 если нет).

Z3.A (apply-step + counters) полностью покрыт
`test_zodchiy_z7id_local_zodchiy_tissues.py::test_zodchiy_apply_step_increments_counters`.
Здесь проверяем именно агрегацию в `diagnostics`, без которой /api/admin/
не покажет cerebellum/amygdala/episodic.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("storage.norg")
pytest.importorskip("core.workbench")
torch = pytest.importorskip("torch")


_ZODCHIY_KEYS = ("cerebellum", "amygdala", "episodic")
_HIGHER_KEYS = (
    "dopamine", "imagination", "planner", "insula",
    "default_mode", "theory_of_mind", "language",
)


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


def test_diag_empty_stub_includes_all_10_higher_keys(seed_file):
    """n=0 stub: higher_sfnn содержит все 7+3 ключа со steps/eta/A."""
    compute, _ = _make_compute(seed_file)
    diag = compute.diagnostics()
    higher = diag["higher_sfnn"]
    for k in _HIGHER_KEYS + _ZODCHIY_KEYS:
        assert k in higher, f"empty diag missing '{k}'"
        block = higher[k]
        assert "steps_total" in block
        assert "eta_avg" in block
        assert "A_avg" in block
        assert block["steps_total"] == 0
        assert block["eta_avg"] == 0.0
        assert block["A_avg"] == 0.0


def test_diag_zodchiy_creature_populates_three_tissues(seed_file):
    """Zodchiy cid + forward + apply-step → steps_total > 0, eta/A > 0."""
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    obs = torch.randn(1, 64, dtype=torch.float32, device=compute.device)
    compute._compute_higher_tissues("c1", obs)
    for t in _ZODCHIY_KEYS:
        compute._higher_tissue_sfnn_update_step(
            t, "c1", dopa_td_mult=1.0,
            r_imm_eff=0.1, r_med_eff=0.0, r_long_eff=0.0)
    diag = compute.diagnostics()
    higher = diag["higher_sfnn"]
    for t in _ZODCHIY_KEYS:
        assert t in higher, f"diag missing '{t}'"
        block = higher[t]
        assert block["steps_total"] >= 1, \
            f"{t}: steps_total {block['steps_total']} not > 0"
        # ROLE_DEFAULTS через for_role → A=1.0, η=1e-3 для всех 3-х.
        assert block["eta_avg"] > 0.0, f"{t}: eta_avg=0 (правило не подцепилось)"
        assert block["A_avg"] > 0.0, f"{t}: A_avg=0 (правило не подцепилось)"


def test_diag_wanderer_zodchiy_keys_zero(seed_file):
    """Wanderer cid: 3 Зодчий-ключа присутствуют, но всё нулями."""
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="wanderer")
    diag = compute.diagnostics()
    higher = diag["higher_sfnn"]
    for t in _ZODCHIY_KEYS:
        assert t in higher, f"wanderer diag missing key '{t}'"
        block = higher[t]
        # rule_store пуст для wanderer → etas/As пустые списки.
        assert block["eta_avg"] == 0.0
        assert block["A_avg"] == 0.0
        # steps_total — глобальный счётчик ткани; до forward = 0.
        assert block["steps_total"] == 0


def test_diag_higher_keys_always_present_after_zodchiy(seed_file):
    """Z3.B regression: 7 высших ключей не исчезли при расширении."""
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    diag = compute.diagnostics()
    higher = diag["higher_sfnn"]
    for k in _HIGHER_KEYS:
        assert k in higher, f"higher key '{k}' lost after Z3.B expansion"


def test_diag_enabled_pct_for_zodchiy_default_on(seed_file):
    """_higher_sfnn_default=True → enabled_pct=1.0 после add_creature."""
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    diag = compute.diagnostics()
    assert diag["higher_sfnn"]["enabled_pct"] == 1.0
