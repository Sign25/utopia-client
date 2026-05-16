"""SFNN S6.8 — diagnostics dict для 10 базовых тканей.

Что проверяем:
  - diagnostics() возвращает basic_sfnn dict с enabled_pct + 10 ролями
  - каждая роль имеет {steps_total, eta_avg, A_avg, tau_avg, w_norm_avg}
  - enabled_pct = доля особей с genome.basic_tissue_sfnn_enabled=True
  - w_norm_avg > 0 для активных тканей (Frobenius ‖W_sub‖ из cell)
  - zero-alive branch тоже возвращает basic_sfnn (default 0.0)
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
pytest.importorskip("torch")


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


_BASIC = ("sensory", "attention", "brain", "memory", "consciousness",
           "communication", "motor", "manipulator", "digestive", "immune")


def test_diagnostics_includes_basic_sfnn_block(seed_file):
    """diagnostics() содержит basic_sfnn dict с enabled_pct + 10 ролей."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)
    d = compute.diagnostics()
    assert "basic_sfnn" in d
    block = d["basic_sfnn"]
    assert "enabled_pct" in block
    for role in _BASIC:
        assert role in block, f"role {role} missing"
        roleinfo = block[role]
        for k in ("steps_total", "eta_avg", "A_avg", "tau_avg", "w_norm_avg"):
            assert k in roleinfo, f"role {role} missing key {k}"


def test_diagnostics_basic_sfnn_enabled_pct(seed_file):
    """enabled_pct отражает долю особей с флагом ON."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org1 = load_founders(seed_file, 1)[0]
    org2 = load_founders(seed_file, 1)[0]
    compute.set_basic_sfnn(False)  # S6.11: дефолт True, сбрасываем для теста
    compute.add_creature("c1", org1, hebbian_enabled=True)
    compute.add_creature("c2", org2, hebbian_enabled=True)
    # После сброса флаг False у всех → enabled_pct = 0.
    d = compute.diagnostics()
    assert d["basic_sfnn"]["enabled_pct"] == pytest.approx(0.0)
    # Включаем флаг у одной → 0.5.
    org1.genome.basic_tissue_sfnn_enabled = True
    d = compute.diagnostics()
    assert d["basic_sfnn"]["enabled_pct"] == pytest.approx(0.5)


def test_diagnostics_basic_sfnn_w_norm_positive(seed_file):
    """w_norm_avg > 0 для тканей с инициализированными весами."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)
    d = compute.diagnostics()
    block = d["basic_sfnn"]
    # Хотя бы одна базовая ткань должна иметь w_norm > 0.
    positive_w = [r for r in _BASIC if block[r]["w_norm_avg"] > 0.0]
    assert positive_w, "ни одна базовая ткань не имеет w_norm > 0"


def test_diagnostics_basic_sfnn_tau_matches_role_defaults(seed_file):
    """tau_avg совпадает с ROLE_DEFAULTS для свежей особи."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    from core.sfnn_rule import ROLE_DEFAULTS
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)
    d = compute.diagnostics()
    block = d["basic_sfnn"]
    for role in _BASIC:
        defaults = ROLE_DEFAULTS[role]
        # ROLE_DEFAULTS[role][0] = tau (по порядку: tau, r_imm, r_med, r_long, td, eta)
        assert block[role]["tau_avg"] == pytest.approx(defaults[0])


def test_diagnostics_basic_sfnn_steps_after_handle_tick(seed_file):
    """basic_sfnn.<role>.steps_total растёт после handle_tick с флагом ON."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)
    org.genome.basic_tissue_sfnn_enabled = True
    rng = np.random.default_rng(0)
    compute.handle_tick(
        {"c1": rng.normal(size=80).astype(np.float32)},
        events_per_cid={"c1": {"ate": 0, "killed": 0,
                                "damage_taken": 0, "delta_energy": 0.0}},
    )
    d = compute.diagnostics()
    block = d["basic_sfnn"]
    incremented = [r for r in _BASIC if block[r]["steps_total"] > 0]
    assert incremented, f"steps_total=0 для всех 10 ролей: {block}"


def test_diagnostics_zero_alive_basic_sfnn_present(seed_file):
    """Пустая колония (n_alive=0) — basic_sfnn dict всё равно вернулся."""
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    d = compute.diagnostics()
    assert d["n_alive"] == 0
    assert "basic_sfnn" in d
    block = d["basic_sfnn"]
    assert block["enabled_pct"] == 0.0
    for role in _BASIC:
        assert role in block
        assert block[role]["steps_total"] == 0
