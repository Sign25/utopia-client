"""motor_policy базовые тесты (после SFNN S4 — Adam/REINFORCE удалены).

После S4 motor_policy обучается локальным правилом пластичности SFNN.
Этот файл оставлен для проверки базовых инвариантов motor_policy:
  - get_phase_emas включает client_motor_delta
  - extract_brain_state_dicts содержит motor_policy
  - inherit_brain_y50 копирует motor_policy в child

REINFORCE-специфичные тесты (M7.1-M7.3, M7.7) удалены в S4.
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


def _compute(seed_file, cid="m1"):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature(cid, org, hebbian_enabled=True)
    return compute, org


def test_add_creature_makes_motor_policy(seed_file):
    """motor_policy создаётся, Adam отсутствует (S4)."""
    compute, _ = _compute(seed_file)
    assert "m1" in compute.motor_policy
    assert compute.motor_policy["m1"] is not None
    assert not hasattr(compute, "motor_policy_opt"), \
        "motor_policy_opt должен быть удалён в S4"


def test_get_phase_emas_includes_client_motor_delta(seed_file):
    compute, _ = _compute(seed_file)
    rng = np.random.default_rng(42)
    compute.handle_tick({"m1": rng.normal(size=80).astype(np.float32)})
    snap = compute.get_phase_emas("m1")
    assert snap is not None
    assert "client_motor_delta" in snap
    vals = snap["client_motor_delta"]
    assert isinstance(vals, list)
    assert len(vals) == 16
    for v in vals:
        assert isinstance(v, float)
        assert -1.0 - 1e-6 <= v <= 1.0 + 1e-6


def test_extract_brain_state_dicts_includes_motor_policy(seed_file):
    compute, _ = _compute(seed_file)
    brain, _ = compute.extract_brain_state_dicts("m1")
    assert "motor_policy" in brain
    sd = brain["motor_policy"]
    assert isinstance(sd, dict) and len(sd) > 0


def test_inherit_brain_y50_copies_motor_policy(seed_file):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    parent, child = load_founders(seed_file, 2)
    compute.add_creature("p1", parent)
    compute.add_creature("c1", child)
    parent_params = [p.detach().clone()
                      for p in compute.motor_policy["p1"].parameters()]
    ok = compute.inherit_brain_y50("p1", "c1")
    assert ok
    child_params = list(compute.motor_policy["c1"].parameters())
    diff = sum((p - c).abs().sum().item()
                for p, c in zip(parent_params, child_params))
    assert diff > 0.0, "Y50 должен внести шум в motor_policy ребёнка"


def test_remove_creature_clears_motor_policy(seed_file):
    """После remove_creature motor_policy и SFNN-state очищены."""
    compute, _ = _compute(seed_file)
    rng = np.random.default_rng(0)
    compute.handle_tick({"m1": rng.normal(size=80).astype(np.float32)})
    assert "m1" in compute.motor_policy
    assert "m1" in compute.motor_sfnn_rule
    compute.remove_creature("m1")
    assert "m1" not in compute.motor_policy
    assert "m1" not in compute.motor_sfnn_rule
    assert "m1" not in compute.last_motor_delta
