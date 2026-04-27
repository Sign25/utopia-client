"""Phase F3.2.b — Hebbian update на клиенте по локальному R3 reward."""
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
    client_seed = tmp_path / "client_seed.norg"
    client_seed.write_bytes(seed_path.read_bytes())

    from utopia_client import seed_loader as cli_loader
    importlib.reload(cli_loader)
    return client_seed


# ── Reward computation: формула R3 immediate ─────────────────────────

def test_reward_zero_event():
    from utopia_client.local_compute import LocalColonyCompute
    r = LocalColonyCompute._compute_immediate_reward({})
    assert r == 0.0


def test_reward_ate_only():
    from utopia_client.local_compute import LocalColonyCompute
    r = LocalColonyCompute._compute_immediate_reward(
        {"ate": True, "delta_energy": 3.0})
    # 3.0*0.05 + 1.0 = 1.15
    assert abs(r - 1.15) < 1e-6


def test_reward_killed_only():
    from utopia_client.local_compute import LocalColonyCompute
    r = LocalColonyCompute._compute_immediate_reward(
        {"killed": True, "delta_energy": 0.0})
    assert abs(r - 5.0) < 1e-6


def test_reward_damage_only():
    from utopia_client.local_compute import LocalColonyCompute
    r = LocalColonyCompute._compute_immediate_reward(
        {"damage_taken": 4.0, "delta_energy": -4.0})
    # -4.0*0.05 - 4.0*0.1 = -0.2 - 0.4 = -0.6
    assert abs(r - (-0.6)) < 1e-6


def test_reward_ate_and_killed():
    from utopia_client.local_compute import LocalColonyCompute
    r = LocalColonyCompute._compute_immediate_reward(
        {"ate": True, "killed": True, "delta_energy": 5.0})
    # 5.0*0.05 + 1.0 + 5.0 = 6.25
    assert abs(r - 6.25) < 1e-6


def test_reward_metabolism_only():
    """Норм метаболизм (-0.05/тик) → ~0 reward (близко к baseline)."""
    from utopia_client.local_compute import LocalColonyCompute
    r = LocalColonyCompute._compute_immediate_reward(
        {"delta_energy": -0.05})
    assert abs(r - (-0.0025)) < 1e-6


# ── Integration: handle_tick c events → heb.update вызывается ───────

def test_handle_tick_calls_heb_update(seed_file):
    """С events_per_cid → heb.update вызывается, total_updates растёт."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)
    heb = compute.hebbian["c1"]
    assert heb is not None
    updates_before = heb.total_updates

    obs = {"c1": np.zeros(80, dtype=np.float32)}
    events = {"c1": {"ate": True, "killed": False,
                      "damage_taken": 0.0, "delta_energy": 2.0}}
    actions = compute.handle_tick(obs, events_per_cid=events)

    assert "c1" in actions
    assert heb.total_updates == updates_before + 1
    assert compute.hebbian_updates == 1


def test_handle_tick_no_events_no_update(seed_file):
    """Без events_per_cid → heb.update НЕ вызывается (старое поведение)."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)
    heb = compute.hebbian["c1"]
    updates_before = heb.total_updates

    obs = {"c1": np.zeros(80, dtype=np.float32)}
    compute.handle_tick(obs)  # без events
    assert heb.total_updates == updates_before
    assert compute.hebbian_updates == 0


def test_handle_tick_hebbian_disabled_skips(seed_file):
    """hebbian_enabled=False → controller None → update пропускается."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=False)
    assert compute.hebbian["c1"] is None

    obs = {"c1": np.zeros(80, dtype=np.float32)}
    events = {"c1": {"ate": True, "delta_energy": 2.0}}
    compute.handle_tick(obs, events_per_cid=events)
    assert compute.hebbian_updates == 0


def test_handle_tick_event_missing_for_cid_skips_only_that(seed_file):
    """Если event только для одного из двух cid — другой получает action, но без update."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    compute = LocalColonyCompute(device="cpu")
    for cid in ("ca", "cb"):
        org = load_founders(seed_file, 1)[0]
        compute.add_creature(cid, org, hebbian_enabled=True)

    obs = {"ca": np.zeros(80, dtype=np.float32),
           "cb": np.zeros(80, dtype=np.float32)}
    events = {"ca": {"ate": True, "delta_energy": 3.0}}  # cb отсутствует
    actions = compute.handle_tick(obs, events_per_cid=events)
    assert set(actions.keys()) == {"ca", "cb"}
    # ca получил update, cb — нет
    assert compute.hebbian_updates == 1


def test_handle_tick_weights_change_after_update(seed_file):
    """Веса tissues изменяются после нескольких heb.update с positive reward."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    import torch

    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)

    # Снэпшот весов первой ткани
    first_tid = next(iter(org.tissues.keys()))
    w_before = org.tissues[first_tid].state_dict()
    w_before_clone = {k: v.clone() for k, v in w_before.items()}

    obs = {"c1": np.random.randn(80).astype(np.float32) * 0.1}
    events = {"c1": {"ate": True, "killed": False,
                      "damage_taken": 0.0, "delta_energy": 2.0}}
    for _ in range(20):
        compute.handle_tick(obs, events_per_cid=events)

    w_after = org.tissues[first_tid].state_dict()
    # Хотя бы один параметр должен заметно сдвинуться (oja или reward update).
    diffs = []
    for k in w_after:
        if k in w_before_clone:
            d = (w_after[k] - w_before_clone[k]).abs().sum().item()
            diffs.append(d)
    assert any(d > 1e-6 for d in diffs), \
        f"После 20 Hebbian updates ни один tissue param не изменился: {diffs}"
