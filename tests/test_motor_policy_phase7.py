"""Phase 7 — motor_policy REINFORCE sidecar (utopia-client сторона).

Бит 11 высших тканей (только wanderer). На клиенте:
  - motor_policy Tissue 21/3/1 + Adam(lr=1e-4) создаются в add_creature
  - handle_tick: forward с grad → motor_delta [-1, 1]^16 → combined logits →
    select → log_prob выбранного действия → pending. На следующем тике —
    REINFORCE update: loss = -log_prob · (r_imm_total - intrinsic_ema).
  - get_phase_emas включает `client_motor_delta` (16 floats).
  - extract_brain_state_dicts / inherit_brain_y50 / save_state хранят
    motor_policy state_dict с тем же ключом 'motor_policy'.

Покрытие:
  M7.1  add_creature создаёт motor_policy + Adam opt
  M7.2  handle_tick: первый тик → pending_log_prob записан, REINFORCE step=0
  M7.3  handle_tick: второй тик с events → REINFORCE step выполнен,
        pending очищен, motor_reinforce_steps_total > 0
  M7.4  get_phase_emas включает client_motor_delta после tick'а
  M7.5  extract_brain_state_dicts содержит motor_policy
  M7.6  inherit_brain_y50 копирует motor_policy в child
  M7.7  remove_creature чистит pending_log_prob / pending_action
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


# ── M7.1: add_creature создаёт motor_policy + opt ──────────────────────

def test_add_creature_makes_motor_policy(seed_file):
    compute, _ = _compute(seed_file)
    assert "m1" in compute.motor_policy
    assert compute.motor_policy["m1"] is not None
    assert "m1" in compute.motor_policy_opt
    assert isinstance(compute.motor_policy_opt["m1"], torch.optim.Adam)
    # lr настроен по _MOTOR_POLICY_LR=1e-4.
    pg = compute.motor_policy_opt["m1"].param_groups[0]
    assert pg["lr"] == pytest.approx(1e-4)


# ── M7.2: первый тик → pending_log_prob записан ────────────────────────

def test_first_tick_stashes_pending_log_prob(seed_file):
    compute, _ = _compute(seed_file)
    rng = np.random.default_rng(7)
    obs = {"m1": rng.normal(size=80).astype(np.float32)}
    # Без events — REINFORCE update не вызывается, но motor forward всё равно
    # создаёт pending_log_prob для следующего тика.
    compute.handle_tick(obs)
    assert "m1" in compute.pending_log_prob
    pl = compute.pending_log_prob["m1"]
    assert isinstance(pl, torch.Tensor)
    assert pl.requires_grad
    assert "m1" in compute.pending_action
    assert 0 <= compute.pending_action["m1"] < 16
    # REINFORCE-шаг ещё не сделан (нет events для t=0).
    assert compute.motor_reinforce_steps == 0
    # last_motor_delta заполнен (для push в actions_batch).
    assert "m1" in compute.last_motor_delta
    md = compute.last_motor_delta["m1"]
    assert md.shape == (16,)
    assert torch.all(md >= -1.0 - 1e-6)
    assert torch.all(md <= 1.0 + 1e-6)


# ── M7.3: второй тик с events → REINFORCE step ──────────────────────────

def test_second_tick_with_events_runs_reinforce(seed_file):
    compute, _ = _compute(seed_file)
    rng = np.random.default_rng(13)
    # tick 1: pending создан, без update.
    compute.handle_tick({"m1": rng.normal(size=80).astype(np.float32)})
    # Сохраним веса до update для контроля изменений.
    params_before = [p.detach().clone()
                     for p in compute.motor_policy["m1"].parameters()]
    # tick 2: с events → REINFORCE применяется к pending_log_prob.
    obs2 = {"m1": rng.normal(size=80).astype(np.float32)}
    events = {"m1": {"ate": True, "killed": False,
                       "damage_taken": 0.0, "delta_energy": 0.1}}
    compute.handle_tick(obs2, events_per_cid=events)
    assert compute.motor_reinforce_steps == 1
    # Новый pending для tick 2 свежесоздан.
    assert "m1" in compute.pending_log_prob
    # Веса изменились — REINFORCE сделал opt.step().
    params_after = list(compute.motor_policy["m1"].parameters())
    diff = sum((b - a).abs().sum().item()
                for b, a in zip(params_before, params_after))
    assert diff > 0.0, "motor_policy weights должны измениться после REINFORCE"


# ── M7.4: get_phase_emas включает client_motor_delta ───────────────────

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


# ── M7.5: extract_brain_state_dicts содержит motor_policy ───────────────

def test_extract_brain_state_dicts_includes_motor_policy(seed_file):
    compute, _ = _compute(seed_file)
    brain, _ = compute.extract_brain_state_dicts("m1")
    assert "motor_policy" in brain
    sd = brain["motor_policy"]
    assert isinstance(sd, dict) and len(sd) > 0


# ── M7.6: inherit_brain_y50 копирует motor_policy ──────────────────────

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
    # После Y50 веса ребёнка отличаются от родителя (0.5·parent + 0.5·noise).
    diff = sum((p - c).abs().sum().item()
                for p, c in zip(parent_params, child_params))
    assert diff > 0.0, "Y50 должен внести шум в motor_policy ребёнка"


# ── M7.7: remove_creature чистит pending ────────────────────────────────

def test_remove_creature_clears_pending(seed_file):
    compute, _ = _compute(seed_file)
    rng = np.random.default_rng(0)
    compute.handle_tick({"m1": rng.normal(size=80).astype(np.float32)})
    assert "m1" in compute.pending_log_prob
    compute.remove_creature("m1")
    assert "m1" not in compute.pending_log_prob
    assert "m1" not in compute.pending_action
    assert "m1" not in compute.last_motor_delta
    assert "m1" not in compute.motor_policy
    assert "m1" not in compute.motor_policy_opt
