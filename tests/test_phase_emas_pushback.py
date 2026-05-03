"""Phase emas pushback (03.05.2026) — get_phase_emas + actions_batch payload.

Покрытие:
  E.1  get_phase_emas() возвращает None для незарегистрированного cid
  E.2  get_phase_emas() возвращает все 4 поля после tick'а
  E.3  finite-guard: NaN/Inf отсеиваются из снимка
  E.4  ws_client._handle_obs_batch включает phase_emas в actions_batch
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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


def test_get_phase_emas_unknown_cid_returns_none(seed_file):
    from utopia_client.local_compute import LocalColonyCompute

    compute = LocalColonyCompute(device="cpu")
    assert compute.get_phase_emas("nonexistent") is None


def test_get_phase_emas_after_tick_has_all_fields(seed_file):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    founders = load_founders(seed_file, n=2)
    compute = LocalColonyCompute(device="cpu")
    for i, org in enumerate(founders):
        compute.add_creature(f"c{i}", org)

    rng = np.random.default_rng(7)
    for _ in range(5):
        obs = {f"c{i}": rng.normal(size=80).astype(np.float32) for i in range(2)}
        events = {f"c{i}": {"ate": True, "killed": False,
                              "damage_taken": 0.0, "delta_energy": 0.1}
                  for i in range(2)}
        compute.handle_tick(obs, events_per_cid=events)

    snap = compute.get_phase_emas("c0")
    assert snap is not None
    for key in ("loss_ema", "entropy_ema", "trace_norm_ema", "intrinsic_ema"):
        assert key in snap, f"missing {key}"
        assert isinstance(snap[key], float)
        assert math.isfinite(snap[key])


def test_get_phase_emas_filters_nonfinite(seed_file):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    org = load_founders(seed_file, n=1)[0]
    compute = LocalColonyCompute(device="cpu")
    compute.add_creature("c0", org)

    compute.loss_ema["c0"] = float("nan")
    compute.entropy_ema["c0"] = 0.5
    compute.trace_norm_ema["c0"] = float("inf")
    compute.intrinsic_ema["c0"] = 0.1

    snap = compute.get_phase_emas("c0")
    assert snap is not None
    assert "loss_ema" not in snap
    assert "trace_norm_ema" not in snap
    assert snap["entropy_ema"] == pytest.approx(0.5)
    assert snap["intrinsic_ema"] == pytest.approx(0.1)


def test_actions_batch_includes_phase_emas(seed_file):
    import asyncio

    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    from utopia_client.ws_client import ColonyWSClient

    founders = load_founders(seed_file, n=2)
    compute = LocalColonyCompute(device="cpu")
    for i, org in enumerate(founders):
        compute.add_creature(f"c{i}", org)
    rng = np.random.default_rng(3)
    for _ in range(3):
        obs = {f"c{i}": rng.normal(size=80).astype(np.float32) for i in range(2)}
        events = {f"c{i}": {"ate": False, "killed": False,
                              "damage_taken": 0.0, "delta_energy": 0.0}
                  for i in range(2)}
        compute.handle_tick(obs, events_per_cid=events)

    client = ColonyWSClient.__new__(ColonyWSClient)
    client.compute = compute
    client._obs_batches_received = 0
    client._actions_batches_sent = 0
    sent: list = []

    class FakeWS:
        async def send(self, data):
            sent.append(json.loads(data))

    client._ws = FakeWS()
    msg = {
        "type": "obs_batch",
        "world_tick": 1234,
        "ts_p40_ns": 9999,
        "creatures": [
            {"cid": f"c{i}", "obs": rng.normal(size=80).astype(np.float32).tolist(),
             "ate": False, "killed": False, "damage_taken": 0.0, "delta_energy": 0.0}
            for i in range(2)
        ],
    }
    asyncio.run(client._handle_obs_batch(msg))

    assert sent, "actions_batch не отправлен"
    out = sent[0]
    assert out["type"] == "actions_batch"
    assert len(out["creatures"]) == 2
    for entry in out["creatures"]:
        assert "phase_emas" in entry
        emas = entry["phase_emas"]
        assert isinstance(emas, dict)
        assert any(k in emas for k in
                   ("loss_ema", "entropy_ema", "trace_norm_ema", "intrinsic_ema"))
