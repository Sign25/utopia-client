"""Тесты клиентской death-логики (Phase F3.6)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("storage.norg")
pytest.importorskip("core.workbench")


@pytest.fixture
def organism(tmp_path, monkeypatch):
    seed_path = tmp_path / "seed.norg"
    monkeypatch.setenv("WORLD_SEED_PATH", str(seed_path))
    monkeypatch.setenv("UTOPIA_SEED_PATH", str(tmp_path / "client_seed.norg"))

    import importlib
    from environment import seed_loader as ns_loader
    importlib.reload(ns_loader)
    ns_loader.ensure_seed(preset="tardigrade")

    client_seed = tmp_path / "client_seed.norg"
    client_seed.write_bytes(seed_path.read_bytes())

    from utopia_client import seed_loader as cli_loader
    importlib.reload(cli_loader)
    return cli_loader.load_founders(client_seed, n=1)[0]


def test_death_envelope_format(organism):
    from utopia_client.death import build_death_envelope

    env = build_death_envelope("c42", organism, reason="hp_zero", fitness=0.812)
    assert env["type"] == "death"
    assert env["cid"] == "c42"
    assert env["reason"] == "hp_zero"
    assert env["fitness"] == pytest.approx(0.812)
    assert isinstance(env["weights_b64"], str)
    assert len(env["weights_b64"]) > 100


def test_death_envelope_roundtrip(organism):
    """Веса в envelope → unpack должен дать те же state_dict."""
    import torch
    from utopia_client.death import build_death_envelope
    from utopia_client.reproduce import (
        _extract_tissues_state_dict,
        unpack_zstd_b64,
    )

    parent_sd = _extract_tissues_state_dict(organism)
    env = build_death_envelope("c1", organism)
    decoded = unpack_zstd_b64(env["weights_b64"])
    out_sd = decoded["tissues_state_dict"]
    assert set(out_sd.keys()) == set(parent_sd.keys())
    for tid, ref in parent_sd.items():
        for k, v in ref.items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, out_sd[tid][k])
