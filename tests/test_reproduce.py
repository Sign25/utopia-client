"""Тесты клиентской reproduce-логики (Phase F3.5).

Pure-функции: extract → mutate → pack → unpack → apply. Без сети.
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


@pytest.fixture
def seed_and_organism(tmp_path, monkeypatch):
    """Bootstrap seed.norg, вернуть один CompositeOrganism для теста."""
    seed_path = tmp_path / "seed.norg"
    monkeypatch.setenv("WORLD_SEED_PATH", str(seed_path))
    monkeypatch.setenv("UTOPIA_SEED_PATH", str(tmp_path / "client_seed.norg"))

    import importlib

    from environment import seed_loader as ns_loader
    importlib.reload(ns_loader)
    ns_loader.ensure_seed(preset="nexus")

    client_seed = tmp_path / "client_seed.norg"
    client_seed.write_bytes(seed_path.read_bytes())

    from utopia_client import seed_loader as cli_loader
    importlib.reload(cli_loader)

    org = cli_loader.load_founders(client_seed, n=1)[0]
    return org


def test_pack_unpack_roundtrip(seed_and_organism):
    import torch

    from utopia_client.reproduce import (
        _extract_tissues_state_dict,
        pack_zstd_b64,
        unpack_zstd_b64,
    )

    org = seed_and_organism
    sd = _extract_tissues_state_dict(org)
    payload = {"tissues_state_dict": sd}
    encoded = pack_zstd_b64(payload)
    assert isinstance(encoded, str) and len(encoded) > 0

    decoded = unpack_zstd_b64(encoded)
    assert "tissues_state_dict" in decoded
    out_sd = decoded["tissues_state_dict"]
    assert set(out_sd.keys()) == set(sd.keys())
    for tid, ref_sd in sd.items():
        for k, v in ref_sd.items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, out_sd[tid][k])


def test_mutate_state_dict_changes_floats(seed_and_organism):
    import torch

    from utopia_client.reproduce import (
        DEFAULT_SIGMA,
        _extract_tissues_state_dict,
        mutate_state_dict,
    )

    torch.manual_seed(123)
    org = seed_and_organism
    parent_sd = _extract_tissues_state_dict(org)
    child_sd = mutate_state_dict(parent_sd, sigma=DEFAULT_SIGMA)

    # хотя бы один float-тензор изменился
    changed = False
    for tid, ref_sd in parent_sd.items():
        for k, v in ref_sd.items():
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                if not torch.equal(v, child_sd[tid][k]):
                    changed = True
                    break
        if changed:
            break
    assert changed, "Y50 mutation should perturb at least one float tensor"

    # σ ≈ 0.09 — изменения малы, не превышают 5×std
    for tid, ref_sd in parent_sd.items():
        for k, v in ref_sd.items():
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                std = float(v.detach().float().std().item())
                if std > 0:
                    delta = (child_sd[tid][k] - v).abs().max().item()
                    assert delta < 5.0 * std + 1e-6, f"{tid}/{k}: too large"


def test_build_reproduce_envelope_format(seed_and_organism):
    from utopia_client.reproduce import build_reproduce_envelope

    env = build_reproduce_envelope("parent_cid_1", seed_and_organism)
    assert env["type"] == "reproduce"
    assert env["parent_cid"] == "parent_cid_1"
    assert isinstance(env["child_weights_b64"], str)
    assert len(env["child_weights_b64"]) > 100  # не пустой
    assert env["sigma"] > 0


def test_apply_state_dict_roundtrip(seed_and_organism):
    import torch

    from utopia_client.reproduce import (
        _extract_tissues_state_dict,
        apply_state_dict,
        mutate_state_dict,
    )

    torch.manual_seed(7)
    org = seed_and_organism
    parent_sd = _extract_tissues_state_dict(org)
    child_sd = mutate_state_dict(parent_sd)

    # Применяем child_sd к organism, проверяем что веса теперь = child_sd
    n = apply_state_dict(org, child_sd)
    assert n == len(parent_sd)
    out_sd = _extract_tissues_state_dict(org)
    for tid, ref_sd in child_sd.items():
        for k, v in ref_sd.items():
            if isinstance(v, torch.Tensor):
                assert torch.equal(v, out_sd[tid][k])
