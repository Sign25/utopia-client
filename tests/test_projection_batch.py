"""E3 (Colony Ownership Migration §5.2): build_projection_batch schema.

Финальная schema projection_batch_draft.md §3 (28.05.2026):
  {cid, species_id, alive, frozen, action, chem{7}, mental_break}

Histamine исключён (mirror infection_severity).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")
pytest.importorskip("core.workbench")
pytest.importorskip("storage.norg")

_PROD_SEED = Path.home() / ".utopia-client" / "seed.norg"
if not _PROD_SEED.exists():
    pytest.skip(f"production seed not present at {_PROD_SEED}",
                allow_module_level=True)


@pytest.fixture
def compute_zodchiy(tmp_path, monkeypatch):
    from utopia_client import config as cli_config
    monkeypatch.setattr(cli_config, "colonies_dir",
                        lambda: tmp_path / "colonies")
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    c = LocalColonyCompute(device="cpu")
    orgs = load_founders(_PROD_SEED, 2)
    for i, org in enumerate(orgs):
        c.add_creature(f"cid-{i}", org, lineage="zodchiy")
    return c


def test_empty_compute_returns_empty_list(tmp_path, monkeypatch):
    from utopia_client import config as cli_config
    monkeypatch.setattr(cli_config, "colonies_dir",
                        lambda: tmp_path / "colonies")
    from utopia_client.local_compute import LocalColonyCompute
    c = LocalColonyCompute(device="cpu")
    assert c.build_projection_batch() == []


def test_projection_schema_fields(compute_zodchiy):
    batch = compute_zodchiy.build_projection_batch()
    assert len(batch) == 2
    p = batch[0]
    # Required schema fields
    for key in ("cid", "species_id", "alive", "frozen", "action",
                "chem", "mental_break"):
        assert key in p, f"missing: {key}"


def test_chem_has_7_ephemeral_no_histamine(compute_zodchiy):
    p = compute_zodchiy.build_projection_batch()[0]
    assert p["chem"] is not None
    expected = {"cortisol", "dopamine", "serotonin", "oxytocin",
                "adrenaline", "glucose", "fatigue"}
    assert set(p["chem"].keys()) == expected
    assert "histamine" not in p["chem"]


def test_projection_reflects_biochem_state(compute_zodchiy):
    c = compute_zodchiy
    bc = c.biochem["cid-0"]
    bc.cortisol = 77.5
    bc.oxytocin = 42.3
    bc.mental_break = "depression"
    batch = c.build_projection_batch()
    proj_a = next(p for p in batch if p["cid"] == "cid-0")
    assert proj_a["chem"]["cortisol"] == 77.5
    assert proj_a["chem"]["oxytocin"] == 42.3
    assert proj_a["mental_break"] == "depression"


def test_no_mental_break_returns_none(compute_zodchiy):
    """Пустая строка mental_break → null в projection."""
    p = compute_zodchiy.build_projection_batch()[0]
    # default — empty
    assert p["mental_break"] is None


def test_alive_default_true(compute_zodchiy):
    batch = compute_zodchiy.build_projection_batch()
    for p in batch:
        assert p["alive"] is True
        assert p["frozen"] is False


def test_cid_is_string(compute_zodchiy):
    batch = compute_zodchiy.build_projection_batch()
    for p in batch:
        assert isinstance(p["cid"], str)


def test_msgpack_serializable(compute_zodchiy):
    """Smoke: batch валидный для msgpack/json transmit."""
    import json
    batch = compute_zodchiy.build_projection_batch()
    encoded = json.dumps({"type": "projection_batch", "creatures": batch})
    decoded = json.loads(encoded)
    assert len(decoded["creatures"]) == 2
