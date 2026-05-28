"""E2 (Colony Ownership Migration §5.1): restore_colony_from_local().

End-to-end:
  - Создать compute_A с N zodchiy → mutate weights + biochem → save_all_states
  - Создать compute_B fresh → restore_colony_from_local → verify bit-exact

Edge cases:
  - empty dir → empty list
  - dir не существует → empty list
  - partial corrupted file → пропускается, остальные restored
  - повторный вызов после restore → skip (cid уже в organisms)
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


def _make_compute(tmp_path, monkeypatch):
    from utopia_client import config as cli_config
    monkeypatch.setattr(cli_config, "colonies_dir",
                        lambda: tmp_path / "colonies")
    from utopia_client.local_compute import LocalColonyCompute
    return LocalColonyCompute(device="cpu")


def test_restore_empty_dir(tmp_path, monkeypatch):
    c = _make_compute(tmp_path, monkeypatch)
    states_dir = tmp_path / "states"
    states_dir.mkdir()
    restored = c.restore_colony_from_local(states_dir)
    assert restored == []
    assert len(c.organisms) == 0


def test_restore_nonexistent_dir(tmp_path, monkeypatch):
    c = _make_compute(tmp_path, monkeypatch)
    restored = c.restore_colony_from_local(tmp_path / "never-existed")
    assert restored == []


def test_save_then_restore_roundtrip(tmp_path, monkeypatch):
    """End-to-end: save → fresh compute → restore → bit-exact biochem."""
    import torch
    from utopia_client.seed_loader import load_founders

    # Phase A: save 2 organisms с mutated state
    c_a = _make_compute(tmp_path, monkeypatch)
    orgs = load_founders(_PROD_SEED, 2)
    for i, org in enumerate(orgs):
        c_a.add_creature(f"cid-{i}", org, lineage="zodchiy")
    # Mutate биохимию
    c_a.biochem["cid-0"].cortisol = 99.0
    c_a.biochem["cid-0"].oxytocin = 33.0
    c_a.biochem["cid-1"].mental_break = "berserk"
    c_a.biochem["cid-1"].fatigue = 75.5

    states_dir = tmp_path / "states"
    n_saved = c_a.save_all_states(states_dir)
    assert n_saved == 2
    assert len(list(states_dir.glob("*.pt"))) == 2

    # Phase B: fresh compute → restore
    c_b = _make_compute(tmp_path, monkeypatch)
    assert len(c_b.organisms) == 0
    restored = c_b.restore_colony_from_local(states_dir)
    assert sorted(restored) == ["cid-0", "cid-1"]
    assert len(c_b.organisms) == 2

    # Verify биохимия restored bit-exact
    assert c_b.biochem["cid-0"].cortisol == 99.0
    assert c_b.biochem["cid-0"].oxytocin == 33.0
    assert c_b.biochem["cid-1"].mental_break == "berserk"
    assert c_b.biochem["cid-1"].fatigue == 75.5


def test_restore_skips_corrupted_file(tmp_path, monkeypatch):
    """Битый .pt в dir → лог warning, остальные restored."""
    import torch
    from utopia_client.seed_loader import load_founders

    c_a = _make_compute(tmp_path, monkeypatch)
    org = load_founders(_PROD_SEED, 1)[0]
    c_a.add_creature("good-cid", org, lineage="zodchiy")
    states_dir = tmp_path / "states"
    c_a.save_all_states(states_dir)

    # Подкинуть битый файл
    (states_dir / "bad-cid.pt").write_bytes(b"not a torch save file")

    c_b = _make_compute(tmp_path, monkeypatch)
    restored = c_b.restore_colony_from_local(states_dir)
    assert restored == ["good-cid"]
    assert "good-cid" in c_b.organisms
    assert "bad-cid" not in c_b.organisms


def test_restore_idempotent(tmp_path, monkeypatch):
    """Повторный restore — skip cids которые уже зарегистрированы."""
    from utopia_client.seed_loader import load_founders

    c_a = _make_compute(tmp_path, monkeypatch)
    org = load_founders(_PROD_SEED, 1)[0]
    c_a.add_creature("cid-x", org, lineage="zodchiy")
    states_dir = tmp_path / "states"
    c_a.save_all_states(states_dir)

    c_b = _make_compute(tmp_path, monkeypatch)
    restored_1 = c_b.restore_colony_from_local(states_dir)
    assert restored_1 == ["cid-x"]
    # Second call — skip (already loaded)
    restored_2 = c_b.restore_colony_from_local(states_dir)
    assert restored_2 == []
    assert len(c_b.organisms) == 1
