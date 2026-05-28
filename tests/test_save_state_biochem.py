"""E1 (Colony Ownership Migration §5.1): biochem в save_state/restore.

save_state(cid) теперь включает biochem field (asdict от
ClientCreatureBiochem). restore_persisted_state(cid, payload)
восстанавливает обратно через ClientCreatureBiochem(**dict).

Tests:
  - save_state включает biochem field когда biochem[cid] существует
  - save_state без biochem — поле отсутствует
  - restore_persisted_state восстанавливает биохимию bit-exact
  - schema mismatch (legacy payload без некоторых полей) → fallback
  - restore for unknown cid → no-op + warning
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
def compute_with_zodchiy(tmp_path, monkeypatch):
    from utopia_client import config as cli_config
    monkeypatch.setattr(cli_config, "colonies_dir",
                        lambda: tmp_path / "colonies")
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    c = LocalColonyCompute(device="cpu")
    org = load_founders(_PROD_SEED, 1)[0]
    c.add_creature("cid-A", org, lineage="zodchiy")
    return c


def test_save_state_includes_biochem(compute_with_zodchiy):
    c = compute_with_zodchiy
    payload = c.save_state("cid-A")
    assert payload is not None
    assert "biochem" in payload
    bc = payload["biochem"]
    assert isinstance(bc, dict)
    # 8 эфемерид + 7 baseline + mental_break
    for key in ("cortisol", "dopamine", "serotonin", "oxytocin",
                "adrenaline", "glucose", "fatigue", "histamine",
                "baseline_cortisol", "baseline_serotonin",
                "mental_break"):
        assert key in bc, f"missing biochem field: {key}"


def test_save_state_no_biochem_field_when_not_zodchiy(tmp_path, monkeypatch):
    """Wanderer/elder lineage — нет biochem → save_state без поля."""
    from utopia_client import config as cli_config
    monkeypatch.setattr(cli_config, "colonies_dir",
                        lambda: tmp_path / "colonies")
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    c = LocalColonyCompute(device="cpu")
    org = load_founders(_PROD_SEED, 1)[0]
    c.add_creature("wanderer-1", org, lineage="wanderer")
    payload = c.save_state("wanderer-1")
    assert payload is not None
    assert "biochem" not in payload, "wanderer не должен иметь biochem"


def test_restore_biochem_bit_exact(compute_with_zodchiy):
    c = compute_with_zodchiy
    # Mutate биохимию: change cortisol/oxytocin/mental_break
    bc = c.biochem["cid-A"]
    bc.cortisol = 42.5
    bc.oxytocin = 78.3
    bc.serotonin = 23.1
    bc.mental_break = "depression"
    bc.mental_break_ticks = 100
    bc.baseline_cortisol = 15.0

    payload = c.save_state("cid-A")
    assert payload is not None
    saved_bc = payload["biochem"]

    # Reset биохимию на default
    from utopia_client.biochemistry import make_default
    c.biochem["cid-A"] = make_default()
    assert c.biochem["cid-A"].cortisol == 0.0

    # Restore
    c.restore_persisted_state("cid-A", payload)
    restored = c.biochem["cid-A"]
    assert restored.cortisol == 42.5
    assert restored.oxytocin == 78.3
    assert restored.serotonin == 23.1
    assert restored.mental_break == "depression"
    assert restored.mental_break_ticks == 100
    assert restored.baseline_cortisol == 15.0


def test_restore_biochem_schema_mismatch_fallback(compute_with_zodchiy):
    """Legacy payload без некоторых полей — restore через make_default + setattr."""
    c = compute_with_zodchiy
    # Partial dict (missing newer fields, like _biochem_close_kin)
    partial_payload = {
        "biochem": {
            "cortisol": 50.0,
            "oxytocin": 60.0,
            # missing: serotonin, mental_break, baselines, etc.
            "this_field_does_not_exist": 999,  # unknown field
        }
    }
    c.restore_persisted_state("cid-A", partial_payload)
    bc = c.biochem["cid-A"]
    # Применённые поля
    assert bc.cortisol == 50.0
    assert bc.oxytocin == 60.0
    # Дефолтные сохранены для unspecified
    assert bc.serotonin == 50.0  # default
    assert bc.mental_break == ""


def test_restore_unknown_cid_no_op(compute_with_zodchiy):
    c = compute_with_zodchiy
    # Не должно крашнуться
    c.restore_persisted_state("never-existed", {"biochem": {"cortisol": 99.0}})
    assert "never-existed" not in c.biochem


def test_save_restore_roundtrip_tissues_too(compute_with_zodchiy, tmp_path):
    """tissues_by_role тоже восстанавливаются bit-exact (без Y50).

    Используем torch.save/load чтобы payload был independent copy —
    реальный use case (state живёт на диске между сессиями).
    """
    import torch
    c = compute_with_zodchiy
    org = c.organisms["cid-A"]
    tid = list(org.tissues.keys())[0]
    tissue = org.tissues[tid]
    sd = tissue.state_dict()
    target_key = next(iter(sd.keys()))
    original = sd[target_key].clone()

    # Mutate weights
    mutated_sd = {k: (v + 7.0 if k == target_key else v.clone())
                  for k, v in sd.items()}
    tissue.load_state_dict(mutated_sd)
    assert torch.allclose(tissue.state_dict()[target_key], original + 7.0)

    # save → disk → reload (как реальный production flow)
    payload = c.save_state("cid-A")
    save_path = tmp_path / "state.pt"
    torch.save(payload, save_path)
    loaded_payload = torch.load(save_path, weights_only=True)

    # Reset weights to original
    reset_sd = {**tissue.state_dict(), target_key: original.clone()}
    tissue.load_state_dict(reset_sd)
    assert torch.allclose(tissue.state_dict()[target_key], original)

    # Restore from loaded payload
    c.restore_persisted_state("cid-A", loaded_payload)
    restored = org.tissues[tid].state_dict()[target_key]
    # Должно вернуться к mutated значению (original + 7)
    assert torch.allclose(restored, original + 7.0)
