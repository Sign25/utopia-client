"""Brain migration Etap 3.2 — Y50 для asexual reproduce envelope.

`build_reproduce_envelope` теперь опционально принимает brain_state_dicts
(predictor + S2.E/G/A/F) — мутирует Y50 на клиенте и пакует в envelope.
`extract_brain_state_dicts` собирает state_dict родителя из LocalColonyCompute.
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


def _compute_with_creature(seed_file, cid="m1"):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature(cid, org, hebbian_enabled=True)
    return compute, org


# ── 1. extract_brain_state_dicts известного cid ──────────────────────

def test_extract_brain_state_dicts_known_cid(seed_file):
    compute, _ = _compute_with_creature(seed_file)
    brain, emas = compute.extract_brain_state_dicts("m1")
    # Predictor + 4 высших ткани (dopamine/imagination/planner/insula)
    # инициализируются для каждой особи в add_creature.
    assert "predictor" in brain
    assert "dopamine" in brain
    assert "imagination" in brain
    assert "planner" in brain
    assert "insula" in brain
    # state_dict содержит реальные тензоры.
    for key in ("predictor", "dopamine", "imagination", "planner", "insula"):
        sd = brain[key]
        assert isinstance(sd, dict) and len(sd) > 0
    # EMA — числа.
    assert isinstance(emas, dict)


# ── 2. extract_brain_state_dicts unknown cid → пустые dicts ──────────

def test_extract_brain_state_dicts_unknown_cid(seed_file):
    compute, _ = _compute_with_creature(seed_file)
    brain, emas = compute.extract_brain_state_dicts("ghost")
    assert brain == {}
    assert emas == {}


# ── 3. build_reproduce_envelope без brain — обратная совместимость ──

def test_envelope_without_brain_legacy(seed_file):
    from utopia_client.reproduce import build_reproduce_envelope, unpack_zstd_b64
    compute, org = _compute_with_creature(seed_file)
    env = build_reproduce_envelope("m1", org)
    assert env["type"] == "reproduce"
    payload = unpack_zstd_b64(env["child_weights_b64"])
    assert "tissues_by_role" in payload
    assert "brain" not in payload  # legacy путь — мозг не шлётся


# ── 4. build_reproduce_envelope с brain — мутированный мозг в payload ──

def test_envelope_with_brain_includes_mutated_state(seed_file):
    from utopia_client.reproduce import build_reproduce_envelope, unpack_zstd_b64
    compute, org = _compute_with_creature(seed_file)
    brain, emas = compute.extract_brain_state_dicts("m1")
    # Снимем эталонный snapshot до Y50.
    parent_pred_w = {k: v.detach().clone() for k, v in brain["predictor"].items()}

    env = build_reproduce_envelope(
        "m1", org, brain_state_dicts=brain, brain_emas=emas)
    payload = unpack_zstd_b64(env["child_weights_b64"])
    assert "brain" in payload
    assert "brain_emas" in payload
    # Все 5 тканей мозга.
    for key in ("predictor", "dopamine", "imagination", "planner", "insula"):
        assert key in payload["brain"]
    # Y50 применён: 2D веса отличаются от parent.
    child_pred = payload["brain"]["predictor"]
    diffs = []
    for k, pv in parent_pred_w.items():
        if pv.dim() >= 2 and "weight" in k:
            diff = (pv - child_pred[k]).abs().mean().item()
            diffs.append(diff)
    assert diffs and all(d > 0 for d in diffs), \
        "Y50 σ·std·noise должен сдвинуть 2D weights"


# ── 5. EMAs корректно сериализуются и десериализуются ────────────────

def test_envelope_brain_emas_roundtrip(seed_file):
    from utopia_client.reproduce import build_reproduce_envelope, unpack_zstd_b64
    compute, org = _compute_with_creature(seed_file)
    # Принудительно проставим EMA, чтобы они не были все 0.
    compute.loss_ema["m1"] = 0.123
    compute.intrinsic_ema["m1"] = 0.456
    brain, emas = compute.extract_brain_state_dicts("m1")
    assert emas["predictor_loss_ema"] == pytest.approx(0.123)
    assert emas["intrinsic_ema"] == pytest.approx(0.456)
    env = build_reproduce_envelope(
        "m1", org, brain_state_dicts=brain, brain_emas=emas)
    payload = unpack_zstd_b64(env["child_weights_b64"])
    assert payload["brain_emas"]["predictor_loss_ema"] == pytest.approx(0.123)
    assert payload["brain_emas"]["intrinsic_ema"] == pytest.approx(0.456)


# ── 6. unknown cid envelope — brain пуст, payload без brain ──────────

def test_envelope_unknown_cid_silent(seed_file):
    from utopia_client.reproduce import build_reproduce_envelope, unpack_zstd_b64
    compute, org = _compute_with_creature(seed_file)
    brain, emas = compute.extract_brain_state_dicts("ghost")
    env = build_reproduce_envelope(
        "m1", org, brain_state_dicts=brain, brain_emas=emas)
    payload = unpack_zstd_b64(env["child_weights_b64"])
    # brain={} → ключ brain не добавляется (truthy check).
    assert "brain" not in payload
