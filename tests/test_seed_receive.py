"""Phase F3.1.b — приём seed_chunk + obs_batch на стороне клиента."""
from __future__ import annotations

import asyncio
import base64
import io
import json
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


def _make_weights_bytes(seed_path: Path) -> bytes:
    """Сериализовать seed-organism в формат _save_member_pt (без hebbian/selector)."""
    import torch
    from utopia_client.seed_loader import load_founders
    org = load_founders(seed_path, 1)[0]
    payload = {
        "tissues_state_dict": {tid: t.state_dict()
                                for tid, t in org.tissues.items()},
    }
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


# ── seed_loader.organism_from_weights ─────────────────────────────────

def test_organism_from_weights_smoke(seed_file):
    from utopia_client.seed_loader import organism_from_weights
    weights = _make_weights_bytes(seed_file)
    org, payload = organism_from_weights(weights, seed_file)
    assert org is not None
    assert hasattr(org, "tissues")
    assert "tissues_state_dict" in payload


def test_organism_from_weights_bad_payload(seed_file):
    from utopia_client.seed_loader import organism_from_weights
    import torch
    buf = io.BytesIO()
    torch.save({"wrong": "format"}, buf)
    with pytest.raises(ValueError):
        organism_from_weights(buf.getvalue(), seed_file)


# ── apply_inherited_state ─────────────────────────────────────────────

def test_apply_inherited_state_unknown_cid(seed_file):
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    # Не должно бросить при несуществующем cid.
    compute.apply_inherited_state("nope", {"hebbian": {}})


def test_apply_inherited_state_missing_keys_ok(seed_file):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org)
    compute.apply_inherited_state("c1", {})  # пустой payload — не падает


# ── ws_client: seed_chunk reassembly ──────────────────────────────────

def _make_client():
    from utopia_client.ws_client import ColonyWSClient
    return ColonyWSClient(server="https://example.com", token="t",
                          colony_name="test", client_version="0.0.0",
                          estimated_population=0)


def test_seed_chunk_reassembly_single_creature(seed_file):
    cli = _make_client()
    weights = _make_weights_bytes(seed_file)
    chunk_size = max(1, len(weights) // 3)
    chunks = [weights[i:i + chunk_size]
              for i in range(0, len(weights), chunk_size)]
    total = len(chunks)

    cli._handle_seed_start({"seed_id": 1, "n_creatures": 1,
                            "total_bytes": len(weights)})
    for seq, blob in enumerate(chunks):
        msg = {
            "type": "seed_chunk", "seed_id": 1, "cid": "c1",
            "seq": seq, "total": total,
            "payload_b64": base64.b64encode(blob).decode("ascii"),
            "last": (seq == total - 1),
        }
        if seq == 0:
            msg["meta"] = {"learning_rate": 1e-4, "trace_decay": 0.9}
        cli._handle_seed_chunk(msg)

    assert cli._seed_accepted == 1
    assert cli._seed_failed == 0
    assert cli.compute is not None
    assert cli.compute.n_alive == 1
    assert "c1" in cli.compute.organisms
    # Буфер очищен после finalize
    assert (1, "c1") not in cli._seed_buffers


def test_seed_chunk_interleaved_creatures(seed_file):
    """Чанки разных особей перемешаны → буферы по (seed_id,cid) разделяются."""
    cli = _make_client()
    weights = _make_weights_bytes(seed_file)
    half = len(weights) // 2
    # 2 особи, по 2 чанка каждая
    parts = {
        "ca": [weights[:half], weights[half:]],
        "cb": [weights[:half], weights[half:]],
    }
    total = 2

    cli._handle_seed_start({"seed_id": 7, "n_creatures": 2,
                            "total_bytes": len(weights) * 2})
    # Перемешанный порядок: ca-0, cb-0, ca-1, cb-1
    for seq in range(total):
        for cid in ("ca", "cb"):
            msg = {
                "type": "seed_chunk", "seed_id": 7, "cid": cid,
                "seq": seq, "total": total,
                "payload_b64": base64.b64encode(parts[cid][seq]).decode("ascii"),
                "last": (seq == total - 1),
            }
            if seq == 0:
                msg["meta"] = {"learning_rate": 1e-4, "trace_decay": 0.9}
            cli._handle_seed_chunk(msg)

    assert cli._seed_accepted == 2
    assert cli._seed_failed == 0
    assert set(cli.compute.organisms.keys()) == {"ca", "cb"}


def test_seed_chunk_bad_b64_skips(seed_file):
    cli = _make_client()
    cli._handle_seed_start({"seed_id": 2, "n_creatures": 1, "total_bytes": 0})
    cli._handle_seed_chunk({"seed_id": 2, "cid": "c1", "seq": 0,
                             "total": 1, "payload_b64": "!!!not-b64!!!",
                             "last": True})
    assert cli._seed_failed >= 1
    assert cli._seed_accepted == 0


def test_seed_chunk_corrupt_weights_failed_counter(seed_file):
    cli = _make_client()
    cli._handle_seed_start({"seed_id": 3, "n_creatures": 1, "total_bytes": 0})
    # Валидный b64, но не torch payload
    cli._handle_seed_chunk({"seed_id": 3, "cid": "c1", "seq": 0,
                             "total": 1,
                             "payload_b64": base64.b64encode(b"garbage").decode("ascii"),
                             "last": True,
                             "meta": {"learning_rate": 1e-4, "trace_decay": 0.9}})
    assert cli._seed_failed == 1
    assert cli._seed_accepted == 0


# ── seed_complete → seed_ack ──────────────────────────────────────────

class _FakeWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def test_seed_complete_sends_ack(seed_file):
    async def _run():
        cli = _make_client()
        cli._ws = _FakeWS()
        cli._seed_accepted = 3
        cli._seed_failed = 1
        await cli._handle_seed_complete({"seed_id": 42})
        sent = cli._ws.sent
        assert len(sent) == 1
        ack = sent[0]
        assert ack["type"] == "seed_ack"
        assert ack["seed_id"] == 42
        assert ack["accepted"] == 3
        assert ack["failed"] == 1

    asyncio.run(_run())


# ── obs_batch → handle_tick → actions_batch ───────────────────────────

def test_obs_batch_no_compute_noop(seed_file):
    async def _run():
        cli = _make_client()
        cli._ws = _FakeWS()
        await cli._handle_obs_batch({"world_tick": 1, "ts_p40_ns": 100,
                                      "creatures": [{"cid": "c1",
                                                     "obs": [0.0] * 64}]})
        assert cli._ws.sent == []
        assert cli._obs_batches_received == 1
        assert cli._actions_batches_sent == 0

    asyncio.run(_run())


def test_obs_batch_produces_actions(seed_file):
    """End-to-end: seed → obs_batch → actions_batch отправлен."""
    async def _run():
        from utopia_client.local_compute import N_ACTIONS

        cli = _make_client()
        cli._ws = _FakeWS()

        # Сидируем 2 особи через прямой finalize.
        weights = _make_weights_bytes(seed_file)
        cli._handle_seed_start({"seed_id": 1, "n_creatures": 2,
                                "total_bytes": 2 * len(weights)})
        for cid in ("ca", "cb"):
            cli._handle_seed_chunk({
                "seed_id": 1, "cid": cid, "seq": 0, "total": 1,
                "payload_b64": base64.b64encode(weights).decode("ascii"),
                "last": True,
                "meta": {"learning_rate": 1e-4, "trace_decay": 0.9},
            })
        assert cli._seed_accepted == 2

        # obs_batch на обе особи
        await cli._handle_obs_batch({
            "world_tick": 137, "ts_p40_ns": 999,
            "creatures": [
                {"cid": "ca", "obs": [0.1] * 64},
                {"cid": "cb", "obs": [0.2] * 64},
            ],
        })
        assert cli._actions_batches_sent == 1
        sent = cli._ws.sent[0]
        assert sent["type"] == "actions_batch"
        assert sent["world_tick"] == 137
        assert sent["ts_p40_ns_echo"] == 999
        assert len(sent["creatures"]) == 2
        cids = {c["cid"] for c in sent["creatures"]}
        assert cids == {"ca", "cb"}
        for c in sent["creatures"]:
            assert isinstance(c["action"], int)
            assert 0 <= c["action"] < N_ACTIONS
            assert "target_id" in c

    asyncio.run(_run())


def test_obs_batch_unknown_cid_ignored(seed_file):
    """obs для cid не зарегистрированной — handle_tick пропускает."""
    async def _run():
        cli = _make_client()
        cli._ws = _FakeWS()
        weights = _make_weights_bytes(seed_file)
        cli._handle_seed_start({"seed_id": 1, "n_creatures": 1,
                                "total_bytes": len(weights)})
        cli._handle_seed_chunk({
            "seed_id": 1, "cid": "ca", "seq": 0, "total": 1,
            "payload_b64": base64.b64encode(weights).decode("ascii"),
            "last": True,
            "meta": {"learning_rate": 1e-4, "trace_decay": 0.9},
        })
        await cli._handle_obs_batch({
            "world_tick": 1, "ts_p40_ns": 1,
            "creatures": [
                {"cid": "ca", "obs": [0.1] * 64},
                {"cid": "ghost", "obs": [0.0] * 64},
            ],
        })
        sent = cli._ws.sent[0]
        cids = {c["cid"] for c in sent["creatures"]}
        # ghost не в compute → не попадает в actions_batch
        assert cids == {"ca"}

    asyncio.run(_run())
