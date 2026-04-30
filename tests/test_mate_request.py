"""Phase F3.3.b.2 — handler mate_request → newborn (client side)."""
from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
from pathlib import Path

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
    monkeypatch.setenv("UTOPIA_COLONIES_DIR", str(tmp_path / "colonies"))
    client_seed = tmp_path / "client_seed.norg"
    client_seed.write_bytes(seed_path.read_bytes())

    from utopia_client import seed_loader as cli_loader
    importlib.reload(cli_loader)
    return client_seed


class _CapturingWS:
    def __init__(self):
        self.sent: list = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _ws_with_compute(colony="c"):
    from utopia_client.ws_client import ColonyWSClient
    ws = ColonyWSClient(server="https://x", token="t",
                         colony_name=colony, client_version="test")
    assert ws._ensure_compute()
    ws._ws = _CapturingWS()
    return ws


def _serialize_father(seed_file):
    """Серилизовать организм отца в формате P40 _save_member_pt (tissues_state_dict)."""
    import torch
    from utopia_client.seed_loader import load_founders
    org = load_founders(seed_file, 1)[0]
    payload = {"tissues_by_role":
               {tid: t.state_dict() for tid, t in org.tissues.items()}}
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


# ── unknown mother → reject ─────────────────────────────────────────

def test_mate_request_unknown_mother_rejects(seed_file):
    async def _run():
        ws = _ws_with_compute()
        father_bytes = _serialize_father(seed_file)
        await ws._handle_mate_request({
            "request_id": "r1",
            "mother_cid": "ghost",
            "father_cid": "f1",
            "father_blob_b64": base64.b64encode(father_bytes).decode("ascii"),
        })
        assert ws._ws.sent
        msg = ws._ws.sent[-1]
        assert msg["type"] == "newborn"
        assert msg["request_id"] == "r1"
        assert msg["ok"] is False
        assert msg["reason"] == "unknown_mother"
        assert ws._mate_requests_received == 1
        assert ws._mate_rejects_sent == 1
        assert ws._mate_newborns_sent == 0

    asyncio.run(_run())


# ── empty mother_cid → no_compute ───────────────────────────────────

def test_mate_request_empty_mother_rejects(seed_file):
    async def _run():
        ws = _ws_with_compute()
        await ws._handle_mate_request({
            "request_id": "r2", "mother_cid": "", "father_cid": "x",
            "father_blob_b64": "AA==",
        })
        assert ws._ws.sent
        assert ws._ws.sent[-1]["ok"] is False
        assert ws._ws.sent[-1]["reason"] == "no_compute"

    asyncio.run(_run())


# ── no compute → reject ─────────────────────────────────────────────

def test_mate_request_no_compute_rejects(seed_file):
    async def _run():
        from utopia_client.ws_client import ColonyWSClient
        ws = ColonyWSClient(server="https://x", token="t",
                             colony_name="c", client_version="test")
        ws._ws = _CapturingWS()
        await ws._handle_mate_request({
            "request_id": "r", "mother_cid": "x", "father_cid": "f",
            "father_blob_b64": "AA==",
        })
        msg = ws._ws.sent[-1]
        assert msg["ok"] is False
        assert msg["reason"] == "no_compute"

    asyncio.run(_run())


# ── bad father blob → reject ────────────────────────────────────────

def test_mate_request_bad_father_blob(seed_file):
    async def _run():
        from utopia_client.seed_loader import load_founders
        ws = _ws_with_compute()
        org = load_founders(seed_file, 1)[0]
        ws.compute.add_creature("m1", org, hebbian_enabled=True)
        await ws._handle_mate_request({
            "request_id": "r", "mother_cid": "m1", "father_cid": "f",
            "father_blob_b64": "@@@bad",
        })
        msg = ws._ws.sent[-1]
        assert msg["ok"] is False
        assert msg["reason"] == "bad_father_blob"

    asyncio.run(_run())


# ── happy path: known mother + valid father → newborn ───────────────

def test_mate_request_known_mother_returns_child(seed_file):
    async def _run():
        import torch
        from utopia_client.seed_loader import (
            load_founders, organism_from_weights)
        ws = _ws_with_compute()
        mother_org = load_founders(seed_file, 1)[0]
        ws.compute.add_creature("m1", mother_org, hebbian_enabled=True)
        father_bytes = _serialize_father(seed_file)

        await ws._handle_mate_request({
            "request_id": "rx",
            "mother_cid": "m1",
            "father_cid": "f1",
            "father_blob_b64": base64.b64encode(father_bytes).decode("ascii"),
            "sigma_scale": 1.0,
        })
        assert ws._ws.sent
        msg = ws._ws.sent[-1]
        assert msg["type"] == "newborn"
        assert msg["request_id"] == "rx"
        assert msg["mother_cid"] == "m1"
        assert msg["father_cid"] == "f1"
        assert msg["ok"] is True
        # child_blob деserialize-ится через тот же путь что seed
        child_blob = base64.b64decode(msg["child_blob_b64"])
        child_org, payload = organism_from_weights(child_blob, seed_file)
        assert "tissues_by_role" in payload
        assert len(payload["tissues_by_role"]) > 0
        assert ws._mate_newborns_sent == 1
        assert ws._mate_rejects_sent == 0

    asyncio.run(_run())


# ── ws is None → silent no crash ────────────────────────────────────

def test_mate_request_no_ws_silent(seed_file):
    async def _run():
        from utopia_client.ws_client import ColonyWSClient
        ws = ColonyWSClient(server="https://x", token="t",
                             colony_name="c", client_version="test")
        ws._ws = None
        await ws._handle_mate_request({
            "request_id": "r", "mother_cid": "m", "father_cid": "f",
            "father_blob_b64": "AA==",
        })
        assert ws._mate_newborns_sent == 0
        assert ws._mate_rejects_sent == 0

    asyncio.run(_run())
