"""Phase F3.3.a — handler weights_request → weights_dump (client side)."""
from __future__ import annotations

import asyncio
import base64
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
    """Минимальная заглушка ws, копит отправленные сообщения."""

    def __init__(self):
        self.sent: list = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _ws_with_compute(colony="c", live_cids=()):
    from utopia_client.ws_client import ColonyWSClient
    ws = ColonyWSClient(server="https://x", token="t",
                         colony_name=colony, client_version="test")
    assert ws._ensure_compute()
    ws._ws = _CapturingWS()
    return ws


# ── unknown cid → ok=False ──────────────────────────────────────────

def test_weights_request_unknown_cid_responds_false(seed_file):
    async def _run():
        ws = _ws_with_compute()
        await ws._handle_weights_request(
            {"cid": "ghost", "request_id": "r1"})
        assert ws._ws.sent
        msg = ws._ws.sent[-1]
        assert msg["type"] == "weights_dump"
        assert msg["cid"] == "ghost"
        assert msg["request_id"] == "r1"
        assert msg["ok"] is False
        assert msg["reason"] == "unknown_cid"
        assert ws._weights_requests_received == 1
        assert ws._weights_dumps_sent == 0

    asyncio.run(_run())


# ── empty cid → no_compute / no answer ──────────────────────────────

def test_weights_request_empty_cid_responds_no_compute(seed_file):
    async def _run():
        ws = _ws_with_compute()
        await ws._handle_weights_request({"cid": "", "request_id": "r"})
        # Пустой cid → reject "no_compute" (защита от плохого msg).
        assert ws._ws.sent
        msg = ws._ws.sent[-1]
        assert msg["ok"] is False

    asyncio.run(_run())


# ── happy path: known cid → ok=True + blob_b64 deserializes ─────────

def test_weights_request_known_cid_returns_blob(seed_file):
    async def _run():
        from utopia_client.seed_loader import load_founders
        from utopia_client.seed_loader import organism_from_weights

        ws = _ws_with_compute()
        org = load_founders(seed_file, 1)[0]
        ws.compute.add_creature("c1", org, hebbian_enabled=True)

        await ws._handle_weights_request(
            {"cid": "c1", "request_id": "req-x"})
        assert ws._ws.sent
        msg = ws._ws.sent[-1]
        assert msg["type"] == "weights_dump"
        assert msg["cid"] == "c1"
        assert msg["request_id"] == "req-x"
        assert msg["ok"] is True
        blob = base64.b64decode(msg["blob_b64"])
        assert len(blob) > 100  # хоть какой-то реальный размер

        # Roundtrip: blob → organism (через тот же контракт что P40 seed).
        org2, payload = organism_from_weights(blob, seed_file)
        assert "tissues_state_dict" in payload
        assert len(payload["tissues_state_dict"]) > 0
        # Hebbian включён → state присутствует.
        assert "hebbian" in payload
        assert ws._weights_dumps_sent == 1

    asyncio.run(_run())


# ── no compute initialized → no_compute reject ──────────────────────

def test_weights_request_no_compute_responds_false(seed_file):
    async def _run():
        from utopia_client.ws_client import ColonyWSClient
        ws = ColonyWSClient(server="https://x", token="t",
                             colony_name="c", client_version="test")
        # Не вызываем _ensure_compute().
        ws._ws = _CapturingWS()
        await ws._handle_weights_request(
            {"cid": "x", "request_id": "r"})
        assert ws._ws.sent
        msg = ws._ws.sent[-1]
        assert msg["ok"] is False
        assert msg["reason"] == "no_compute"

    asyncio.run(_run())


# ── ws is None → silently skip (no crash) ───────────────────────────

def test_weights_request_no_ws_silent(seed_file):
    async def _run():
        from utopia_client.ws_client import ColonyWSClient
        ws = ColonyWSClient(server="https://x", token="t",
                             colony_name="c", client_version="test")
        ws._ws = None
        # Не должен крашить.
        await ws._handle_weights_request(
            {"cid": "x", "request_id": "r"})
        assert ws._weights_dumps_sent == 0

    asyncio.run(_run())
