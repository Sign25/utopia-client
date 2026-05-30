"""Evolved-traits recovery — ws_client пути доставки (контракт Хьюберт ca3e3b2).

Покрывает client-side glue поверх LocalColonyCompute store:
  - top-level owned_traits_snapshot (reason=self_heal/pull) → FILL-ONLY ingest
    + force re-announce (guard игнорируется — стор наполнился после welcome)
  - welcome inline snapshot
  - traits_ack через _handle снимает pending
  - pull safety-net (traits_request) при пустом сторе; skip когда наполнен

Без production seed — stub-организмы прямо в compute.organisms + fake ws.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402
import utopia_client.ws_client as wsm  # noqa: E402


_FULL = {
    "vision_radius": 7, "smell_radius": 20, "attack_radius": 2,
    "move_speed": 4, "attack_power": 3, "armor": 1,
    "efficiency": 6, "camel": 12, "diet_gene": 0.45,
}


class _FakeWS:
    def __init__(self):
        self.sent: list[dict] = []

    async def send(self, s: str) -> None:
        self.sent.append(json.loads(s))


def _make_client():
    return wsm.ColonyWSClient(
        server="https://example.com", token="t",
        colony_name="test", client_version="0.0.0",
        estimated_population=0)


def _compute_with(cid="c1", traits=None, generation=1):
    c = LocalColonyCompute(device="cpu")
    c.organisms[cid] = types.SimpleNamespace(generation=generation)
    if traits:
        c.ingest_owned_traits(cid, traits)
    return c


def _sent_types(ws):
    return [m["type"] for m in ws.sent]


# ── top-level owned_traits_snapshot (self_heal) ──────────────────────────

def test_self_heal_snapshot_fill_only_then_force_reannounce():
    cli = _make_client()
    c = _compute_with("c1", {"vision_radius": 12})  # client evolved
    cli.compute = c
    cli._ws = _FakeWS()
    cli._traits_announced_conn = True  # welcome уже выставил guard

    # P40 self_heal push: baseline vision_radius=3 + новое move_speed=8
    msg = {"type": "owned_traits_snapshot", "reason": "self_heal",
           "creatures": [{"cid": "c1",
                          "traits": {"vision_radius": 3, "move_speed": 8}}]}
    asyncio.run(cli._handle(msg))

    # FILL-ONLY: evolved vision_radius сохранён, move_speed заполнен
    assert c.get_traits("c1")["vision_radius"] == 12
    assert c.get_traits("c1")["move_speed"] == 8
    # force re-announce сработал несмотря на guard
    assert "traits_announce" in _sent_types(cli._ws)
    ann = next(m for m in cli._ws.sent if m["type"] == "traits_announce")
    item = ann["creatures"][0]
    assert item["cid"] == "c1"
    assert item["traits"]["vision_radius"] == 12  # evolved отправлен поверх baseline


def test_self_heal_snapshot_empty_store_fills_baseline():
    cli = _make_client()
    c = _compute_with("c1", None)  # стор пуст (старый .pt без traits)
    cli.compute = c
    cli._ws = _FakeWS()
    msg = {"type": "owned_traits_snapshot", "reason": "self_heal",
           "creatures": [{"cid": "c1", "traits": _FULL}]}
    asyncio.run(cli._handle(msg))
    # пустой стор → baseline заполняется (нечего было защищать)
    assert c.get_traits("c1") == _FULL
    # и re-announce отправляет его
    assert "traits_announce" in _sent_types(cli._ws)


def test_pull_snapshot_reason_routed_same_handler():
    cli = _make_client()
    c = _compute_with("c1", None)
    cli.compute = c
    cli._ws = _FakeWS()
    msg = {"type": "owned_traits_snapshot", "reason": "pull",
           "creatures": [{"cid": "c1", "traits": {"camel": 30}}]}
    asyncio.run(cli._handle(msg))
    assert c.get_traits("c1")["camel"] == 30


# ── traits_ack через _handle ─────────────────────────────────────────────

def test_traits_ack_via_handle_clears_pending():
    cli = _make_client()
    c = _compute_with("c1", _FULL)
    c.mark_traits_announce_sent(["c1"])
    cli.compute = c
    asyncio.run(cli._handle({"type": "traits_ack", "applied_cids": ["c1"],
                             "invalid": 0, "unknown": 0}))
    assert "c1" not in c._pending_traits_announce


# ── pull safety-net ──────────────────────────────────────────────────────

def test_request_traits_resync_sends_traits_request():
    cli = _make_client()
    cli._ws = _FakeWS()
    ok = asyncio.run(cli.request_traits_resync())
    assert ok is True
    assert cli._ws.sent[0]["type"] == "traits_request"


def test_maybe_pull_when_store_empty(monkeypatch):
    monkeypatch.setattr(wsm, "TRAITS_PULL_GRACE_SEC", 0.0)
    cli = _make_client()
    cli._ws = _FakeWS()
    cli._traits_pull_sent = False
    cli.compute = _compute_with("c1", None)  # organisms есть, traits нет
    asyncio.run(cli._maybe_pull_traits())
    assert any(m["type"] == "traits_request" for m in cli._ws.sent)
    assert cli._traits_pull_sent is True


def test_maybe_pull_skips_when_store_populated(monkeypatch):
    monkeypatch.setattr(wsm, "TRAITS_PULL_GRACE_SEC", 0.0)
    cli = _make_client()
    cli._ws = _FakeWS()
    cli._traits_pull_sent = False
    cli.compute = _compute_with("c1", _FULL)  # стор наполнен
    asyncio.run(cli._maybe_pull_traits())
    assert not any(m["type"] == "traits_request" for m in cli._ws.sent)


def test_maybe_pull_idempotent_per_connect(monkeypatch):
    monkeypatch.setattr(wsm, "TRAITS_PULL_GRACE_SEC", 0.0)
    cli = _make_client()
    cli._ws = _FakeWS()
    cli._traits_pull_sent = True  # уже слали в этом коннекте
    cli.compute = _compute_with("c1", None)
    asyncio.run(cli._maybe_pull_traits())
    assert not any(m["type"] == "traits_request" for m in cli._ws.sent)


# ── welcome inline snapshot всё ещё работает ─────────────────────────────

def test_welcome_inline_snapshot_ingests():
    cli = _make_client()
    c = _compute_with("c1", None)
    cli.compute = c
    cli._ws = _FakeWS()
    cli._traits_announced_conn = False
    msg = {"type": "welcome", "world_tick": 100, "n_creatures": 1,
           "mode": "normal",
           "owned_traits_snapshot": [{"cid": "c1", "traits": {"armor": 5}}]}
    asyncio.run(cli._handle(msg))
    assert c.get_traits("c1")["armor"] == 5
