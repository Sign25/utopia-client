"""Evolved-traits recovery (30.05.2026, Бендер; согласовано Фрай/Хьюберт).

Closure Body Migration: тело (9 body-traits) переживает рестарт как мозг.
Контракт P40 (Хьюберт f673741): owned-handoff в seed.meta + welcome.
owned_traits_snapshot; client→P40 batch {type:traits_announce, creatures:[]};
P40 → {type:traits_ack, applied_cids, invalid, unknown}.

Тесты на client-сторону — без production seed (stub-организм в compute.organisms):
  - ingest_owned_traits: sanitize/clamp/extra-key drop; overwrite vs FILL-ONLY
  - build_traits_announce(_envelope) / mark_sent / handle_traits_ack (batch)
  - save_state / restore_persisted_state — traits+generation переживают рестарт
  - _inherit_traits_for_newborn читает authoritative стор (не baseline-атрибуты)
  - remove_creature / reset_all чистят стор
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402


_FULL_TRAITS = {
    "vision_radius": 7, "smell_radius": 20, "attack_radius": 2,
    "move_speed": 4, "attack_power": 3, "armor": 1,
    "efficiency": 6, "camel": 12, "diet_gene": 0.45,
}


def _compute():
    return LocalColonyCompute(device="cpu")


def _stub_org(generation: int = 0):
    return types.SimpleNamespace(generation=generation)


# ── ingest_owned_traits ──────────────────────────────────────────────────

def test_ingest_stores_and_setattrs():
    c = _compute()
    org = _stub_org()
    c.organisms["c1"] = org
    n = c.ingest_owned_traits("c1", _FULL_TRAITS)
    assert n == 9
    assert c.get_traits("c1") == _FULL_TRAITS
    assert org.vision_radius == 7
    assert org.diet_gene == 0.45


def test_ingest_clamps_and_drops_unknown():
    c = _compute()
    c.organisms["c1"] = _stub_org()
    n = c.ingest_owned_traits("c1", {
        "vision_radius": 999,    # > hi=12 → clamp 12
        "move_speed": -5,        # < lo=1 → clamp 1
        "garbage_field": 123,    # не trait → ignore
        "diet_gene": 2.0,        # > hi=1.0 → clamp 1.0
    })
    assert n == 3
    t = c.get_traits("c1")
    assert t["vision_radius"] == 12
    assert t["move_speed"] == 1
    assert t["diet_gene"] == 1.0
    assert "garbage_field" not in t


def test_ingest_overwrite_replaces():
    c = _compute()
    c.organisms["c1"] = _stub_org()
    c.ingest_owned_traits("c1", _FULL_TRAITS)
    c.ingest_owned_traits("c1", {"vision_radius": 10})  # overwrite=True default
    assert c.get_traits("c1")["vision_radius"] == 10


def test_ingest_fill_only_keeps_existing():
    """FILL-ONLY (overwrite=False): client authoritative — handoff baseline НЕ
    затирает уже известный evolved, но заполняет НЕизвестные поля."""
    c = _compute()
    org = _stub_org()
    c.organisms["c1"] = _stub_org()
    # client уже держит evolved vision_radius=12
    c.ingest_owned_traits("c1", {"vision_radius": 12})
    # P40 handoff (baseline) несёт vision_radius=3 + новое move_speed=8
    n = c.ingest_owned_traits(
        "c1", {"vision_radius": 3, "move_speed": 8}, overwrite=False)
    assert n == 1  # только move_speed применён
    t = c.get_traits("c1")
    assert t["vision_radius"] == 12   # evolved сохранён, baseline отброшен
    assert t["move_speed"] == 8       # новое поле заполнено


def test_ingest_int_fields_stay_int():
    c = _compute()
    c.organisms["c1"] = _stub_org()
    c.ingest_owned_traits("c1", {"vision_radius": 7.9})
    assert c.get_traits("c1")["vision_radius"] == 8
    assert isinstance(c.get_traits("c1")["vision_radius"], int)


# ── build / envelope / mark / ack (batch контракт Хьюберта) ──────────────

def test_build_traits_announce_items_schema():
    c = _compute()
    c.organisms["c1"] = _stub_org(generation=4)
    c.ingest_owned_traits("c1", _FULL_TRAITS)
    items = c.build_traits_announce()
    assert items == [{"cid": "c1", "traits": _FULL_TRAITS}]


def test_build_envelope_batch_schema():
    c = _compute()
    c.organisms["c1"] = _stub_org()
    c.organisms["c2"] = _stub_org()
    c.ingest_owned_traits("c1", _FULL_TRAITS)
    c.ingest_owned_traits("c2", _FULL_TRAITS)
    env = c.build_traits_announce_envelope()
    assert env["type"] == "traits_announce"
    assert {e["cid"] for e in env["creatures"]} == {"c1", "c2"}


def test_build_envelope_none_when_empty():
    c = _compute()
    c.organisms["c1"] = _stub_org()  # нет traits в сторе
    assert c.build_traits_announce_envelope() is None


def test_mark_sent_registers_pending():
    c = _compute()
    c.organisms["c1"] = _stub_org()
    c.ingest_owned_traits("c1", _FULL_TRAITS)
    c.mark_traits_announce_sent(["c1"])
    assert "c1" in c._pending_traits_announce


def test_handle_traits_ack_clears_pending():
    c = _compute()
    c.organisms["c1"] = _stub_org()
    c.organisms["c2"] = _stub_org()
    c.ingest_owned_traits("c1", _FULL_TRAITS)
    c.ingest_owned_traits("c2", _FULL_TRAITS)
    c.mark_traits_announce_sent(["c1", "c2"])
    cleared = c.handle_traits_ack({
        "type": "traits_ack", "applied_cids": ["c1", "c2"],
        "invalid": 0, "unknown": 0,
    })
    assert cleared == 2
    assert c._pending_traits_announce == {}


def test_handle_traits_ack_partial_and_invalid():
    c = _compute()
    c.organisms["c1"] = _stub_org()
    c.ingest_owned_traits("c1", _FULL_TRAITS)
    c.mark_traits_announce_sent(["c1"])
    cleared = c.handle_traits_ack({
        "type": "traits_ack", "applied_cids": ["c1"],
        "invalid": 1, "unknown": 2,
    })
    assert cleared == 1
    assert "c1" not in c._pending_traits_announce


def test_handle_traits_ack_non_dict():
    c = _compute()
    assert c.handle_traits_ack("nope") is False  # type: ignore


# ── persist roundtrip (client-restart) ───────────────────────────────────

def test_save_state_includes_traits_and_generation():
    c = _compute()
    c.organisms["c1"] = _stub_org(generation=5)
    c.ingest_owned_traits("c1", _FULL_TRAITS)
    payload = c.save_state("c1")
    assert payload["traits"] == _FULL_TRAITS
    assert payload["generation"] == 5


def test_restore_repopulates_store_and_attrs():
    # save в одном compute, restore в свежем — эмуляция client-restart
    c1 = _compute()
    c1.organisms["c1"] = _stub_org(generation=5)
    c1.ingest_owned_traits("c1", _FULL_TRAITS)
    payload = c1.save_state("c1")

    c2 = _compute()
    org2 = _stub_org(generation=0)
    c2.organisms["c1"] = org2  # add_creature эмулируется stub'ом
    c2.restore_persisted_state("c1", payload)
    assert c2.get_traits("c1") == _FULL_TRAITS
    assert org2.vision_radius == 7
    assert org2.generation == 5  # generation тоже восстановлен


# ── crossover читает стор, не baseline-атрибуты ──────────────────────────

def test_inherit_reads_store_over_baseline_attr():
    c = _compute()
    # «self-heal» организм: baseline-атрибут vision_radius=3 (минимум),
    # но authoritative стор держит evolved=12.
    mother = types.SimpleNamespace(generation=2, vision_radius=3)
    father = types.SimpleNamespace(generation=2, vision_radius=3)
    c.organisms["m"] = mother
    c.organisms["f"] = father
    c.ingest_owned_traits("m", {"vision_radius": 12})
    c.ingest_owned_traits("f", {"vision_radius": 12})
    seen_high = False
    for _ in range(20):
        child = c._inherit_traits_for_newborn(
            mother, father, mother_cid="m", father_cid="f")
        assert 3 <= child["vision_radius"] <= 12
        if child["vision_radius"] >= 10:
            seen_high = True
    assert seen_high, "crossover читает baseline-атрибут вместо стора"


def test_inherit_fallback_to_attr_when_no_store():
    c = _compute()
    mother = types.SimpleNamespace(generation=0, move_speed=8)
    father = types.SimpleNamespace(generation=0, move_speed=8)
    child = c._inherit_traits_for_newborn(
        mother, father, mother_cid="m", father_cid="f")  # нет в сторе
    assert 1 <= child["move_speed"] <= 10


# ── cleanup ──────────────────────────────────────────────────────────────

def test_remove_creature_clears_traits():
    c = _compute()
    c.organisms["c1"] = _stub_org()
    c.ingest_owned_traits("c1", _FULL_TRAITS)
    c._pending_traits_announce["c1"] = 1.0
    c.remove_creature("c1")
    assert c.get_traits("c1") is None
    assert "c1" not in c._pending_traits_announce


def test_reset_all_clears_traits():
    c = _compute()
    c.organisms["c1"] = _stub_org()
    c.ingest_owned_traits("c1", _FULL_TRAITS)
    c.reset_all()
    assert c.traits == {}
    assert c._pending_traits_announce == {}
