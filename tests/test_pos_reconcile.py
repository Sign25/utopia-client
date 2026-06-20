"""pos-реконсиляция owned-Адама (Хьюберт §20.6, P0-блокер halo). Корень drink_sum=0:
owned-Адам резолвил позицию на obs_batch c[row/col] = ЛОКАЛЬНЫЙ projected sim-pos
(drift ~247), а не серверную проекцию (canon ~48 из snap.creatures→creature_pos).
~200 клеток мимо → near_water/flora/income от пустого места. Фикс: pos_reconcile —
owned-Адаму канон creature_pos + кэш last-known, БЕЗ drift-fallback. OFF dormant →
bit-identical. Канон подтверждён live: creature_pos=(47,48)→resolved(48,47)=P40(48,47).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

import utopia_client.ws_client as wsm  # noqa: E402
from utopia_client.local_compute import LocalColonyCompute  # noqa: E402


class _Cfg:
    def __init__(self, size): self.size = size


class _FakeWC:
    def __init__(self, size=256, creature_pos=None):
        self.config = _Cfg(size)
        self.terrain = bytes(size * size)
        self.creature_pos = dict(creature_pos or {})  # cid → (x=col, y=row)


def _client(wc=None):
    cli = wsm.ColonyWSClient(server="https://e.com", token="t",
                             colony_name="cheef", client_version="0.0.0",
                             estimated_population=0)
    cli.world_cache = wc
    return cli


def _compute(single=True, reconcile=False):
    c = LocalColonyCompute(device="cpu")
    c._single_organism = single
    if reconcile:
        c.set_pos_reconcile(True)
    return c


# ── сеттер / dormant ──────────────────────────────────────────────────────

def test_setter_default_off():
    c = LocalColonyCompute(device="cpu")
    assert c._pos_reconcile_enabled is False
    assert c.set_pos_reconcile(True) is True
    assert c._pos_reconcile_enabled is True
    assert c.set_pos_reconcile(False) is False


# ── OFF → legacy bit-identical ────────────────────────────────────────────

def test_off_legacy_creature_pos():
    # флаг OFF: creature_pos есть → берём его (как раньше)
    cli = _client(_FakeWC(creature_pos={"c1": (47, 48)}))  # (x=47, y=48)
    cli.compute = _compute(reconcile=False)
    assert cli._resolve_pos("c1", {"row": 200, "col": 200}) == (48, 47)


def test_off_legacy_fallback_to_obs_batch():
    # флаг OFF: creature_pos пуст → fallback на obs_batch c (drift) — legacy
    cli = _client(_FakeWC(creature_pos={}))
    cli.compute = _compute(reconcile=False)
    assert cli._resolve_pos("c1", {"row": 247, "col": 74}) == (247, 74)


# ── ON → канон + кэш, без drift-fallback ──────────────────────────────────

def test_on_canon_from_creature_pos_and_caches():
    cli = _client(_FakeWC(creature_pos={"c1": (47, 48)}))  # canon (row=48,col=47)
    cli.compute = _compute(reconcile=True)
    # obs_batch говорит (247,74) drift, но reconcile берёт канон
    assert cli._resolve_pos("c1", {"row": 247, "col": 74}) == (48, 47)
    # и кэширует last-known серверную позицию
    assert cli._adam_server_pos == (48, 47)


def test_on_uses_cache_when_creature_pos_drops():
    # 1-й вызов: канон есть → кэш (48,47)
    wc = _FakeWC(creature_pos={"c1": (47, 48)})
    cli = _client(wc)
    cli.compute = _compute(reconcile=True)
    assert cli._resolve_pos("c1", {"row": 247, "col": 74}) == (48, 47)
    # 2-й: creature_pos выпал из snap-кадра (owned интермиттентно) → НЕ drift,
    # берём кэш last-known канона (а не obs_batch 247)
    wc.creature_pos.clear()
    assert cli._resolve_pos("c1", {"row": 247, "col": 74}) == (48, 47)


def test_on_startup_no_canon_no_cache_falls_to_obs():
    # до первого канона (старт) кэша нет → last-resort obs_batch (один-два тика)
    cli = _client(_FakeWC(creature_pos={}))
    cli.compute = _compute(reconcile=True)
    assert cli._resolve_pos("c1", {"row": 99, "col": 12}) == (99, 12)


def test_on_only_for_single_organism():
    # reconcile гейтится single_organism — в колонии legacy (drift-fallback)
    cli = _client(_FakeWC(creature_pos={}))
    cli.compute = _compute(single=False, reconcile=True)
    assert cli._resolve_pos("c1", {"row": 247, "col": 74}) == (247, 74)
