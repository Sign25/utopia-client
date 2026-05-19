"""Variant B leak fix (19.05.2026): GC ghost cid'ов в LocalColonyCompute.

Unit-тесты на `_gc_orphan_cids`. Мокируем `ws.compute` — нам нужен только
интерфейс `organisms: dict[str, ...]` + `remove_creature(cid)`. Реальный
LocalColonyCompute требует storage/core deps, которых в этом окружении нет.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utopia_client.ws_client import (  # noqa: E402
    CID_GC_INTERVAL_TICKS,
    STALE_CID_TICKS,
    ColonyWSClient,
)


class _FakeCompute:
    """Минимум для GC: словарь organisms + remove_creature."""

    def __init__(self) -> None:
        self.organisms: dict[str, str] = {}
        self.removed: list[str] = []

    def add(self, cid: str) -> None:
        self.organisms[cid] = "stub"

    def remove_creature(self, cid: str) -> None:
        self.organisms.pop(cid, None)
        self.removed.append(cid)


@pytest.fixture
def ws() -> ColonyWSClient:
    return ColonyWSClient(
        server="https://example.com",
        token="t",
        colony_name="test",
        client_version="0.0.0",
    )


def test_gc_removes_stale_cid_absent_from_obs(ws: ColonyWSClient) -> None:
    """cid в organisms, last_seen старее threshold → удаляется."""
    fake = _FakeCompute()
    fake.add("ghost")
    fake.add("alive")
    ws.compute = fake  # type: ignore[assignment]

    ws._cid_last_seen_tick["ghost"] = 100
    ws._cid_last_seen_tick["alive"] = 5000
    world_tick = 100 + STALE_CID_TICKS + 50

    ws._gc_orphan_cids(world_tick)

    assert fake.removed == ["ghost"]
    assert "ghost" not in fake.organisms
    assert "alive" in fake.organisms
    assert "ghost" not in ws._cid_last_seen_tick
    assert ws._cid_gc_total == 1


def test_gc_skips_fresh_cid(ws: ColonyWSClient) -> None:
    """cid с свежим last_seen остаётся, ничего не удалено."""
    fake = _FakeCompute()
    fake.add("a")
    fake.add("b")
    ws.compute = fake  # type: ignore[assignment]
    ws._cid_last_seen_tick["a"] = 4900
    ws._cid_last_seen_tick["b"] = 4990

    ws._gc_orphan_cids(world_tick=5000)

    assert fake.removed == []
    assert ws._cid_gc_total == 0


def test_gc_removes_never_seen_cid(ws: ColonyWSClient) -> None:
    """cid в organisms, но last_seen вообще не выставлен → удаляется.

    Это покрывает редкий путь: add_creature был вызван минуя ws callsite
    (например, через прямой compute API в тестах) — GC всё равно подберёт.
    """
    fake = _FakeCompute()
    fake.add("orphan")
    ws.compute = fake  # type: ignore[assignment]

    ws._gc_orphan_cids(world_tick=STALE_CID_TICKS + 10)

    assert fake.removed == ["orphan"]


def test_gc_respects_interval(ws: ColonyWSClient) -> None:
    """Повторный GC до CID_GC_INTERVAL_TICKS — noop, last_run_tick без изменений."""
    fake = _FakeCompute()
    fake.add("ghost")
    ws.compute = fake  # type: ignore[assignment]
    ws._cid_last_seen_tick["ghost"] = 100

    first_tick = STALE_CID_TICKS + 1000
    ws._gc_orphan_cids(first_tick)
    assert fake.removed == ["ghost"]
    assert ws._cid_gc_last_run_tick == first_tick

    fake.add("another")
    ws._cid_last_seen_tick["another"] = 100
    # Слишком близко по тикам — GC не должен сработать.
    ws._gc_orphan_cids(first_tick + CID_GC_INTERVAL_TICKS - 1)
    assert fake.removed == ["ghost"]  # без изменений
    assert "another" in fake.organisms


def test_gc_skips_when_world_young(ws: ColonyWSClient) -> None:
    """До STALE_CID_TICKS мира GC выключен — новорождённые ещё не успели в obs."""
    fake = _FakeCompute()
    fake.add("never_seen")
    ws.compute = fake  # type: ignore[assignment]

    ws._gc_orphan_cids(world_tick=STALE_CID_TICKS - 1)

    assert fake.removed == []
    assert ws._cid_gc_total == 0


def test_gc_noop_without_compute(ws: ColonyWSClient) -> None:
    """compute=None — GC должен молча выйти."""
    assert ws.compute is None
    ws._gc_orphan_cids(world_tick=10_000_000)
    assert ws._cid_gc_total == 0
