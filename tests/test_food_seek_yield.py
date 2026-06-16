"""Food-seek YIELD-TO-EAT (16.06.2026, Фрай GO; Бендер диагноз).

Регресс edible-seeking: client `_apply_food_seek` (energy<450) перебивал
ЛЕГИТИМНЫЙ EAT на MOVE-к-ягоде. Его «на ягоде» = точное (r,c) in berry_pos
(client-кэш) рассинхронен с серверным on_flora (→ compute._on_food → eat-reflex).
В зазоре food-seek перетирал EAT → укус не стартовал (EAT_PROGRESS kind=None) →
голод среди ягод в режиме energy<450.

Фикс: food-seek УСТУПАЕТ eat — skip если compute._on_food[cid] (тот же серверный
сигнал, что eat-reflex). Тест: на еде НЕ перебивает, не на еде — перебивает (как
раньше). Конвенция направлений: N=0,S=1,E=2,W=3 (world.py:261).
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


class _Cfg:
    def __init__(self, size): self.size = size


class _BC:
    def __init__(self, energy, hydration=99.0):
        self.energy = energy
        self.hydration = hydration


class _FakeWC:
    def __init__(self, size, flora, creature_pos):
        self.config = _Cfg(size)
        self.flora = list(flora)          # [(row, col, kind), ...]
        # creature_pos: dict[cid → (x=col, y=row)] (world_cache.py:384)
        self.creature_pos = dict(creature_pos)


class _FakeCompute:
    def __init__(self, biochem, on_food):
        self.biochem = dict(biochem)
        self._on_food = dict(on_food)


def _client(wc, compute):
    cli = wsm.ColonyWSClient(server="https://e.com", token="t",
                             colony_name="cheef", client_version="0.0.0",
                             estimated_population=0)
    cli.world_cache = wc
    cli.compute = compute
    return cli


def _setup(on_food, energy=100.0):
    # Адам (10,10), голоден (energy<450); ягода (kind=2) восточнее на (10,11).
    wc = _FakeWC(20, flora=[(10, 11, 2)], creature_pos={"a": (10, 10)})
    compute = _FakeCompute(biochem={"a": _BC(energy)}, on_food=on_food)
    cli = _client(wc, compute)
    creatures = [{"cid": "a", "row": 10, "col": 10}]
    creatures_out = [{"cid": "a", "action": 14}]   # eat-reflex уже поставил EAT
    cli._apply_food_seek(creatures, creatures_out)
    return creatures_out[0]["action"]


def test_yields_to_eat_when_on_food():
    """compute._on_food[cid] истинно → food-seek НЕ перебивает EAT (14 остаётся)."""
    assert _setup(on_food={"a": 1}) == 14


def test_overrides_when_not_on_food():
    """НЕ на еде → food-seek перебивает на MOVE к ягоде (восток=2). Регресс-гард:
    фикс не сломал основную функцию ведения к высокоценной флоре."""
    assert _setup(on_food={}) == 2


def test_no_override_when_fed():
    """energy ≥ 450 → food-seek вообще не активен (EAT/любое действие нетронуто)."""
    wc = _FakeWC(20, flora=[(10, 11, 2)], creature_pos={"a": (10, 10)})
    compute = _FakeCompute(biochem={"a": _BC(500.0)}, on_food={})
    cli = _client(wc, compute)
    creatures = [{"cid": "a", "row": 10, "col": 10}]
    creatures_out = [{"cid": "a", "action": 14}]
    cli._apply_food_seek(creatures, creatures_out)
    assert creatures_out[0]["action"] == 14
