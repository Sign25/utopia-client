"""anti-freeze (Бендер 20.06): Адам рекуррентно застревает в карманах/краях, что
nav_repellent (obs wall-канал) ПРОПУСКАЕТ (стена вне канала → server move-no-op → pos
заморожен несмотря на move-эмит). ФИКС: position-based детект (pos не меняется N тиков)
→ коммит-ротация направлений ПО ФАКТУ неподвижности → выбивает из кармана. OFF dormant.
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

_N = wsm.ColonyWSClient._ANTIFREEZE_STUCK_N


class _Cfg:
    def __init__(self, size): self.size = size


class _FakeWC:
    def __init__(self, creature_pos=None, size=256):
        self.config = _Cfg(size)
        self.terrain = bytes(size * size)
        self.creature_pos = dict(creature_pos or {})  # cid → (x=col, y=row)


def _client(wc):
    cli = wsm.ColonyWSClient(server="https://e.com", token="t",
                             colony_name="cheef", client_version="0.0.0",
                             estimated_population=0)
    cli.world_cache = wc
    return cli


def _compute(single=True, on=False):
    c = LocalColonyCompute(device="cpu")
    c._single_organism = single
    if on:
        c.set_anti_freeze(True)
    return c


def test_setter_default_off():
    c = LocalColonyCompute(device="cpu")
    assert c._anti_freeze_enabled is False
    assert c.set_anti_freeze(True) is True
    assert c._anti_freeze_enabled is True
    assert c.set_anti_freeze(False) is False


def test_frozen_triggers_escape():
    # pos фиксирован → после N тиков застоя → коммит-побег (move-направление)
    cli = _client(_FakeWC(creature_pos={"c1": (50, 50)}))
    cli.compute = _compute(on=True)
    creatures = [{"cid": "c1"}]
    co = None
    for _ in range(_N + 2):
        co = [{"cid": "c1", "action": 4}]              # STAY каждый тик
        cli._apply_anti_freeze(creatures, co)
    assert co[0]["action"] in (0, 1, 2, 3)             # побег = move
    assert co[0].get("life_critical") is True          # bypass §3


def test_not_frozen_no_override():
    # pos меняется каждый тик → не застрял → STAY не тронут
    wc = _FakeWC(creature_pos={"c1": (50, 50)})
    cli = _client(wc)
    cli.compute = _compute(on=True)
    creatures = [{"cid": "c1"}]
    co = None
    for i in range(_N + 5):
        wc.creature_pos["c1"] = (50 + i, 50)           # двигается
        co = [{"cid": "c1", "action": 4}]
        cli._apply_anti_freeze(creatures, co)
    assert co[0]["action"] == 4                         # не тронут (движется)


def test_pos_change_resets_stuck():
    # застрял почти до порога, потом сдвинулся → счётчик сброс → нет побега
    wc = _FakeWC(creature_pos={"c1": (50, 50)})
    cli = _client(wc)
    cli.compute = _compute(on=True)
    creatures = [{"cid": "c1"}]
    for _ in range(_N - 1):                             # почти застрял
        cli._apply_anti_freeze(creatures, [{"cid": "c1", "action": 4}])
    wc.creature_pos["c1"] = (60, 60)                    # сдвинулся
    co = [{"cid": "c1", "action": 4}]
    cli._apply_anti_freeze(creatures, co)
    assert co[0]["action"] == 4                         # сброшен, нет побега


def test_off_no_op():
    # флаг OFF → _apply_anti_freeze не зовётся из пайплайна; прямой вызов гейтит single,
    # но при OFF метод вызывается из caller только под флагом — проверяем bit-identical
    # через caller-гейт: здесь компьют OFF, метод сам не гейтит флаг (гейт в caller),
    # но single-гейт держит. Проверяем что флаг-сеттер OFF не активирует.
    c = _compute(single=True, on=False)
    assert c._anti_freeze_enabled is False             # dormant


def test_not_single_no_op():
    cli = _client(_FakeWC(creature_pos={"c1": (50, 50)}))
    cli.compute = _compute(single=False, on=True)       # колония
    creatures = [{"cid": "c1"}]
    co = None
    for _ in range(_N + 2):
        co = [{"cid": "c1", "action": 4}]
        cli._apply_anti_freeze(creatures, co)
    assert co[0]["action"] == 4                          # single-гейт → no-op


def test_escape_rotates_directions():
    # под застоем направление меняется по циклу (систематический перебор сторон)
    cli = _client(_FakeWC(creature_pos={"c1": (50, 50)}))
    cli.compute = _compute(on=True)
    creatures = [{"cid": "c1"}]
    dirs = set()
    for _ in range(_N + wsm.ColonyWSClient._ANTIFREEZE_PERSIST * 4 + 2):
        co = [{"cid": "c1", "action": 4}]
        cli._apply_anti_freeze(creatures, co)
        if co[0]["action"] in (0, 1, 2, 3):
            dirs.add(co[0]["action"])
    assert len(dirs) >= 2                                # перебрал ≥2 направления
