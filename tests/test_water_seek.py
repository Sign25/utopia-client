"""Water-seek рефлекс (31.05.2026, Хьюберт+Бендер). zodchiy не навигируют к
воде (obs без water-градиента) → drink_sum=0 → дегидратация. Рефлекс: при
hydration<30% override action к ближайшему WATER-тайлу.

Конвенция Мира (environment/world.py:261): NORTH=0→(-1,0), SOUTH=1→(+1,0),
EAST=2→(0,+1), WEST=3→(0,-1).
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


class _FakeWC:
    def __init__(self, size, water_cells):
        self.config = _Cfg(size)
        t = bytearray(size * size)  # 0 = PLAIN
        for (r, c) in water_cells:
            t[r * size + c] = 1  # WATER
        self.terrain = bytes(t)


def _client(wc=None):
    cli = wsm.ColonyWSClient(server="https://e.com", token="t",
                             colony_name="cheef", client_version="0.0.0",
                             estimated_population=0)
    cli.world_cache = wc
    return cli


def test_water_north():
    cli = _client(_FakeWC(20, [(2, 10)]))  # water севернее (меньше row)
    assert cli._water_seek_action(10, 10) == 0  # NORTH


def test_water_south():
    cli = _client(_FakeWC(20, [(18, 10)]))
    assert cli._water_seek_action(10, 10) == 1  # SOUTH


def test_water_east():
    cli = _client(_FakeWC(20, [(10, 16)]))
    assert cli._water_seek_action(10, 10) == 2  # EAST


def test_water_west():
    cli = _client(_FakeWC(20, [(10, 4)]))
    assert cli._water_seek_action(10, 10) == 3  # WEST


def test_nearest_water_chosen():
    # вода и близко (восток, d=2) и далеко (север, d=7) → ближняя (восток)
    cli = _client(_FakeWC(20, [(3, 10), (10, 12)]))
    assert cli._water_seek_action(10, 10) == 2  # EAST (ближе)


def test_no_water_in_radius():
    cli = _client(_FakeWC(40, [(39, 39)]))  # вода далеко (>radius 8)
    assert cli._water_seek_action(5, 5) is None


def test_no_world_cache():
    cli = _client(None)
    assert cli._water_seek_action(5, 5) is None


def test_apply_water_seek_overrides_thirsty():
    pytest.importorskip("torch")
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.biochemistry import make_default
    import types
    c = LocalColonyCompute(device="cpu")
    bc = make_default(); bc.hydration = 10.0  # жаждущий (<30)
    c.biochem["c1"] = bc
    c.organisms["c1"] = types.SimpleNamespace(generation=0)
    cli = _client(_FakeWC(20, [(10, 16)]))  # вода восточнее
    cli.compute = c
    creatures = [{"cid": "c1", "row": 10, "col": 10}]
    creatures_out = [{"cid": "c1", "action": 1, "target_id": None}]  # был SOUTH
    cli._apply_water_seek(creatures, creatures_out)
    assert creatures_out[0]["action"] == 2  # override → EAST (к воде)


def test_near_water_on_tile():
    cli = _client(_FakeWC(20, [(10, 10)]))  # организм прямо на воде
    assert cli._near_water(10, 10) is True


def test_near_water_adjacent():
    cli = _client(_FakeWC(20, [(10, 11)]))  # вода — сосед справа (радиус 1)
    assert cli._near_water(10, 10) is True


def test_near_water_far_false():
    cli = _client(_FakeWC(20, [(10, 13)]))  # 3 клетки — вне радиуса-1
    assert cli._near_water(10, 10) is False


def test_near_water_no_cache():
    cli = _client(None)
    assert cli._near_water(5, 5) is False


def test_apply_water_seek_skips_hydrated():
    pytest.importorskip("torch")
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.biochemistry import make_default
    import types
    c = LocalColonyCompute(device="cpu")
    bc = make_default(); bc.hydration = 90.0  # НЕ жаждущий
    c.biochem["c1"] = bc
    c.organisms["c1"] = types.SimpleNamespace(generation=0)
    cli = _client(_FakeWC(20, [(10, 16)]))
    cli.compute = c
    creatures = [{"cid": "c1", "row": 10, "col": 10}]
    creatures_out = [{"cid": "c1", "action": 1, "target_id": None}]
    cli._apply_water_seek(creatures, creatures_out)
    assert creatures_out[0]["action"] == 1  # НЕ тронут (не жаждущий)
