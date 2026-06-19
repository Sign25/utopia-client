"""φ-арбитраж голод↔жажда (Хьюберт 19.06): single-Адам фиксировался на ОДНОМ драйве —
water-seek=рефлекс-backstop (всегда обслуживался), food-seek=OFF (еда без backstop) →
XOR-асимметрия, необслуженный драйв уходит в 0 → §3. Фикс: симметричный анти­ципаторный
арбитраж — обслужить МЕНЬШИЙ-относительный (energy/max_e vs hydration/max_h), sticky-latch
+ φ-deadband гистерезис, «не гнать один в 0». Флаг need_arbitration (OFF dormant).

Конвенция Мира: NORTH=0→(-1,0), SOUTH=1→(+1,0), EAST=2→(0,+1), WEST=3→(0,-1).
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
from utopia_client.biochemistry import make_default  # noqa: E402
import types  # noqa: E402


class _Cfg:
    def __init__(self, size): self.size = size


class _FakeWC:
    def __init__(self, size, water_cells=(), flora=(), creature_pos=None):
        self.config = _Cfg(size)
        t = bytearray(size * size)  # 0 = PLAIN
        for (r, c) in water_cells:
            t[r * size + c] = 1  # WATER
        self.terrain = bytes(t)
        self.flora = list(flora)            # [(row, col, kind)]
        self.creature_pos = dict(creature_pos or {})


def _client(wc=None):
    cli = wsm.ColonyWSClient(server="https://e.com", token="t",
                             colony_name="cheef", client_version="0.0.0",
                             estimated_population=0)
    cli.world_cache = wc
    return cli


def _compute_with(energy, hyd, max_e=1000.0, max_h=100.0, cid="c1"):
    c = LocalColonyCompute(device="cpu")
    c._single_organism = True
    bc = make_default()
    bc.energy = float(energy)
    bc.hydration = float(hyd)
    bc.max_energy = float(max_e)
    bc.max_hydration = float(max_h)
    c.biochem[cid] = bc
    c.organisms[cid] = types.SimpleNamespace(generation=0)
    return c


# ── сеттер / dormant default ──────────────────────────────────────────────

def test_setter_default_off():
    c = LocalColonyCompute(device="cpu")
    assert c._need_arbitration_enabled is False           # dormant
    assert c.set_need_arbitration(True) is True
    assert c._need_arbitration_enabled is True
    assert c.set_need_arbitration(False) is False


# ── оба драйва сыты (выше onset φ⁻¹) → брейн ведёт, арбитраж молчит ────────

def test_both_satisfied_no_override():
    c = _compute_with(energy=700, hyd=70)   # e_rel=0.70, h_rel=0.70 ≥ 0.618
    cli = _client(_FakeWC(20, water_cells=[(10, 16)], flora=[(10, 12, 2)],
                          creature_pos={"c1": (10, 10)}))
    cli.compute = c
    co = [{"cid": "c1", "action": 1, "target_id": None}]   # был SOUTH
    cli._apply_need_arbitration([{"cid": "c1"}], co)
    assert co[0]["action"] == 1                            # не тронут
    assert cli._need_arb_target.get("c1") is None          # latch очищен


# ── меньший-относительный = голод → food-seek-навигация ────────────────────

def test_hunger_lower_seeks_food():
    c = _compute_with(energy=100, hyd=90)    # e_rel=0.10 << h_rel=0.90 → food
    cli = _client(_FakeWC(20, water_cells=[(10, 16)], flora=[(10, 12, 2)],
                          creature_pos={"c1": (10, 10)}))   # berry восточнее
    cli.compute = c
    co = [{"cid": "c1", "action": 1, "target_id": None}]
    cli._apply_need_arbitration([{"cid": "c1"}], co)
    assert cli._need_arb_target["c1"] == "food"
    assert co[0]["action"] == 2                            # EAST → к berry


# ── меньший-относительный = жажда → water-seek-навигация ───────────────────

def test_thirst_lower_seeks_water():
    c = _compute_with(energy=900, hyd=10)    # e_rel=0.90 >> h_rel=0.10 → water
    cli = _client(_FakeWC(20, water_cells=[(10, 16)], flora=[(10, 12, 2)],
                          creature_pos={"c1": (10, 10)}))   # вода восточнее
    cli.compute = c
    co = [{"cid": "c1", "action": 1, "target_id": None}]
    cli._apply_need_arbitration([{"cid": "c1"}], co)
    assert cli._need_arb_target["c1"] == "water"
    assert co[0]["action"] == 2                            # EAST → к воде


# ── yield-to-consume: у ресурса нужного драйва → не оверрайдим ─────────────

def test_yield_to_drink_near_water():
    c = _compute_with(energy=900, hyd=10)    # жажда-winner
    cli = _client(_FakeWC(20, water_cells=[(10, 11)], flora=[],
                          creature_pos={"c1": (10, 10)}))   # вода — сосед
    cli.compute = c
    co = [{"cid": "c1", "action": 4, "target_id": None}]    # STAY (пьёт)
    cli._apply_need_arbitration([{"cid": "c1"}], co)
    assert co[0]["action"] == 4                            # не тронут (пьёт)


def test_yield_to_eat_on_berry():
    c = _compute_with(energy=100, hyd=90)    # голод-winner
    cli = _client(_FakeWC(20, water_cells=[], flora=[(10, 10, 2)],
                          creature_pos={"c1": (10, 10)}))   # на berry
    cli.compute = c
    co = [{"cid": "c1", "action": 4, "target_id": None}]    # ест
    cli._apply_need_arbitration([{"cid": "c1"}], co)
    assert co[0]["action"] == 4                            # не тронут (ест)


# ── гистерезис: sticky-latch не переключается на маргинальной разнице ──────

def test_hysteresis_keeps_target_within_deadband():
    # 1-й тик: голод чуть ниже → latch=food
    c = _compute_with(energy=100, hyd=120)   # e_rel=0.10, h_rel=0.12 → food
    cli = _client(_FakeWC(20, water_cells=[(10, 16)], flora=[(10, 12, 2)],
                          creature_pos={"c1": (10, 10)}))
    cli.compute = c
    co = [{"cid": "c1", "action": 1}]
    cli._apply_need_arbitration([{"cid": "c1"}], co)
    assert cli._need_arb_target["c1"] == "food"
    # 2-й тик: жажда теперь чуть НИЖЕ (h_rel=0.09 < e_rel=0.10), но в пределах
    # deadband (φ⁻⁷≈0.0344): 0.09 ≥ 0.10−0.0344 → ДЕРЖИМ food (без thrash)
    c.biochem["c1"].hydration = 9.0
    co2 = [{"cid": "c1", "action": 1}]
    cli._apply_need_arbitration([{"cid": "c1"}], co2)
    assert cli._need_arb_target["c1"] == "food"            # гистерезис держит


def test_hysteresis_switches_when_clearly_more_urgent():
    c = _compute_with(energy=100, hyd=120)   # latch=food
    cli = _client(_FakeWC(20, water_cells=[(10, 16)], flora=[(10, 12, 2)],
                          creature_pos={"c1": (10, 10)}))
    cli.compute = c
    cli._apply_need_arbitration([{"cid": "c1"}], [{"cid": "c1", "action": 1}])
    assert cli._need_arb_target["c1"] == "food"
    # жажда обвалилась ниже deadband: h_rel=0.05 < e_rel(0.10)−0.0344=0.0656 → switch
    c.biochem["c1"].hydration = 5.0
    co = [{"cid": "c1", "action": 1}]
    cli._apply_need_arbitration([{"cid": "c1"}], co)
    assert cli._need_arb_target["c1"] == "water"           # переключился
    assert co[0]["action"] == 2                            # EAST → к воде


# ── life_critical → bypass §3-force-STAY ──────────────────────────────────

def test_food_life_critical_when_energy_critical():
    c = _compute_with(energy=50, hyd=90)     # energy ≤89 критично, голод-winner
    cli = _client(_FakeWC(20, water_cells=[], flora=[(10, 12, 2)],
                          creature_pos={"c1": (10, 10)}))
    cli.compute = c
    co = [{"cid": "c1", "action": 1}]
    cli._apply_need_arbitration([{"cid": "c1"}], co)
    assert co[0]["action"] == 2                            # к berry
    assert co[0]["life_critical"] is True                  # bypass §3


def test_water_life_critical_when_hydration_critical():
    c = _compute_with(energy=900, hyd=10)    # hyd ≤15 критично, жажда-winner
    cli = _client(_FakeWC(20, water_cells=[(10, 16)], flora=[],
                          creature_pos={"c1": (10, 10)}))
    cli.compute = c
    co = [{"cid": "c1", "action": 1}]
    cli._apply_need_arbitration([{"cid": "c1"}], co)
    assert co[0]["action"] == 2                            # к воде
    assert co[0]["life_critical"] is True


def test_no_life_critical_when_moderate():
    # голод-winner, но energy=300 >89 (не критично) → life_critical False
    c = _compute_with(energy=300, hyd=95)
    cli = _client(_FakeWC(20, water_cells=[], flora=[(10, 12, 2)],
                          creature_pos={"c1": (10, 10)}))
    cli.compute = c
    co = [{"cid": "c1", "action": 1}]
    cli._apply_need_arbitration([{"cid": "c1"}], co)
    assert co[0]["action"] == 2
    assert co[0]["life_critical"] is False
