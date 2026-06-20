"""food_income_server (Хьюберт §20.6.6): КОРЕНЬ «Адам не ест» — клиент кредитил energy
из cache.flora (ДЕСИНК: on_flora=1 100%, но income~0), а не из server delta_energy/
on_flora. ФИКС: честим server delta_energy + on_flora+flora_kind GROUND-TRUTH (зеркало
delta_hydration воды), бросаем cache.flora. _flora_kind_income считает income по СЕРВЕРНОМУ
flora_kind. OFF dormant → legacy cache.flora (bit-identical).
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

_PHI = 1.618033988749895


def _client():
    return wsm.ColonyWSClient(server="https://e.com", token="t",
                              colony_name="cheef", client_version="0.0.0",
                              estimated_population=0)


def _expect(base, diet, eff, vision):
    kleiber = (max(1.0, eff) / 10.0) ** (1.0 / _PHI)
    vbonus = 1.0 + (vision - 3.0) * 0.02
    return base * (1.0 - diet) * kleiber * vbonus


# ── сеттер / dormant ──────────────────────────────────────────────────────

def test_setter_default_off():
    c = LocalColonyCompute(device="cpu")
    assert c._food_income_server_enabled is False
    assert c.set_food_income_server(True) is True
    assert c._food_income_server_enabled is True
    assert c.set_food_income_server(False) is False


# ── _flora_kind_income по серверному kind (не cache.flora) ─────────────────

def test_grass_income():
    cli = _client()
    got = cli._flora_kind_income(1, diet=0.618, eff=10, vision=7)
    assert abs(got - _expect(_PHI ** -2, 0.618, 10, 7)) < 1e-9


def test_berry_income():
    cli = _client()
    got = cli._flora_kind_income(2, diet=0.618, eff=10, vision=7)
    assert abs(got - _expect(_PHI, 0.618, 10, 7)) < 1e-9


def test_fruit_income_kinds_3_to_6():
    cli = _client()
    for k in (3, 4, 5, 6):
        got = cli._flora_kind_income(k, diet=0.618, eff=10, vision=7)
        assert abs(got - _expect(_PHI ** 3, 0.618, 10, 7)) < 1e-9


def test_unknown_kind_zero():
    cli = _client()
    assert cli._flora_kind_income(0, 0.5, 10, 7) == 0.0
    assert cli._flora_kind_income(99, 0.5, 10, 7) == 0.0


def test_carnivore_diet_penalty():
    # diet 0.618 (хищник) режет растительный income ×(1-0.618)=×0.382
    cli = _client()
    herb = cli._flora_kind_income(2, diet=0.0, eff=10, vision=7)
    carn = cli._flora_kind_income(2, diet=0.618, eff=10, vision=7)
    assert abs(carn - herb * (1.0 - 0.618)) < 1e-9


def test_fruit_sustains_grass_does_not():
    # для хищника (diet 0.618) при cost ~0.3-0.4/тик: fruit кормит, grass — нет
    cli = _client()
    grass = cli._flora_kind_income(1, 0.618, 10, 7)   # ≈0.16/тик
    berry = cli._flora_kind_income(2, 0.618, 10, 7)   # ≈0.67/тик
    fruit = cli._flora_kind_income(3, 0.618, 10, 7)   # ≈1.75/тик
    assert grass < 0.3 < berry < fruit                # grass не кормит, berry/fruit да
