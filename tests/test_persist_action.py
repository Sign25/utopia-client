"""persist-opt-in (Хьюберт c8c2af8, 11.06): Адам — client-authoritative, его
мозг-клиент тикает медленнее world-loop → без persist P40 force-STAY'ит через
30 world-тиков тишины → Адам застывает/прыгает в feed. persist=true на локомоции
Адама → сервер дед-реконит последний MOVE каждый world-тик → feed плотный.

scope: ТОЛЬКО single-organism Адам (default false safe для прочих owned).
"""
from __future__ import annotations
import sys, types
from pathlib import Path
import pytest
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
pytest.importorskip("torch")
import utopia_client.ws_client as wsm  # noqa: E402


def _ws_with(actions, single=True):
    ws = wsm.ColonyWSClient(server="https://e.com", token="t",
                            colony_name="cheef", client_version="0.0.0",
                            estimated_population=0)
    comp = types.SimpleNamespace()
    comp._single_organism = single
    comp.handle_tick = lambda *a, **k: actions
    comp.get_phase_emas = lambda cid: None
    ws.compute = comp
    return ws


def _build(ws):
    out = ws._run_tick_and_build({}, {}, {}, world_tick=0)
    return {e["cid"]: e for e in (out or [])}


def test_adam_move_gets_persist():
    ws = _ws_with({"a": {"action": 2}})        # MOVE кардинальный
    e = _build(ws)["a"]
    assert e.get("persist") is True


def test_adam_flee_gets_persist_and_life_critical():
    ws = _ws_with({"a": {"action": 10}})       # FLEE — локомоция + survival
    e = _build(ws)["a"]
    assert e.get("persist") is True
    assert e.get("life_critical") is True      # коэкзистит


def test_adam_stay_no_persist():
    ws = _ws_with({"a": {"action": 4}})        # STAY — не локомоция
    e = _build(ws)["a"]
    assert "persist" not in e                  # default false (30т окно)


def test_adam_eat_no_persist():
    ws = _ws_with({"a": {"action": 14}})       # EAT — не дед-реконим
    e = _build(ws)["a"]
    assert "persist" not in e


def test_non_single_no_persist():
    ws = _ws_with({"z": {"action": 2}}, single=False)  # не Адам → opt-out
    e = _build(ws)["z"]
    assert "persist" not in e
