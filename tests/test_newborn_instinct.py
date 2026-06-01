"""Newborn-инстинкт GATHER/EAT (01.06.2026, Фрай — порт server phase_a.py:748-755).

Client-рождённые особи тянутся к GATHER (на флоре) и EAT (с едой в инвентаре)
первые 500 тиков, затем инстинкт затухает → motor_policy ест сама на выученном
eat-reward. carried_food — клиентское зеркало (P40 его не шлёт).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")
import torch  # noqa: E402

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402

# Action enum (world.py): GATHER=13, EAT=14, SHARE_FOOD=15
GATHER, EAT, SHARE_FOOD = 13, 14, 15


def _c():
    return LocalColonyCompute(device="cpu")


# ── _apply_newborn_instinct ───────────────────────────────────────────

def test_instinct_noop_for_untracked():
    # Особь без birth_tick (старая/restored) — инстинкта нет.
    c = _c()
    lg = torch.zeros(16)
    c._apply_newborn_instinct("old", lg, world_tick=10, on_flora=True)
    assert lg[13].item() == 0.0 and lg[14].item() == 0.0


def test_instinct_gather_on_flora_fresh():
    # birth_tick=age0 → instinct=1.0, на флоре, carried=0 → GATHER +2, EAT 0.
    c = _c()
    c._birth_tick["n"] = 100
    c._carried_food["n"] = 0
    lg = torch.zeros(16)
    c._apply_newborn_instinct("n", lg, world_tick=100, on_flora=True)
    assert abs(lg[13].item() - 2.0) < 1e-6
    assert lg[14].item() == 0.0  # нечего есть


def test_instinct_uses_p40_carried_food_over_mirror():
    # P40 authoritative carried_food (9f8d99d) > client-зеркало (убирает desync).
    c = _c()
    c._birth_tick["n"] = 0
    c._carried_food["n"] = 0          # зеркало говорит 0
    lg = torch.zeros(16)
    # P40 говорит carried=3 → EAT bias, несмотря на зеркало=0
    c._apply_newborn_instinct("n", lg, world_tick=0, on_flora=False,
                              carried_food=3)
    assert abs(lg[14].item() - 2.0) < 1e-6   # EAT по P40-истине


def test_instinct_fallback_to_mirror_when_p40_none():
    # carried_food=None (P40 не шлёт) → fallback на зеркало.
    c = _c()
    c._birth_tick["n"] = 0
    c._carried_food["n"] = 2
    lg = torch.zeros(16)
    c._apply_newborn_instinct("n", lg, world_tick=0, on_flora=False,
                              carried_food=None)
    assert abs(lg[14].item() - 2.0) < 1e-6   # EAT по зеркалу


def test_instinct_eat_when_carrying():
    # carried=3 → EAT +2; на флоре и <5 → ещё и GATHER +2.
    c = _c()
    c._birth_tick["n"] = 0
    c._carried_food["n"] = 3
    lg = torch.zeros(16)
    c._apply_newborn_instinct("n", lg, world_tick=0, on_flora=True)
    assert abs(lg[13].item() - 2.0) < 1e-6
    assert abs(lg[14].item() - 2.0) < 1e-6


def test_instinct_decays_linearly():
    # age=250 → instinct=0.5 → GATHER +1.0.
    c = _c()
    c._birth_tick["n"] = 0
    c._carried_food["n"] = 0
    lg = torch.zeros(16)
    c._apply_newborn_instinct("n", lg, world_tick=250, on_flora=True)
    assert abs(lg[13].item() - 1.0) < 1e-6


def test_instinct_expired_removes_tracking():
    # age>=500 → instinct<=0 → no-op + трекинг снят (особь «выросла»).
    c = _c()
    c._birth_tick["n"] = 0
    c._carried_food["n"] = 2
    lg = torch.zeros(16)
    c._apply_newborn_instinct("n", lg, world_tick=500, on_flora=True)
    assert lg[13].item() == 0.0 and lg[14].item() == 0.0
    assert "n" not in c._birth_tick
    assert "n" not in c._carried_food


def test_instinct_gather_capped_at_full():
    # carried=5 (cap) → GATHER не бустится (нечего собирать), EAT бустится.
    c = _c()
    c._birth_tick["n"] = 0
    c._carried_food["n"] = 5
    lg = torch.zeros(16)
    c._apply_newborn_instinct("n", lg, world_tick=0, on_flora=True)
    assert lg[13].item() == 0.0
    assert abs(lg[14].item() - 2.0) < 1e-6


# ── _update_carried_food_mirror ───────────────────────────────────────

def test_mirror_gather_on_flora_increments():
    c = _c()
    c._carried_food["n"] = 0
    c._update_carried_food_mirror("n", GATHER, on_flora=True)
    assert c._carried_food["n"] == 1


def test_mirror_gather_off_flora_noop():
    c = _c()
    c._carried_food["n"] = 2
    c._update_carried_food_mirror("n", GATHER, on_flora=False)
    assert c._carried_food["n"] == 2


def test_mirror_gather_capped():
    c = _c()
    c._carried_food["n"] = 5
    c._update_carried_food_mirror("n", GATHER, on_flora=True)
    assert c._carried_food["n"] == 5  # cap F(5)


def test_mirror_eat_decrements():
    c = _c()
    c._carried_food["n"] = 3
    c._update_carried_food_mirror("n", EAT, on_flora=False)
    assert c._carried_food["n"] == 2


def test_mirror_eat_floor_zero():
    c = _c()
    c._carried_food["n"] = 0
    c._update_carried_food_mirror("n", EAT, on_flora=False)
    assert c._carried_food["n"] == 0


def test_mirror_share_food_decrements():
    c = _c()
    c._carried_food["n"] = 1
    c._update_carried_food_mirror("n", SHARE_FOOD, on_flora=False)
    assert c._carried_food["n"] == 0


# ── lifecycle: remove_creature чистит трекинг ─────────────────────────

def test_remove_creature_clears_instinct_state():
    c = _c()
    c._birth_tick["n"] = 10
    c._carried_food["n"] = 2
    c.remove_creature("n")
    assert "n" not in c._birth_tick
    assert "n" not in c._carried_food


# ── bootstrap: омоложение restored-особей ─────────────────────────────

def test_bootstrap_rejuvenates_registered_alive():
    import types
    c = _c()
    c.organisms["r1"] = types.SimpleNamespace()
    c._bootstrap_pending.add("r1")
    c._apply_bootstrap_pending(world_tick=1000)
    assert c._birth_tick.get("r1") == 1000   # омоложён → инстинкт активен
    assert c._carried_food.get("r1") == 0
    assert c._n_bootstrap_rejuv == 1
    assert "r1" not in c._bootstrap_pending   # pending очищен


def test_bootstrap_skips_dead_or_unregistered():
    import types
    c = _c()
    c.organisms["alive"] = types.SimpleNamespace()
    c._dead_cids.add("dead")
    c.organisms["dead"] = types.SimpleNamespace()
    c._bootstrap_pending.update({"alive", "dead", "ghost"})
    c._apply_bootstrap_pending(world_tick=500)
    assert "alive" in c._birth_tick
    assert "dead" not in c._birth_tick      # мёртвого не омолаживаем
    assert "ghost" not in c._birth_tick     # незарегистрированного нет
    assert c._n_bootstrap_rejuv == 1


def test_bootstrap_noop_when_empty():
    c = _c()
    c._apply_bootstrap_pending(world_tick=10)
    assert c._n_bootstrap_rejuv == 0
