"""root-3 рефлекс-rest-floor (Хьюберт §20.6.3): single-Адаму был отключён force-STAY@
exhaustion (absorbing-страх слепил glucose/catatonic/exhaustion) → нет рефлекс-отдыха →
fat пин 85 (выносл=15) + gate-2 без bootstrap-примеров отдыха. Client-половина (моя):
(2) force-STAY@exhaustion для single → видимый отдых, (3) blocked-move→0-fatigue. ОДИН
флаг fatigue_rest_floor (OFF dormant → bit-identical). bio-половина (decay/гистерезис) —
в environment.biochemistry (Хьюберт), вызывается из сеттера.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute, STAY, _MOVE_WALL_OBS,
)
from utopia_client.biochemistry import make_default  # noqa: E402


def _c(single=True, rest_floor=False):
    c = LocalColonyCompute(device="cpu")
    c._single_organism = single
    if rest_floor:
        c.set_fatigue_rest_floor(True)
    return c


# ── сеттер / dormant ──────────────────────────────────────────────────────

def test_setter_default_off():
    c = LocalColonyCompute(device="cpu")
    assert c._fatigue_rest_floor_enabled is False
    assert c.set_fatigue_rest_floor(True) is True
    assert c._fatigue_rest_floor_enabled is True
    assert c.set_fatigue_rest_floor(False) is False


# ── force-STAY@exhaustion (часть 2) ───────────────────────────────────────

def test_rest_floor_forces_stay_on_exhaustion_single():
    c = _c(single=True, rest_floor=True)
    bc = make_default(); bc.mental_break = "exhaustion"
    c.biochem["c1"] = bc
    out = {"c1": {"action": 2, "target_id": None}}        # move E
    c._maybe_force_stay("c1", out)
    assert out["c1"]["action"] == STAY                     # отдых


def test_rest_floor_off_no_force_single():
    # OFF: для single biochem force-STAY остаётся отключён (bit-identical) → move цел
    c = _c(single=True, rest_floor=False)
    bc = make_default(); bc.mental_break = "exhaustion"
    c.biochem["c1"] = bc
    out = {"c1": {"action": 2, "target_id": None}}
    c._maybe_force_stay("c1", out)
    assert out["c1"]["action"] == 2                         # НЕ тронут


def test_rest_floor_only_exhaustion_not_catatonic():
    # rest-floor фаирит ТОЛЬКО на exhaustion; catatonic/glucose остаются OFF для single
    c = _c(single=True, rest_floor=True)
    bc = make_default(); bc.mental_break = "catatonic"
    c.biochem["c1"] = bc
    out = {"c1": {"action": 2, "target_id": None}}
    c._maybe_force_stay("c1", out)
    assert out["c1"]["action"] == 2                         # catatonic не форсит (single)


def test_rest_floor_not_for_colony():
    # гейт single_organism: в колонии rest-floor-ветка не применяется
    c = _c(single=False, rest_floor=True)
    bc = make_default(); bc.mental_break = "exhaustion"
    c.biochem["c1"] = bc
    out = {"c1": {"action": 2, "target_id": None}}
    c._maybe_force_stay("c1", out)
    # колония: biochem should_force_stay путь (не rest-floor) — exhaustion → STAY ИЛИ
    # (если env недоступен) move цел; в любом случае rest-floor-ветка не сработала.
    # Проверяем что rest_floor-счётчик НЕ инкрементнулся (ветка пропущена):
    assert getattr(c, "_rest_floor_n", 0) == 0


# ── blocked-move → 0-fatigue (часть 3) ────────────────────────────────────

def _fat_setup(rest_floor: bool):
    c = _c(single=True, rest_floor=rest_floor)
    c._fatigue_b = 0.081
    bc = make_default()
    bc.mental_break = ""        # НЕ exhaustion → incap-gate не скипает
    bc.fatigue = 20.0
    c.biochem["c1"] = bc
    return c, bc


def test_blocked_move_no_fatigue():
    c, bc = _fat_setup(rest_floor=True)
    c._nav_blocked_move = {"c1": True}        # удар-в-стену без смещения
    c._apply_action_fatigue("c1", 2)          # move E ∈ _MOVE_WALL_OBS
    assert bc.fatigue == 20.0                 # НЕ копит (нет смещения)


def test_unblocked_move_accumulates():
    c, bc = _fat_setup(rest_floor=True)
    c._nav_blocked_move = {"c1": False}       # смещение есть
    c._apply_action_fatigue("c1", 2)
    assert bc.fatigue > 20.0                  # копит (реальное движение)


def test_blocked_move_off_still_accumulates():
    # флаг OFF → blocked move копит как раньше (bit-identical)
    c, bc = _fat_setup(rest_floor=False)
    c._nav_blocked_move = {"c1": True}
    c._apply_action_fatigue("c1", 2)
    assert bc.fatigue > 20.0


def test_blocked_constant_mapping():
    assert _MOVE_WALL_OBS == {0: 2, 1: 18, 2: 10, 3: 26}
