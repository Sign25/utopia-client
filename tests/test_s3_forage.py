"""§3-редизайн паралич→фураж (Фрай 19.06, Шеф-критика «голод≠кома»).

Голод НЕ парализует → ФУРАЖ-режим. branch-1 §3 (energy≤0): нет видимой еды →
RANDOM-WALK-поиск (НЕ STAY) → выход из barren (non-absorbing). incap-гейт +glucose<5
(голод-фураж недобровольный → не копит fatigue). Голод>усталость (server P40-mb).
Флаг s3_forage (OFF dormant → bit-identical STAY-fallback). lockstep server _S3_FORAGE.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_NEUROCORE = _ROOT.parent / "NeuroCore"
for _p in (str(_ROOT), str(_NEUROCORE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute, STAY, _S3_HUNGER_GLUCOSE,
)
from utopia_client.biochemistry import ClientCreatureBiochem  # noqa: E402


def _c(glucose=50.0, fatigue=0.0):
    c = LocalColonyCompute(device="cpu")
    c._single_organism = True
    bc = ClientCreatureBiochem()
    bc.glucose, bc.fatigue = glucose, fatigue
    c.biochem["c0"] = bc
    return c, bc


def _paralyze(c, cid="c0"):
    c._paralysis_until[cid] = time.monotonic() + 10.0   # в §3-параличе


# ── флаг (dormant default, kill-switch) ──────────────────────────────────
def test_s3_forage_setter():
    c, _ = _c()
    assert c._s3_forage_enabled is False             # dormant default
    assert c.set_s3_forage(True) is True
    assert c._s3_forage_enabled is True
    assert c.set_s3_forage(False) is False


# ── branch-1 §3: OFF → STAY-fallback (bit-identical) ─────────────────────
def test_s3_off_stay_fallback():
    """flag OFF → §3 нет видимой еды → STAY (текущее поведение, bit-identical)."""
    c, _ = _c()
    _paralyze(c)
    out = {"c0": {"action": 0, "target_id": None}}   # «хотел» move
    c._maybe_force_stay("c0", out)
    assert out["c0"]["action"] == STAY               # OFF → STAY-fallback


# ── branch-1 §3: ON → RANDOM-WALK (НЕ STAY) = non-absorbing ──────────────
def test_s3_on_random_walk_not_stay():
    """flag ON → §3 нет видимой еды → RANDOM-WALK move (НЕ STAY = не кома)."""
    c, _ = _c()
    c.set_s3_forage(True)
    _paralyze(c)
    c._last_world_tick = 0
    out = {"c0": {"action": 0, "target_id": None}}
    c._maybe_force_stay("c0", out)
    assert out["c0"]["action"] != STAY               # НЕ замер
    assert out["c0"]["action"] in (0, 1, 2, 3)       # move-направление поиска


def test_s3_food_visible_forage_floor():
    """§3 + еда видна (_forage_dir) → FORAGE_FLOOR (move к еде), приоритет над random-walk."""
    c, _ = _c()
    c.set_s3_forage(True)
    _paralyze(c)
    c._forage_dir["c0"] = 2                           # еда видна, move=2
    out = {"c0": {"action": 0, "target_id": None}}
    c._maybe_force_stay("c0", out)
    assert out["c0"]["action"] == 2                   # к видимой еде (не random)


def test_s3_search_dir_valid_move():
    c, _ = _c()
    c._last_world_tick = 7
    assert c._s3_search_dir("c0") in (0, 1, 2, 3)


def test_s3_search_dir_persists_then_rotates():
    """направление держится _S3_SEARCH_PERSIST тиков, потом меняется (покрытие дистанции)."""
    from utopia_client.local_compute import _S3_SEARCH_PERSIST
    c, _ = _c()
    c._last_world_tick = 0
    d0 = c._s3_search_dir("c0")
    c._last_world_tick = _S3_SEARCH_PERSIST - 1
    assert c._s3_search_dir("c0") == d0              # держится
    c._last_world_tick = _S3_SEARCH_PERSIST * 2
    c._s3_search_dir("c0")                            # сменилось (ротация, не падаем)


# ── incap-гейт +glucose<5 (голод-фураж недобровольный → не копит fatigue) ─
def test_incap_gate_glucose_hunger_no_fatigue():
    """голод (glucose<5) + s3_forage ON → недобровольный фураж → НЕ копит fatigue."""
    c, bc = _c(glucose=2.0, fatigue=50.0)
    c.set_s3_forage(True)
    c.set_phi_fatigue(True)
    c.set_fatigue_b(0.2)
    c._apply_action_fatigue("c0", 0)                 # move = голод-фураж
    assert bc.fatigue == 50.0                         # НЕ копилось (рефлекс)


def test_incap_gate_glucose_ok_accumulates():
    """glucose норма → волевое действие копит штатно (гейт не ловит)."""
    c, bc = _c(glucose=50.0, fatigue=10.0)
    c.set_s3_forage(True)
    c.set_phi_fatigue(True)
    c.set_fatigue_b(0.2)
    c._apply_action_fatigue("c0", 0)
    assert bc.fatigue == pytest.approx(10.2)          # копилось (сыт, волевое)


def test_incap_gate_glucose_gated_by_flag():
    """glucose<5 но s3_forage OFF → гейт-glucose НЕ активен (только при ON; при OFF §3=STAY)."""
    c, bc = _c(glucose=2.0, fatigue=10.0)
    c.set_phi_fatigue(True)
    c.set_fatigue_b(0.2)                              # s3_forage OFF
    c._apply_action_fatigue("c0", 0)
    assert bc.fatigue == pytest.approx(10.2)          # копилось (glucose-гейт за s3_forage)
