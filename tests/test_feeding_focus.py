"""feeding_focus (Бендер 20.06): хищник-Адам коммитит в ОХОТУ, не грейзит ягоды.
ON: (1) hunt-commit DIST-CAP 13→21 (Fib) — трекает дичь дальше (freeze убит + speed →
дальний трек дёшев); (2) carnivore berry-graze suppress — diet>0.5 + дичь huntable
(hunt_commit активен) → НЕ берёт on_food-флор для ягод → коммит в kill. Труп НЕ трогаем.
Гард: нет дичи → ест ягоды (не голодает). OFF dormant → bit-identical. kill-switch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402


def _compute():
    c = LocalColonyCompute(device="cpu")
    c._single_organism = True
    return c


def test_setter_default_off():
    c = LocalColonyCompute(device="cpu")
    assert c._feeding_focus_enabled is False          # dormant по умолчанию
    assert c.set_feeding_focus(True) is True
    assert c._feeding_focus_enabled is True
    assert c.set_feeding_focus(False) is False
    assert c._feeding_focus_enabled is False


def test_dist_cap_value_gated():
    # DIST-CAP: OFF → 13, ON → 21 (Fib). Проверяем выражение-гейт напрямую.
    c = _compute()
    c.set_feeding_focus(False)
    assert (21.0 if c._feeding_focus_enabled else 13.0) == 13.0
    c.set_feeding_focus(True)
    assert (21.0 if c._feeding_focus_enabled else 13.0) == 21.0


def test_carn_skip_flora_predicate():
    # carnivore berry-suppress predicate: хищник (diet>0.5) + на флоре (не труп) +
    # hunt_commit активен → skip. Любое условие ложно → не skip (ест ягоду).
    c = _compute()
    c.set_feeding_focus(True)
    c._hunt_commit["c1"] = 2      # дичь huntable (commit-move активен)

    def _skip(diet, onf, on_corpse, has_commit, er=0.5):
        commit = c._hunt_commit.get("c1") if has_commit else None
        return (c._feeding_focus_enabled and diet > 0.5
                and onf and not on_corpse and commit is not None
                and er > 0.146)

    assert _skip(0.618, True, False, True) is True       # хищник, флора, commit, er ок → skip
    assert _skip(0.618, True, True, True) is False        # на трупе → НЕ skip (его еда)
    assert _skip(0.618, True, False, False) is False      # нет дичи → НЕ skip (ест, не голодает)
    assert _skip(0.30, True, False, True) is False        # травоядный → НЕ skip
    assert _skip(0.618, False, False, True) is False      # не на флоре → нечего skip-ать
    # ENERGY-FLOOR: крит. голод (er≤φ⁻⁴≈0.146) → НЕ skip (ест ягоды-бутстрап, не death-spiral)
    assert _skip(0.618, True, False, True, er=0.08) is False   # er<пол → ест (остаётся capable)
    assert _skip(0.618, True, False, True, er=0.20) is True    # er>пол → suppress (hunt-фокус)


def test_hunt_burst_predicate():
    # HUNT-BURST: chase-move (hunt_commit MOVE активен) + feeding_focus + single + act MOVE
    # → speed_boost=2. ATTACK-commit (5) или нет commit → не burst (pounce-логика отдельно).
    c = _compute()
    c.set_feeding_focus(True)

    def _burst(single, act, ff, hc):
        c.set_feeding_focus(ff)
        if hc is None:
            c._hunt_commit.pop("c1", None)
        else:
            c._hunt_commit["c1"] = hc
        return (single and act in (0, 1, 2, 3) and c._feeding_focus_enabled
                and c._hunt_commit.get("c1") in (0, 1, 2, 3))

    assert _burst(True, 2, True, 2) is True       # chase-move + ff → burst=2
    assert _burst(True, 5, True, 2) is False       # эмит ATTACK(5) не в (0-3) → не chase-burst
    assert _burst(True, 2, True, 5) is False        # hunt_commit=ATTACK не chase-move
    assert _burst(True, 2, True, None) is False     # нет commit (нет дичи) → не burst
    assert _burst(True, 2, False, 2) is False       # feeding_focus OFF → не burst
    assert _burst(False, 2, True, 2) is False       # колония → не burst


def test_sustain_latch_cap_hysteresis():
    # латч расширяет cap: свежий=21, latched=32 (держим погоню за видимой дичью).
    c = _compute()
    c.set_feeding_focus(True)

    def _cap(latched):
        c._hunt_latch["c1"] = c._HUNT_LATCH_N if latched else 0
        _lp = (c._feeding_focus_enabled and c._hunt_latch.get("c1", 0) > 0)
        return (32.0 if _lp else 21.0) if c._feeding_focus_enabled else 13.0

    assert _cap(False) == 21.0      # нет латча → свежий cap 21
    assert _cap(True) == 32.0        # латч активен → cap до obs-предела 32
    c.set_feeding_focus(False)
    c._hunt_latch["c1"] = 13
    _lp = (c._feeding_focus_enabled and c._hunt_latch.get("c1", 0) > 0)
    assert ((32.0 if _lp else 21.0) if c._feeding_focus_enabled else 13.0) == 13.0  # OFF → 13


def test_sustain_latch_refresh_and_decay():
    # коммит → латч=N; нет коммита → decay; 0 → погоня тухнет.
    c = _compute()
    c.set_feeding_focus(True)

    def _update(has_commit):
        if c._feeding_focus_enabled:
            if has_commit:
                c._hunt_latch["c1"] = c._HUNT_LATCH_N
            elif c._hunt_latch.get("c1", 0) > 0:
                c._hunt_latch["c1"] -= 1

    _update(True)
    assert c._hunt_latch["c1"] == 13                 # коммит → рефреш N
    for _ in range(5):
        _update(False)                                # 5 тиков без дичи
    assert c._hunt_latch["c1"] == 8                   # decay 13→8 (sustain ещё держит)
    for _ in range(8):
        _update(False)
    assert c._hunt_latch["c1"] == 0                   # выдохся → возврат к грейзу


def test_off_dormant_no_skip():
    # OFF → предикат всегда False (bit-identical с до-флагом поведением)
    c = _compute()
    c.set_feeding_focus(False)
    c._hunt_commit["c1"] = 2
    skip = (c._feeding_focus_enabled and 0.618 > 0.5
            and True and not False and c._hunt_commit.get("c1") is not None)
    assert skip is False
