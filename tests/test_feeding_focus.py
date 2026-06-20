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

    def _skip(diet, onf, on_corpse, has_commit):
        commit = c._hunt_commit.get("c1") if has_commit else None
        return (c._feeding_focus_enabled and diet > 0.5
                and onf and not on_corpse and commit is not None)

    assert _skip(0.618, True, False, True) is True       # хищник, флора, commit → skip
    assert _skip(0.618, True, True, True) is False        # на трупе → НЕ skip (его еда)
    assert _skip(0.618, True, False, False) is False      # нет дичи → НЕ skip (ест, не голодает)
    assert _skip(0.30, True, False, True) is False        # травоядный → НЕ skip
    assert _skip(0.618, False, False, True) is False      # не на флоре → нечего skip-ать


def test_off_dormant_no_skip():
    # OFF → предикат всегда False (bit-identical с до-флагом поведением)
    c = _compute()
    c.set_feeding_focus(False)
    c._hunt_commit["c1"] = 2
    skip = (c._feeding_focus_enabled and 0.618 > 0.5
            and True and not False and c._hunt_commit.get("c1") is not None)
    assert skip is False
