"""Phase 4 этап E — mate detection tests.

Pure functions. Тестируем:
  - is_reproduction_ready критерии (energy / oxytocin / serotonin / mental_break)
  - detect_mate_pairs cooldown enforcement
  - cross-owner запрет (через scope: только own dict)
  - deterministic pairing (sort by cid)
  - max_pairs rate limit
  - asexual detection variant
  - mark_mate_event cooldown update
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


class _FakeBiochem:
    """Минимальный duck-type под ClientCreatureBiochem."""

    def __init__(self, **kwargs):
        defaults = {
            "energy": 100.0,
            "oxytocin": 60.0,
            "serotonin": 60.0,
            "mental_break": "",
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


# ────────────────────────────────────────────────────────────────────
# is_reproduction_ready
# ────────────────────────────────────────────────────────────────────

def test_healthy_organism_is_ready():
    from utopia_client.mate_detection import is_reproduction_ready
    assert is_reproduction_ready(_FakeBiochem()) is True


def test_low_energy_not_ready():
    from utopia_client.mate_detection import is_reproduction_ready
    assert is_reproduction_ready(_FakeBiochem(energy=15.0)) is False


def test_low_oxytocin_not_ready():
    from utopia_client.mate_detection import is_reproduction_ready
    assert is_reproduction_ready(_FakeBiochem(oxytocin=10.0)) is False


def test_low_serotonin_not_ready():
    from utopia_client.mate_detection import is_reproduction_ready
    assert is_reproduction_ready(_FakeBiochem(serotonin=10.0)) is False


def test_catatonic_not_ready():
    from utopia_client.mate_detection import is_reproduction_ready
    assert is_reproduction_ready(_FakeBiochem(mental_break="catatonic")) is False


def test_depression_not_ready():
    from utopia_client.mate_detection import is_reproduction_ready
    assert is_reproduction_ready(_FakeBiochem(mental_break="depression")) is False


def test_mild_mental_break_still_ready():
    """berserk / mania / loner — не блокируют (force_STAY не действует)."""
    from utopia_client.mate_detection import is_reproduction_ready
    assert is_reproduction_ready(_FakeBiochem(mental_break="berserk")) is True
    assert is_reproduction_ready(_FakeBiochem(mental_break="loner")) is True


# ────────────────────────────────────────────────────────────────────
# detect_mate_pairs
# ────────────────────────────────────────────────────────────────────

def test_empty_dict_no_pairs():
    from utopia_client.mate_detection import detect_mate_pairs
    assert detect_mate_pairs({}, {}, world_tick=1000) == []


def test_single_organism_no_pair():
    from utopia_client.mate_detection import detect_mate_pairs
    pairs = detect_mate_pairs({"a": _FakeBiochem()}, {}, world_tick=1000)
    assert pairs == []


def test_two_ready_form_pair():
    from utopia_client.mate_detection import detect_mate_pairs
    biochems = {"alpha": _FakeBiochem(), "beta": _FakeBiochem()}
    pairs = detect_mate_pairs(biochems, {}, world_tick=1000)
    assert len(pairs) == 1
    # Mother = алфавитно меньший cid
    assert pairs[0] == ("alpha", "beta")


def test_pair_blocked_by_cooldown():
    from utopia_client.mate_detection import detect_mate_pairs
    biochems = {"a": _FakeBiochem(), "b": _FakeBiochem()}
    last_mate = {"a": 800}  # 200 < 500 cooldown
    pairs = detect_mate_pairs(biochems, last_mate, world_tick=1000)
    # a в cooldown → b остался один → нет пары
    assert pairs == []


def test_pair_after_cooldown_elapsed():
    from utopia_client.mate_detection import detect_mate_pairs
    biochems = {"a": _FakeBiochem(), "b": _FakeBiochem()}
    last_mate = {"a": 100, "b": 100}  # 1000 - 100 = 900 >= 500
    pairs = detect_mate_pairs(biochems, last_mate, world_tick=1000)
    assert len(pairs) == 1


def test_unready_organism_not_in_pair():
    """Если b unready, a остался один → нет пары."""
    from utopia_client.mate_detection import detect_mate_pairs
    biochems = {
        "a": _FakeBiochem(),
        "b": _FakeBiochem(energy=10.0),  # unready
    }
    pairs = detect_mate_pairs(biochems, {}, world_tick=1000)
    assert pairs == []


def test_three_ready_max_one_pair():
    """Default max_pairs=1, lone third остаётся unpaired."""
    from utopia_client.mate_detection import detect_mate_pairs
    biochems = {f"cid-{i}": _FakeBiochem() for i in range(3)}
    pairs = detect_mate_pairs(biochems, {}, world_tick=1000)
    assert len(pairs) == 1
    # Deterministic: sorted → cid-0 + cid-1
    assert pairs[0] == ("cid-0", "cid-1")


def test_max_pairs_override():
    """С max_pairs=2 в группе 4 — получаем 2 пары."""
    from utopia_client.mate_detection import detect_mate_pairs
    biochems = {f"cid-{i}": _FakeBiochem() for i in range(4)}
    pairs = detect_mate_pairs(biochems, {}, world_tick=1000, max_pairs=2)
    assert len(pairs) == 2
    # Pair 1: cid-0+cid-1, Pair 2: cid-2+cid-3
    assert pairs[0] == ("cid-0", "cid-1")
    assert pairs[1] == ("cid-2", "cid-3")


def test_organism_in_only_one_pair():
    """Каждый organism — в максимум одной паре per tick."""
    from utopia_client.mate_detection import detect_mate_pairs
    biochems = {f"cid-{i}": _FakeBiochem() for i in range(4)}
    pairs = detect_mate_pairs(biochems, {}, world_tick=1000, max_pairs=10)
    used = set()
    for m, f in pairs:
        assert m not in used and f not in used
        used.add(m)
        used.add(f)


def test_deterministic_pairing_alpha_order():
    """Same input → same pairs (для repeatability)."""
    from utopia_client.mate_detection import detect_mate_pairs
    biochems = {"z": _FakeBiochem(), "a": _FakeBiochem(), "m": _FakeBiochem()}
    pairs_1 = detect_mate_pairs(biochems, {}, world_tick=1000)
    pairs_2 = detect_mate_pairs(biochems, {}, world_tick=1000)
    assert pairs_1 == pairs_2
    # Sorted alphabetically — a + m в паре, z остаётся
    assert pairs_1[0] == ("a", "m")


# ────────────────────────────────────────────────────────────────────
# Asexual candidates
# ────────────────────────────────────────────────────────────────────

def test_asexual_requires_more_energy():
    """Asexual default min_energy = 50 (vs 30 для sexual)."""
    from utopia_client.mate_detection import detect_asexual_candidates
    # energy=40 — достаточно для sexual, не для asexual
    biochems = {"a": _FakeBiochem(energy=40.0)}
    cands = detect_asexual_candidates(biochems, {}, world_tick=1000)
    assert cands == []
    # energy=60 — OK для asexual
    biochems_high = {"a": _FakeBiochem(energy=60.0)}
    cands_high = detect_asexual_candidates(biochems_high, {}, world_tick=1000)
    assert cands_high == ["a"]


def test_asexual_respects_cooldown():
    from utopia_client.mate_detection import detect_asexual_candidates
    biochems = {"a": _FakeBiochem()}
    cands = detect_asexual_candidates(
        biochems, last_mate_tick={"a": 999}, world_tick=1000)
    assert cands == []


def test_asexual_max_births_limit():
    from utopia_client.mate_detection import detect_asexual_candidates
    biochems = {f"c-{i}": _FakeBiochem() for i in range(5)}
    cands = detect_asexual_candidates(biochems, {}, world_tick=1000, max_births=2)
    assert len(cands) == 2


# ────────────────────────────────────────────────────────────────────
# mark_mate_event
# ────────────────────────────────────────────────────────────────────

def test_mark_updates_cooldown():
    from utopia_client.mate_detection import mark_mate_event
    last_mate: dict = {}
    mark_mate_event(last_mate, ["a", "b"], world_tick=5000)
    assert last_mate == {"a": 5000, "b": 5000}


def test_mark_overwrites_previous():
    from utopia_client.mate_detection import mark_mate_event
    last_mate = {"a": 100}
    mark_mate_event(last_mate, ["a"], world_tick=2000)
    assert last_mate["a"] == 2000


def test_mark_empty_list_no_op():
    from utopia_client.mate_detection import mark_mate_event
    last_mate = {"a": 100}
    mark_mate_event(last_mate, [], world_tick=2000)
    assert last_mate == {"a": 100}


# ────────────────────────────────────────────────────────────────────
# Integration — full cycle
# ────────────────────────────────────────────────────────────────────

def test_detect_mark_detect_cycle():
    """E2E: detect → mark → detect (после cooldown) → second detect."""
    from utopia_client.mate_detection import (
        detect_mate_pairs, mark_mate_event,
    )
    biochems = {"a": _FakeBiochem(), "b": _FakeBiochem()}
    last_mate: dict = {}

    # Tick 1000: пара формируется
    pairs1 = detect_mate_pairs(biochems, last_mate, world_tick=1000)
    assert len(pairs1) == 1
    mark_mate_event(last_mate, [pairs1[0][0], pairs1[0][1]], world_tick=1000)

    # Tick 1100: cooldown не прошёл → нет пары
    pairs2 = detect_mate_pairs(biochems, last_mate, world_tick=1100)
    assert pairs2 == []

    # Tick 1600 (>1000+500): cooldown прошёл → снова пара
    pairs3 = detect_mate_pairs(biochems, last_mate, world_tick=1600)
    assert len(pairs3) == 1
    assert pairs3[0] == ("a", "b")


def test_no_cross_owner_in_pure_function():
    """detect_mate_pairs принимает только один dict — caller гарантирует
    что это own organisms (cross-owner запрет на уровне caller)."""
    from utopia_client.mate_detection import detect_mate_pairs
    # Этот тест проверяет интерфейс — функция не принимает "other_user" arg
    own_biochems = {"a": _FakeBiochem(), "b": _FakeBiochem()}
    pairs = detect_mate_pairs(own_biochems, {}, world_tick=1000)
    # Pure function: всё что передано — кандидаты, никаких filter по
    # owner_user_id. Cross-owner enforcement = scope каллера.
    assert len(pairs) == 1
