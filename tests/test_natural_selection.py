"""Phase 4 этап G — естественный отбор tracking tests.

Pure scoring functions, не задействуют LocalColonyCompute:
  - normalize функции (low/high boundaries)
  - mental_break severity classification
  - score_organism per cid
  - rank descending
  - weakest_n top-N
  - snapshot для diag

Vision §3.1 «вариант D»: client не вмешивается — только наблюдает.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ────────────────────────────────────────────────────────────────────
# Fake biochem (duck-type)
# ────────────────────────────────────────────────────────────────────

class _FakeBiochem:
    """Минимальный duck-type под ClientCreatureBiochem."""

    def __init__(self, **kwargs):
        # дефолты соответствуют healthy organism
        defaults = {
            "energy": 100.0,
            "cortisol": 0.0,
            "fatigue": 0.0,
            "glucose": 100.0,
            "infected": False,
            "infection_severity": 0.0,
            "mental_break": "",
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


# ────────────────────────────────────────────────────────────────────
# Normalize functions
# ────────────────────────────────────────────────────────────────────

def test_norm_low_boundary_at_threshold_returns_zero():
    from utopia_client.natural_selection import _norm_low
    assert _norm_low(30.0, threshold=30.0) == 0.0
    assert _norm_low(100.0, threshold=30.0) == 0.0  # выше — тоже 0


def test_norm_low_at_floor_returns_one():
    from utopia_client.natural_selection import _norm_low
    assert _norm_low(0.0, threshold=30.0) == 1.0


def test_norm_low_linear_mid():
    from utopia_client.natural_selection import _norm_low
    # threshold=30, floor=0, value=15 → 0.5
    result = _norm_low(15.0, threshold=30.0, floor=0.0)
    assert abs(result - 0.5) < 1e-6


def test_norm_high_at_threshold_returns_zero():
    from utopia_client.natural_selection import _norm_high
    assert _norm_high(70.0, threshold=70.0) == 0.0


def test_norm_high_at_ceil_returns_one():
    from utopia_client.natural_selection import _norm_high
    assert _norm_high(100.0, threshold=70.0, ceil=100.0) == 1.0


# ────────────────────────────────────────────────────────────────────
# Mental break severity
# ────────────────────────────────────────────────────────────────────

def test_mental_break_empty_state_zero():
    from utopia_client.natural_selection import _mental_break_severity
    assert _mental_break_severity("") == 0.0
    assert _mental_break_severity(None) == 0.0


def test_mental_break_catatonic_max():
    from utopia_client.natural_selection import _mental_break_severity
    assert _mental_break_severity("catatonic") == 1.0


def test_mental_break_depression_high():
    from utopia_client.natural_selection import _mental_break_severity
    assert _mental_break_severity("depression") == 0.7


def test_mental_break_case_insensitive():
    from utopia_client.natural_selection import _mental_break_severity
    assert _mental_break_severity("CATATONIC") == 1.0
    assert _mental_break_severity("Berserk") == 0.4


# ────────────────────────────────────────────────────────────────────
# score_organism
# ────────────────────────────────────────────────────────────────────

def test_healthy_organism_low_score():
    from utopia_client.natural_selection import score_organism
    bc = _FakeBiochem()  # все дефолты — healthy
    f = score_organism(bc, cid="alpha")
    assert f.weakness_score < 0.01
    assert f.cid == "alpha"


def test_low_energy_increases_score():
    from utopia_client.natural_selection import score_organism
    healthy = score_organism(_FakeBiochem(energy=100.0))
    weak = score_organism(_FakeBiochem(energy=5.0))
    assert weak.weakness_score > healthy.weakness_score
    assert weak.energy > 0.5


def test_chronic_cortisol_increases_score():
    from utopia_client.natural_selection import score_organism
    healthy = score_organism(_FakeBiochem(cortisol=0.0))
    stressed = score_organism(_FakeBiochem(cortisol=95.0))
    assert stressed.weakness_score > healthy.weakness_score
    assert stressed.cortisol_chronic > 0.5


def test_catatonic_dominates_score():
    """mental_break=catatonic с weight=1.5 — самый тяжёлый компонент."""
    from utopia_client.natural_selection import score_organism
    bc = _FakeBiochem(mental_break="catatonic")  # всё остальное healthy
    f = score_organism(bc)
    # catatonic = 1.0 * 1.5 (weight) = 1.5
    assert f.mental_break == 1.0
    assert f.weakness_score >= 1.5


def test_infection_increases_score():
    from utopia_client.natural_selection import score_organism
    f = score_organism(_FakeBiochem(infected=True, infection_severity=0.8))
    assert f.infection > 0.5
    assert f.weakness_score > 0.5


def test_multiple_factors_sum_up():
    """Несколько критериев суммируются — терминальная особь."""
    from utopia_client.natural_selection import score_organism
    terminal = _FakeBiochem(
        energy=5.0,
        cortisol=90.0,
        fatigue=85.0,
        glucose=10.0,
        infected=True,
        infection_severity=0.9,
        mental_break="catatonic",
    )
    f = score_organism(terminal)
    # Все 6 факторов на максимуме → score должен быть > sum веса
    assert f.weakness_score > 3.0


def test_score_uses_weights_override():
    from utopia_client.natural_selection import score_organism

    bc = _FakeBiochem(energy=0.0)  # max energy weakness = 1.0
    # с default weights: energy_low=1.0 → contribution 1.0
    default = score_organism(bc)
    # double weight
    custom = score_organism(bc, weights={
        "energy_low": 2.0, "cortisol_chronic": 0.0, "fatigue_high": 0.0,
        "glucose_low": 0.0, "infection": 0.0, "mental_break": 0.0,
    })
    assert custom.weakness_score == 2.0
    assert default.weakness_score < custom.weakness_score


# ────────────────────────────────────────────────────────────────────
# rank_organisms
# ────────────────────────────────────────────────────────────────────

def test_rank_descending_by_score():
    from utopia_client.natural_selection import rank_organisms
    biochems = {
        "healthy": _FakeBiochem(),
        "weak": _FakeBiochem(energy=10.0),
        "terminal": _FakeBiochem(mental_break="catatonic"),
    }
    ranked = rank_organisms(biochems)
    # weakest first
    assert ranked[0].cid == "terminal"
    assert ranked[-1].cid == "healthy"
    # Scores monotonically decreasing
    for i in range(len(ranked) - 1):
        assert ranked[i].weakness_score >= ranked[i + 1].weakness_score


def test_rank_empty_dict():
    from utopia_client.natural_selection import rank_organisms
    assert rank_organisms({}) == []


def test_weakest_n_returns_top_n():
    from utopia_client.natural_selection import weakest_n
    biochems = {f"cid-{i}": _FakeBiochem(energy=100.0 - i * 20)
                for i in range(5)}
    weak3 = weakest_n(biochems, n=3)
    assert len(weak3) == 3
    # cid-4 имеет лучшую energy=20.0 → самый weak
    assert "cid-4" in weak3


# ────────────────────────────────────────────────────────────────────
# Diag snapshot
# ────────────────────────────────────────────────────────────────────

def test_snapshot_empty_returns_zeros():
    from utopia_client.natural_selection import natural_selection_snapshot
    s = natural_selection_snapshot({}, capacity=10)
    assert s["n_organisms"] == 0
    assert s["capacity"] == 10
    assert s["weakest_cids"] == []
    assert s["scores"] == {}
    assert s["mean_score"] == 0.0
    assert s["max_score"] == 0.0


def test_snapshot_includes_capacity():
    from utopia_client.natural_selection import natural_selection_snapshot
    biochems = {"a": _FakeBiochem()}
    s = natural_selection_snapshot(biochems, capacity=12)
    assert s["capacity"] == 12


def test_snapshot_emits_weakest_3_by_default():
    from utopia_client.natural_selection import natural_selection_snapshot
    biochems = {
        "h1": _FakeBiochem(),
        "h2": _FakeBiochem(),
        "w1": _FakeBiochem(energy=10.0),
        "w2": _FakeBiochem(mental_break="catatonic"),
        "w3": _FakeBiochem(cortisol=95.0),
    }
    s = natural_selection_snapshot(biochems, capacity=10)
    assert s["n_organisms"] == 5
    assert len(s["weakest_cids"]) == 3
    # Catatonic должен быть в top-3
    assert "w2" in s["weakest_cids"]


def test_snapshot_max_score_correct():
    from utopia_client.natural_selection import natural_selection_snapshot
    biochems = {
        "h": _FakeBiochem(),
        "t": _FakeBiochem(mental_break="catatonic", energy=5.0),
    }
    s = natural_selection_snapshot(biochems, capacity=2)
    # max должен соответствовать terminal organism
    assert s["max_score"] >= 1.5  # catatonic + low energy
    assert s["max_score"] == max(s["scores"].values())


def test_snapshot_top_n_configurable():
    from utopia_client.natural_selection import natural_selection_snapshot
    biochems = {f"cid-{i}": _FakeBiochem(energy=100.0 - i * 10)
                for i in range(10)}
    s = natural_selection_snapshot(biochems, capacity=10, top_n_to_emit=5)
    assert len(s["weakest_cids"]) == 5


def test_snapshot_does_not_mutate_input():
    from utopia_client.natural_selection import natural_selection_snapshot
    bc = _FakeBiochem(energy=50.0)
    biochems = {"x": bc}
    _ = natural_selection_snapshot(biochems, capacity=1)
    # Energy не должен измениться
    assert bc.energy == 50.0
