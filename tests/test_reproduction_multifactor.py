"""Многофакторный gate размножения (30.05.2026, Шеф + vision body_migration.md §).

Размножение — совокупность факторов (energy + cortisol + fatigue + glucose +
infection + mental_break), не безконтрольное (одна энергия). Без этого колония
плодилась 9× сверх ёмкости железа → perf-коллапс. Естественная регуляция:
плодятся только fit особи, server-side смерть слабых освобождает место.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

from utopia_client.biochemistry import make_default  # noqa: E402
from utopia_client.mate_detection import (  # noqa: E402
    is_reproduction_ready, detect_mate_pairs, MIN_ENERGY_FOR_REPRO,
)


def _fit_biochem():
    """Здоровая, готовая к размножению особь."""
    bc = make_default()
    bc.energy = MIN_ENERGY_FOR_REPRO + 100.0  # выше порога стоимости
    bc.cortisol = 10.0
    bc.fatigue = 10.0
    bc.glucose = 60.0
    bc.infected = False
    bc.infection_severity = 0.0
    return bc


def test_fit_organism_ready():
    assert is_reproduction_ready(_fit_biochem()) is True


def test_low_energy_blocks():
    bc = _fit_biochem()
    bc.energy = 100.0  # < порога
    assert is_reproduction_ready(bc) is False


def test_high_cortisol_blocks():
    bc = _fit_biochem()
    bc.cortisol = 85.0  # хронический стресс > 70
    assert is_reproduction_ready(bc) is False


def test_high_fatigue_blocks():
    bc = _fit_biochem()
    bc.fatigue = 80.0  # переутомление > 70
    assert is_reproduction_ready(bc) is False


def test_low_glucose_blocks():
    bc = _fit_biochem()
    bc.glucose = 10.0  # голод < 25
    assert is_reproduction_ready(bc) is False


def test_infected_blocks():
    bc = _fit_biochem()
    bc.infected = True
    assert is_reproduction_ready(bc) is False
    bc2 = _fit_biochem()
    bc2.infection_severity = 0.5  # > 0.1
    assert is_reproduction_ready(bc2) is False


def test_mental_break_blocks():
    bc = _fit_biochem()
    bc.mental_break = "catatonic"
    assert is_reproduction_ready(bc) is False


def test_detect_skips_unfit_in_pairing():
    """detect_mate_pairs не парит больных/стрессованных даже при энергии."""
    fit_a = _fit_biochem()
    fit_b = _fit_biochem()
    sick = _fit_biochem()
    sick.infected = True            # болен — не должен попасть в пару
    stressed = _fit_biochem()
    stressed.cortisol = 90.0        # хронический стресс — не в пару
    biochems = {"a": fit_a, "b": fit_b, "sick": sick, "stressed": stressed}
    pairs = detect_mate_pairs(biochems, {}, world_tick=10000, max_pairs=8)
    paired = {c for pair in pairs for c in pair}
    assert "sick" not in paired
    assert "stressed" not in paired
    # два здоровых спарились
    assert paired == {"a", "b"}


def test_default_biochem_repro_friendly():
    """make_default дефолты не блокируют (cortisol=0, glucose=50, fatigue=0,
    не болен) — backward-compat с energy-only тестами."""
    bc = make_default()
    bc.energy = MIN_ENERGY_FOR_REPRO + 50.0
    assert is_reproduction_ready(bc) is True
