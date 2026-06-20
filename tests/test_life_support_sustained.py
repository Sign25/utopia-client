"""sustained-life_support (i) (Хьюберт §20.6.5): держит energy/hyd выше порога КАЖДЫЙ
тик (не разовый edge как life_support) → выживание DECOUPLED (нет §3/дегидратации) →
ось stamina (усталость/отдых) наблюдаем в ИЗОЛЯЦИИ без food-halo-follows-дрейф-конфаундов.
НЕ трогает fatigue/rest (max() только energy/hyd). OFF dormant → bit-identical. single-Адам.
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
    LocalColonyCompute, _LIFE_SUSTAIN_ENERGY, _LIFE_SUSTAIN_HYDRATION,
)
from utopia_client.biochemistry import make_default  # noqa: E402


def _c(single=True, sustained=False):
    c = LocalColonyCompute(device="cpu")
    c._single_organism = single
    if sustained:
        c.set_life_support_sustained(True)
    return c


def test_setter_default_off():
    c = LocalColonyCompute(device="cpu")
    assert c._life_support_sustained_enabled is False
    assert c.set_life_support_sustained(True) is True
    assert c._life_support_sustained_enabled is True
    assert c.set_life_support_sustained(False) is False


def test_thresholds_phi_aligned():
    # φ⁻²·1309 ≈ 500, φ⁻¹·100 ≈ 61.8 (выше §3/critical и thirst-онсета 55)
    assert abs(_LIFE_SUSTAIN_ENERGY - 1309.0 * (1.618033988749895 ** -2)) < 1e-6
    assert _LIFE_SUSTAIN_ENERGY > 89.0                        # выше critical
    assert _LIFE_SUSTAIN_HYDRATION > 55.0                     # выше thirst-yield


def test_sustain_tops_up_low_energy_hyd():
    c = _c(single=True, sustained=True)
    bc = make_default()
    bc.energy = 5.0; bc.hydration = 2.0; bc.fatigue = 77.0   # краш + усталость
    c.biochem["c1"] = bc
    c._apply_life_sustain("c1")
    assert bc.energy >= _LIFE_SUSTAIN_ENERGY                  # подтянут
    assert bc.hydration >= _LIFE_SUSTAIN_HYDRATION
    assert bc.fatigue == 77.0                                 # ось НЕ тронута


def test_sustain_does_not_lower():
    # max() не понижает: высокие energy/hyd остаются
    c = _c(single=True, sustained=True)
    bc = make_default()
    bc.energy = 1000.0; bc.hydration = 95.0
    c.biochem["c1"] = bc
    c._apply_life_sustain("c1")
    assert bc.energy == 1000.0
    assert bc.hydration == 95.0


def test_off_no_topup():
    c = _c(single=True, sustained=False)              # dormant
    bc = make_default()
    bc.energy = 5.0; bc.hydration = 2.0
    c.biochem["c1"] = bc
    c._apply_life_sustain("c1")
    assert bc.energy == 5.0                            # не тронут (bit-identical)
    assert bc.hydration == 2.0


def test_not_for_colony():
    c = _c(single=False, sustained=True)              # колония
    bc = make_default()
    bc.energy = 5.0; bc.hydration = 2.0
    c.biochem["c1"] = bc
    c._apply_life_sustain("c1")
    assert bc.energy == 5.0                            # гейт single → no-op
    assert bc.hydration == 2.0


def test_clears_paralysis_latch():
    import time
    c = _c(single=True, sustained=True)
    bc = make_default(); bc.energy = 1.0
    c.biochem["c1"] = bc
    c._paralysis_until["c1"] = time.monotonic() + 999  # §3-защёлка
    c._apply_life_sustain("c1")
    assert "c1" not in c._paralysis_until               # снята
