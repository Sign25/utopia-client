"""stamina 1b.2 (Фрай/Хьюберт §18.6/18.11) — hp → §3-paralysis-триггер + closure.

1b.2a (hp_paralysis): hp≤max_hp/φ⁷≈45 ДОБАВЛЯЕТСЯ как §3-триггер РЯДОМ с energy≤0
(overlap-guardrail, immortal обоими) + passive_water-backstop в параличе (закрывает
водную absorbing-дыру §18.11) + §3-recovery грант +hp/+hydration. 1b.2b (hp_death):
energy-§3 СНЯТ → hp единственный триггер. Death-equiv для immortal-Адама = §3-paralysis.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute, _HP_S3_THRESHOLD, _RECOVERY_HP, _PASSIVE_WATER,
)
from utopia_client.biochemistry import ClientCreatureBiochem  # noqa: E402


def _compute(energy=500.0, hydration=100.0, hp=500.0):
    c = LocalColonyCompute(device="cpu")
    c._single_organism = True
    c._damage_factor = 1.0
    bc = ClientCreatureBiochem()
    bc.energy, bc.hydration, bc.hp = energy, hydration, hp
    c.biochem["c0"] = bc
    return c, bc


def _rates(**kw):
    base = {"step_cost_per_tick": 0.0, "basal_drain_per_tick": 0.0,
            "thirst_per_tick": 0.0, "telomere_decay_per_tick": 0.0}
    base.update(kw)
    return base


# ── _s3_trigger split (energy vs hp по флагам) ──────────────────────────────
def test_s3_trigger_default_energy_only():
    c, bc = _compute(energy=0.0, hp=1000.0)
    assert c._s3_trigger(bc) is True                 # energy≤0 → §3
    c2, bc2 = _compute(energy=100.0, hp=10.0)        # hp≤45 но флаги OFF
    assert c2._s3_trigger(bc2) is False              # hp НЕ триггер (default)


def test_s3_trigger_1b2a_overlap():
    """1b.2a: energy≤0 ИЛИ hp≤порог (оба активны = overlap-guardrail)."""
    c, bc = _compute(energy=100.0, hp=10.0)          # hp≤45
    c.set_hp_paralysis(True)
    assert c._s3_trigger(bc) is True                 # hp триггерит
    c2, bc2 = _compute(energy=0.0, hp=1000.0)        # energy≤0
    c2.set_hp_paralysis(True)
    assert c2._s3_trigger(bc2) is True               # energy ВСЁ ЕЩЁ триггерит (overlap)
    c3, bc3 = _compute(energy=100.0, hp=1000.0)
    c3.set_hp_paralysis(True)
    assert c3._s3_trigger(bc3) is False              # оба ok


def test_s3_trigger_1b2b_removes_energy():
    """1b.2b: energy-§3 СНЯТ → hp единственный триггер."""
    c, bc = _compute(energy=0.0, hp=1000.0)          # energy≤0 но hp ok
    c.set_hp_death(True)
    assert c._s3_trigger(bc) is False                # energy-§3 снят → НЕ триггер
    c2, bc2 = _compute(energy=100.0, hp=10.0)
    c2.set_hp_death(True)
    assert c2._s3_trigger(bc2) is True               # hp≤45 → триггер


def test_threshold_value():
    """hp-порог ≈45 (max_hp/φ⁷)."""
    assert _HP_S3_THRESHOLD == pytest.approx(45.0, abs=1.0)
    assert _RECOVERY_HP == pytest.approx(73.0, abs=1.5)  # max_hp/φ⁶, выше порога


# ── флаги ───────────────────────────────────────────────────────────────────
def test_flags_default_off_and_setters():
    c = LocalColonyCompute(device="cpu")
    assert c._hp_paralysis_enabled is False and c._hp_death_enabled is False
    assert c.set_hp_paralysis(True) is True and c._hp_paralysis_enabled is True
    assert c.set_hp_death(True) is True and c._hp_death_enabled is True


# ── hp-§3 вход + passive_water + recovery ───────────────────────────────────
def test_hp_low_enters_paralysis():
    """hp≤порог + hp_paralysis → _apply_metabolism вводит §3 (по hp)."""
    c, bc = _compute(energy=100.0, hydration=50.0, hp=10.0)
    c.set_hp_paralysis(True)
    c._apply_metabolism("c0", _rates())
    assert "c0" in c._paralysis_until                # вошёл в §3 по hp


def test_passive_water_during_paralysis():
    """В параличе (hp_paralysis) is_adam → passive_water влил hydration (закрытие
    absorbing-дыры §18.11: вода-далеко тоже восстанавливается)."""
    c, bc = _compute(energy=100.0, hydration=0.0, hp=500.0)
    c.set_hp_paralysis(True)
    c._paralysis_until["c0"] = time.monotonic() + 100.0   # в параличе (не истёк)
    c._apply_metabolism("c0", _rates())
    assert bc.hydration == pytest.approx(_PASSIVE_WATER)  # вода восстановилась (была 0)


def test_recovery_grant_lifts_hp():
    """§3-recovery (паралич истёк) + hp_paralysis → грант +hp (climb из §3-зоны) +
    hydration. Выход из hp-§3 (non-absorbing)."""
    c, bc = _compute(energy=10.0, hydration=5.0, hp=20.0)  # hp в §3-зоне
    c.set_hp_paralysis(True)
    c._paralysis_until["c0"] = time.monotonic() - 1.0      # истёк → recovery
    c._apply_metabolism("c0", _rates())
    assert bc.hp == pytest.approx(_RECOVERY_HP)            # hp поднят выше порога
    assert bc.hp > _HP_S3_THRESHOLD                        # вышел из §3-зоны
    assert bc.hydration >= 30.0                            # hydration restored
    assert "c0" not in c._paralysis_until                  # recovery снял паралич


def test_passive_water_off_when_dormant():
    """Флаги OFF → passive_water НЕ льёт (инертно)."""
    c, bc = _compute(energy=100.0, hydration=0.0, hp=500.0)
    c._paralysis_until["c0"] = time.monotonic() + 100.0
    c._apply_metabolism("c0", _rates())
    assert bc.hydration == 0.0                             # нет passive_water (dormant)
