"""stamina шаг 1b.1 (Фрай/Хьюберт §18) — hp оживает (AUTHORITATIVE).

ON (hp_authoritative): снять hp=energy зеркало + урон→hp (не energy) + дренаж нужд
(сытость/вода)=0→hp φ³ + лечение φ² при нуждах в норме. death/§3 ЕЩЁ на energy
(guardrail) → hp вниз НЕ убивает (живой recoverable-замер). OFF → инертно (зеркало).
Выносливость-дренаж INERT до φ-расход (§18.7). LOCKSTEP: Хьюберт снимает
creature.hp=energy из decay_step per-creature (is_adam).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_NEUROCORE = _ROOT.parent / "NeuroCore"
for _p in (str(_ROOT), str(_NEUROCORE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute, _CLIENT_MAX_ENERGY, _HP_NEED_DRAIN, _HP_HEAL,
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
    """Минимальный per-tick rates (путь _tick_metab: dt=1, без wall-dt)."""
    base = {"step_cost_per_tick": 0.0, "basal_drain_per_tick": 0.0,
            "thirst_per_tick": 0.0, "telomere_decay_per_tick": 0.0}
    base.update(kw)
    return base


def test_flag_default_off():
    c = LocalColonyCompute(device="cpu")
    assert c._hp_authoritative_enabled is False


def test_flag_setter():
    c = LocalColonyCompute(device="cpu")
    assert c.set_hp_authoritative(True) is True
    assert c._hp_authoritative_enabled is True
    assert c.set_hp_authoritative(False) is False


def test_is_adam_field():
    """ClientCreatureBiochem.is_adam=True (client тикает только Адама → lockstep)."""
    bc = ClientCreatureBiochem()
    assert bc.is_adam is True


def test_energy_ratios_mirror_gate():
    """OFF → _energy_ratios зеркалит hp=energy; ON → НЕ зеркалит (hp живёт сам)."""
    c, bc = _compute(energy=700.0, hp=123.0)
    c.set_four_scale(True)
    # OFF: зеркало → hp перезаписывается = energy
    c._energy_ratios("c0")
    assert bc.hp == 700.0
    # ON: НЕ зеркалит → hp остаётся как был (divergence)
    bc.hp = 123.0
    c.set_hp_authoritative(True)
    c._energy_ratios("c0")
    assert bc.hp == 123.0                 # не тронут зеркалом


def test_damage_to_hp_when_on():
    """ON: predator-урон → hp (не energy). OFF: → energy (legacy). Нужды в dead-zone
    (0<need<=норма) → ни дренаж ни лечение → урон изолирован."""
    c, bc = _compute(energy=0.5, hydration=0.5, hp=500.0)   # dead-zone нужды
    c.set_hp_authoritative(True)
    c._apply_metabolism("c0", _rates(damage_per_tick=10.0))
    assert bc.hp == pytest.approx(490.0)      # урон в hp (без heal/drain)
    assert bc.energy == pytest.approx(0.5)    # energy не тронута уроном
    # OFF: урон в energy (legacy)
    c2, bc2 = _compute(energy=500.0, hp=500.0)
    c2._apply_metabolism("c0", _rates(damage_per_tick=10.0))
    assert bc2.energy == pytest.approx(490.0)  # legacy: урон в energy
    assert bc2.hp == 500.0


def test_needs_drain_to_hp():
    """Сытость=0 → дренаж hp φ³; вода=0 → φ³; обе=0 → 2×φ³ (выносливость INERT)."""
    c, bc = _compute(energy=0.0, hydration=50.0, hp=500.0)
    c.set_hp_authoritative(True)
    c._apply_metabolism("c0", _rates())
    assert bc.hp == pytest.approx(500.0 - _HP_NEED_DRAIN)   # сытость=0 → 1×φ³
    # обе нужды=0 → 2×φ³
    c2, bc2 = _compute(energy=0.0, hydration=0.0, hp=500.0)
    c2.set_hp_authoritative(True)
    c2._apply_metabolism("c0", _rates())
    assert bc2.hp == pytest.approx(500.0 - 2 * _HP_NEED_DRAIN)


def test_heal_when_needs_in_norm():
    """Все активные нужды>норма → лечение φ²; hp не превышает max_hp."""
    c, bc = _compute(energy=500.0, hydration=100.0, hp=500.0)
    c.set_hp_authoritative(True)
    c._apply_metabolism("c0", _rates())
    assert bc.hp == pytest.approx(500.0 + _HP_HEAL)
    # cap на max_hp
    c2, bc2 = _compute(energy=500.0, hydration=100.0, hp=1308.5)
    c2.set_hp_authoritative(True)
    c2._apply_metabolism("c0", _rates())
    assert bc2.hp == pytest.approx(1309.0)        # не выше max_hp


def test_death_guardrail_still_energy():
    """1b.1: hp→0 НЕ убивает (death/§3 на energy). cid НЕ в _dead_cids при hp=0,
    energy>0."""
    c, bc = _compute(energy=300.0, hydration=0.5, hp=2.0)   # hyd dead-zone → нет heal
    c.set_hp_authoritative(True)
    c._apply_metabolism("c0", _rates(damage_per_tick=100.0))  # hp → 0
    assert bc.hp == 0.0
    assert "c0" not in c._dead_cids                # death всё ещё на energy (guardrail)
    assert bc.energy == pytest.approx(300.0)       # energy цела


def test_off_inert_no_hp_dynamics():
    """OFF: дренаж/лечение/урон-в-hp НЕ применяются (инертно)."""
    c, bc = _compute(energy=0.0, hydration=0.0, hp=500.0)  # нужды=0, но флаг OFF
    c._apply_metabolism("c0", _rates(damage_per_tick=10.0))
    assert bc.hp == 500.0                          # hp не тронут (нет дренажа)
    assert bc.energy == pytest.approx(0.0)         # урон в energy (legacy, был 0)


def test_recoverable_structure_starve_then_feed():
    """RECOVERABLE-структура (§18.6): голод → hp дренится; откорм → hp лечится назад.
    Не absorbing (равновесие в пользу жизни при доступной еде)."""
    c, bc = _compute(energy=0.0, hydration=50.0, hp=500.0)
    c.set_hp_authoritative(True)
    # фаза голода: hp дренится
    for _ in range(10):
        c._apply_metabolism("c0", _rates())
    assert bc.hp < 500.0
    drained = bc.hp
    # фаза откорма: energy>норма → hp лечится назад
    bc.energy = 500.0
    for _ in range(10):
        c._apply_metabolism("c0", _rates())
    assert bc.hp > drained                          # восстанавливается (не absorbing)


def test_projection_includes_hp():
    """build_projection_batch несёт hp/max_hp (frozen-fallback P40)."""
    c, bc = _compute(hp=742.0)
    c.organisms["c0"] = object()                    # минимальный org-заглушка
    projs = c.build_projection_batch()
    p = next((x for x in projs if x["cid"] == "c0"), None)
    assert p is not None and p["hp"] == 742.0 and p["max_hp"] == 1309.0
