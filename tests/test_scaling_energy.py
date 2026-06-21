"""scaling_energy (Фрай/Хьюберт/Шеф 20.06 §Закон 1): энергия как ПРОИЗВОДНАЯ от нужд,
а не независимый дренаж-бак. Чинит парадокс Шефа «сыт (satiety 55) но энергия 43».

E = E_max·(1 − [φ²·w^φ + φ·s^φ + 1·f^φ] / (φ³+1)), w/s = провал воды/сытости, f = усталость.

Client-фундамент (Фаза 1, dormant): satiety-поле + флаг + setter (зеркало Хьюберт 05f5401/
cfcc0aa). Формула зовётся из neurocore напрямую (бит-в-бит) — в venv neurocore нет, потому
тут тестируем поле/флаг/setter + чистую φ-математику как sanity-контракт ожидаемых значений.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

from utopia_client.biochemistry import ClientCreatureBiochem  # noqa: E402
from utopia_client.local_compute import LocalColonyCompute  # noqa: E402

_PHI = 1.618033988749895
_NORM = _PHI ** 3 + 1.0


def _energy_from_needs(hyd, sat, fat, e_max=1309.0):
    """Чистое зеркало contract-формулы (для sanity; runtime зовёт neurocore-функцию)."""
    w = max(0.0, 1.0 - hyd / 100.0)
    s = max(0.0, 1.0 - sat / 100.0)
    f = max(0.0, min(1.0, fat / 100.0))
    penalty = (_PHI ** 2) * (w ** _PHI) + _PHI * (s ** _PHI) + 1.0 * (f ** _PHI)
    return max(0.0, e_max * (1.0 - penalty / _NORM))


def test_satiety_field_default():
    b = ClientCreatureBiochem()
    assert b.satiety == 100.0          # сыт при рождении (зеркало CreatureState cfcc0aa)
    assert b.max_satiety == 100.0


def test_flag_default_off_and_toggle():
    c = LocalColonyCompute(device="cpu")
    assert c._scaling_energy_enabled is False          # dormant по умолчанию
    # setter работает даже БЕЗ neurocore в venv (lockstep degrade-gracefully)
    assert c.set_scaling_energy(True) is True
    assert c._scaling_energy_enabled is True
    assert c.set_scaling_energy(False) is False
    assert c._scaling_energy_enabled is False


def test_formula_fixes_shef_paradox():
    # Парадокс Шефа: Сытость=55, Вода=95, Усталость=68 → старый бак дал E=43 (3%).
    # Новая формула должна дать ВЫСОКУЮ E (сыт+напоён → энергичен).
    e = _energy_from_needs(hyd=95, sat=55, fat=68)
    assert e > 900.0                   # ~1059, НЕ 43 — парадокс снят
    assert e < 1309.0                  # есть провалы → не максимум


def test_formula_boundaries():
    # все нужды full → E_max; все пусты → 0 (контракт Хьюберта)
    assert _energy_from_needs(100, 100, 0) == pytest.approx(1309.0)
    assert _energy_from_needs(0, 0, 100) == pytest.approx(0.0)


def test_formula_monotonic_in_satiety():
    # выше сытость → выше энергия (при прочих равных)
    lo = _energy_from_needs(90, 30, 40)
    hi = _energy_from_needs(90, 80, 40)
    assert hi > lo


def test_hunger_ratio_switches_to_satiety_9th_site():
    # 9-й сайт (Фрай блокер): scaling ON → голод-ratio = satiety (не energy). Голодный
    # Адам (satiety=30) с ВЫСОКОЙ energy должен быть HUNGRY (иначе голод-регрессия).
    c = LocalColonyCompute(device="cpu")
    c._four_scale_enabled = True
    c._er_norm_enabled = True
    bc = ClientCreatureBiochem(energy=1000.0, satiety=30.0, hp=1000.0)
    c.biochem["a"] = bc
    _phi_inv = 1.0 / 1.618033988749895
    # OFF: голод-ratio = energy-based → energy 1000/1309≈0.76 > φ⁻¹ → НЕ голоден (старое)
    c.set_scaling_energy(False)
    sat_off, _ = c._energy_ratios("a")
    assert sat_off > _phi_inv
    # ON: голод-ratio = satiety → 30/100=0.30 < φ⁻¹ → ГОЛОДЕН несмотря на высокую energy
    c.set_scaling_energy(True)
    sat_on, hp_on = c._energy_ratios("a")
    assert sat_on < _phi_inv          # satiety-голод (ключевой фикс блокера)
    assert hp_on > _phi_inv           # capability/hp остаётся на energy (combat→energy)
