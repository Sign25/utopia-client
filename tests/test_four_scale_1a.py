"""stamina шаг 1a (Фрай/Хьюберт §15, 17.06) — 4-шкальная модель здоровья, LOCKSTEP.

Контракт 1a: HP-бак `hp` зеркалом energy (dormant, бит-в-бит), единый max=1309
(§12.5 фикс er /1000→/1309), классификация читателей energy_ratio (ГОЛОД→
сытость_ratio / ЖИЗНЬ→hp_ratio). HARD GATE: при OFF поведение бит-в-бит старое
(er=/1000, оба ratio равны); при ON hp=energy → сытость_ratio==hp_ratio (нет
разъезда на 1a). Разъезд hp/energy — шаг 1b. Гейт client_flag `four_scale`.
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

pytest.importorskip("torch")

from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute, _CLIENT_MAX_ENERGY,
)
from utopia_client.biochemistry import ClientCreatureBiochem  # noqa: E402


def _compute_with_energy(e):
    c = LocalColonyCompute(device="cpu")
    bc = ClientCreatureBiochem()
    bc.energy = float(e)
    bc.hp = float(e)
    c.biochem["c0"] = bc
    return c


def test_default_off():
    c = LocalColonyCompute(device="cpu")
    assert c._four_scale_enabled is False


def test_hp_field_default_mirrors_energy():
    """ClientCreatureBiochem.hp дефолт = energy дефолт (зеркало с рождения)."""
    bc = ClientCreatureBiochem()
    assert bc.hp == bc.energy == 500.0
    assert "hp" in bc.as_snapshot()


def test_off_bitexact_legacy():
    """OFF (dormant): оба ratio = energy/1000 (legacy er, бит-в-бит старое)."""
    c = _compute_with_energy(618.0)
    sat, hp = c._energy_ratios("c0")
    assert sat == hp == pytest.approx(618.0 / 1000.0)


def test_on_unified_1309_and_equal():
    """ON (1a): оба ratio = energy/1309 и РАВНЫ (hp=energy зеркало)."""
    c = _compute_with_energy(809.0)
    c.set_four_scale(True)
    sat, hp = c._energy_ratios("c0")
    assert sat == pytest.approx(809.0 / _CLIENT_MAX_ENERGY)
    assert hp == pytest.approx(809.0 / _CLIENT_MAX_ENERGY)
    assert sat == hp                     # нет разъезда на 1a


def test_on_mirrors_hp_field():
    """ON: _energy_ratios поддерживает hp=energy зеркало в поле (scaffolding 1b)."""
    c = _compute_with_energy(500.0)
    c.biochem["c0"].energy = 742.0       # energy сдвинулась, hp ещё 500
    c.set_four_scale(True)
    c._energy_ratios("c0")               # должно отзеркалить hp←energy
    assert c.biochem["c0"].hp == 742.0


def test_normalization_shift_off_to_on():
    """§12.5: OFF=/1000, ON=/1309 — фикс рассинхрона (значение РАЗНОЕ, ожидаемо)."""
    c = _compute_with_energy(1000.0)
    off_sat, _ = c._energy_ratios("c0")
    assert off_sat == pytest.approx(1.0)             # 1000/1000
    c.set_four_scale(True)
    on_sat, _ = c._energy_ratios("c0")
    assert on_sat == pytest.approx(1000.0 / 1309.0)  # ≈0.764 (выровнено на server-канон)


def test_no_biochem_default():
    """bc None → (0.5, 0.5) (старый дефолт _er)."""
    c = LocalColonyCompute(device="cpu")
    assert c._energy_ratios("missing") == (0.5, 0.5)
    c.set_four_scale(True)
    assert c._energy_ratios("missing") == (0.5, 0.5)


def test_flag_setter():
    c = LocalColonyCompute(device="cpu")
    assert c.set_four_scale(True) is True
    assert c._four_scale_enabled is True
    assert c.set_four_scale(False) is False
    assert c._four_scale_enabled is False


def test_shape_action_logits_hp_ratio_fallback():
    """_shape_action_logits: hp_ratio=None → fallback energy_ratio (legacy бит-в-бит)."""
    import numpy as np
    import torch
    c = LocalColonyCompute(device="cpu")
    obs = np.zeros(64, dtype=np.float32)
    lg_legacy = torch.zeros(16)
    c._shape_action_logits(lg_legacy, obs, 0.3, 0.2)               # без hp_ratio
    lg_explicit = torch.zeros(16)
    c._shape_action_logits(lg_explicit, obs, 0.3, 0.2, hp_ratio=0.2)  # hp=energy
    assert torch.allclose(lg_legacy, lg_explicit)                 # идентичны
