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

assert _CLIENT_MAX_ENERGY == 1309.0      # паритет server max_hp/max_energy


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
    """ClientCreatureBiochem.hp дефолт = energy дефолт; max_hp=1309 (паритет server,
    guard-skip в decay_step). Оба в snapshot."""
    bc = ClientCreatureBiochem()
    assert bc.hp == bc.energy == 500.0
    assert bc.max_hp == 1309.0
    snap = bc.as_snapshot()
    assert "hp" in snap and "max_hp" in snap


def test_off_bitexact_legacy():
    """OFF (dormant): оба ratio = energy/1000 (legacy er, бит-в-бит старое)."""
    c = _compute_with_energy(618.0)
    sat, hp = c._energy_ratios("c0")
    assert sat == hp == pytest.approx(618.0 / 1000.0)


def test_on_bitidentical_and_equal():
    """ON (1a, dormant): оба ratio = energy/1000 (бит-в-бит OFF) и РАВНЫ (hp=energy).
    HARD GATE: флип four_scale ИНЕРТЕН (§12.5 re-source 1309 — net-zero, отдельно)."""
    c = _compute_with_energy(809.0)
    off_sat, off_hp = c._energy_ratios("c0")
    c.set_four_scale(True)
    on_sat, on_hp = c._energy_ratios("c0")
    assert on_sat == pytest.approx(809.0 / 1000.0)
    assert on_sat == on_hp == off_sat == off_hp      # инертно: ON≡OFF, sat≡hp


def test_on_mirrors_hp_field():
    """ON: _energy_ratios поддерживает hp=energy зеркало в поле (scaffolding 1b)."""
    c = _compute_with_energy(500.0)
    c.biochem["c0"].energy = 742.0       # energy сдвинулась, hp ещё 500
    c.set_four_scale(True)
    c._energy_ratios("c0")               # должно отзеркалить hp←energy
    assert c.biochem["c0"].hp == 742.0


def test_flip_inert_bitexact():
    """§14.2/HARD GATE: для диапазона энергий флип four_scale НЕ меняет ratio."""
    for e in (50.0, 200.0, 618.0, 809.0, 1000.0, 1309.0):
        c = _compute_with_energy(e)
        off = c._energy_ratios("c0")
        c.set_four_scale(True)
        on = c._energy_ratios("c0")
        assert on == off, f"флип сдвинул ratio при energy={e}: {off}→{on}"


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


def test_shape_life_reader_keys_on_hp():
    """1a.2 lock: ЖИЗНЬ-ридер defensive-BUILD (logits[12]) реагирует на hp_ratio,
    НЕ на сытость. Низкий hp + высокая сытость → BUILD растёт; высокий hp + низкая
    сытость → BUILD НЕ растёт (порог hp<0.3). Готовность к разъезду 1b (hp≠сытость).
    На 1a hp==energy → оба пути совпадают, но routing уже на hp."""
    import numpy as np
    import torch
    c = LocalColonyCompute(device="cpu")
    c._bias_scale = 1.0
    obs = np.zeros(64, dtype=np.float32)
    lo_hp = torch.zeros(16)
    c._shape_action_logits(lo_hp, obs, 0.3, 0.9, hp_ratio=0.2)   # сыт, HP НИЗКИЙ
    hi_hp = torch.zeros(16)
    c._shape_action_logits(hi_hp, obs, 0.3, 0.2, hp_ratio=0.9)   # голоден, HP высокий
    assert float(lo_hp[12]) > float(hi_hp[12])   # BUILD от HP (ЖИЗНЬ), не сытости


def test_decay_step_hp_mirror_parity():
    """math-equiv (client-половина HARD GATE): общий server decay_step ставит
    hp=energy + НЕ перетирает max_hp=1309 (guard max_hp<=0 ложен на дефолте 1309).
    Зеркало Хьюбертова tests/test_stamina_1a. Требует environment (skip в dev-venv)."""
    decay_step = pytest.importorskip("environment.biochemistry").decay_step
    from utopia_client.biochemistry import _FakeWorld
    for e in (137.0, 500.0, 999.0):
        bc = ClientCreatureBiochem()
        bc.energy = e
        bc.hp = 0.0                       # decay_step должен отзеркалить ←energy
        decay_step(bc, _FakeWorld())
        assert bc.hp == pytest.approx(e)  # hp == energy (зеркало)
        assert bc.max_hp == 1309.0        # guard-skip → паритет с server, не 100
        assert bc.energy == pytest.approx(e)  # energy-траектория НЕ тронута


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
