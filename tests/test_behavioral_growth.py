"""Рост-от-ПОВЕДЕНИЯ, Путь 2 (Фрай 15.06.2026) — Segment 1: фундамент.

Axis-agnostic режим: ткань рождается/удерживается/graduate'ится от ПОВЕДЕНЧЕСКОЙ
verdict-dim (не predictor-loss). Ритм = первая ось (neg_dark_loss = energy-drop
за is_night-окно). S1 = реестр осей + флаг + метрика, DORMANT. Mint/retention/
graduation — S2-S4.

S1-гейт: метрика neg_dark_loss считается верно + флаг dormant default OFF +
ось зарегистрирована axis-agnostic + sample её отдаёт. Рост-поведения ещё нет.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute,
    _BRAIN_INPUT_DIM,
)


def _c():
    return LocalColonyCompute(device="cpu")


# ── Реестр осей (axis-agnostic) ──────────────────────────────────────────
def test_axis_registry_has_rhythm():
    c = _c()
    assert "rhythm" in c._beh_axes
    ax = c._beh_axes["rhythm"]
    assert ax["cum_dim"] == "dark_loss_cum"
    assert ax["input_dim"] == _BRAIN_INPUT_DIM   # сайдкар читает obs72 (ритм@[68:72])
    assert ax["sign"] == -1                       # cost-метрика (ablate↑ → keep)


def test_register_beh_axis_generic():
    c = _c()
    n0 = len(c._beh_axes)
    d = c.register_beh_axis("fatigue", "exertion_cum", sign=-1)
    assert d["key"] == "fatigue" and d["cum_dim"] == "exertion_cum"
    assert "fatigue" in c._beh_axes
    assert len(c._beh_axes) == n0 + 1
    # axis-agnostic: машинерия не пересобирается, просто +регистрация


# ── Флаг (dormant default, kill-switch) ──────────────────────────────────
def test_set_behavioral_growth_flag():
    c = _c()
    assert c._behavioral_growth_enabled is False   # DORMANT по умолчанию
    assert c.set_behavioral_growth(True) is True
    assert c._behavioral_growth_enabled is True
    assert c.set_behavioral_growth(False) is False
    assert c._behavioral_growth_enabled is False


# ── Метрика neg_dark_loss (energy-drop за is_night-окно) ──────────────────
def test_dark_loss_accumulates_over_night():
    c = _c()
    cid = "c0"
    c.biochem[cid] = types.SimpleNamespace(energy=600.0)
    # ДЕНЬ: ничего не копится, окно не открыто
    c._world_is_night = False
    c._update_dark_loss(cid)
    assert cid not in c._dark_win_e0
    assert c._beh_dark_loss_cum.get(cid, 0.0) == 0.0
    # ВХОД В НОЧЬ: e0 = 600
    c._world_is_night = True
    c._update_dark_loss(cid)
    assert c._dark_win_e0[cid] == 600.0
    # ночь идёт, энергия падает — e0 НЕ меняется
    c.biochem[cid].energy = 540.0
    c._update_dark_loss(cid)
    assert c._dark_win_e0[cid] == 600.0
    # ВЫХОД ИЗ НОЧИ при energy=520 → drop=80 в cum, окно закрыто
    c.biochem[cid].energy = 520.0
    c._world_is_night = False
    c._update_dark_loss(cid)
    assert cid not in c._dark_win_e0
    assert abs(c._beh_dark_loss_cum[cid] - 80.0) < 1e-6


def test_dark_loss_no_negative_on_night_gain():
    """Запас перед ночью / прирост за ночь → drop=max(0,...)=0 (forage-ahead платит)."""
    c = _c()
    cid = "c0"
    c.biochem[cid] = types.SimpleNamespace(energy=500.0)
    c._world_is_night = True
    c._update_dark_loss(cid)              # e0=500
    c.biochem[cid].energy = 560.0         # вырос за ночь
    c._world_is_night = False
    c._update_dark_loss(cid)              # drop=max(0,500-560)=0
    assert c._beh_dark_loss_cum.get(cid, 0.0) == 0.0
    assert cid not in c._dark_win_e0


def test_dark_loss_no_biochem_safe():
    c = _c()
    c._world_is_night = True
    c._update_dark_loss("ghost")          # нет biochem → no-op, без краша
    assert "ghost" not in c._dark_win_e0


# ── Метрика в _beh_gc_sample (verdict-dim, как прочие оси) ────────────────
def test_beh_gc_sample_exposes_dark_loss():
    c = _c()
    c.biochem["c0"] = types.SimpleNamespace(energy=600.0)
    c._beh_dark_loss_cum["c0"] = 42.0
    s = c._beh_gc_sample("c0")
    assert "dark_loss_cum" in s
    assert s["dark_loss_cum"] == 42.0
    # прочие оси не сломаны
    for k in ("damage_cum", "meat_cum", "thermal_cum", "predkill_cum"):
        assert k in s


# ── Dormant: метрика пассивна (копится), но рост-поведения ещё нет (S1) ───
def test_metric_passive_independent_of_flag():
    """Метрика копится ВСЕГДА (наблюдаемость), не зависит от флага роста."""
    c = _c()
    cid = "c0"
    c.biochem[cid] = types.SimpleNamespace(energy=400.0)
    assert c._behavioral_growth_enabled is False   # рост OFF
    c._world_is_night = True
    c._update_dark_loss(cid)
    c.biochem[cid].energy = 370.0
    c._world_is_night = False
    c._update_dark_loss(cid)
    assert abs(c._beh_dark_loss_cum[cid] - 30.0) < 1e-6   # метрика всё равно считалась
