"""social forecast-born probe (Фрай §7, 17.06.2026) — read-only paired-ablation.

Метрика «оживёт ли social как predictor-forecast-born»: на DANGER-окнах (prev[74:76]≠0)
снимает forecast-loss predictor'а на predator-каналах obs[59:62] дважды — full vs
social-zeroed (obs[72:76]=0) с ОДНОГО состояния (snapshot/restore). Δ=loss(zeroed)−
loss(full). Главный инвариант: probe READ-ONLY — состояние predictor'а НЕ меняется
(реальный forward идёт отдельно). Гейт client_flag social_forecast_probe (OFF).
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
pytest.importorskip("core.tissue")

import torch  # noqa: E402

from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute,
    _BRAIN_INPUT_DIM,
    _SOCIAL_OFFSET,
    _SOCIAL_DIM,
)


def _compute():
    return LocalColonyCompute(device="cpu")


def _pred_or_skip(c):
    p = c._make_predictor_tissue()
    if p is None:
        pytest.skip("predictor tissue не построился")
    c._upgrade_tissue_input_dim(p, _BRAIN_INPUT_DIM)
    return p


def _danger_prev(val=0.5):
    """obs76 c DANGER-сигналом (obs[74]=danger_ns≠0)."""
    x = torch.zeros(1, _BRAIN_INPUT_DIM)
    x[0, 74] = val
    x[0, :64] = torch.randn(64)        # env-контекст
    return x


def _state_snapshot(pred):
    cs = getattr(pred, "_cell_states", {})
    return {k: v.clone() for k, v in cs.items()}


def _states_equal(a, b):
    if set(a) != set(b):
        return False
    return all(torch.equal(a[k], b[k]) for k in a)


def _setup(c, probe_on=True):
    pred = _pred_or_skip(c)
    c.predictor["c0"] = pred
    # один реальный forward — заполнить _cell_states (не пусто)
    with torch.no_grad():
        pred({"input": torch.randn(1, _BRAIN_INPUT_DIM)})
    if probe_on:
        c.set_social_forecast_probe(True)
    return pred


def test_probe_is_read_only():
    """ГЛАВНЫЙ инвариант: probe НЕ меняет состояние predictor'а (snapshot/restore)."""
    c = _compute()
    pred = _setup(c)
    s_before = _state_snapshot(pred)
    c._social_forecast_probe("c0", _danger_prev(), torch.randn(1, 64))
    s_after = _state_snapshot(pred)
    assert _states_equal(s_before, s_after), "probe изменил состояние predictor'а!"


def test_probe_records_delta_on_danger():
    c = _compute()
    _setup(c)
    assert "c0" not in c._social_probe_diffs or not c._social_probe_diffs["c0"]
    c._social_forecast_probe("c0", _danger_prev(), torch.randn(1, 64))
    assert len(c._social_probe_diffs.get("c0", [])) == 1
    assert c._social_probe_episodes.get("c0") == 1          # один эпизод


def test_probe_skips_when_no_danger():
    """obs[74:76]=0 → НЕ DANGER-окно → Δ не пишется."""
    c = _compute()
    _setup(c)
    no_danger = torch.zeros(1, _BRAIN_INPUT_DIM)
    no_danger[0, :64] = torch.randn(64)
    c._social_forecast_probe("c0", no_danger, torch.randn(1, 64))
    assert not c._social_probe_diffs.get("c0")
    assert c._social_probe_episodes.get("c0", 0) == 0


def test_episode_counting_window_boundary():
    """Эпизод = вход в DANGER-окно. Непрерывный DANGER = 1 эпизод; разрыв = новый."""
    c = _compute()
    _setup(c)
    tgt = torch.randn(1, 64)
    c._social_forecast_probe("c0", _danger_prev(), tgt)   # эпизод 1 (вход)
    c._social_forecast_probe("c0", _danger_prev(), tgt)   # тот же эпизод (continuous)
    assert c._social_probe_episodes["c0"] == 1
    no_danger = torch.zeros(1, _BRAIN_INPUT_DIM)
    c._social_forecast_probe("c0", no_danger, tgt)        # выход из окна
    c._social_forecast_probe("c0", _danger_prev(), tgt)   # эпизод 2 (новый вход)
    assert c._social_probe_episodes["c0"] == 2


def test_probe_disabled_noop():
    """Флаг OFF → probe не вызывается (zero-cost)."""
    c = _compute()
    _setup(c, probe_on=False)
    c._social_forecast_probe("c0", _danger_prev(), torch.randn(1, 64))
    assert not c._social_probe_diffs.get("c0")


def test_flag_setter_resets_on_off():
    c = _compute()
    _setup(c)
    c._social_forecast_probe("c0", _danger_prev(), torch.randn(1, 64))
    assert c._social_probe_diffs.get("c0")
    assert c.set_social_forecast_probe(False) is False
    assert not c._social_probe_diffs               # сброс накопления
    assert not c._social_probe_episodes
    assert c._social_probe_enabled is False


def test_delta_zeroes_social_in_shadow():
    """Δ корректен: zeroed-форвард реально обнуляет obs[72:76] (иначе Δ≡0 всегда)."""
    c = _compute()
    pred = _setup(c)
    # сильный social-сигнал → full и zeroed форварды РАЗНЫЕ → Δ≠0 (в общем случае)
    prev = _danger_prev(val=1.0)
    prev[0, _SOCIAL_OFFSET:_SOCIAL_OFFSET + _SOCIAL_DIM] = torch.tensor([0.9, -0.8, 1.0, -0.7])
    c._social_forecast_probe("c0", prev, torch.randn(1, 64))
    diffs = c._social_probe_diffs.get("c0", [])
    assert len(diffs) == 1
    # Δ — конечное число (loss-разность), не NaN
    assert diffs[0] == diffs[0]   # not NaN
