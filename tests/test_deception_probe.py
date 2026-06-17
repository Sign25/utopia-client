"""deception-exploit probe (Фрай v0.6, 17.06.2026) — READ-ONLY within-subject
paired counterfactual: учится ли Адам каузально ЭКСПЛУАТИРОВАТЬ обман.

Контекст = tribe-FOOD активен (obs[72:74]≠0). Кандидат-тик: EMIT (action==
SIGNAL_DANGER=7 → ложный сигнал тревоги) vs NO-EMIT. Метрика = Δenergy за
следующие K=8 тиков. Накопление gain|EMIT и gain|NO-EMIT → робастный
двухвыборочный Δ. Инвариант: probe НИЧЕГО не меняет (только наблюдает action/
energy/контекст). Гейт client_flag deception_probe (OFF zero-cost).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_NEUROCORE = _ROOT.parent / "NeuroCore"
for _p in (str(_ROOT), str(_NEUROCORE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("torch")

from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute,
    _BRAIN_INPUT_DIM,
    _SOCIAL_FOOD,
    _DECEP_K,
    _SIGNAL_DANGER,
)


def _compute():
    return LocalColonyCompute(device="cpu")


def _set_energy(c, cid, e):
    c.biochem[cid] = types.SimpleNamespace(energy=float(e))


def _obs(food=False):
    """obs76; food=True → tribe-FOOD контекст (obs[72]≠0)."""
    x = np.zeros(_BRAIN_INPUT_DIM, dtype=np.float32)
    x[:64] = np.random.randn(64).astype(np.float32)
    if food:
        x[_SOCIAL_FOOD[0]] = 0.7      # tribe_food_NS≠0 → контекст активен
    return x


def _tick(c, cid, energy, food, action):
    """один probe-тик: выставить энергию, прогнать probe."""
    _set_energy(c, cid, energy)
    c._deception_exploit_probe(cid, _obs(food=food), action)


def test_default_off():
    c = _compute()
    assert c._decep_probe_enabled is False


def test_flag_setter_resets_on_off():
    c = _compute()
    assert c.set_deception_probe(True) is True
    assert c._decep_probe_enabled is True
    _tick(c, "c0", 500.0, True, _SIGNAL_DANGER)
    assert c._decep_pending.get("c0")
    assert c.set_deception_probe(False) is False
    assert not c._decep_pending
    assert not c._decep_gain_emit
    assert not c._decep_emit_episodes
    assert c._decep_probe_enabled is False


def test_disabled_noop():
    """Флаг OFF → probe не пишет (zero-cost)."""
    c = _compute()
    _tick(c, "c0", 500.0, True, _SIGNAL_DANGER)
    assert not c._decep_pending.get("c0")


def test_no_context_no_candidate():
    """Нет tribe-FOOD контекста → кандидат не заводится."""
    c = _compute()
    c.set_deception_probe(True)
    _tick(c, "c0", 500.0, False, _SIGNAL_DANGER)
    assert not c._decep_pending.get("c0")
    assert c._decep_ctx_ticks.get("c0", 0) == 0
    assert c._decep_emit_episodes.get("c0", 0) == 0


def test_emit_in_context_counts_episode():
    """EMIT в контексте → ctx-тик + EMIT-эпизод (rising-edge) + emit_log world_tick."""
    c = _compute()
    c.set_deception_probe(True)
    c._last_world_tick = 12345
    _tick(c, "c0", 500.0, True, _SIGNAL_DANGER)
    assert c._decep_ctx_ticks["c0"] == 1
    assert c._decep_emit_ticks["c0"] == 1
    assert c._decep_emit_episodes["c0"] == 1
    assert c._decep_emit_log["c0"] == [12345]


def test_episode_rising_edge():
    """Непрерывный EMIT = 1 эпизод; разрыв (NO-EMIT) → новый эпизод на следующем EMIT."""
    c = _compute()
    c.set_deception_probe(True)
    _tick(c, "c0", 500.0, True, _SIGNAL_DANGER)   # эпизод 1
    _tick(c, "c0", 500.0, True, _SIGNAL_DANGER)   # тот же (continuous)
    assert c._decep_emit_episodes["c0"] == 1
    _tick(c, "c0", 500.0, True, 4)                # NO-EMIT (STAY) в контексте → разрыв
    _tick(c, "c0", 500.0, True, _SIGNAL_DANGER)   # эпизод 2
    assert c._decep_emit_episodes["c0"] == 2


def test_gain_window_emit():
    """EMIT-кандидат закрывается через K тиков: Δenergy = e[t+K] − e[t0]."""
    c = _compute()
    c.set_deception_probe(True)
    _tick(c, "c0", 500.0, True, _SIGNAL_DANGER)   # t0: EMIT, e0=500
    # K−1 промежуточных тиков (вне контекста, чтобы не плодить кандидатов)
    for i in range(_DECEP_K - 1):
        _tick(c, "c0", 500.0 + i, False, 4)
    assert not c._decep_gain_emit.get("c0")       # ещё не созрело
    _tick(c, "c0", 560.0, False, 4)               # возраст == K → закрытие, gain=560−500
    assert c._decep_gain_emit.get("c0") == [60.0]
    assert not c._decep_gain_noemit.get("c0")


def test_gain_window_noemit():
    """NO-EMIT кандидат (контекст, но action≠7) → gain в no-emit выборку."""
    c = _compute()
    c.set_deception_probe(True)
    _tick(c, "c0", 500.0, True, 4)                # t0: контекст, NO-EMIT
    for i in range(_DECEP_K - 1):
        _tick(c, "c0", 500.0, False, 4)
    _tick(c, "c0", 480.0, False, 4)               # возраст == K → gain=480−500=−20
    assert c._decep_gain_noemit.get("c0") == [-20.0]
    assert not c._decep_gain_emit.get("c0")


def test_read_only_no_energy_mutation():
    """Инвариант: probe НЕ трогает biochem.energy (чистое наблюдение)."""
    c = _compute()
    c.set_deception_probe(True)
    _set_energy(c, "c0", 777.0)
    c._deception_exploit_probe("c0", _obs(food=True), _SIGNAL_DANGER)
    assert c.biochem["c0"].energy == 777.0


def test_robust_two_sample():
    """Δ = med(a) − med(b); знак t совпадает со знаком Δ; разделимые выборки → |t|≥порог."""
    c = _compute()
    a = [10.0, 11.0, 9.0, 10.5, 10.2, 9.8, 10.1, 10.3]   # ~10
    b = [1.0, 1.2, 0.8, 1.1, 0.9, 1.05, 0.95, 1.0]       # ~1
    ma, mb, delta, t = c._robust_two_sample(a, b)
    assert abs(delta - (ma - mb)) < 1e-9
    assert delta > 0 and t > 0
    assert abs(t) >= 2.0                                  # хорошо разделимы
    # симметрия знака
    _, _, d2, t2 = c._robust_two_sample(b, a)
    assert d2 < 0 and t2 < 0


def test_two_sample_underpowered():
    """n<2 в выборке → t=0 (нет вердикта)."""
    c = _compute()
    _, _, _, t = c._robust_two_sample([5.0], [1.0, 2.0])
    assert t == 0.0
