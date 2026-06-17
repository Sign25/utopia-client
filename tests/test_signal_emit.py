"""social этап B ОБМАН (Фрай v0.6, 17.06): signal_emit — Адам эмитит ложный DANGER.

Механизм: структурный φ-штраф на logits[7] (SIGNAL_DANGER) СНИМАЕТСЯ при
client_flag signal_emit → Адам получает действие эмиссии (→ Старшие уходят →
еда свободна → ест; выгода эмерджентна). Default OFF (dormant, штраф держит).
МОТОР-касание → flip по «да» Шефа + joint-flip B с билдом Хьюберта.

Тест: penalty[7] применяется при OFF, снимается при ON, остальные логиты не
тронуты гейтом. + флаг-сеттер.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_NEUROCORE = _ROOT.parent / "NeuroCore"
for _p in (str(_ROOT), str(_NEUROCORE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("torch")

import torch  # noqa: E402

from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute, _PHI, N_ACTIONS,
)

_PENALTY = 1.0 / (_PHI * _PHI)   # структурный штраф SIGNAL_DANGER (≈0.382)


def _shape(c, enabled):
    c._signal_emit_enabled = enabled
    logits = torch.zeros(N_ACTIONS)
    obs = np.zeros(64, dtype=np.float32)   # нулевые градиенты → без контекстных boost
    c._shape_action_logits(logits, obs, diet=0.3, energy_ratio=0.5)
    return logits


def test_default_off():
    c = LocalColonyCompute(device="cpu")
    assert c._signal_emit_enabled is False          # dormant по умолчанию


def test_penalty_applied_when_off():
    """OFF (default): logits[7] получает φ-штраф (эмиссия подавлена)."""
    c = LocalColonyCompute(device="cpu")
    logits = _shape(c, False)
    assert abs(float(logits[7]) - (-_PENALTY)) < 1e-5


def test_penalty_removed_when_on():
    """ON: штраф с logits[7] СНЯТ → Адам получает действие эмиссии. Разница ровно
    +penalty; logits[7]_on == 0 (нет др. правок [7])."""
    c = LocalColonyCompute(device="cpu")
    off = _shape(c, False)
    on = _shape(c, True)
    assert abs((float(on[7]) - float(off[7])) - _PENALTY) < 1e-5
    assert abs(float(on[7])) < 1e-5                  # на нулевом обзоре [7]=0 при ON


def test_gate_isolated_to_action7():
    """Гейт трогает ТОЛЬКО [7]; прочие структурные штрафы (STAY[4]/SIGNAL_FOOD[6]/
    SHARE[8]/DIG[11]/BUILD[12]) идентичны при OFF и ON."""
    c = LocalColonyCompute(device="cpu")
    off = _shape(c, False)
    on = _shape(c, True)
    for i in (4, 6, 8, 11, 12):
        assert abs(float(on[i]) - float(off[i])) < 1e-6, f"гейт задел logits[{i}]"


def test_flag_setter():
    c = LocalColonyCompute(device="cpu")
    assert c.set_signal_emit(True) is True
    assert c._signal_emit_enabled is True
    assert c.set_signal_emit(False) is False
    assert c._signal_emit_enabled is False
