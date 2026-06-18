"""obs O2 (stamina §19.2/§20): миграция входа предиктора 76→78 (Фрай GO 18.06).

HARD GATE перед деплоем (как social 72→76): math-equivalence на КАЖДОЙ расширенной
ткани. obs O2 = INTERO [76:78] (выносливость + HP), строго в хвосте; env64/self4/
rhythm4/social4 [0:76] НЕ двигаются. Расширение СОХРАНЯЮЩЕЕ: 76 колонок целы,
intero-колонки=0 → при нулевом intero (dormant) вход ИДЕНТИЧЕН довходу 76.
Зеркало test_social_migration. Без зелёного — миграция НЕ деплоится.
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
pytest.importorskip("core.tissue")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute, _BRAIN_INPUT_DIM, _SOCIAL_OFFSET, _SOCIAL_DIM,
    _INTERO_OFFSET, _INTERO_DIM, _FATIGUE_MAX, _CLIENT_MAX_ENERGY,
)
from utopia_client.biochemistry import ClientCreatureBiochem  # noqa: E402

_SOCIAL_END = _SOCIAL_OFFSET + _SOCIAL_DIM    # 76 — довходовой dim (до intero)


def _compute():
    return LocalColonyCompute(device="cpu")


def _proj(pred, x1d):
    return pred.input_proj(x1d.unsqueeze(0)).reshape(-1)


def _pred_or_skip(c):
    p = c._make_predictor_tissue()
    if p is None:
        pytest.skip("predictor tissue не построился")
    return p


def test_constants_layout():
    assert _INTERO_OFFSET == 76
    assert _INTERO_DIM == 2
    assert _BRAIN_INPUT_DIM == 78
    assert _INTERO_OFFSET == _SOCIAL_END       # intero ровно где social кончился


def test_preserve_expansion_76_to_78_identical():
    """HARD GATE: 76 колонок целы ТОЧНО, intero-колонки [76:78]=0 → проекция
    [obs76|intero0] идентична проекции_76(obs76)."""
    c = _compute()
    pred = _pred_or_skip(c)
    assert c._upgrade_tissue_input_dim(pred, _SOCIAL_END) is True   # 64→76
    with torch.no_grad():
        pred.input_proj.weight.copy_(torch.randn_like(pred.input_proj.weight))
        pred.input_proj.bias.copy_(torch.randn_like(pred.input_proj.bias))
    w76 = pred.input_proj.weight.detach().clone()
    b76 = pred.input_proj.bias.detach().clone()
    x76 = torch.randn(_SOCIAL_END)
    with torch.no_grad():
        proj76 = F.linear(x76.unsqueeze(0), w76, b76).reshape(-1)
    assert c._upgrade_tissue_input_dim(pred, _BRAIN_INPUT_DIM) is True   # 76→78
    assert int(pred.data_dim) == _BRAIN_INPUT_DIM
    assert torch.allclose(pred.input_proj.weight[:, :_SOCIAL_END], w76, atol=0), \
        "preserve обнулил выученные колонки = brain-reset"
    assert torch.allclose(pred.input_proj.bias, b76, atol=0)
    assert torch.count_nonzero(
        pred.input_proj.weight[:, _INTERO_OFFSET:_INTERO_OFFSET + _INTERO_DIM]) == 0
    x78 = torch.cat([x76, torch.zeros(_INTERO_DIM)])
    with torch.no_grad():
        proj78 = _proj(pred, x78)
    assert torch.allclose(proj76, proj78, atol=1e-6), float((proj76 - proj78).abs().max())


def test_load_transition_76_saved_to_78():
    c = _compute()
    src = _pred_or_skip(c)
    c._upgrade_tissue_input_dim(src, _SOCIAL_END)
    with torch.no_grad():
        src.input_proj.weight.copy_(torch.randn_like(src.input_proj.weight))
        src.input_proj.bias.copy_(torch.randn_like(src.input_proj.bias))
    sd = {k: v.detach().clone() for k, v in src.state_dict().items()}
    w76 = src.input_proj.weight.detach().clone()
    x76 = torch.randn(_SOCIAL_END)
    with torch.no_grad():
        proj_src = F.linear(x76.unsqueeze(0), w76, src.input_proj.bias).reshape(-1)
    fresh = _pred_or_skip(c)
    c.predictor["t"] = fresh
    c._load_predictor_sd("t", sd)
    loaded = c.predictor["t"]
    assert int(loaded.data_dim) == _BRAIN_INPUT_DIM
    assert torch.allclose(loaded.input_proj.weight[:, :_SOCIAL_END], w76, atol=0)
    assert torch.count_nonzero(
        loaded.input_proj.weight[:, _INTERO_OFFSET:_INTERO_OFFSET + _INTERO_DIM]) == 0
    x78 = torch.cat([x76, torch.zeros(_INTERO_DIM)])
    with torch.no_grad():
        assert torch.allclose(proj_src, _proj(loaded, x78), atol=1e-6)


def test_build_intero_obs_dormant_and_values():
    """OFF → zeros[2] (dormant). ON → [выносливость_norm, hp_norm]."""
    c = _compute()
    bc = ClientCreatureBiochem()
    bc.fatigue = 40.0
    bc.hp = 654.5
    bc.max_hp = _CLIENT_MAX_ENERGY
    c.biochem["c0"] = bc
    # OFF → zeros
    assert np.allclose(c._build_intero_obs("c0"), 0.0)
    assert c._build_intero_obs("c0").shape[0] == _INTERO_DIM
    # ON → значения
    c.set_intero_obs(True)
    out = c._build_intero_obs("c0")
    assert np.isclose(out[0], 1.0 - 40.0 / _FATIGUE_MAX)   # выносливость=0.6
    assert np.isclose(out[1], 654.5 / _CLIENT_MAX_ENERGY)  # hp_norm=0.5


def test_set_intero_obs_flag():
    c = _compute()
    assert c._intero_obs_enabled is False
    assert c.set_intero_obs(True) is True
    assert c._intero_obs_enabled is True
    assert c.set_intero_obs(False) is False
