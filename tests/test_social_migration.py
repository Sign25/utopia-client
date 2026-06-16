"""social_signals этап A: миграция входа предиктора 72→76 (Фрай 16.06.2026).

HARD GATE перед деплоем (Фрай, как ритм 68→72): математическая эквивалентность
на КАЖДОЙ расширенной ткани. Social = INPUT-only [72:76] (tribe FOOD/DANGER NS/EW
градиенты), строго в хвосте; env64/self4/rhythm4 [0:72] НЕ двигаются. Расширение
СОХРАНЯЮЩЕЕ: выученные 72 колонки целы, social-колонки=0 → при нулевом social
спроецированный вход ИДЕНТИЧЕН довходу 72.

Ядро ткани SFNN STATEFUL → проверяем на ПРОЕКЦИИ (input_proj — чистый Linear):
input_proj([obs72|social0]) бит-идентичен input_proj_72(obs72) → ядро получает
ТОТ ЖЕ вход. Зеркалит test_rhythm_migration.

Без зелёного этого файла миграция НЕ деплоится.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent          # utopia-client
_NEUROCORE = _ROOT.parent / "NeuroCore"                  # sibling: core.tissue
for _p in (str(_ROOT), str(_NEUROCORE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("torch")
pytest.importorskip("core.tissue")

import torch                                             # noqa: E402
import torch.nn.functional as F                          # noqa: E402

from utopia_client.local_compute import (                # noqa: E402
    LocalColonyCompute,
    _BRAIN_INPUT_DIM,
    _RHYTHM_OFFSET,
    _RHYTHM_DIM,
    _SOCIAL_OFFSET,
    _SOCIAL_DIM,
)

_RHYTHM_END = _RHYTHM_OFFSET + _RHYTHM_DIM               # 72 — довходовой dim


def _compute():
    return LocalColonyCompute(device="cpu")


def _proj(pred, x1d):
    return pred.input_proj(x1d.unsqueeze(0)).reshape(-1)


def _pred_or_skip(c):
    p = c._make_predictor_tissue()
    if p is None:
        pytest.skip("predictor tissue не построился (core недоступен)")
    return p


# ── Контракт констант ────────────────────────────────────────────────────
def test_constants_layout():
    assert _SOCIAL_OFFSET == 72          # строго после rhythm4, в хвосте
    assert _SOCIAL_DIM == 4              # food NS/EW + danger NS/EW
    assert _BRAIN_INPUT_DIM == 76
    assert _SOCIAL_OFFSET == _RHYTHM_END  # social начинается ровно где ритм кончился


# ── Тест 1: СОХРАНЯЮЩЕЕ расширение 72→76 — критичный путь (HARD GATE) ──────
def test_preserve_expansion_72_to_76_identical():
    """Наивный fresh-[I|0] обнулил бы выученные env64/self4/rhythm (brain-reset).
    Презерв сохраняет 72 колонки ТОЧНО, social-колонки=0 → проекция идентична."""
    c = _compute()
    pred = _pred_or_skip(c)
    # 64→72: создаёт input_proj, СИМУЛИРУЕМ обученные веса (env64+self4+rhythm4)
    assert c._upgrade_tissue_input_dim(pred, _RHYTHM_END) is True
    with torch.no_grad():
        pred.input_proj.weight.copy_(torch.randn_like(pred.input_proj.weight))
        pred.input_proj.bias.copy_(torch.randn_like(pred.input_proj.bias))
    w72 = pred.input_proj.weight.detach().clone()
    b72 = pred.input_proj.bias.detach().clone()
    x72 = torch.randn(_RHYTHM_END)
    with torch.no_grad():
        proj72 = F.linear(x72.unsqueeze(0), w72, b72).reshape(-1)   # эталон
    # 72→76 ПРЕЗЕРВ
    assert c._upgrade_tissue_input_dim(pred, _BRAIN_INPUT_DIM) is True
    assert int(pred.data_dim) == _BRAIN_INPUT_DIM
    # выученные 72 колонки целы ТОЧНО, social-колонки [72:76] = 0, bias цел
    assert torch.allclose(pred.input_proj.weight[:, :_RHYTHM_END], w72, atol=0), \
        "preserve обнулил/исказил выученные колонки = brain-reset"
    assert torch.allclose(pred.input_proj.bias, b72, atol=0)
    assert torch.count_nonzero(
        pred.input_proj.weight[:, _SOCIAL_OFFSET:_SOCIAL_OFFSET + _SOCIAL_DIM]) == 0
    # math-equivalence: проекция [obs72 | social=0] == проекция_72(obs72)
    x76 = torch.cat([x72, torch.zeros(_SOCIAL_DIM)])
    with torch.no_grad():
        proj76 = _proj(pred, x76)
    assert torch.allclose(proj72, proj76, atol=1e-6), \
        float((proj72 - proj76).abs().max())


# ── Тест 2: load-transition (72-saved .pt → load → preserve-expand 76) ────
def test_load_transition_72_saved_to_76():
    c = _compute()
    src = _pred_or_skip(c)
    c._upgrade_tissue_input_dim(src, _RHYTHM_END)
    with torch.no_grad():
        src.input_proj.weight.copy_(torch.randn_like(src.input_proj.weight))
        src.input_proj.bias.copy_(torch.randn_like(src.input_proj.bias))
    sd = {k: v.detach().clone() for k, v in src.state_dict().items()}
    w72 = src.input_proj.weight.detach().clone()
    b72 = src.input_proj.bias.detach().clone()
    x72 = torch.randn(_RHYTHM_END)
    with torch.no_grad():
        proj_src = F.linear(x72.unsqueeze(0), w72, b72).reshape(-1)
    fresh = _pred_or_skip(c)
    c.predictor["t"] = fresh
    c._load_predictor_sd("t", sd)
    loaded = c.predictor["t"]
    assert int(loaded.data_dim) == _BRAIN_INPUT_DIM
    assert torch.allclose(loaded.input_proj.weight[:, :_RHYTHM_END], w72, atol=0)
    assert torch.allclose(loaded.input_proj.bias, b72, atol=0)
    assert torch.count_nonzero(
        loaded.input_proj.weight[:, _SOCIAL_OFFSET:_SOCIAL_OFFSET + _SOCIAL_DIM]) == 0
    x76 = torch.cat([x72, torch.zeros(_SOCIAL_DIM)])
    with torch.no_grad():
        proj_loaded = _proj(loaded, x76)
    assert torch.allclose(proj_src, proj_loaded, atol=1e-6)


def test_load_transition_already_76_noop():
    """Уже сохранён на 76 (после деплоя+рестарта) → грузим как есть."""
    c = _compute()
    src = _pred_or_skip(c)
    c._upgrade_tissue_input_dim(src, _BRAIN_INPUT_DIM)
    with torch.no_grad():
        src.input_proj.weight.copy_(torch.randn_like(src.input_proj.weight))
    sd = {k: v.detach().clone() for k, v in src.state_dict().items()}
    w76 = src.input_proj.weight.detach().clone()
    fresh = _pred_or_skip(c)
    c.predictor["t"] = fresh
    c._load_predictor_sd("t", sd)
    assert int(c.predictor["t"].data_dim) == _BRAIN_INPUT_DIM
    assert torch.allclose(c.predictor["t"].input_proj.weight, w76, atol=0)


# ── Тест 3: КАЖДАЯ расширенная ткань social-дормантна (требование Фрая) ────
def test_every_expanded_tissue_social_dormant():
    c = _compute()
    preds = []
    for i in range(3):
        p = _pred_or_skip(c)
        c._upgrade_tissue_input_dim(p, _RHYTHM_END)
        with torch.no_grad():
            p.input_proj.weight.copy_(torch.randn_like(p.input_proj.weight))
        c.predictor[f"c{i}"] = p
        preds.append((p, p.input_proj.weight.detach().clone()))
    for p, _ in preds:
        c._upgrade_tissue_input_dim(p, _BRAIN_INPUT_DIM)
    expanded = [p for p in c.predictor.values()
                if int(getattr(p, "data_dim", 64)) == _BRAIN_INPUT_DIM]
    assert len(expanded) == 3
    for p, w72 in preds:
        assert int(p.data_dim) == _BRAIN_INPUT_DIM
        assert torch.allclose(p.input_proj.weight[:, :_RHYTHM_END], w72, atol=0)
        assert torch.count_nonzero(
            p.input_proj.weight[:, _SOCIAL_OFFSET:_SOCIAL_OFFSET + _SOCIAL_DIM]) == 0


# ── Тест 4: _extract_social — дефолт нули, чтение [72:76] ─────────────────
def test_extract_social_defaults_and_read():
    c = _compute()
    assert np.allclose(c._extract_social(np.zeros(64, np.float32)), 0.0)
    assert np.allclose(c._extract_social(np.zeros(72, np.float32)), 0.0)  # узкий
    assert np.allclose(c._extract_social(None), 0.0)
    o = np.zeros(80, np.float32)
    o[_SOCIAL_OFFSET:_SOCIAL_OFFSET + _SOCIAL_DIM] = [0.1, -0.2, 0.3, -0.4]
    assert np.allclose(c._extract_social(o), [0.1, -0.2, 0.3, -0.4])
    assert c._extract_social(o).shape[0] == _SOCIAL_DIM


# ── Тест 5: client_flag set_social_signals (независимый rollback) ─────────
def test_set_social_signals_flag():
    c = _compute()
    assert c._social_enabled is False           # дефолт OFF (dormant)
    assert c.set_social_signals(True) is True
    assert c._social_enabled is True
    assert c.set_social_signals(False) is False
    assert c._social_enabled is False


# ── Тест 6: ws_client инжект social в obs[72:76] ─────────────────────────
def test_ws_apply_social_to_obs():
    from utopia_client.ws_client import ColonyWSClient
    o = np.zeros(80, np.float32)
    assert ColonyWSClient._apply_social_to_obs(o, None) is o   # None → no-op
    assert np.count_nonzero(o) == 0
    soc = {"food_ns": 0.5, "food_ew": -0.5, "danger_ns": 0.25, "danger_ew": -0.25}
    o2 = ColonyWSClient._apply_social_to_obs(np.zeros(80, np.float32), soc)
    assert np.allclose(o2[72:76], [0.5, -0.5, 0.25, -0.25])
    # узкий obs (64) → паддинг до 80, social на месте, env64 не тронут
    o3 = ColonyWSClient._apply_social_to_obs(np.zeros(64, np.float32), soc)
    assert o3.shape[0] == 80
    assert np.allclose(o3[72:76], [0.5, -0.5, 0.25, -0.25])
    assert np.count_nonzero(o3[:64]) == 0
    # obs ровно 72 (ритм-падд, social ещё нет) → должен ПЕРЕпаддить до 80 (не OOB)
    o4 = ColonyWSClient._apply_social_to_obs(np.zeros(72, np.float32), soc)
    assert o4.shape[0] == 80
    assert np.allclose(o4[72:76], [0.5, -0.5, 0.25, -0.25])
