"""Ритм-аффорданс: миграция входа предиктора 68→72 (Фрай 14.06.2026).

HARD GATE перед деплоем (Фрай): математическая эквивалентность на КАЖДОЙ
расширенной ткани. Ритм = INPUT-only [68:72] (day/year phase sin/cos), строго
в хвосте; self4@[64:68] не двигается. Преширение СОХРАНЯЮЩЕЕ: выученные колонки
целы, ритм-колонки=0 → при нулевом ритме спроецированный вход ИДЕНТИЧЕН довходу.

ВАЖНО: ядро ткани SFNN — STATEFUL (membrane state переносится между forward) →
сам forward НЕ чист. Поэтому эквивалентность проверяем на ПРОЕКЦИИ (input_proj —
чистый Linear): если input_proj([obs68|rhythm0]) бит-идентичен input_proj_68(obs68),
ядро получает ТОТ ЖЕ вход → та же траектория при любом внутреннем состоянии. Это
строже full-forward (изолирует ровно то, что меняет миграция).

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
    _SELF_OBS_OFFSET,
    _SELF_OBS_DIM,
    _RHYTHM_OFFSET,
    _RHYTHM_DIM,
)


def _compute():
    return LocalColonyCompute(device="cpu")


def _proj(pred, x1d):
    """Чистая проекция input_proj на 1-D входе → 1-D спроецированный 64-вектор."""
    return pred.input_proj(x1d.unsqueeze(0)).reshape(-1)


def _pred_or_skip(c):
    p = c._make_predictor_tissue()
    if p is None:
        pytest.skip("predictor tissue не построился (core недоступен)")
    return p


# ── Контракт констант ────────────────────────────────────────────────────
def test_constants_layout():
    assert _SELF_OBS_OFFSET == 64
    assert _RHYTHM_OFFSET == 68          # строго после self4, в хвосте
    assert _RHYTHM_DIM == 4              # day sin/cos + year sin/cos
    # 16.06: social_signals расширил хвост 72→76 (rhythm [68:72] остался на месте,
    # social [72:76] добавлен). _BRAIN_INPUT_DIM теперь 76 — ритм-колонки целы.
    assert _BRAIN_INPUT_DIM == 76


# ── Тест 1: первое расширение 64→72 = [I_64|0], проекция = identity ───────
def test_first_expansion_identity_64_to_72():
    c = _compute()
    pred = _pred_or_skip(c)
    assert int(getattr(pred, "data_dim", 64)) == 64
    assert c._upgrade_tissue_input_dim(pred, _BRAIN_INPUT_DIM) is True
    assert int(pred.data_dim) == _BRAIN_INPUT_DIM
    # [I_64|0]: первые 64 = единичная, новые 8 (self4+rhythm4) = 0, bias = 0
    assert torch.allclose(
        pred.input_proj.weight[:, :_SELF_OBS_OFFSET],
        torch.eye(_SELF_OBS_OFFSET), atol=0)
    assert torch.count_nonzero(pred.input_proj.weight[:, _SELF_OBS_OFFSET:]) == 0
    assert torch.count_nonzero(pred.input_proj.bias) == 0
    # проекция [x64 | 0] == x64 (ядро видит ровно тот же вход, что и на 64)
    x64 = torch.randn(_SELF_OBS_OFFSET)
    x72 = torch.cat([x64, torch.zeros(_BRAIN_INPUT_DIM - _SELF_OBS_OFFSET)])
    with torch.no_grad():
        proj = _proj(pred, x72)
    assert torch.allclose(proj, x64, atol=1e-6), float((proj - x64).abs().max())


# ── Тест 2: СОХРАНЯЮЩЕЕ расширение 68→72 — критичный путь (KEY FIX) ───────
def test_preserve_expansion_68_to_72_identical():
    """Наивный fresh-[I|0] обнулил бы выученный self4 (частичный brain-reset).
    Презерв сохраняет 68 колонок ТОЧНО, ритм-колонки=0 → проекция идентична."""
    c = _compute()
    pred = _pred_or_skip(c)
    # 64→68: создаёт input_proj, затем СИМУЛИРУЕМ обученный self4 (рандом весов)
    assert c._upgrade_tissue_input_dim(pred, 68) is True
    with torch.no_grad():
        pred.input_proj.weight.copy_(torch.randn_like(pred.input_proj.weight))
        pred.input_proj.bias.copy_(torch.randn_like(pred.input_proj.bias))
    w68 = pred.input_proj.weight.detach().clone()
    b68 = pred.input_proj.bias.detach().clone()
    x68 = torch.randn(68)
    with torch.no_grad():
        proj68 = F.linear(x68.unsqueeze(0), w68, b68).reshape(-1)   # эталон (чистый)
    # 68→72 ПРЕЗЕРВ
    assert c._upgrade_tissue_input_dim(pred, _BRAIN_INPUT_DIM) is True
    assert int(pred.data_dim) == _BRAIN_INPUT_DIM
    # выученные 68 колонок целы ТОЧНО, ритм-колонки [68:72] = 0, bias цел
    assert torch.allclose(pred.input_proj.weight[:, :68], w68, atol=0), \
        "preserve обнулил/исказил выученные колонки = brain-reset"
    assert torch.allclose(pred.input_proj.bias, b68, atol=0)
    assert torch.count_nonzero(
        pred.input_proj.weight[:, _RHYTHM_OFFSET:_RHYTHM_OFFSET + _RHYTHM_DIM]) == 0
    # math-equivalence: проекция [obs68 | rhythm=0] == проекция_68(obs68)
    x72 = torch.cat([x68, torch.zeros(_BRAIN_INPUT_DIM - 68)])  # rhythm4+social4=0
    with torch.no_grad():
        proj72 = _proj(pred, x72)
    assert torch.allclose(proj68, proj72, atol=1e-6), \
        float((proj68 - proj72).abs().max())


# ── Тест 3: load-transition (68-saved .pt → load → preserve-expand 72) ────
def test_load_transition_68_saved_to_72():
    c = _compute()
    src = _pred_or_skip(c)
    c._upgrade_tissue_input_dim(src, 68)
    with torch.no_grad():
        src.input_proj.weight.copy_(torch.randn_like(src.input_proj.weight))
        src.input_proj.bias.copy_(torch.randn_like(src.input_proj.bias))
    sd = {k: v.detach().clone() for k, v in src.state_dict().items()}
    w68 = src.input_proj.weight.detach().clone()
    b68 = src.input_proj.bias.detach().clone()
    x68 = torch.randn(68)
    with torch.no_grad():
        proj_src = F.linear(x68.unsqueeze(0), w68, b68).reshape(-1)
    # свежий предиктор (data_dim=64) + load через миграционный путь
    fresh = _pred_or_skip(c)
    c.predictor["t"] = fresh
    c._load_predictor_sd("t", sd)
    loaded = c.predictor["t"]
    assert int(loaded.data_dim) == _BRAIN_INPUT_DIM
    assert torch.allclose(loaded.input_proj.weight[:, :68], w68, atol=0)
    assert torch.allclose(loaded.input_proj.bias, b68, atol=0)
    assert torch.count_nonzero(
        loaded.input_proj.weight[:, _RHYTHM_OFFSET:_RHYTHM_OFFSET + _RHYTHM_DIM]) == 0
    x72 = torch.cat([x68, torch.zeros(_BRAIN_INPUT_DIM - 68)])  # rhythm4+social4=0
    with torch.no_grad():
        proj_loaded = _proj(loaded, x72)
    assert torch.allclose(proj_src, proj_loaded, atol=1e-6)


def test_load_transition_already_72_noop():
    """Уже сохранён на 72 (после первого деплоя+рестарта) → грузим как есть."""
    c = _compute()
    src = _pred_or_skip(c)
    c._upgrade_tissue_input_dim(src, _BRAIN_INPUT_DIM)
    with torch.no_grad():
        src.input_proj.weight.copy_(torch.randn_like(src.input_proj.weight))
    sd = {k: v.detach().clone() for k, v in src.state_dict().items()}
    w72 = src.input_proj.weight.detach().clone()
    fresh = _pred_or_skip(c)
    c.predictor["t"] = fresh
    c._load_predictor_sd("t", sd)
    assert int(c.predictor["t"].data_dim) == _BRAIN_INPUT_DIM
    assert torch.allclose(c.predictor["t"].input_proj.weight, w72, atol=0)


# ── Тест 4: КАЖДАЯ расширенная ткань дормантна (требование Фрая) ──────────
def test_every_expanded_tissue_rhythm_dormant():
    """Презерв должен примениться ВЕЗДЕ, не только головному предиктору —
    иначе тихий частичный сброс где-то сбоку. Проверяем инвариант по всем."""
    c = _compute()
    preds = []
    for i in range(3):
        p = _pred_or_skip(c)
        c._upgrade_tissue_input_dim(p, 68)
        with torch.no_grad():
            p.input_proj.weight.copy_(torch.randn_like(p.input_proj.weight))
        c.predictor[f"c{i}"] = p
        preds.append((p, p.input_proj.weight.detach().clone()))
    for p, _ in preds:
        c._upgrade_tissue_input_dim(p, _BRAIN_INPUT_DIM)
    expanded = [p for p in c.predictor.values()
                if int(getattr(p, "data_dim", 64)) == _BRAIN_INPUT_DIM]
    assert len(expanded) == 3
    for p, w68 in preds:
        assert int(p.data_dim) == _BRAIN_INPUT_DIM
        assert torch.allclose(p.input_proj.weight[:, :68], w68, atol=0)
        assert torch.count_nonzero(
            p.input_proj.weight[:, _RHYTHM_OFFSET:_RHYTHM_OFFSET + _RHYTHM_DIM]) == 0


# ── Тест 5: _extract_rhythm — дефолт нули, чтение [68:72] ─────────────────
def test_extract_rhythm_defaults_and_read():
    c = _compute()
    assert np.allclose(c._extract_rhythm(np.zeros(64, np.float32)), 0.0)
    assert np.allclose(c._extract_rhythm(None), 0.0)
    o = np.zeros(80, np.float32)
    o[_RHYTHM_OFFSET:_RHYTHM_OFFSET + _RHYTHM_DIM] = [0.1, 0.2, 0.3, 0.4]
    assert np.allclose(c._extract_rhythm(o), [0.1, 0.2, 0.3, 0.4])
    assert c._extract_rhythm(o).shape[0] == _RHYTHM_DIM


# ── Тест 5b: client_flag set_rhythm (независимый rollback, predator-стиль) ─
def test_set_rhythm_flag():
    c = _compute()
    assert c._rhythm_enabled is False           # дефолт OFF (dormant)
    assert c.set_rhythm(True) is True
    assert c._rhythm_enabled is True
    assert c.set_rhythm(False) is False
    assert c._rhythm_enabled is False


# ── Тест 6: ws_client инжект ритма в obs[68:72] ──────────────────────────
def test_ws_apply_rhythm_to_obs():
    from utopia_client.ws_client import ColonyWSClient
    o = np.zeros(80, np.float32)
    assert ColonyWSClient._apply_rhythm_to_obs(o, None) is o   # None → no-op
    assert np.count_nonzero(o) == 0
    rh = {"day_phase_sin": 0.5, "day_phase_cos": -0.5,
          "year_phase_sin": 0.25, "year_phase_cos": -0.25}
    o2 = ColonyWSClient._apply_rhythm_to_obs(np.zeros(80, np.float32), rh)
    assert np.allclose(o2[68:72], [0.5, -0.5, 0.25, -0.25])
    # узкий obs (64) → паддинг до 80, ритм на месте, env64 не тронут
    o3 = ColonyWSClient._apply_rhythm_to_obs(np.zeros(64, np.float32), rh)
    assert o3.shape[0] == 80
    assert np.allclose(o3[68:72], [0.5, -0.5, 0.25, -0.25])
    assert np.count_nonzero(o3[:64]) == 0


# ── Тест 7: predictor train-step с расширенным входом (dormant, end-to-end) ─
def test_predictor_train_step_72_dormant():
    """То, что handle_tick вызывает для предиктора: train-step на obs72-входе
    (env64 | self4 | rhythm0). Расширенный предиктор обучается без краша, prev_obs
    роундтрипит на 72, loss копится (target=obs64 неизменён)."""
    c = _compute()
    pred = _pred_or_skip(c)
    c._upgrade_tissue_input_dim(pred, _BRAIN_INPUT_DIM)
    c.predictor["c0"] = pred
    c.predictor_opt["c0"] = torch.optim.Adam(pred.parameters(), lr=1e-3)
    c.pred_loss_history["c0"] = []        # add_creature инициализирует в проде
    rng = np.random.default_rng(0)
    for _ in range(6):
        obs64 = torch.from_numpy(
            rng.standard_normal(64).astype(np.float32)).unsqueeze(0)
        so = torch.zeros(1, _SELF_OBS_DIM)
        rh = torch.zeros(1, _RHYTHM_DIM)                 # ритм дормантен
        soc = torch.zeros(1, _BRAIN_INPUT_DIM - _SELF_OBS_OFFSET
                          - _SELF_OBS_DIM - _RHYTHM_DIM)  # social дормантен
        obs72 = torch.cat([obs64, so, rh, soc], dim=1)
        c._predictor_train_step("c0", obs64, obs72)
    assert len(c.pred_loss_history["c0"]) >= 1           # ≥1 train-шаг прошёл
    assert int(c.predictor["c0"].data_dim) == _BRAIN_INPUT_DIM
    assert int(c.prev_obs["c0"].shape[-1]) == _BRAIN_INPUT_DIM  # вход роундтрипит на 72
