"""stamina rest-response — вторая beh-ось (Путь 2 шаг 4, Фрай 18.06.2026).

Зеркало ритм-оси, но: окно = rolling-N тиков (НЕ is_night), cost-метрика =
fatigue-интеграл НАД exhaustion-онсетом (cum_dim B, Фрай OPEN-1: «порог=mental_break-
онсет = реальное последствие»). Осцилляция ЭМЕРДЖЕНТНА из обучения rest-ответа (мозг
учит STAY-при-усталости), не из b-тюна (fatigue=открытый интегратор).

ИНЕРТ gate-1: forecast-pretrain (motor-isolated), GRADUATE (gate-2 motor-touch) на «да»
Шефа. Тест: метрика/окно/реестр + AXIS-ISOLATION forecast (stamina не загрязняет
rhythm-сайдкар и наоборот — критично для 2+ осей).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_NEUROCORE = _ROOT.parent / "NeuroCore"              # sibling: core.tissue
for _p in (str(_ROOT), str(_NEUROCORE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("torch")

from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute,
    _BRAIN_INPUT_DIM,
    _STAMINA_EXHAUSTION_ONSET,
    _STAMINA_WIN_N,
    _STAMINA_POOR_WIN,
)
from utopia_client.biochemistry import ClientCreatureBiochem  # noqa: E402

_PHI = 1.6180339887498949


def _c():
    return LocalColonyCompute(device="cpu")


def _bc(c, cid="c0", fatigue=0.0):
    b = ClientCreatureBiochem()
    b.fatigue = fatigue
    c.biochem[cid] = b
    return b


def _skip_if_no_core(c):
    if c._make_higher_tissue("probe", data_dim=64) is None:
        pytest.skip("core.tissue недоступен")


# ── Реестр: вторая ось stamina (axis-agnostic, зеркало ритма) ────────────
def test_stamina_axis_registered():
    c = _c()
    assert "stamina" in c._beh_axes
    ax = c._beh_axes["stamina"]
    assert ax["cum_dim"] == "stamina_cost_cum"
    assert ax["input_dim"] == _BRAIN_INPUT_DIM      # сайдкар читает obs O2 выносливость@[76]
    assert ax["sign"] == -1                          # cost-метрика (poor=высокая стоимость)
    assert ax["poor_win_thresh"] == _STAMINA_POOR_WIN
    assert ax["hist_n"] == 8                          # Fib (зеркало ритма)
    assert abs(ax["poor_frac"] - _PHI ** -2) < 1e-6  # φ⁻²


def test_onset_matches_neurocore_contract():
    """Порог = neurocore MENTAL_BREAK_EXHAUSTION_FATIGUE_MIN (контракт-матч, не прокси)."""
    assert _STAMINA_EXHAUSTION_ONSET == 85.0


# ── Метрика: fatigue-интеграл над онсетом за rolling-N окно ───────────────
def test_stamina_cost_accumulates_over_window():
    c = _c()
    c._phi_fatigue_enabled = True
    bc = _bc(c, fatigue=100.0)                        # excess = 100−85 = 15/тик
    # N−1 тиков: окно НЕ закрыто, cum ещё 0
    for _ in range(_STAMINA_WIN_N - 1):
        c._update_stamina_cost("c0")
    assert c._beh_stamina_cost_cum.get("c0", 0.0) == 0.0
    assert c._stam_win_ticks["c0"] == _STAMINA_WIN_N - 1
    # N-й тик → закрытие: cost = N×15, cum += cost, окно reset
    c._update_stamina_cost("c0")
    assert abs(c._beh_stamina_cost_cum["c0"] - _STAMINA_WIN_N * 15.0) < 1e-6
    assert c._stam_win_ticks["c0"] == 0              # окно сброшено
    assert c._stam_win_cost["c0"] == 0.0


def test_stamina_cost_zero_below_onset():
    """fatigue ≤ онсет → excess=0 → cost не копится (нет exhaustion = нет давления)."""
    c = _c()
    c._phi_fatigue_enabled = True
    _bc(c, fatigue=_STAMINA_EXHAUSTION_ONSET - 5.0)  # 80 < 85
    for _ in range(_STAMINA_WIN_N):
        c._update_stamina_cost("c0")
    assert c._beh_stamina_cost_cum.get("c0", 0.0) == 0.0


def test_stamina_gated_phi_fatigue_off():
    """phi_fatigue OFF → fatigue инертна → ось no-op (не копит, окно не тикает)."""
    c = _c()
    assert c._phi_fatigue_enabled is False
    _bc(c, fatigue=100.0)
    for _ in range(_STAMINA_WIN_N):
        c._update_stamina_cost("c0")
    assert c._beh_stamina_cost_cum.get("c0", 0.0) == 0.0
    assert c._stam_win_ticks.get("c0", 0) == 0


def test_stamina_no_biochem_safe():
    c = _c()
    c._phi_fatigue_enabled = True
    c._update_stamina_cost("ghost")                  # нет biochem → no-op, без краша
    assert "ghost" not in c._stam_win_ticks


def test_stamina_window_resets_and_continues():
    """После закрытия окна — новое окно копит заново (rolling)."""
    c = _c()
    c._phi_fatigue_enabled = True
    _bc(c, fatigue=95.0)                             # excess 10/тик
    for _ in range(_STAMINA_WIN_N * 2):             # 2 полных окна
        c._update_stamina_cost("c0")
    assert abs(c._beh_stamina_cost_cum["c0"] - 2 * _STAMINA_WIN_N * 10.0) < 1e-6


# ── Метрика пассивна (копится без флага роста, наблюдаемость) ─────────────
def test_metric_passive_independent_of_growth_flag():
    c = _c()
    c._phi_fatigue_enabled = True
    assert c._behavioral_growth_enabled is False     # рост OFF
    _bc(c, fatigue=100.0)
    for _ in range(_STAMINA_WIN_N):
        c._update_stamina_cost("c0")
    assert c._beh_stamina_cost_cum["c0"] > 0.0       # метрика всё равно копится


def test_stamina_cost_in_gc_sample():
    c = _c()
    _bc(c, fatigue=0.0)
    c._beh_stamina_cost_cum["c0"] = 777.0
    s = c._beh_gc_sample("c0")
    assert s.get("stamina_cost_cum") == 777.0
    assert "dark_loss_cum" in s                       # ритм-ось не сломана


# ── КРИТИЧНО: axis-isolation forecast (2+ осей не загрязняют друг друга) ──
def test_forecast_train_axis_isolated():
    c = _c()
    _skip_if_no_core(c)
    c.set_behavioral_growth(True)
    c.organisms["c0"] = object()
    c._last_world_tick = 0
    # минтим по сайдкару на КАЖДУЮ ось напрямую (one-at-a-time обходим)
    assert c._propose_behavioral_tissue("c0", c._beh_axes["rhythm"]) is True
    assert c._propose_behavioral_tissue("c0", c._beh_axes["stamina"]) is True
    axis_of = c._beh_grown_axis["c0"]
    r_role = next(k for k, v in axis_of.items() if v == "rhythm")
    s_role = next(k for k, v in axis_of.items() if v == "stamina")
    import torch
    # per-axis форкаст-входы
    c._beh_forecast_input["c0"] = {
        "rhythm": torch.randn(1, _BRAIN_INPUT_DIM),
        "stamina": torch.randn(1, _BRAIN_INPUT_DIM),
    }
    assert c._beh_forecast_err["c0"][r_role] is None
    assert c._beh_forecast_err["c0"][s_role] is None
    # тренируем ТОЛЬКО stamina → rhythm-сайдкар НЕ тронут
    c._beh_forecast_train("c0", 3000.0, "stamina")
    assert c._beh_forecast_err["c0"][s_role] is not None   # stamina обучен
    assert c._beh_forecast_err["c0"][r_role] is None       # rhythm НЕ загрязнён
    # теперь rhythm → его сайдкар обучается, stamina не тикает повторно
    s_trained0 = c._beh_forecast_trained["c0"][s_role]
    c._beh_forecast_train("c0", 15.0, "rhythm")
    assert c._beh_forecast_err["c0"][r_role] is not None   # rhythm обучен
    assert c._beh_forecast_trained["c0"][s_role] == s_trained0  # stamina не двинулся


def test_forecast_train_default_axis_rhythm_backcompat():
    """Старый вызов _beh_forecast_train(cid, drop) (flat input, default rhythm) цел."""
    c = _c()
    _skip_if_no_core(c)
    c.set_behavioral_growth(True)
    c.organisms["c0"] = object()
    assert c._propose_behavioral_tissue("c0", c._beh_axes["rhythm"]) is True
    role = next(iter(c._beh_grown_tissues["c0"]))
    import torch
    c._beh_forecast_input["c0"] = torch.randn(1, _BRAIN_INPUT_DIM)  # FLAT (старый формат)
    c._beh_forecast_train("c0", 15.0)                # default axis_key="rhythm"
    assert c._beh_forecast_err["c0"][role] is not None   # isinstance-fallback работает


# ── gate-2: per-axis grace + poor-приоритет graduation (Фрай 19.06) ──────
def _set_baseline(c, cid, axis, mean, n=40):
    c._beh_axis_drop_sum.setdefault(cid, {})[axis] = float(mean) * n
    c._beh_axis_drop_n.setdefault(cid, {})[axis] = n


def _grad_ready(c, cid="c0"):
    c.set_behavioral_graduation(True)
    c.organisms[cid] = object()
    c.biochem[cid] = types.SimpleNamespace(energy=999.0)  # ≥_GRAD_HEALTH_ENERGY


def test_stamina_grace_nights_in_descriptor():
    c = _c()
    assert c._beh_axes["stamina"]["grace_nights"] == 3     # bootstrap-рано (транзиент)
    assert c._beh_axes["rhythm"]["grace_nights"] is None    # → глобальный 34


def test_graduate_stamina_at_low_grace():
    """stamina grace_nights=3 → созревает на 3 окнах (vs глобальные 34)."""
    c = _c()
    _grad_ready(c)
    c._beh_grown_tissues["c0"] = {"beh7": object()}
    c._beh_forecast_err["c0"] = {"beh7": 1.0}
    c._beh_grown_axis["c0"] = {"beh7": "stamina"}
    c._beh_forecast_trained["c0"] = {"beh7": 3}            # =grace_nights stamina
    _set_baseline(c, "c0", "stamina", 4.0)
    assert c._maybe_behavioral_graduate("c0", None) is True
    assert "beh7" in c._beh_graduated["c0"]


def test_graduate_stamina_blocked_below_grace():
    c = _c()
    _grad_ready(c)
    c._beh_grown_tissues["c0"] = {"beh7": object()}
    c._beh_forecast_err["c0"] = {"beh7": 1.0}
    c._beh_grown_axis["c0"] = {"beh7": "stamina"}
    c._beh_forecast_trained["c0"] = {"beh7": 2}            # <3 → не созрел
    _set_baseline(c, "c0", "stamina", 4.0)
    assert c._maybe_behavioral_graduate("c0", None) is False


def test_graduate_rhythm_still_needs_global_grace():
    """rhythm grace_nights=None → глобальный 34 (per-axis НЕ ослабил ритм)."""
    c = _c()
    _grad_ready(c)
    c._beh_grown_tissues["c0"] = {"beh1": object()}
    c._beh_forecast_err["c0"] = {"beh1": 1.0}
    c._beh_grown_axis["c0"] = {"beh1": "rhythm"}
    c._beh_forecast_trained["c0"] = {"beh1": 3}           # <34 → ритм НЕ созрел
    _set_baseline(c, "c0", "rhythm", 4.0)
    assert c._maybe_behavioral_graduate("c0", None) is False


def test_graduate_poor_axis_priority():
    """Вариант a (Фрай): poor-ось (stamina под давлением) graduate'ит ПОПЕРЁД зрелого
    rhythm с ЛУЧШЕЙ skill (не poor). Гарантия: лечим ось, что болит (расшибаем пин)."""
    c = _c()
    _grad_ready(c)
    c._beh_grown_tissues["c0"] = {"beh1": object(), "beh7": object()}
    c._beh_grown_axis["c0"] = {"beh1": "rhythm", "beh7": "stamina"}
    c._beh_forecast_err["c0"] = {"beh1": 0.5, "beh7": 2.0}  # rhythm skill ЛУЧШЕ
    c._beh_forecast_trained["c0"] = {"beh1": 34, "beh7": 3}  # оба созрелы (per-axis)
    _set_baseline(c, "c0", "rhythm", 4.0)
    _set_baseline(c, "c0", "stamina", 4.0)
    # stamina POOR (costly-окна >233), rhythm НЕ poor (штиль)
    c._beh_axis_hist["c0"] = {"stamina": [3483.0] * 8, "rhythm": [0.0] * 8}
    assert c._maybe_behavioral_graduate("c0", None) is True
    assert "beh7" in c._beh_graduated.get("c0", {})        # stamina, НЕ rhythm
    assert "beh1" not in c._beh_graduated.get("c0", {})


def test_graduate_skill_tiebreak_when_neither_poor():
    """Равный poor-статус (оба НЕ poor) → лучший skill (существующее поведение цело)."""
    c = _c()
    _grad_ready(c)
    c._beh_grown_tissues["c0"] = {"beh1": object(), "beh7": object()}
    c._beh_grown_axis["c0"] = {"beh1": "rhythm", "beh7": "stamina"}
    c._beh_forecast_err["c0"] = {"beh1": 0.5, "beh7": 2.0}
    c._beh_forecast_trained["c0"] = {"beh1": 34, "beh7": 3}
    _set_baseline(c, "c0", "rhythm", 4.0)
    _set_baseline(c, "c0", "stamina", 4.0)
    c._beh_axis_hist["c0"] = {"stamina": [0.0] * 8, "rhythm": [0.0] * 8}  # оба НЕ poor
    assert c._maybe_behavioral_graduate("c0", None) is True
    assert "beh1" in c._beh_graduated.get("c0", {})        # лучший skill (rhythm 0.5)
