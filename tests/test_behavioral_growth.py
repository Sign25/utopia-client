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
    _SELF_OBS_OFFSET,
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


# ── S2: пороги mint-предиката ЗАЛОЧЕНЫ из replay-калибровки (Фрай 15.06) ──
def test_rhythm_axis_thresholds_locked():
    c = _c()
    ax = c._beh_axes["rhythm"]
    assert ax["poor_win_thresh"] == 8.0            # Fib, долина над winter-mean 6.4
    assert ax["hist_n"] == 8                        # Fib, ~полгода
    assert abs(ax["poor_frac"] - (1.6180339887 ** -2)) < 1e-6  # φ⁻²≈0.382 (=4/8)


# ── S2: rolling-история окон (trim до hist_n) ────────────────────────────
def test_record_axis_window_trims():
    c = _c()
    for i in range(12):
        c._record_axis_window("c0", "rhythm", float(i))
    h = c._beh_axis_hist["c0"]["rhythm"]
    assert len(h) == 8                             # держим только hist_n
    assert h == [4., 5., 6., 7., 8., 9., 10., 11.]  # последние 8


# ── S2: _axis_poor — БАЙТ-В-БАЙТ replay-предикат ─────────────────────────
def test_axis_poor_replay_predicate():
    c = _c()
    ax = c._beh_axes["rhythm"]
    # <hist_n сэмплов → False (не родим на холодную)
    c._beh_axis_hist["c0"] = {"rhythm": [20., 20., 20.]}
    assert c._axis_poor("c0", ax) is False
    # 4/8 costly (>8) → доля 0.5 ≥ φ⁻² → True
    c._beh_axis_hist["c0"] = {"rhythm": [10., 10., 10., 10., 0., 0., 0., 0.]}
    assert c._axis_poor("c0", ax) is True
    # 3/8 costly → доля 0.375 < φ⁻²(0.382) → False
    c._beh_axis_hist["c0"] = {"rhythm": [10., 10., 10., 0., 0., 0., 0., 0.]}
    assert c._axis_poor("c0", ax) is False
    # ровно =8 НЕ costly (строго >thresh) → 0 costly → False
    c._beh_axis_hist["c0"] = {"rhythm": [8.] * 8}
    assert c._axis_poor("c0", ax) is False


# ── S2: mint-гейты self-limiting (без core — гейты до propose) ───────────
def test_mint_gated_dormant_flag_off():
    c = _c()
    c._beh_axis_hist["c0"] = {"rhythm": [20.] * 8}   # poor=True
    c.organisms["c0"] = object()
    assert c._behavioral_growth_enabled is False
    assert c._maybe_behavioral_mint("c0", None) is False   # флаг OFF → dormant
    assert not c._beh_grown_tissues.get("c0")


def test_mint_gated_paralysis():
    c = _c()
    c.set_behavioral_growth(True)
    c._beh_axis_hist["c0"] = {"rhythm": [20.] * 8}
    c.organisms["c0"] = object()
    c._paralysis_window_n = 1                        # §3 активен → абс-гейт
    assert c._maybe_behavioral_mint("c0", None) is False


def test_mint_gated_cooldown():
    c = _c()
    c.set_behavioral_growth(True)
    c._beh_axis_hist["c0"] = {"rhythm": [20.] * 8}
    c.organisms["c0"] = object()
    c._last_world_tick = 1000
    c._beh_mint_last["c0"] = 900                     # 100 < cooldown 233
    assert c._maybe_behavioral_mint("c0", None) is False


# ── S2: реальное рождение сайдкара obs72 (guarded — нужен core.tissue) ────
def _skip_if_no_core(c):
    if c._make_higher_tissue("probe", data_dim=64) is None:
        pytest.skip("core.tissue недоступен (dev-venv) — tissue-creation skip")


def test_mint_creates_obs72_sidecar_math_equiv():
    import torch
    c = _c()
    _skip_if_no_core(c)
    c.set_behavioral_growth(True)
    c.organisms["c0"] = object()
    c._beh_axis_hist["c0"] = {"rhythm": [20.] * 8}   # poor=True
    assert c._maybe_behavioral_mint("c0", None) is True
    grown = c._beh_grown_tissues["c0"]
    assert len(grown) == 1
    role = next(iter(grown))
    t = grown[role]
    assert int(t.data_dim) == _BRAIN_INPUT_DIM        # component 3: читает obs76
    # math-equivalence: ритм+social-колонки input_proj = 0 (zero-init [I_64|0])
    assert torch.count_nonzero(t.input_proj.weight[:, 68:_BRAIN_INPUT_DIM]) == 0
    assert c._beh_grown_axis["c0"][role] == "rhythm"
    # one-at-a-time + cooldown: повторный вызов не родит вторую
    assert c._maybe_behavioral_mint("c0", None) is False


def test_killswitch_removes_sidecars():
    c = _c()
    _skip_if_no_core(c)
    c.set_behavioral_growth(True)
    c.organisms["c0"] = object()
    c._beh_axis_hist["c0"] = {"rhythm": [20.] * 8}
    c._maybe_behavioral_mint("c0", None)
    assert c._beh_grown_tissues.get("c0")
    assert c._beh_forecast_head.get("c0")            # S3 forecast-состояние создано
    c.set_behavioral_growth(False)                   # kill-switch
    assert not c._beh_grown_tissues.get("c0")        # все сайдкары сняты
    assert not c._beh_forecast_head.get("c0")        # + forecast-состояние
    assert c._behavioral_growth_enabled is False


# ── S3: gate-2 флаг (graduation — отдельный гейт от gate-1) ───────────────
def test_set_behavioral_graduation_flag():
    c = _c()
    assert c._behavioral_graduation_enabled is False   # dormant default
    assert c.set_behavioral_graduation(True) is True
    assert c._behavioral_graduation_enabled is True
    assert c.set_behavioral_graduation(False) is False
    assert c._behavioral_graduation_enabled is False


def test_gates_independent():
    """gate-1 (growth) и gate-2 (graduation) — раздельные флаги."""
    c = _c()
    c.set_behavioral_growth(True)
    assert c._behavioral_growth_enabled is True
    assert c._behavioral_graduation_enabled is False   # gate-2 НЕ включился вместе
    c.set_behavioral_graduation(True)
    c.set_behavioral_growth(False)
    assert c._behavioral_graduation_enabled is True     # независим


# ── S3: pool-cull мёртвого форкастера (логика, без core) ─────────────────
def _set_baseline(c, cid, axis, mean, n=40):
    """global baseline = drop_sum/drop_n = mean (Фрай: шип=replay-определение)."""
    c._beh_axis_drop_sum.setdefault(cid, {})[axis] = float(mean) * n
    c._beh_axis_drop_n.setdefault(cid, {})[axis] = n


def test_pool_cull_clearly_dead_phi_margin():
    """grace созрел + err ≥ φ×baseline (clearly-dead) → cull; маржинальный выживает."""
    c = _c()
    c._beh_grown_tissues["c0"] = {"beh1": object(), "beh2": object()}
    c._beh_grown_axis["c0"] = {"beh1": "rhythm", "beh2": "rhythm"}
    _set_baseline(c, "c0", "rhythm", 4.0)                  # global baseline=4 → φ×=6.47
    c._beh_forecast_trained["c0"] = {"beh1": 34, "beh2": 34}  # созрели (≥grace)
    c._beh_forecast_err["c0"] = {"beh1": 10.0, "beh2": 5.0}   # beh1 dead(≥6.47), beh2 маржа(<6.47)
    c._beh_pool_cull("c0")
    assert "beh1" not in c._beh_grown_tissues["c0"]        # clearly-dead снят
    assert "beh2" in c._beh_grown_tissues["c0"]            # маржинальный ВЫЖИЛ (warm-start)


def test_pool_cull_grace_protects_young():
    """Незрелый форкастер (trained<grace) НЕ cull'ится даже с высокой err."""
    c = _c()
    c._beh_grown_tissues["c0"] = {"beh1": object()}
    c._beh_grown_axis["c0"] = {"beh1": "rhythm"}
    _set_baseline(c, "c0", "rhythm", 4.0)
    c._beh_forecast_trained["c0"] = {"beh1": 10}          # <grace 34 — младенец
    c._beh_forecast_err["c0"] = {"beh1": 99.0}            # высокая err, но grace защищает
    c._beh_pool_cull("c0")
    assert "beh1" in c._beh_grown_tissues["c0"]           # не тронут до созревания


def test_pool_cull_skips_untrained():
    c = _c()
    c._beh_grown_tissues["c0"] = {"beh1": object()}
    c._beh_grown_axis["c0"] = {"beh1": "rhythm"}
    _set_baseline(c, "c0", "rhythm", 5.0)
    c._beh_forecast_err["c0"] = {"beh1": None}             # ещё не тренился (err=None)
    c._beh_pool_cull("c0")
    assert "beh1" in c._beh_grown_tissues["c0"]            # нетренированного не трогаем


def test_axis_baseline_global_mean():
    c = _c()
    _set_baseline(c, "c0", "rhythm", 3.9, n=100)
    assert abs(c._beh_axis_baseline("c0", "rhythm") - 3.9) < 1e-6
    assert c._beh_axis_baseline("c0", "missing") == 0.0


# ── S3: forecast pipeline (guarded — нужен core.tissue) ──────────────────
def test_forecast_pipeline_obs72():
    import torch
    c = _c()
    _skip_if_no_core(c)
    c.set_behavioral_growth(True)
    c.organisms["c0"] = object()
    c._beh_axis_hist["c0"] = {"rhythm": [20.] * 8}        # poor
    c._last_world_tick = 0
    assert c._maybe_behavioral_mint("c0", None) is True
    role = next(iter(c._beh_grown_tissues["c0"]))
    head = c._beh_forecast_head["c0"][role]
    assert int(torch.count_nonzero(head.weight)) == 0     # zero-init readout
    # per-tick инференс: obs76 → живой форкаст
    obs72 = torch.randn(1, _BRAIN_INPUT_DIM)
    c._beh_forecast_infer("c0", obs72)
    assert role in c._beh_forecast_live["c0"]
    # разреженный тренинг пары (obs72@night-start, drop=15)
    c._beh_forecast_input["c0"] = obs72
    assert c._beh_forecast_err["c0"][role] is None
    c._beh_forecast_train("c0", 15.0)
    assert c._beh_forecast_err["c0"][role] is not None     # err (skill) измерен


def test_pool_cap_blocks_mint():
    c = _c()
    _skip_if_no_core(c)
    c.set_behavioral_growth(True)
    c.organisms["c0"] = object()
    c._beh_axis_hist["c0"] = {"rhythm": [20.] * 8}
    for _ in range(c._BEH_POOL_CAP):                       # заполнить пул до cap
        c._propose_behavioral_tissue("c0", c._beh_axes["rhythm"])
    assert len(c._beh_grown_tissues["c0"]) == c._BEH_POOL_CAP
    c._beh_mint_last.pop("c0", None)                       # снять cooldown
    assert c._maybe_behavioral_mint("c0", None) is False   # пул полон → не родит


# ── S4 gate-2: graduation (мотор-голова) self-limiting (без core) ─────────
def test_graduate_gated_dormant():
    c = _c()
    c.organisms["c0"] = object()
    c._beh_grown_tissues["c0"] = {"beh1": object()}
    c._beh_forecast_err["c0"] = {"beh1": 1.0}
    c._beh_grown_axis["c0"] = {"beh1": "rhythm"}
    c._beh_axis_hist["c0"] = {"rhythm": [10.] * 8}
    assert c._behavioral_graduation_enabled is False
    assert c._maybe_behavioral_graduate("c0", None) is False   # gate-2 OFF → dormant
    assert not c._beh_graduated.get("c0")


def test_graduate_gated_paralysis():
    c = _c()
    c.set_behavioral_graduation(True)
    c.organisms["c0"] = object()
    c._beh_grown_tissues["c0"] = {"beh1": object()}
    c._beh_forecast_err["c0"] = {"beh1": 1.0}
    c._beh_grown_axis["c0"] = {"beh1": "rhythm"}
    c._beh_axis_hist["c0"] = {"rhythm": [10.] * 8}
    c._paralysis_window_n = 1                              # §3 → абс-гейт
    assert c._maybe_behavioral_graduate("c0", None) is False


def test_graduate_skips_clearly_dead_forecaster():
    c = _c()
    c.set_behavioral_graduation(True)
    c.organisms["c0"] = object()
    c.biochem["c0"] = types.SimpleNamespace(energy=999.0)
    c._beh_grown_tissues["c0"] = {"beh1": object()}
    c._beh_forecast_err["c0"] = {"beh1": 20.0}            # err ≥ φ×baseline → clearly-dead
    c._beh_grown_axis["c0"] = {"beh1": "rhythm"}
    c._beh_forecast_trained["c0"] = {"beh1": 34}          # созрел
    _set_baseline(c, "c0", "rhythm", 4.0)                 # φ×4=6.47, err20≥ → не выпускаем
    assert c._maybe_behavioral_graduate("c0", None) is False


def test_graduate_skips_immature_forecaster():
    """Зрелость (grace): незрелый форкастер НЕ выпускается даже с хорошей err."""
    c = _c()
    c.set_behavioral_graduation(True)
    c.organisms["c0"] = object()
    c.biochem["c0"] = types.SimpleNamespace(energy=999.0)
    c._beh_grown_tissues["c0"] = {"beh1": object()}
    c._beh_forecast_err["c0"] = {"beh1": 1.0}             # хорошая err
    c._beh_grown_axis["c0"] = {"beh1": "rhythm"}
    c._beh_forecast_trained["c0"] = {"beh1": 10}          # <grace 34 — незрел
    _set_baseline(c, "c0", "rhythm", 4.0)
    assert c._maybe_behavioral_graduate("c0", None) is False  # grace держит


def test_revert_behavioral_node():
    c = _c()
    c._beh_motor_head["c0"] = {"beh1": object()}
    c._beh_graduated["c0"] = {"beh1": object()}
    c._beh_motor_opt["c0"] = {"beh1": object()}
    c._revert_behavioral_node("c0", "beh1")
    assert "beh1" not in c._beh_motor_head["c0"]           # голова снята (база бит-в-бит)
    assert "beh1" not in c._beh_graduated["c0"]


def test_gate1_off_reverts_graduations():
    c = _c()
    c.set_behavioral_growth(True)
    c._beh_motor_head["c0"] = {"beh1": object()}
    c._beh_graduated["c0"] = {"beh1": object()}
    c.set_behavioral_growth(False)                         # мастер-стоп
    assert not c._beh_motor_head.get("c0")                 # + мотор-головы сняты
    assert not c._beh_graduated.get("c0")


def test_gate2_off_reverts_only_graduations():
    c = _c()
    c.set_behavioral_growth(True)
    c.set_behavioral_graduation(True)
    c._beh_grown_tissues["c0"] = {"beh1": "tissue-ref"}    # сайдкар-пул
    c._beh_motor_head["c0"] = {"beh1": object()}
    c._beh_graduated["c0"] = {"beh1": object()}
    c.set_behavioral_graduation(False)                     # gate-2 OFF (не мастер)
    assert not c._beh_graduated.get("c0")                  # graduations сняты
    assert c._beh_grown_tissues.get("c0")                  # сайдкары ЖИВЫ (re-graduate)


# ── S4 gate-2: реальная graduation zero-init головы (guarded — core) ─────
def test_graduate_creates_zero_init_motor_head():
    import torch
    c = _c()
    _skip_if_no_core(c)
    c.set_behavioral_growth(True)
    c.set_behavioral_graduation(True)
    c.organisms["c0"] = object()
    c.biochem["c0"] = types.SimpleNamespace(energy=999.0)
    c._beh_axis_hist["c0"] = {"rhythm": [10.] * 8}
    c._propose_behavioral_tissue("c0", c._beh_axes["rhythm"])
    role = next(iter(c._beh_grown_tissues["c0"]))
    c._beh_forecast_err["c0"][role] = 1.0                  # хороший skill
    c._beh_forecast_trained["c0"][role] = 34              # созрел (grace)
    _set_baseline(c, "c0", "rhythm", 4.0)                 # err1 < φ×4 → не-мёртв, выпускаем
    assert c._maybe_behavioral_graduate("c0", None) is True
    assert role in c._beh_graduated["c0"]
    head = c._beh_motor_head["c0"][role]
    assert int(torch.count_nonzero(head.weight)) == 0      # zero-init → NO-OP на флипе
    assert int(torch.count_nonzero(head.bias)) == 0
    assert c._maybe_behavioral_graduate("c0", None) is False  # one-at-a-time
    c.set_behavioral_graduation(False)                     # revert
    assert not c._beh_graduated.get("c0")                  # голова снята бит-в-бит


def test_beh_motor_reinforce_learns():
    """REINFORCE-путь головы (рисковейший новый код): zero-init + ненулевой
    advantage + grad → веса сдвигаются (влияние растёт по заслуге)."""
    import torch
    from utopia_client.local_compute import N_ACTIONS
    c = _c()
    _skip_if_no_core(c)
    c.set_behavioral_growth(True)
    c.set_behavioral_graduation(True)
    c.organisms["c0"] = object()
    c.biochem["c0"] = types.SimpleNamespace(energy=999.0)
    c._beh_axis_hist["c0"] = {"rhythm": [10.] * 8}
    c._propose_behavioral_tissue("c0", c._beh_axes["rhythm"])
    role = next(iter(c._beh_grown_tissues["c0"]))
    c._beh_forecast_err["c0"][role] = 1.0
    c._beh_forecast_trained["c0"][role] = 34             # созрел (grace)
    _set_baseline(c, "c0", "rhythm", 4.0)
    c._maybe_behavioral_graduate("c0", None)
    head = c._beh_motor_head["c0"][role]
    w0 = head.weight.detach().clone()
    to = torch.randn(1, _SELF_OBS_OFFSET)
    base = torch.zeros(N_ACTIONS)
    c._beh_motor_reinforce("c0", role, (to, 3, base), advantage=1.0)
    assert not torch.equal(head.weight.detach(), w0)       # голова обучилась
    # ctx с action=None → no-op (защита от незаполненного ctx)
    c._beh_motor_reinforce("c0", role, (to, None, base), advantage=1.0)


# ── Независимая петля _behavioral_growth_step (фикс births=0) ────────────
def test_behavioral_growth_step_dormant():
    c = _c()
    c.organisms["c0"] = object()
    assert c._behavioral_growth_enabled is False
    c._behavioral_growth_step("c0")                       # флаг OFF → no-op, без краша
    c.set_behavioral_growth(True)
    c._behavioral_growth_step("ghost")                    # org None → no-op


def test_behavioral_growth_step_reaches_mint():
    """Петля независима от tissue_growth/predictor-GC → mint достигается."""
    c = _c()
    _skip_if_no_core(c)
    c.set_behavioral_growth(True)
    c.organisms["c0"] = object()
    c._beh_axis_hist["c0"] = {"rhythm": [20.] * 8}        # poor
    c._behavioral_growth_step("c0")
    assert c._beh_grown_tissues.get("c0")                 # mint сработал через петлю


# ── S4b: retention-СЕЛЕКТОР (GC-ablation на neg_dark_loss, без core) ──────
def test_head_gc_start_sets_ablate_mask():
    c = _c()
    c.set_behavioral_graduation(True)
    c.organisms["c0"] = object()
    c.biochem["c0"] = types.SimpleNamespace(energy=999.0)
    c._beh_graduated["c0"] = {"beh1": object()}
    c._last_world_tick = 10 ** 9                           # вне §3/cooldown
    assert c._maybe_start_beh_head_gc("c0", None) is True
    assert c._beh_motor_ablate["c0"] == "beh1"             # старт = ablate-фаза
    assert "c0" in c._beh_head_gc_state


def test_head_gc_resolve_keep_when_action_helps():
    """ablate dark-loss >> restore (голова-действие снижает dark-loss) → KEEP."""
    c = _c()
    c._beh_graduated["c0"] = {"beh1": object()}
    c._beh_motor_head["c0"] = {"beh1": object()}
    st = {"role": "beh1", "phase": "ablate", "pairs_done": 13,
          "samples": {
              "ablate": [20., 18., 22., 19., 21., 20., 18., 23., 19., 21., 20., 18., 22.],
              "restore": [3., 2., 4., 2., 3., 3., 2., 4., 2., 3., 3., 2., 4.]}}
    c._beh_head_gc_state["c0"] = st
    c._resolve_beh_head_gc("c0", st)
    assert "beh1" in c._beh_graduated.get("c0", {})        # KEEP (не снят)
    assert "c0" not in c._beh_head_gc_state                # resolve завершён


def test_head_gc_resolve_cull_when_no_help():
    """ablate ≈ restore (голова не снижает dark-loss) → CULL (revert бит-в-бит)."""
    c = _c()
    c._beh_graduated["c0"] = {"beh1": object()}
    c._beh_motor_head["c0"] = {"beh1": object()}
    st = {"role": "beh1", "phase": "ablate", "pairs_done": 13,
          "samples": {
              "ablate": [5., 4., 6., 5., 4., 6., 5., 4., 6., 5., 4., 6., 5.],
              "restore": [5., 5., 4., 6., 5., 4., 5., 6., 4., 5., 5., 4., 6.]}}
    c._beh_head_gc_state["c0"] = st
    c._resolve_beh_head_gc("c0", st)
    assert "beh1" not in c._beh_graduated.get("c0", {})    # CULL (снят)


def test_head_gc_year_gate_blocks_early_resolve():
    """pairs готовы, но <1 год накоплен → НЕ резолвит (сезонная честность #3)."""
    c = _c()
    c._beh_dark_loss_cum["c0"] = 0.0
    c._last_world_tick = 100
    st = {"role": "beh1", "phase": "ablate", "pairs_done": c._BEH_GC_PAIRS,
          "win_ticks": c._BEH_GC_WINDOW - 1, "win0_dark": 0.0,
          "gc_start": 50,                                   # 50 тиков ≪ 1 год
          "samples": {"ablate": [], "restore": []}}
    c._beh_head_gc_state["c0"] = st
    c._beh_motor_ablate["c0"] = "beh1"
    c._beh_head_gc_step("c0", st)                          # окно закрывается
    assert "c0" in c._beh_head_gc_state                    # год-гейт держит → НЕ резолв


def test_head_gc_window_toggles_phase():
    c = _c()
    c._beh_dark_loss_cum["c0"] = 10.0
    st = {"role": "beh1", "phase": "ablate", "pairs_done": 0,
          "win_ticks": c._BEH_GC_WINDOW - 1, "win0_dark": 0.0,
          "gc_start": 0, "samples": {"ablate": [], "restore": []}}
    c._beh_head_gc_state["c0"] = st
    c._beh_motor_ablate["c0"] = "beh1"
    c._beh_head_gc_step("c0", st)                          # закрыть ablate-окно
    assert len(st["samples"]["ablate"]) == 1               # сэмпл записан
    assert st["samples"]["ablate"][0] == 10.0              # dark-loss за окно
    assert st["phase"] == "restore"                        # фаза переключилась
    assert "c0" not in c._beh_motor_ablate                 # маска снята (restore)
