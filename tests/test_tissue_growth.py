"""§10.8 рост ТКАНЕЙ (Фрай 09.06): Адам растит мозг УЗЛАМИ (не только рёбрами).
После насыщения связей (19/19→cerebellum) петля минтит новую ткань-кандидата
(sidecar, читает obs как сенсор) + проводит {роль}→cerebellum → dwell → Δloss_ema →
keep/backoff (как рёбра). Драйвер prediction: keep-путь спит, пока мир не вернёт
давление (loss 0.046 = пол). Дисциплина: одна за раз, dwell, durable-персист,
kill-switch. Client-local.

Тесты: флаг/сеттер + персист счётчиков/спеков + no-op гейты (dev-venv). Полный
mint/insert/remove — neurocore-gated (skip без core).
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

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402


def _c():
    return LocalColonyCompute(device="cpu")


# ── kill-switch ─────────────────────────────────────────────────────────

def test_tissue_growth_default_off():
    c = _c()
    assert c._tissue_growth_enabled is False     # dormant дефолт


def test_set_tissue_growth_toggle():
    c = _c()
    assert c.set_tissue_growth(True) is True
    assert c._tissue_growth_enabled is True
    assert c.set_tissue_growth(False) is False
    assert c._tissue_growth_enabled is False


# ── no-op гейты (узлы после рёбер, только при насыщении) ─────────────────

def test_step_noop_when_disabled():
    c = _c()
    c.organisms["a"] = types.SimpleNamespace(tissues={}, generation=0)
    c._tissue_growth_enabled = False
    c._tissue_growth_step("a")                   # не падает, ничего не делает
    assert "a" not in c._tissue_growth_state


def test_step_noop_when_connections_not_saturated():
    # флаг ON, но связи ещё не насыщены → ткани ждут (узлы после рёбер).
    c = _c()
    c.set_tissue_growth(True)
    c._growth_saturated = False
    c.organisms["a"] = types.SimpleNamespace(tissues={}, generation=0)
    c.predictor["a"] = object()
    # _cerebellum_tissue_id вернёт None (нет cerebellum) → ранний выход в любом случае
    c._tissue_growth_step("a")
    assert "a" not in c._tissue_growth_state


# ── персист счётчиков + спеков (durable через рестарт) ──────────────────

def test_tissue_counters_persist_round_trip():
    src = _c()
    src.organisms["a"] = types.SimpleNamespace(generation=0)
    src._tissue_kept = 3
    src._tissue_reverted = 2
    payload = src.save_state("a")
    gl = payload["growth_loop"]
    assert gl["tissue_kept"] == 3 and gl["tissue_reverted"] == 2
    assert gl["grown_tissues"] == []             # ещё ни одной KEEP'нутой

    dst = _c()
    dst.organisms["a"] = types.SimpleNamespace(generation=0)
    assert dst._tissue_kept == 0
    dst.restore_persisted_state("a", payload)
    assert dst._tissue_kept == 3 and dst._tissue_reverted == 2


def test_grown_tissue_specs_persisted():
    src = _c()
    src.organisms["a"] = types.SimpleNamespace(generation=0)
    src._tissue_grown_specs["a"] = [{"role": "grown1", "data_dim": 64, "n_embd": 21}]
    payload = src.save_state("a")
    assert payload["growth_loop"]["grown_tissues"] == [
        {"role": "grown1", "data_dim": 64, "n_embd": 21}]


# ── редизайн a: predictor-сайдкар, ИЗОЛИРОВАН от графа/мотора (neurocore-gated) ──

def test_propose_sidecar_isolated_from_graph():
    pytest.importorskip("core.tissue")
    import torch
    c = _c()
    cb = c._make_higher_tissue("cerebellum", data_dim=64)
    if cb is None:
        pytest.skip("cerebellum tissue build failed (neurocore?)")
    org = types.SimpleNamespace(
        tissues={"cb": cb}, connections=[], tissue_topology_genes=[])
    c.organisms["a"] = org
    c.predictor["a"] = object()
    c.loss_ema["a"] = 0.05

    assert c._propose_growth_tissue("a", org) is True
    role = c._tissue_growth_state["a"]["role"]
    # сайдкар в _grown_tissues, НЕ в графе (мотор изолирован)
    assert role in c._grown_tissues["a"]
    assert role not in {getattr(t, "role", None) for t in org.tissues.values()}
    assert org.tissue_topology_genes == []          # НЕТ ребра →cerebellum
    # вклад сайдкара во вход предиктора: [1, DATA_DIM]
    gc = c._grown_pred_contribution("a", torch.zeros(1, 64))
    assert gc is not None and gc.shape[-1] == 64

    # backoff: сайдкар удаляется, граф нетронут
    c._remove_grown_tissue("a", role=role)
    assert not c._grown_tissues.get("a")
    assert org.tissue_topology_genes == []


def test_grown_contribution_none_without_sidecars():
    c = _c()
    import torch
    assert c._grown_pred_contribution("a", torch.zeros(1, 64)) is None


# ── durability GC (Фрай 10.06): dwell ≥ погодный цикл + leave-one-out ре-оценка ──

def test_tissue_dwell_is_full_weather_cycle():
    # dwell 89<цикл 233 судил по фрагменту → фазовый шум. Окно ≥ цикл → durable.
    c = _c()
    assert c._tissue_growth_dwell_ticks == 233          # = период sin/233


def test_gc_start_paired_soft_ablate():
    # PAIRED (Фрай 14.06): старт = soft-маска роли (спек ЦЕЛ в _grown_tissues),
    # НЕ held-aside removal. Машина: phase=ablate, пустые paired-samples.
    c = _c()
    c._grown_tissues["a"] = {"grown1": object(), "grown2": object()}
    c.loss_ema["a"] = 0.05
    c._last_world_tick = 1000
    assert c._maybe_start_tissue_gc("a") is True
    gc = c._tissue_gc_state["a"]
    assert gc["role"] in ("grown1", "grown2")
    assert gc["role"] in c._grown_tissues["a"]           # soft-маска: спек/веса целы
    assert c._tissue_gc_ablate["a"] == gc["role"]         # роль маскирована из вклада
    assert gc["phase"] == "ablate" and gc["pairs_done"] == 0
    assert gc["samples"] == {"ablate": [], "restore": []}


def test_gc_step_toggles_mask_each_window():
    # машина: окно ablate (transient+window) → закрыть → toggle в restore (маска снята).
    c = _c()
    c._grown_tissues["a"] = {"grown1": object()}
    c.loss_ema["a"] = 0.05
    c._last_world_tick = 0
    c._maybe_start_tissue_gc("a")
    gc = c._tissue_gc_state["a"]
    assert c._tissue_gc_ablate["a"] == "grown1"          # старт ablate
    for _ in range(c._BEH_GC_TRANSIENT + c._BEH_GC_WINDOW):
        c._tissue_gc_step("a", gc)
    assert len(gc["samples"]["ablate"]) == 1             # окно ablate закрылось
    assert gc["phase"] == "restore"                       # toggle
    assert "a" not in c._tissue_gc_ablate                 # маска снята (restore-фаза)


def test_gc_paired_prune_when_no_durable_effect():
    # ablate ≈ restore (удаление НЕ меняет loss) → median≈0 → PRUNE (noise).
    c = _c()
    c._grown_tissues["a"] = {"grown1": object()}
    c._tissue_grown_specs["a"] = [{"role": "grown1", "data_dim": 64, "n_embd": 21}]
    c._tissue_kept = 1
    c._last_world_tick = 0
    c._maybe_start_tissue_gc("a")
    gc = c._tissue_gc_state["a"]
    gc["samples"]["ablate"] = [0.050, 0.051, 0.049, 0.050, 0.051, 0.049, 0.050, 0.050]
    gc["samples"]["restore"] = [0.050, 0.049, 0.051, 0.050, 0.049, 0.051, 0.050, 0.050]
    c._resolve_tissue_gc("a", gc)
    assert "a" not in c._tissue_gc_state
    assert c._tissue_gc_pruned == 1 and c._tissue_kept == 0
    assert c._tissue_grown_specs.get("a") in (None, [])   # спек убран
    assert "grown1" not in c._grown_tissues.get("a", {})
    assert "a" not in c._tissue_gc_ablate


def test_gc_paired_keep_when_durable():
    # ablate СТАБИЛЬНО выше restore на ≥abs_floor → median≥floor, t≥t_keep → KEEP.
    c = _c()
    t = object()
    c._grown_tissues["a"] = {"grown1": t}
    c._tissue_grown_specs["a"] = [{"role": "grown1", "data_dim": 64, "n_embd": 21}]
    c._tissue_kept = 1
    c._last_world_tick = 0
    c._maybe_start_tissue_gc("a")
    gc = c._tissue_gc_state["a"]
    # diff ≈ +0.010 с реалистичным шумом (MAD>0 → t считается, эффект чёткий >t_keep)
    gc["samples"]["ablate"] = [0.060, 0.063, 0.058, 0.061, 0.059, 0.062, 0.060, 0.057]
    gc["samples"]["restore"] = [0.050, 0.051, 0.049, 0.052, 0.048, 0.051, 0.050, 0.049]
    c._resolve_tissue_gc("a", gc)
    assert "a" not in c._tissue_gc_state
    assert c._grown_tissues["a"]["grown1"] is t           # остался (soft-маска снята)
    assert c._tissue_kept == 1 and c._tissue_gc_pruned == 0
    assert c._tissue_gc_keep_rise["a"]["grown1"] >= c._TISSUE_GC_ABS_FLOOR
    assert "a" not in c._tissue_gc_ablate


def test_gc_paired_abs_floor_prunes_tiny_effect():
    # abs-floor (Фрай): эффект ЗНАЧИМ (низкий шум, t высок) НО мал (<abs_floor) →
    # немеряемо-малый = noise → PRUNE (не KEEP мизерный вклад).
    c = _c()
    c._grown_tissues["a"] = {"grown1": object()}
    c._tissue_grown_specs["a"] = [{"role": "grown1", "data_dim": 64, "n_embd": 21}]
    c._tissue_kept = 1
    c._last_world_tick = 0
    c._maybe_start_tissue_gc("a")
    gc = c._tissue_gc_state["a"]
    # стабильная разница ~0.001 (< abs_floor 0.005), крошечный шум → t высок
    gc["samples"]["ablate"] = [0.0510, 0.0511, 0.0509, 0.0510, 0.0511, 0.0509, 0.0510, 0.0510]
    gc["samples"]["restore"] = [0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500]
    c._resolve_tissue_gc("a", gc)
    assert c._tissue_gc_pruned == 1                       # значим, но <abs-floor → noise
    assert "grown1" not in c._grown_tissues.get("a", {})


def test_gc_epoch_rest_lets_growth_resume():
    # sweep завершён (все живые протестированы) → отдых: новый sweep не стартует,
    # idle уходит в propose (рост новых durable между эпохами).
    c = _c()
    c._grown_tissues["a"] = {"grown1": object()}
    c._tissue_gc_tested["a"] = {"grown1"}
    c._last_world_tick = 5000
    assert c._maybe_start_tissue_gc("a") is False
    assert c._tissue_gc_sweep_done.get("a") == 5000
    assert c._tissue_gc_tested["a"] == set()
    assert c._maybe_start_tissue_gc("a") is False         # отдых держит


def test_tissue_growth_metrics_for_ui():
    # UI-контракт: live-счёт + cumulative + in-flight флаги, отдельно от связей.
    c = _c()
    c._tissue_growth_enabled = True
    c._grown_tissues["a"] = {"grown4": object(), "grown10": object()}
    c._tissue_kept = 2
    c._tissue_reverted = 5
    c._tissue_gc_pruned = 3
    c._tissue_propose_count = 12
    m = c._tissue_growth_metrics("a")
    assert m["tissue_grown_live"] == 2          # 2 живых сайдкара
    assert m["tissue_kept"] == 2 and m["tissue_reverted"] == 5
    assert m["tissue_gc_pruned"] == 3 and m["tissue_propose_total"] == 12
    assert m["tissue_growing"] == 0 and m["tissue_gc_evaluating"] == 0
    assert m["tissue_growth_enabled"] is True
    # paired-GC: сайдкар = soft-маска, остаётся в _grown_tissues (уже в live=2),
    # НЕ +1 (иначе двойной счёт); флаг evaluating=1.
    c._tissue_gc_state["a"] = {"role": "grown4", "phase": "ablate", "pairs_done": 0,
                               "win_ticks": 0, "acc": 0.0, "acc_n": 0,
                               "samples": {"ablate": [], "restore": []}}
    m2 = c._tissue_growth_metrics("a")
    assert m2["tissue_gc_evaluating"] == 1
    assert m2["tissue_grown_live"] == 2          # сайдкар в dict (soft-маска), не +1
    # агрегат (cid=None) по всем организмам
    assert c._tissue_growth_metrics()["tissue_grown_live"] == 2


def test_gc_pruned_counter_persists():
    src = _c()
    src.organisms["a"] = types.SimpleNamespace(generation=0)
    src._tissue_gc_pruned = 7
    payload = src.save_state("a")
    assert payload["growth_loop"]["tissue_gc_pruned"] == 7
    dst = _c()
    dst.organisms["a"] = types.SimpleNamespace(generation=0)
    dst.restore_persisted_state("a", payload)
    assert dst._tissue_gc_pruned == 7
