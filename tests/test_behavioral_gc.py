"""§10.3 Stage 3 behavioral-GC (Фрай go 10.06): парный interleaved leave-one-out
по самочувствию. ablate=soft edge-weight→0 (НЕ removal → нет churn/§3); discard
post-toggle transient; §3-abort; порог=paired-t по окнам; specialist-keep +
veto net-harm. Измерения live-variance (cortisol↓/glucose↑/hydration↑/income↑).
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


# ── флаг / дефолты ───────────────────────────────────────────────────────

def test_beh_gc_default_off():
    c = _c()
    assert c._behavioral_gc_enabled is False
    assert c._BEH_GC_PAIRS == 13 and c._BEH_GC_WINDOW == 233


def test_set_beh_gc_toggle():
    c = _c()
    assert c.set_behavioral_gc(True) is True
    assert c.set_behavioral_gc(False) is False


# ── income-rate монотонный аккумулятор ──────────────────────────────────

def test_sample_income_is_cumulative():
    c = _c()
    c.biochem["a"] = types.SimpleNamespace(cortisol=60, glucose=80, hydration=90)
    c._beh_income_cum["a"] = 12.5
    s = c._beh_gc_sample("a")
    assert s["neg_cortisol"] == -60 and s["glucose"] == 80
    assert s["income_cum"] == 12.5


# ── soft edge-weight (НЕ removal, neurocore-gated) ──────────────────────

def _graduated_setup():
    pytest.importorskip("core.tissue")
    c = _c()
    cb = c._make_higher_tissue("cerebellum", data_dim=64)
    if cb is None:
        pytest.skip("cerebellum build failed")
    t = c._make_higher_tissue("grown1", data_dim=64, n_embd=21)
    from core.tissue_topology import (TissueConnectionGene,
                                       apply_topology_overlay_to_org)
    from core.connection import ConnectionType
    gene = TissueConnectionGene(innovation=0, source_role="grown1",
                                target_role="cerebellum",
                                conn_type=ConnectionType.DIRECT,
                                weight=0.382, enabled=True)
    org = types.SimpleNamespace(
        tissues={getattr(cb, "tissue_id", "cb"): cb,
                 getattr(t, "tissue_id"): t},
        connections=[], tissue_topology_genes=[gene], _cached_levels=None)
    apply_topology_overlay_to_org(org)
    c.organisms["a"] = org
    c._tissue_graduated["a"] = {"grown1": t}
    c.biochem["a"] = types.SimpleNamespace(cortisol=60.0, glucose=80.0,
                                           hydration=90.0)
    return c, org, t, gene


def test_soft_ablate_changes_weight_not_topology():
    c, org, t, gene = _graduated_setup()
    tid = getattr(t, "tissue_id")
    n_before = len(org.tissues)
    c._beh_gc_set_edge_weight(org, "grown1", 0.0)
    assert gene.weight == 0.0
    assert tid in org.tissues                       # узел НЕ удалён (нет churn)
    assert len(org.tissues) == n_before
    # вес 0 → ребро ушло из conn (overlay убирает нулевые? нет — weight 0 остаётся)
    c._beh_gc_set_edge_weight(org, "grown1", 0.382)
    assert gene.weight == 0.382                     # restore


def test_maybe_start_sets_ablate_phase():
    c, org, t, gene = _graduated_setup()
    c._last_world_tick = 10**6
    assert c._maybe_start_behavioral_gc("a", org) is True
    st = c._beh_gc_state["a"]
    assert st["role"] == "grown1" and st["phase"] == "ablate"
    assert gene.weight == 0.0                        # стартовал в ablate


def test_start_skips_cooldown_role():
    c, org, t, gene = _graduated_setup()
    c._last_world_tick = 1000
    c._beh_gc_rejected["a"] = {"grown1": 1000}       # только что prune'нут
    assert c._maybe_start_behavioral_gc("a", org) is False


# ── §3-abort + transient discard (parametric, без neurocore) ────────────

def test_step_aborts_on_paralysis(monkeypatch):
    c = _c()
    org = types.SimpleNamespace(tissue_topology_genes=[], connections=[])
    c.organisms["a"] = org
    monkeypatch.setattr(c, "_beh_gc_set_edge_weight", lambda *a, **k: None)
    c._beh_gc_state["a"] = {"role": "grown1", "phase": "ablate", "pairs_done": 0,
                            "win_ticks": 50, "win0_income": 0.0, "acc": {},
                            "acc_n": 0, "samples": {"ablate": {}, "restore": {}},
                            "par_before": 0}
    c._paralysis_window_n = 1
    c._behavioral_gc_step("a", org)
    assert "a" not in c._beh_gc_state                # abort


def test_transient_ticks_not_sampled(monkeypatch):
    c = _c()
    org = types.SimpleNamespace()
    c.organisms["a"] = org
    c.biochem["a"] = types.SimpleNamespace(cortisol=60, glucose=80, hydration=90)
    c._beh_gc_state["a"] = {"role": "grown1", "phase": "ablate", "pairs_done": 0,
                            "win_ticks": 0, "win0_income": 0.0, "acc": {},
                            "acc_n": 0, "samples": {"ablate": {}, "restore": {}},
                            "par_before": 0}
    for _ in range(c._BEH_GC_TRANSIENT):
        c._behavioral_gc_step("a", org)
    assert c._beh_gc_state["a"]["acc_n"] == 0        # transient не сэмплился
    c._behavioral_gc_step("a", org)
    assert c._beh_gc_state["a"]["acc_n"] == 1        # после transient — сэмпл


# ── resolve: specialist-keep / prune / veto net-harm ────────────────────

def _resolve_with(c, ablate, restore):
    """Прогнать resolve с заданными per-dim окнами (ablate/restore списки)."""
    org = types.SimpleNamespace(tissue_topology_genes=[], connections=[])
    c.organisms["a"] = org
    c._tissue_graduated["a"] = {"grown1": object()}
    st = {"role": "grown1", "samples": {"ablate": ablate, "restore": restore}}
    # заглушки графовых правок
    c._beh_gc_set_edge_weight = lambda *a, **k: None
    c._degraduate_node = lambda *a, **k: c._tissue_graduated["a"].pop("grown1", None)
    c._resolve_behavioral_gc("a", org, st)


def test_resolve_keep_when_ablate_worsens_cortisol():
    # neg_cortisol ablate НИЖЕ restore стабильно → ткань снижает стресс → KEEP
    c = _c()
    abl = {"neg_cortisol": [-70, -72, -68, -71, -69]}
    res = {"neg_cortisol": [-60, -59, -61, -60, -60]}   # diffs≈-10±2 → t≈-10
    _resolve_with(c, abl, res)
    assert c._beh_gc_done == 1 and c._beh_gc_pruned == 0


def test_resolve_prune_soft_when_underpowered():
    # тихие измерения, но n=5 малый → MDE может быть adequate если rsd крошечный.
    # Здесь cortisol underpowered (нет данных) → SOFT (не permanent).
    c = _c()
    abl = {"glucose": [80, 81, 79, 80, 80]}           # cortisol отсутствует
    res = {"glucose": [80, 80, 81, 79, 80]}
    _resolve_with(c, abl, res)
    assert c._beh_gc_pruned == 1 and c._beh_gc_done == 0
    assert "grown1" in c._beh_gc_rejected["a"]        # cooldown ВСЕГДА
    assert not c._tissue_graduated.get("a")           # degraduated
    # cortisol n<2 → не powered → SOFT → НЕ permanent
    assert "grown1" not in c._beh_rejected_roles.get("a", set())


def test_prune_permanent_only_when_all_powered():
    # все 4 измерения с малым rsd (adequate power) и без benefit → PERMANENT
    c = _c()
    base = lambda v: [v, v+0.2, v-0.2, v+0.1, v-0.1, v, v+0.15, v-0.15]
    abl = {"neg_cortisol": base(-60), "glucose": base(80),
           "hydration": base(95), "income": base(2.0)}
    res = {"neg_cortisol": base(-60), "glucose": base(80),
           "hydration": base(95), "income": base(2.0)}
    _resolve_with(c, abl, res)
    assert c._beh_gc_pruned == 1
    assert "grown1" in c._beh_rejected_roles.get("a", set())   # PERMANENT


def test_resolve_veto_net_harm():
    # польза по glucose, но БОЛЬШИЙ вред по hydration → net-harm → PRUNE
    c = _c()
    abl = {"glucose": [70, 72, 68, 71, 69],          # ablate ниже = польза (слабая, Δ≈3)
           "hydration": [95, 97, 93, 96, 94]}        # ablate ВЫШЕ = вред (сильный, Δ≈15)
    res = {"glucose": [73, 74, 72, 73, 73],
           "hydration": [80, 81, 79, 80, 80]}
    _resolve_with(c, abl, res)
    assert c._beh_gc_pruned == 1                      # veto net-harm сработал


# ── анти-осцилляция (Фрай): prune → метка → re-graduate заблокирован ────

def test_prune_permanent_marks_behavior_rejected():
    # ВСЕ измерения adequately powered + no-benefit → PERMANENT метка
    c = _c()
    base = lambda v: [v, v+0.2, v-0.2, v+0.1, v-0.1, v, v+0.15, v-0.15]
    abl = {"neg_cortisol": base(-60), "glucose": base(80),
           "hydration": base(95), "income": base(2.0)}
    res = {"neg_cortisol": base(-60), "glucose": base(80),
           "hydration": base(95), "income": base(2.0)}
    _resolve_with(c, abl, res)                       # → prune permanent
    assert "grown1" in c._beh_rejected_roles.get("a", set())


def test_graduate_skips_behavior_rejected():
    c = _c()
    org = types.SimpleNamespace(tissues={}, connections=[],
                                tissue_topology_genes=[])
    c.organisms["a"] = org
    c._grown_tissues["a"] = {"grown1": object()}
    c._tissue_gc_keep_rise["a"] = {"grown1": 0.05}   # prediction-good!
    c._beh_rejected_roles["a"] = {"grown1"}          # но behavior-rejected
    assert c._graduate_tissue("a", org) is False     # повторный выпуск блокирован


def test_behavior_rejected_persists_restart():
    src = _c()
    src.organisms["a"] = types.SimpleNamespace(generation=0)
    src._beh_rejected_roles["a"] = {"grown7", "grown9"}
    payload = src.save_state("a")
    assert payload["growth_loop"]["beh_rejected"] == ["grown7", "grown9"]
    dst = _c()
    dst.organisms["a"] = types.SimpleNamespace(generation=0)
    dst.restore_persisted_state("a", payload)
    assert dst._beh_rejected_roles["a"] == {"grown7", "grown9"}


def test_reset_behavior_rejected_on_world_change():
    c = _c()
    c._beh_rejected_roles["a"] = {"grown7"}
    assert c.reset_behavior_rejected() == 1
    assert not c._beh_rejected_roles


def test_metrics_expose_beh_gc():
    c = _c()
    c._beh_gc_done = 2
    c._beh_gc_pruned = 1
    c._beh_gc_state["a"] = {"role": "grown1"}
    m = c._tissue_growth_metrics("a")
    assert m["beh_gc_active"] == 1
    assert m["beh_gc_kept"] == 2 and m["beh_gc_pruned"] == 1
    assert m["behavioral_gc_enabled"] is False


def test_retest_clears_rejection_and_limit():
    c = _c()
    c._beh_rejected_roles["a"] = {"grown133"}
    c._beh_gc_rejected["a"] = {"grown133": 100}
    c._tissue_grad_done = 1
    assert c.behavioral_gc_retest() == 1
    assert not c._beh_rejected_roles
    assert not c._beh_gc_rejected
    assert c._tissue_grad_done == 0          # лимит сброшен → re-graduate


# ── ПЕТЛЯ №1 (Фрай): SOFT → эскалация cooldown + досветка 34-пар ────────

def test_soft_prune_escalates_graduation_cooldown():
    c = _c()
    abl = {"glucose": [80, 81, 79, 80, 80]}          # cortisol нет → underpowered
    res = {"glucose": [80, 80, 81, 79, 80]}
    _resolve_with(c, abl, res)                       # SOFT
    assert c._grad_revert_count["a"]["grown1"] == 1  # re-graduate разрежен
    assert "grown1" in c._grad_rejected["a"]
    assert c._beh_soft_count["a"]["grown1"] == 1
    assert "grown1" not in c._beh_rejected_roles.get("a", set())   # НЕ permanent


def test_repeat_soft_routes_to_deep_retest():
    pytest.importorskip("core.tissue")
    c, org, t, gene = _graduated_setup()
    c._last_world_tick = 10**6
    c._beh_soft_count["a"] = {"grown1": 2}           # repeat-soft
    assert c._maybe_start_behavioral_gc("a", org) is True
    st = c._beh_gc_state["a"]
    assert st["pairs_target"] == 34                  # досветка
    assert abs(st["t_keep"] - 2.035) < 1e-9
    # обычная роль → 13 пар
    c._beh_gc_state.clear()
    c._beh_soft_count["a"] = {"grown1": 1}
    c._maybe_start_behavioral_gc("a", org)
    assert c._beh_gc_state["a"]["pairs_target"] == 13


def test_powered_keep_clears_counters():
    c = _c()
    c._beh_soft_count["a"] = {"grown1": 2}
    c._grad_revert_count["a"] = {"grown1": 2}
    abl = {"income": [1.2, 1.25, 1.15, 1.3, 1.1, 1.2, 1.22, 1.18]}   # ниже = польза
    res = {"income": [2.0, 1.95, 2.1, 1.9, 2.05, 2.0, 1.98, 2.02]}   # rsd>0, t<<−2
    _resolve_with(c, abl, res)
    assert c._beh_gc_done == 1                       # KEEP
    assert "grown1" not in c._beh_soft_count.get("a", {})
    assert "grown1" not in c._grad_revert_count.get("a", {})


def test_soft_count_persists():
    src = _c()
    src.organisms["a"] = types.SimpleNamespace(generation=0)
    src._beh_soft_count["a"] = {"grown156": 1}
    payload = src.save_state("a")
    assert payload["growth_loop"]["beh_soft_count"] == {"grown156": 1}
    dst = _c()
    dst.organisms["a"] = types.SimpleNamespace(generation=0)
    dst.restore_persisted_state("a", payload)
    assert dst._beh_soft_count["a"] == {"grown156": 1}


# ── KEEP-cooldown (Фрай: подтверждённый узел не ре-ревизуется по кругу) ──

def test_keep_sets_long_cooldown():
    pytest.importorskip("core.tissue")
    c, org, t, gene = _graduated_setup()
    c._last_world_tick = 10**6
    # KEEP-резолв (cortisol польза)
    st = {"role": "grown1", "samples": {
        "ablate": {"neg_cortisol": [-70, -72, -68, -71, -69]},
        "restore": {"neg_cortisol": [-60, -59, -61, -60, -60]}}}
    c._resolve_behavioral_gc("a", org, st)
    assert c._beh_gc_done == 1
    assert "grown1" in c._beh_gc_keep_cd["a"]              # cooldown поставлен
    # сразу — НЕ ре-стартует (в keep-cooldown)
    assert c._maybe_start_behavioral_gc("a", org) is False
    # после длинной паузы — ре-валидация разрешена
    c._last_world_tick += c._BEH_GC_KEEP_COOLDOWN
    assert c._maybe_start_behavioral_gc("a", org) is True


def test_keep_cooldown_persists():
    src = _c()
    src.organisms["a"] = types.SimpleNamespace(generation=0)
    src._beh_gc_keep_cd["a"] = {"grown4": 12345}
    payload = src.save_state("a")
    assert payload["growth_loop"]["beh_gc_keep_cd"] == {"grown4": 12345}
    dst = _c()
    dst.organisms["a"] = types.SimpleNamespace(generation=0)
    dst.restore_persisted_state("a", payload)
    assert dst._beh_gc_keep_cd["a"] == {"grown4": 12345}
