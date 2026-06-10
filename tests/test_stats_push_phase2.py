"""/stats v1.3 Фаза 2 (10.06): client-push новых полей — §3-статус, felt-thirst,
insula-T, трейты, тех-паспорт, pred_region (per-creature); ledger/§3-счётчики/
поведенческие вердикты/лента/железо (owner). Поля едут в diagnostics → сервер
extra="allow" → БД → UI.
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


def _org():
    return types.SimpleNamespace(tissues={}, generation=0, connections=[],
                                 tissue_topology_genes=[], _cached_levels=None)


# ── per-creature extra ──────────────────────────────────────────────────

def test_felt_thirst_inline_from_hydration():
    c = _c()
    org = _org()
    c.organisms["a"] = org
    c.biochem["a"] = types.SimpleNamespace(hydration=20.0, energy=500.0)
    ex = c._stat_per_creature_extra("a", org, {}, 18736)
    # felt = ((38.2-20)/38.2)^0.618 ≈ 0.63
    assert 0.6 < ex["felt_thirst"] < 0.66
    # сыт водой → felt 0
    c.biochem["a"].hydration = 90.0
    assert c._stat_per_creature_extra("a", org, {}, 0)["felt_thirst"] == 0.0


def test_paralysis_status_fields():
    import time
    c = _c()
    org = _org()
    c.organisms["a"] = org
    c.biochem["a"] = types.SimpleNamespace(hydration=90.0, energy=0.0)
    ex = c._stat_per_creature_extra("a", org, {}, 0)
    assert ex["paralyzed"] is False
    c._paralysis_until["a"] = time.monotonic() + 2.0    # в параличе
    ex2 = c._stat_per_creature_extra("a", org, {}, 0)
    assert ex2["paralyzed"] is True and ex2["paralysis_ticks_remaining"] > 0


def test_traits_and_techspec_fields():
    c = _c()
    org = _org()
    c.organisms["a"] = org
    c.biochem["a"] = types.SimpleNamespace(hydration=90.0, energy=500.0)
    c.traits["a"] = {"move_speed": 10, "attack_power": 7, "efficiency": 15}
    c._stat_pred_region["a"] = {"temp35": 0.04, "dens44_55": 0.1, "rest": 0.06}
    ex = c._stat_per_creature_extra("a", org, {}, 18736)
    assert ex["traits"] == {"move_speed": 10, "attack_power": 7, "efficiency": 15}
    assert ex["n_synapses"] == 18736 and ex["device"] == "cpu"
    assert ex["pred_region"]["temp35"] == 0.04


# ── owner extra ─────────────────────────────────────────────────────────

def test_owner_ledger_and_paralysis_counters():
    c = _c()
    c._stat_ledger = {"net_true": 159.0, "residual": 0.0, "srv_cost": 213.0}
    c._stat_paralysis_count = 14
    c._stat_recovery_count = 14
    ow = c._stat_owner_extra()
    assert ow["ledger_net_true"] == 159.0 and ow["ledger_srv_cost"] == 213.0
    assert ow["paralysis_count"] == 14 and ow["recovery_count"] == 14
    assert ow["immortal"] is False        # single_organism дефолт False


def test_owner_beh_verdicts_and_events():
    c = _c()
    c._stat_beh_verdicts = {"grown133": {"verdict": "KEEP",
                                          "dims": {"neg_cortisol": {"t": 3.9}}}}
    c._last_world_tick = 1000
    c._stat_event("graduate-ok", "grown9", "выпущена")
    ow = c._stat_owner_extra()
    assert ow["beh_verdicts"][0]["role"] == "grown133"
    assert ow["beh_verdicts"][0]["verdict"] == "KEEP"
    assert ow["growth_events"][-1]["role"] == "grown9"
    assert ow["growth_events"][-1]["kind"] == "graduate-ok"


def test_owner_hardware_present():
    c = _c()
    hw = c._stat_owner_extra()["hardware"]
    assert "client_version" in hw and "os_name" in hw and "device" in hw


# ── интеграция: счётчики §3 инкрементятся на событиях ───────────────────

def test_paralysis_counter_increments_on_enter():
    c = _c()
    c.organisms["a"] = _org()
    before = c._stat_paralysis_count
    c._enter_paralysis("a", reason="test")
    assert c._stat_paralysis_count == before + 1
    # idempotent: повторный вызов в активном параличе не копит
    c._enter_paralysis("a", reason="test")
    assert c._stat_paralysis_count == before + 1


def test_beh_verdict_recorded_on_resolve():
    c = _c()
    org = types.SimpleNamespace(tissue_topology_genes=[], connections=[])
    c.organisms["a"] = org
    c._tissue_graduated["a"] = {"grown1": object()}
    c._beh_gc_set_edge_weight = lambda *a, **k: None
    c._degraduate_node = lambda *a, **k: c._tissue_graduated["a"].pop("grown1", None)
    st = {"role": "grown1", "samples": {
        "ablate": {"glucose": [80, 81, 79, 80, 80]},
        "restore": {"glucose": [80, 80, 81, 79, 80]}}}
    c._resolve_behavioral_gc("a", org, st)
    assert "grown1" in c._stat_beh_verdicts
    assert c._stat_beh_verdicts["grown1"]["verdict"].startswith("SOFT")


# ── открытый хвост §4.4: foraging / growth_history / lifetime ───────────

def test_growth_history_snapshot_and_export():
    c = _c()
    org = types.SimpleNamespace(tissues={}, generation=0)
    c.organisms["a"] = org
    c._last_world_tick = 5000
    hist = c._stat_snapshot_growth_history()
    assert len(hist) == 1
    assert hist[0]["t"] == 5000 and hist[0]["n_tissues"] == 0
    assert "grown" in hist[0] and "graduated" in hist[0]
    c._last_world_tick = 5100
    hist2 = c._stat_snapshot_growth_history()
    assert len(hist2) == 2 and hist2[-1]["t"] == 5100


def test_lifetime_odometer():
    c = _c()
    org = types.SimpleNamespace(tissues={}, generation=0)
    c.organisms["a"] = org
    c._stat_ate_total["a"] = 1842
    c._stat_recovery_count = 14
    c._birth_tick["a"] = 1000
    c._last_world_tick = 42280
    lf = c._stat_lifetime()
    assert lf["ate_total"] == 1842 and lf["paralysis_survived"] == 14
    assert lf["ticks_lived"] == 41280


def test_foraging_in_owner_extra():
    c = _c()
    c._stat_foraging = {"onf_rate": 0.12, "sees_flora_rate": 0.93, "active_eat_rate": 0.04}
    ow = c._stat_owner_extra()
    assert ow["foraging"]["onf_rate"] == 0.12
    assert "growth_history" in ow and "lifetime" in ow


def test_lifetime_counters_persist():
    src = _c()
    src.organisms["a"] = types.SimpleNamespace(generation=0)
    src._stat_ate_total["a"] = 500
    src._stat_recovery_count = 7
    src._stat_paralysis_count = 7
    payload = src.save_state("a")
    gl = payload["growth_loop"]
    assert gl["ate_total"] == 500 and gl["recovery_count"] == 7
    dst = _c()
    dst.organisms["a"] = types.SimpleNamespace(generation=0)
    dst.restore_persisted_state("a", payload)
    assert dst._stat_ate_total["a"] == 500 and dst._stat_recovery_count == 7
