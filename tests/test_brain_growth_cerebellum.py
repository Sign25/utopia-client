"""Вариант A (08.06.2026, Фрай/Хьюберт): рост мозга при жизни — predictor
становится узлом графа ЧЕРЕЗ cerebellum. forward-hook ловит выход cerebellum-
ткани → он становится входом обученного predictor'а; связи {ткань}→cerebellum
двигают вход прогноза → Δloss_ema реагирует напрямую (драйвер петли роста).

Тесты проверяют ВСТРАИВАНИЕ (шаг 1): резолв ткани по роли, регистрацию хука,
захват выхода (detached), no-op при отсутствии cerebellum и маршрутизацию
cerebellum-выхода в prev_obs predictor'а. Петля propose/keep — отдельно (шаг 2).
"""
from __future__ import annotations

import sys
import types
from collections import deque
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

import torch  # noqa: E402

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402


def _cerebellum_tissue(role: str = "cerebellum"):
    from core.connection import CellGene
    from core.tissue import Tissue, TissuePort, TissueSpec
    cg = CellGene(innovation=1, n_embd=21, n_head=3, n_layer=1)
    spec = TissueSpec(
        name=role, role=role, cell_genes=[cg], connection_genes=[],
        input_ports=[TissuePort("input", 1)],
        output_ports=[TissuePort("output", 1)],
    )
    return Tissue(spec)


def test_resolve_cerebellum_by_role():
    c = LocalColonyCompute(device="cpu")
    org = types.SimpleNamespace(tissues={"t_cer": _cerebellum_tissue()})
    assert c._cerebellum_tissue_id("c1", org) == "t_cer"
    assert c._cerebellum_tissue_id("c1", org) == "t_cer"  # cache stable


def test_resolve_cerebellum_absent_none():
    c = LocalColonyCompute(device="cpu")
    org = types.SimpleNamespace(tissues={"t_x": _cerebellum_tissue(role="motor")})
    assert c._cerebellum_tissue_id("c1", org) is None


def test_hook_captures_cerebellum_output_detached():
    c = LocalColonyCompute(device="cpu")
    cer = _cerebellum_tissue()
    org = types.SimpleNamespace(tissues={"t_cer": cer})
    c._ensure_cerebellum_hook("c1", org)
    assert "c1" in c._cerebellum_hooked
    out = cer({"input": torch.zeros(1, 64)})  # forward → hook наполняет
    assert isinstance(out, dict) and out
    assert "c1" in c._cerebellum_out
    assert int(c._cerebellum_out["c1"].shape[-1]) == 64
    assert not c._cerebellum_out["c1"].requires_grad  # detached


def test_hook_noop_when_absent():
    c = LocalColonyCompute(device="cpu")
    org = types.SimpleNamespace(tissues={})
    c._ensure_cerebellum_hook("c1", org)  # не падает
    assert "c1" in c._cerebellum_hooked
    assert "c1" not in c._cerebellum_out


def test_hook_registered_once():
    c = LocalColonyCompute(device="cpu")
    cer = _cerebellum_tissue()
    org = types.SimpleNamespace(tissues={"t_cer": cer})
    c._ensure_cerebellum_hook("c1", org)
    c._ensure_cerebellum_hook("c1", org)  # повторно — idempotent
    cer({"input": torch.zeros(1, 64)})
    # один hook → один захват без дублей/ошибок
    assert int(c._cerebellum_out["c1"].shape[-1]) == 64


def test_cerebellum_output_becomes_predictor_prev():
    """Маршрутизация: cerebellum-выход, поданный input_tensor'ом, сохраняется
    как prev_obs = вход следующего прогноза (target остаётся obs_{t+1})."""
    c = LocalColonyCompute(device="cpu")
    pred = c._make_predictor_tissue()
    assert pred is not None
    c.predictor["c1"] = pred
    c.predictor_opt["c1"] = c._torch.optim.Adam(pred.parameters(), lr=1e-3)
    c.loss_ema["c1"] = 0.0
    c.pred_loss_history["c1"] = deque(maxlen=100)
    obs = torch.zeros(1, 64)
    cer = torch.ones(1, 64) * 0.5
    c._predictor_train_step("c1", obs, cer)
    assert torch.allclose(c.prev_obs["c1"], cer)


def test_kill_switch_attr_default_on():
    c = LocalColonyCompute(device="cpu")
    assert c._predictor_from_cerebellum is True


# ── Шаг 2 — петля роста связей (state machine) ────────────────────────────

def _fake_org_with_cerebellum():
    org = types.SimpleNamespace(
        tissues={
            "t_cer": _cerebellum_tissue("cerebellum"),
            "t_brain": _cerebellum_tissue("brain"),
            "t_mem": _cerebellum_tissue("memory"),
        },
        connections=[],
        tissue_topology_genes=[],
        _cached_levels=None,
    )
    return org


def test_growth_default_enabled_after_flip():
    # Флип 0.13.36 (Фрай 08.06): рост ВКЛ по умолчанию. Доп.гейт single_organism
    # в handle_tick → активен только для Адама. Kill-switch _growth_enabled=False.
    c = LocalColonyCompute(device="cpu")
    assert c._growth_enabled is True


def test_growth_no_propose_before_plateau():
    c = LocalColonyCompute(device="cpu")
    c._growth_plateau_ticks = 3
    cid = "c1"
    c.predictor[cid] = c._make_predictor_tissue()
    c.organisms[cid] = _fake_org_with_cerebellum()
    c.loss_ema[cid] = 0.2
    c.intrinsic_ema[cid] = 0.0  # плато
    c._brain_growth_step(cid)
    c._brain_growth_step(cid)  # 2 < 3 тиков плато
    assert cid not in c._growth_state
    assert c.organisms[cid].tissue_topology_genes == []


def test_growth_propose_after_plateau():
    c = LocalColonyCompute(device="cpu")
    c._growth_plateau_ticks = 3
    cid = "c1"
    c.predictor[cid] = c._make_predictor_tissue()
    org = _fake_org_with_cerebellum()
    c.organisms[cid] = org
    c.loss_ema[cid] = 0.2
    c.intrinsic_ema[cid] = 0.0
    for _ in range(3):
        c._brain_growth_step(cid)
    assert cid in c._growth_state                      # перешёл в dwell
    assert len(org.tissue_topology_genes) == 1
    g = org.tissue_topology_genes[0]
    assert g.target_role == "cerebellum" and g.enabled
    assert g.source_role in ("brain", "memory")
    assert c._growth_state[cid]["loss_before"] == 0.2


def test_growth_resets_plateau_when_learning():
    c = LocalColonyCompute(device="cpu")
    c._growth_plateau_ticks = 3
    cid = "c1"
    c.predictor[cid] = c._make_predictor_tissue()
    c.organisms[cid] = _fake_org_with_cerebellum()
    c.loss_ema[cid] = 0.2
    c.intrinsic_ema[cid] = 0.0
    c._brain_growth_step(cid)            # плато n=1
    c.intrinsic_ema[cid] = 0.5           # intrinsic поднялся над floor → прогресс → сброс
    c._brain_growth_step(cid)
    assert c._growth_stagnation_n[cid] == 0
    assert cid not in c._growth_state


def test_growth_no_new_propose_during_dwell():
    c = LocalColonyCompute(device="cpu")
    c._growth_plateau_ticks = 1
    c._growth_dwell_ticks = 100
    cid = "c1"
    c.predictor[cid] = c._make_predictor_tissue()
    org = _fake_org_with_cerebellum()
    c.organisms[cid] = org
    c.loss_ema[cid] = 0.2
    c.intrinsic_ema[cid] = 0.0
    c._brain_growth_step(cid)   # propose
    c._brain_growth_step(cid)   # dwell tick, не новый propose
    assert len(org.tissue_topology_genes) == 1


def test_growth_keep_on_significant_improvement():
    from utopia_client.biochemistry import make_default
    c = LocalColonyCompute(device="cpu")
    c._growth_dwell_ticks = 1
    cid = "c1"
    c.predictor[cid] = c._make_predictor_tissue()
    org = _fake_org_with_cerebellum()
    c.organisms[cid] = org
    from core.connection import ConnectionType
    from core.tissue_topology import TissueConnectionGene
    g = TissueConnectionGene(innovation=1, source_role="memory",
                             target_role="cerebellum",
                             conn_type=ConnectionType.DIRECT, weight=1.0, enabled=True)
    org.tissue_topology_genes.append(g)
    bc = make_default(); bc.energy = 100.0
    c.biochem[cid] = bc
    c._growth_state[cid] = {"gene": g, "loss_before": 0.20, "ticks": 0,
                            "par_before": 0, "energy_before": 100.0}
    c.loss_ema[cid] = 0.15                 # улучшение 25% > φ⁻⁵≈9%
    c._brain_growth_step(cid)
    assert g.enabled is True               # связь оставлена
    assert c._growth_kept == 1
    assert cid not in c._growth_state


def test_growth_backoff_on_no_improvement():
    from utopia_client.biochemistry import make_default
    c = LocalColonyCompute(device="cpu")
    c._growth_dwell_ticks = 1
    cid = "c1"
    c.predictor[cid] = c._make_predictor_tissue()
    org = _fake_org_with_cerebellum()
    c.organisms[cid] = org
    from core.connection import ConnectionType
    from core.tissue_topology import TissueConnectionGene
    g = TissueConnectionGene(innovation=1, source_role="memory",
                             target_role="cerebellum",
                             conn_type=ConnectionType.DIRECT, weight=1.0, enabled=True)
    org.tissue_topology_genes.append(g)
    from core.tissue_topology import apply_topology_overlay_to_org
    apply_topology_overlay_to_org(org)    # связь активна в графе
    n_conn_before = len(org.connections)
    bc = make_default(); bc.energy = 100.0
    c.biochem[cid] = bc
    c._growth_state[cid] = {"gene": g, "loss_before": 0.20, "ticks": 0,
                            "par_before": 0, "energy_before": 100.0}
    c.loss_ema[cid] = 0.199                # улучшение 0.5% < порога → backoff
    c._brain_growth_step(cid)
    assert g.enabled is False              # ребро откатано
    assert c._growth_reverted == 1
    assert len(org.connections) < n_conn_before  # overlay убрал disabled-ребро
    assert cid not in c._growth_state


def test_growth_backoff_on_net_collapse():
    from utopia_client.biochemistry import make_default
    c = LocalColonyCompute(device="cpu")
    c._growth_dwell_ticks = 1
    cid = "c1"
    c.predictor[cid] = c._make_predictor_tissue()
    org = _fake_org_with_cerebellum()
    c.organisms[cid] = org
    from core.connection import ConnectionType
    from core.tissue_topology import TissueConnectionGene
    g = TissueConnectionGene(innovation=1, source_role="brain",
                             target_role="cerebellum",
                             conn_type=ConnectionType.DIRECT, weight=1.0, enabled=True)
    org.tissue_topology_genes.append(g)
    bc = make_default(); bc.energy = 30.0  # обвал (< 100*0.618)
    c.biochem[cid] = bc
    c._growth_state[cid] = {"gene": g, "loss_before": 0.20, "ticks": 0,
                            "par_before": 0, "energy_before": 100.0}
    c.loss_ema[cid] = 0.10                 # прогноз улучшился, НО net обвалился
    c._brain_growth_step(cid)
    assert g.enabled is False              # backoff несмотря на Δloss (net guard)
    assert c._growth_reverted == 1


def test_growth_relative_trigger_at_nonzero_floor():
    """Суть относительного триггера (Фрай 08.06): intrinsic floor у embodied-
    Адама НЕ ноль (~0.005-0.009). Абсолютный порог 1e-3 НЕ сработал бы; относит.
    ловит стагнацию у СВОЕГО floor → propose рождается."""
    c = LocalColonyCompute(device="cpu")
    c._growth_plateau_ticks = 3
    cid = "c1"
    c.predictor[cid] = c._make_predictor_tissue()
    org = _fake_org_with_cerebellum()
    c.organisms[cid] = org
    c.loss_ema[cid] = 0.2
    for v in (0.005, 0.006, 0.0055):       # дрейф у floor ~0.005 >> 1e-3
        c.intrinsic_ema[cid] = v
        c._brain_growth_step(cid)
    assert cid in c._growth_state          # propose сработал у nonzero-floor
    assert len(org.tissue_topology_genes) == 1


def test_growth_spike_above_floor_resets():
    """intrinsic поднялся значимо над трейлинг-floor = прогресс вернулся → сброс
    стагнации (не плато). Drift-robust self-referencing."""
    c = LocalColonyCompute(device="cpu")
    c._growth_plateau_ticks = 5
    cid = "c1"
    c.predictor[cid] = c._make_predictor_tissue()
    org = _fake_org_with_cerebellum()
    c.organisms[cid] = org
    c.loss_ema[cid] = 0.2
    for _ in range(2):                      # у floor → стагнация копится
        c.intrinsic_ema[cid] = 0.005
        c._brain_growth_step(cid)
    assert c._growth_stagnation_n[cid] == 2
    c.intrinsic_ema[cid] = 0.05             # спайк прогресса >> floor → сброс
    c._brain_growth_step(cid)
    assert c._growth_stagnation_n[cid] == 0
    assert cid not in c._growth_state
