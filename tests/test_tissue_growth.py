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


# ── полный mint/insert/remove (neurocore-gated) ─────────────────────────

def test_propose_and_remove_tissue_node():
    pytest.importorskip("core.tissue")
    pytest.importorskip("core.tissue_topology")
    c = _c()
    cb = c._make_higher_tissue("cerebellum", data_dim=64)
    if cb is None:
        pytest.skip("cerebellum tissue build failed (neurocore?)")
    org = types.SimpleNamespace(
        tissues={"cb": cb}, connections=[], tissue_topology_genes=[])
    c.organisms["a"] = org
    c.loss_ema["a"] = 0.05

    assert c._propose_growth_tissue("a", org) is True
    st = c._tissue_growth_state["a"]
    role, tid = st["role"], st["tid"]
    # узел вставлен в граф
    assert tid in org.tissues
    assert any(getattr(t, "role", None) == role for t in org.tissues.values())
    # ребро {роль}→cerebellum проведено (overlay)
    assert any(g.source_role == role and g.target_role == "cerebellum"
               for g in org.tissue_topology_genes)

    # backoff: узел + ребро удаляются
    c._remove_grown_tissue("a", org, tid, role)
    assert tid not in org.tissues
    assert not any(g.source_role == role for g in org.tissue_topology_genes)
