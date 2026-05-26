"""TZ B Phase 2 (26.05.2026, Бендер): unit-тесты per-role Hebbian metrics tracker.

Проверяем:
  - Empty stub diagnostics(): hebbian_per_tissue содержит все 20 ролей с нулями.
  - SFNN basic trace contribution (Tensor per role per cid).
  - SFNN higher + zodchiy trace (dict synapse→Tensor, сумма norms).
  - Classic Hebbian fallback (_tissue_info.trace для не-SFNN ролей).
  - Zero trace → n_total++ но n_learning не++ (creature имеет роль, но не учится).
  - Reset accumulators в _build_hebbian_per_tissue_snapshot.
  - delta_mean = sum(|delta|) / samples через несколько ticks.
  - SFNN priority над classic (нет двойного счёта той же роли).
  - Алиасы ролей вне 20 known тихо игнорируются.
  - Safety: creature без Hebbian и без SFNN traces — не падает.
  - reset_all() обнуляет accumulators.

Schema согласована с Хьюбертом (Option B, ТЗ §3.2 commit 8238b06):
  hebbian_per_tissue: {role: {n_learning, n_total, delta_mean}}
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

torch = pytest.importorskip("torch")

from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute,
    _BASIC_SFNN_TISSUES,
    _HIGHER_SFNN_TISSUES,
    _ZODCHIY_EXTRA_TISSUES,
    _HEB_PT_ALL_ROLES,
)


_ALL_ROLES = _BASIC_SFNN_TISSUES + _HIGHER_SFNN_TISSUES + _ZODCHIY_EXTRA_TISSUES


# ── Empty stub ──────────────────────────────────────────────────────────────


def test_empty_stub_includes_all_20_roles_with_zeros():
    """n=0 → hebbian_per_tissue содержит все 20 ролей со схемой {n_learning,
    n_total, delta_mean} и нулями. Это backward compat для VPS merge."""
    compute = LocalColonyCompute(device="cpu")
    diag = compute.diagnostics()
    assert "hebbian_per_tissue" in diag
    hpt = diag["hebbian_per_tissue"]
    assert len(_ALL_ROLES) == 20, "ожидаем ровно 20 ролей в _HEB_PT_ALL_ROLES"
    for role in _ALL_ROLES:
        assert role in hpt, f"role {role!r} missing в empty stub"
        block = hpt[role]
        assert block["n_learning"] == 0
        assert block["n_total"] == 0
        assert block["delta_mean"] == 0.0


def test_heb_pt_all_roles_constant_matches_3_lists():
    """_HEB_PT_ALL_ROLES = basic (10) + higher (7) + zodchiy (3) = 20 unique."""
    assert tuple(_HEB_PT_ALL_ROLES) == (
        _BASIC_SFNN_TISSUES + _HIGHER_SFNN_TISSUES + _ZODCHIY_EXTRA_TISSUES
    )
    assert len(set(_HEB_PT_ALL_ROLES)) == 20


# ── Safety: no crash ────────────────────────────────────────────────────────


def test_record_sample_no_crash_when_no_hebbian_and_no_traces():
    """Creature без Hebbian и без SFNN traces — silent skip, не падает."""
    compute = LocalColonyCompute(device="cpu")
    compute.hebbian["c1"] = None
    compute._record_hebbian_per_tissue_sample("c1")
    # ни один accumulator не изменился
    for role in _ALL_ROLES:
        assert compute._heb_pt_n_total[role] == 0
        assert compute._heb_pt_n_learning[role] == 0
        assert compute._heb_pt_delta_sum[role] == 0.0


def test_record_sample_no_crash_on_missing_cid():
    """cid не зарегистрирован в self.hebbian → silent skip."""
    compute = LocalColonyCompute(device="cpu")
    # cid вообще не в hebbian dict — heb.get(cid) вернёт None
    compute._record_hebbian_per_tissue_sample("phantom_cid")
    for role in _ALL_ROLES:
        assert compute._heb_pt_n_total[role] == 0


# ── SFNN basic (10 ролей: Tensor per role per cid) ──────────────────────────


def test_basic_sfnn_trace_records_n_learning():
    """SFNN basic trace с norm > epsilon → n_total=1, n_learning=1."""
    compute = LocalColonyCompute(device="cpu")
    cid, role = "c1", "brain"
    compute.basic_tissue_sfnn_trace[role][cid] = torch.tensor([0.5, 0.0])
    compute._record_hebbian_per_tissue_sample(cid)
    assert compute._heb_pt_n_total[role] == 1
    assert compute._heb_pt_n_learning[role] == 1
    assert abs(compute._heb_pt_delta_sum[role] - 0.5) < 1e-6
    assert compute._heb_pt_samples[role] == 1


def test_zero_basic_sfnn_trace_increments_n_total_not_n_learning():
    """Trace = zeros → n_total++, n_learning не++."""
    compute = LocalColonyCompute(device="cpu")
    cid, role = "c1", "memory"
    compute.basic_tissue_sfnn_trace[role][cid] = torch.zeros(3)
    compute._record_hebbian_per_tissue_sample(cid)
    assert compute._heb_pt_n_total[role] == 1
    assert compute._heb_pt_n_learning[role] == 0
    assert compute._heb_pt_delta_sum[role] == 0.0


def test_multiple_creatures_aggregated_per_role():
    """3 creatures с trace в одной роли → n_total=3."""
    compute = LocalColonyCompute(device="cpu")
    role = "motor"
    for i, cid in enumerate(["c1", "c2", "c3"]):
        compute.basic_tissue_sfnn_trace[role][cid] = torch.tensor([0.1 * (i + 1)])
        compute._record_hebbian_per_tissue_sample(cid)
    assert compute._heb_pt_n_total[role] == 3
    assert compute._heb_pt_n_learning[role] == 3
    # delta_sum = 0.1 + 0.2 + 0.3
    assert abs(compute._heb_pt_delta_sum[role] - 0.6) < 1e-6


# ── SFNN higher + zodchiy (dict synapse → Tensor) ───────────────────────────


def test_higher_sfnn_trace_dict_summed_norms():
    """Higher trace = dict synapse→Tensor → sum norms across synapses."""
    compute = LocalColonyCompute(device="cpu")
    cid, role = "c1", "dopamine"
    compute.higher_tissue_sfnn_trace[role][cid] = {
        "input_proj": torch.tensor([0.3]),
        "output_proj": torch.tensor([0.4]),
    }
    compute._record_hebbian_per_tissue_sample(cid)
    assert compute._heb_pt_n_total[role] == 1
    assert compute._heb_pt_n_learning[role] == 1
    # 0.3 + 0.4 = 0.7
    assert abs(compute._heb_pt_delta_sum[role] - 0.7) < 1e-6


def test_zodchiy_extra_tissue_tracked():
    """3 Zodchiy роли (cerebellum/amygdala/episodic) — учитываются как higher."""
    compute = LocalColonyCompute(device="cpu")
    cid = "z1"
    for role in _ZODCHIY_EXTRA_TISSUES:
        compute.higher_tissue_sfnn_trace[role][cid] = {
            "syn": torch.tensor([1.0])
        }
    compute._record_hebbian_per_tissue_sample(cid)
    for role in _ZODCHIY_EXTRA_TISSUES:
        assert compute._heb_pt_n_total[role] == 1
        assert compute._heb_pt_n_learning[role] == 1


def test_higher_sfnn_empty_dict_skipped():
    """Higher trace = пустой dict → пропускаем (нет данных, нет sample)."""
    compute = LocalColonyCompute(device="cpu")
    cid, role = "c1", "planner"
    compute.higher_tissue_sfnn_trace[role][cid] = {}
    compute._record_hebbian_per_tissue_sample(cid)
    assert compute._heb_pt_n_total[role] == 0


# ── delta_mean averaging ────────────────────────────────────────────────────


def test_delta_mean_averaging_across_samples():
    """delta_mean = sum(norms) / samples через несколько последовательных tick'ов."""
    compute = LocalColonyCompute(device="cpu")
    cid, role = "c1", "motor"
    # 3 ticks с разными trace norms
    norms = [0.2, 0.4, 0.6]
    for n in norms:
        compute.basic_tissue_sfnn_trace[role][cid] = torch.tensor([n])
        compute._record_hebbian_per_tissue_sample(cid)
    snap = compute._build_hebbian_per_tissue_snapshot()
    # mean = (0.2 + 0.4 + 0.6) / 3 = 0.4
    assert abs(snap[role]["delta_mean"] - 0.4) < 1e-5
    assert snap[role]["n_total"] == 3
    assert snap[role]["n_learning"] == 3


# ── Reset semantics ─────────────────────────────────────────────────────────


def test_snapshot_resets_accumulators():
    """После _build_hebbian_per_tissue_snapshot все accumulators обнулены."""
    compute = LocalColonyCompute(device="cpu")
    cid, role = "c1", "brain"
    compute.basic_tissue_sfnn_trace[role][cid] = torch.tensor([0.5])
    compute._record_hebbian_per_tissue_sample(cid)

    snap = compute._build_hebbian_per_tissue_snapshot()
    assert snap[role]["n_total"] == 1
    # Reset: accumulators обнулены
    assert compute._heb_pt_n_total[role] == 0
    assert compute._heb_pt_n_learning[role] == 0
    assert compute._heb_pt_delta_sum[role] == 0.0
    assert compute._heb_pt_samples[role] == 0


def test_reset_all_clears_accumulators():
    """reset_all() обнуляет tracker (вместе с остальным state)."""
    compute = LocalColonyCompute(device="cpu")
    cid, role = "c1", "sensory"
    compute.basic_tissue_sfnn_trace[role][cid] = torch.tensor([0.3])
    compute._record_hebbian_per_tissue_sample(cid)
    assert compute._heb_pt_n_total[role] == 1

    compute.reset_all()
    for r in _ALL_ROLES:
        assert compute._heb_pt_n_total[r] == 0
        assert compute._heb_pt_n_learning[r] == 0
        assert compute._heb_pt_delta_sum[r] == 0.0
        assert compute._heb_pt_samples[r] == 0


# ── Classic Hebbian fallback ────────────────────────────────────────────────


class _StubHeb:
    """Минимальный mock HebbianController с одним полем _tissue_info."""

    def __init__(self, tissue_info):
        self._tissue_info = tissue_info


def test_classic_hebbian_fallback_for_role_without_sfnn():
    """Если нет SFNN trace, но есть classic _tissue_info.trace — fallback работает."""
    compute = LocalColonyCompute(device="cpu")
    cid = "c1"
    compute.hebbian[cid] = _StubHeb([
        {"role": "sensory", "trace": torch.tensor([0.3])},
    ])
    compute._record_hebbian_per_tissue_sample(cid)
    assert compute._heb_pt_n_total["sensory"] == 1
    assert compute._heb_pt_n_learning["sensory"] == 1
    assert abs(compute._heb_pt_delta_sum["sensory"] - 0.3) < 1e-6


def test_classic_alias_role_outside_20_ignored():
    """Роль 'hippocampus' (алиас memory в ROLE_ALGORITHM) не в 20 known → ignored."""
    compute = LocalColonyCompute(device="cpu")
    cid = "c1"
    compute.hebbian[cid] = _StubHeb([
        {"role": "hippocampus", "trace": torch.tensor([0.5])},
    ])
    compute._record_hebbian_per_tissue_sample(cid)
    # Никакая из 20 known ролей не получила sample
    for role in _ALL_ROLES:
        assert compute._heb_pt_n_total[role] == 0


def test_classic_trace_none_treated_as_zero():
    """trace=None в _tissue_info → n_total++, n_learning не++, delta=0."""
    compute = LocalColonyCompute(device="cpu")
    cid = "c1"
    compute.hebbian[cid] = _StubHeb([
        {"role": "manipulator", "trace": None},
    ])
    compute._record_hebbian_per_tissue_sample(cid)
    assert compute._heb_pt_n_total["manipulator"] == 1
    assert compute._heb_pt_n_learning["manipulator"] == 0


# ── SFNN priority ───────────────────────────────────────────────────────────


def test_sfnn_takes_priority_over_classic_for_same_role():
    """Если есть и SFNN trace и classic для одной роли — берём SFNN, не дублируем."""
    compute = LocalColonyCompute(device="cpu")
    cid, role = "c1", "brain"

    compute.basic_tissue_sfnn_trace[role][cid] = torch.tensor([0.5])
    compute.hebbian[cid] = _StubHeb([
        {"role": role, "trace": torch.tensor([0.99])},
    ])

    compute._record_hebbian_per_tissue_sample(cid)
    # ровно один sample, и значение из SFNN (0.5), не classic (0.99)
    assert compute._heb_pt_n_total[role] == 1
    assert abs(compute._heb_pt_delta_sum[role] - 0.5) < 1e-6


# ── Snapshot schema ─────────────────────────────────────────────────────────


def test_snapshot_schema_has_expected_keys_per_role():
    """Snapshot всегда возвращает {role: {n_learning, n_total, delta_mean}}."""
    compute = LocalColonyCompute(device="cpu")
    snap = compute._build_hebbian_per_tissue_snapshot()
    for role in _ALL_ROLES:
        assert role in snap
        block = snap[role]
        assert set(block.keys()) == {"n_learning", "n_total", "delta_mean"}
        # типы по schema
        assert isinstance(block["n_learning"], int)
        assert isinstance(block["n_total"], int)
        assert isinstance(block["delta_mean"], float)


def test_snapshot_zero_samples_delta_mean_zero():
    """samples=0 для роли → delta_mean=0.0 (защита от div-by-zero)."""
    compute = LocalColonyCompute(device="cpu")
    snap = compute._build_hebbian_per_tissue_snapshot()
    for role in _ALL_ROLES:
        assert snap[role]["delta_mean"] == 0.0
