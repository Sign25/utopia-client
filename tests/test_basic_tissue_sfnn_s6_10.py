"""SFNN S6.10 — математическая эквивалентность SFNN ↔ Phase 5d Oja-Hebbian.

При `SFNNRule.default()` (A=1, B=C=D=0, η=1e-3, R3-веса=0, td=0, τ=21)
формула `_basic_sfnn_update_step` редуцируется к классическому Oja с
trace decay:
    e_t = exp(-1/τ)·e_{t-1} + (outer(post_c, pre_c) − post_c²·W)
    ΔW  = clip(η · e_t, ±0.01)

Проверки в этом файле:
  - на первом тике (e_0=0) ΔW ≈ η · (outer(post_c, pre_c) − post_c²·W) с
    точностью до clip
  - td_coupling=0 → η_eff не зависит от dopa_td_mult
  - r3-веса=0 → r_eff=0 → ΔW не зависит от r_imm/med/long_eff
  - A=B=C=D=0 → ΔW = 0
  - mutate(0.1) сохраняет dataclass-структуру и не выкидывает значения
    за clip-пределы
  - clone (from_dict(to_dict)) → независимая копия, мутации не делятся
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("storage.norg")
pytest.importorskip("core.workbench")
torch = pytest.importorskip("torch")


@pytest.fixture
def seed_file(tmp_path, monkeypatch):
    seed_path = tmp_path / "seed.norg"
    monkeypatch.setenv("WORLD_SEED_PATH", str(seed_path))
    import importlib
    from environment import seed_loader as ns_loader
    importlib.reload(ns_loader)
    ns_loader.ensure_seed(preset="nexus")
    monkeypatch.setenv("UTOPIA_SEED_PATH", str(tmp_path / "client_seed.norg"))
    monkeypatch.setenv("UTOPIA_WANDERER_SEED_PATH",
                        str(tmp_path / "client_seed.norg"))
    monkeypatch.setenv("UTOPIA_COLONIES_DIR", str(tmp_path / "colonies"))
    client_seed = tmp_path / "client_seed.norg"
    client_seed.write_bytes(seed_path.read_bytes())
    from utopia_client import seed_loader as cli_loader
    importlib.reload(cli_loader)
    return client_seed


def _setup(seed_file, override_rule=None):
    """Создаёт colony c одной особью, прогоняет один tick чтобы заполнились
    heb._last_input/_last_output, перезаписывает basic_tissue_sfnn_rule на
    `override_rule` (если указано) и возвращает (compute, cid, heb).
    """
    import numpy as np
    from utopia_client.local_compute import (
        LocalColonyCompute, _BASIC_SFNN_TISSUES)
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)
    rng = np.random.default_rng(0)
    # Один tick прокачивает hebbian._last_input / _last_output.
    compute.handle_tick(
        {"c1": rng.normal(size=80).astype(np.float32)},
        events_per_cid={"c1": {"ate": 0, "killed": 0,
                                "damage_taken": 0, "delta_energy": 0.0}},
    )
    heb = compute.hebbian["c1"]
    if override_rule is not None:
        for role in _BASIC_SFNN_TISSUES:
            store = compute.basic_tissue_sfnn_rule.get(role)
            if store is not None and "c1" in store:
                store["c1"] = override_rule
    # Очистим trace — следующий вызов _basic_sfnn_update_step стартует с e_0=0.
    for role in _BASIC_SFNN_TISSUES:
        compute.basic_tissue_sfnn_trace[role].pop("c1", None)
    return compute, "c1", heb


def _expected_oja_dW(W_sub, pre, post, eta: float, clip: float):
    """Эталонная Phase 5d формула: ΔW = clip(η · (outer(post_c, pre_c)
    − post_c²·W), ±clip), e_0=0 (первый тик после очистки трасса)."""
    post_c = post - post.mean()
    pre_c = pre - pre.mean()
    hebb_A = (torch.outer(post_c, pre_c)
                - post_c.square().unsqueeze(1) * W_sub)
    dW = eta * hebb_A
    return dW.clamp(-clip, clip)


def test_default_rule_first_tick_matches_oja_formula(seed_file):
    """Дефолтное `SFNNRule.default()` при первом apply-step → ΔW идентичен
    Phase 5d Oja-Hebbian с точностью до clip."""
    from core.sfnn_rule import SFNNRule
    compute, cid, heb = _setup(seed_file, override_rule=SFNNRule.default())
    # Снимок W для всех oja_input/reward_output тканей.
    snapshots = {}
    for info in heb._tissue_info:
        role = info['role']
        from utopia_client.local_compute import _BASIC_SFNN_TISSUES
        if role not in _BASIC_SFNN_TISSUES:
            continue
        cell = info['cell']
        algo = info['algorithm']
        if algo == 'oja_input':
            W = cell.input_proj.weight.data
        else:
            W = cell.output_proj.weight.data
        snapshots[role] = (W.clone(), info)

    compute._basic_sfnn_update_step(cid, heb, dopa_td_mult=1.0,
                                      r_imm_eff=0.0, r_med_eff=0.0,
                                      r_long_eff=0.0)
    # Сравним ΔW против эталонной Oja-формулы для каждой ткани.
    for role, (W_before, info) in snapshots.items():
        cell = info['cell']
        algo = info['algorithm']
        if algo == 'oja_input':
            W_full = cell.input_proj.weight.data
            x = heb._last_input
            data_cols = min(x.shape[1], W_before.shape[1])
            W_sub_before = W_before[:, :data_cols]
            W_sub_after = W_full[:, :data_cols]
            pre = x[0, :data_cols]
            post = W_sub_before @ pre
        else:
            W_full = cell.output_proj.weight.data
            x = heb._last_output
            n_embd = W_before.shape[1]
            n_actions = min(16, W_before.shape[0])
            W_sub_before = W_before[:n_actions, :]
            W_sub_after = W_full[:n_actions, :]
            pre = x[0, :n_embd]
            post = W_sub_before @ pre
        actual_dW = W_sub_after - W_sub_before
        expected_dW = _expected_oja_dW(
            W_sub_before, pre, post,
            eta=1e-3, clip=compute._MOTOR_SFNN_DW_CLIP)
        # Точность ослаблена — float32 cumulative ops.
        assert torch.allclose(actual_dW, expected_dW, atol=1e-6), (
            f"role={role}: ΔW не совпадает с Oja-формулой")


def test_td_coupling_zero_ignores_dopa_mult(seed_file):
    """td_coupling=0 → разные dopa_td_mult дают одинаковый ΔW."""
    from core.sfnn_rule import SFNNRule
    # Прогон 1: dopa_td_mult=1.0.
    compute1, cid1, heb1 = _setup(seed_file, override_rule=SFNNRule.default())
    info1 = next(i for i in heb1._tissue_info
                 if i['algorithm'] == 'oja_input'
                 and i['role'] in {"sensory", "digestive"})
    W1_before = info1['cell'].input_proj.weight.data.clone()
    compute1._basic_sfnn_update_step(cid1, heb1, dopa_td_mult=1.0)
    W1_after = info1['cell'].input_proj.weight.data.clone()
    dW1 = W1_after - W1_before

    # Прогон 2: dopa_td_mult=5.0 на свежей colony.
    compute2, cid2, heb2 = _setup(seed_file, override_rule=SFNNRule.default())
    info2 = next(i for i in heb2._tissue_info
                 if i['algorithm'] == 'oja_input'
                 and i['role'] == info1['role'])
    W2_before = info2['cell'].input_proj.weight.data.clone()
    compute2._basic_sfnn_update_step(cid2, heb2, dopa_td_mult=5.0)
    W2_after = info2['cell'].input_proj.weight.data.clone()
    dW2 = W2_after - W2_before
    # td_coupling=0 → η_eff одинаков → ΔW идентичны.
    assert torch.allclose(dW1, dW2, atol=1e-8)


def test_r3_weights_zero_ignore_rewards(seed_file):
    """При r_imm/med/long_weight=0 разные r_*_eff не влияют на ΔW."""
    from core.sfnn_rule import SFNNRule
    compute1, cid1, heb1 = _setup(seed_file, override_rule=SFNNRule.default())
    info1 = next(i for i in heb1._tissue_info
                 if i['role'] in {"sensory", "digestive"}
                 and i['algorithm'] == 'oja_input')
    W1_before = info1['cell'].input_proj.weight.data.clone()
    compute1._basic_sfnn_update_step(cid1, heb1,
                                       r_imm_eff=0.0, r_med_eff=0.0,
                                       r_long_eff=0.0)
    dW1 = info1['cell'].input_proj.weight.data - W1_before

    compute2, cid2, heb2 = _setup(seed_file, override_rule=SFNNRule.default())
    info2 = next(i for i in heb2._tissue_info
                 if i['role'] == info1['role']
                 and i['algorithm'] == 'oja_input')
    W2_before = info2['cell'].input_proj.weight.data.clone()
    compute2._basic_sfnn_update_step(cid2, heb2,
                                       r_imm_eff=10.0, r_med_eff=-5.0,
                                       r_long_eff=3.0)
    dW2 = info2['cell'].input_proj.weight.data - W2_before
    assert torch.allclose(dW1, dW2, atol=1e-8)


def test_zero_abcd_rule_yields_zero_dW(seed_file):
    """A=B=C=D=0 → ΔW = 0 при любом η, r, td."""
    from core.sfnn_rule import (SFNNRule, SFNNSynapseCoeffs, SYNAPSE_TYPES)
    rule = SFNNRule(
        coeffs={k: SFNNSynapseCoeffs(eta=1e-3, A=0.0, B=0.0, C=0.0, D=0.0)
                for k in SYNAPSE_TYPES},
    )
    compute, cid, heb = _setup(seed_file, override_rule=rule)
    snapshots = []
    from utopia_client.local_compute import _BASIC_SFNN_TISSUES
    for info in heb._tissue_info:
        if info['role'] not in _BASIC_SFNN_TISSUES:
            continue
        cell = info['cell']
        if info['algorithm'] == 'oja_input':
            snapshots.append((cell.input_proj.weight.data.clone(),
                              cell.input_proj.weight.data))
        else:
            snapshots.append((cell.output_proj.weight.data.clone(),
                              cell.output_proj.weight.data))
    compute._basic_sfnn_update_step(cid, heb, dopa_td_mult=3.0,
                                      r_imm_eff=2.0, r_med_eff=1.0,
                                      r_long_eff=0.5)
    for before, current in snapshots:
        assert torch.allclose(before, current, atol=0.0), (
            "ΔW≠0 при A=B=C=D=0")


def test_mutate_preserves_dataclass_and_clips(seed_file):
    """mutate(σ=0.5, seed=0) даёт SFNNRule с теми же типами полей и без
    выхода за clip-границы."""
    from core.sfnn_rule import (SFNNRule, SYNAPSE_TYPES,
                                  ETA_MIN, ETA_MAX, ABCD_MIN, ABCD_MAX,
                                  TEMP_MIN, TEMP_MAX, TAU_MIN, TAU_MAX,
                                  R3_MIN, R3_MAX, TD_MIN, TD_MAX)
    rule = SFNNRule.for_role("sensory")
    rng = random.Random(0)
    for _ in range(20):
        rule = rule.mutate(sigma=0.5, rng=rng)
        assert isinstance(rule, SFNNRule)
        assert set(rule.coeffs.keys()) == set(SYNAPSE_TYPES)
        for c in rule.coeffs.values():
            assert ETA_MIN <= c.eta <= ETA_MAX
            assert ABCD_MIN <= c.A <= ABCD_MAX
            assert ABCD_MIN <= c.B <= ABCD_MAX
            assert ABCD_MIN <= c.C <= ABCD_MAX
            assert ABCD_MIN <= c.D <= ABCD_MAX
        assert TEMP_MIN <= rule.temperature <= TEMP_MAX
        assert TAU_MIN <= rule.tau <= TAU_MAX
        assert R3_MIN <= rule.r_imm_weight <= R3_MAX
        assert R3_MIN <= rule.r_med_weight <= R3_MAX
        assert R3_MIN <= rule.r_long_weight <= R3_MAX
        assert TD_MIN <= rule.td_coupling <= TD_MAX
        assert rule.algorithm == "oja_input"  # структурный признак не мутирует


def test_to_from_dict_roundtrip_independent(seed_file):
    """to_dict + from_dict → независимая копия (mutate origin не трогает копию)."""
    from core.sfnn_rule import SFNNRule
    origin = SFNNRule.for_role("memory")
    clone = SFNNRule.from_dict(origin.to_dict())
    # Поля совпадают.
    assert clone.to_dict() == origin.to_dict()
    # Мутируем origin — clone не меняется.
    mutated = origin.mutate(sigma=0.5, rng=random.Random(42))
    assert clone.to_dict() != mutated.to_dict()
    # clone сам по себе остался прежним.
    assert clone.to_dict() == SFNNRule.from_dict(origin.to_dict()).to_dict()


def test_role_defaults_match_table(seed_file):
    """Для каждой из 10 базовых: ROLE_DEFAULTS → SFNNRule.for_role() даёт
    те же tau / R3 / td / algorithm, что в таблице."""
    from core.sfnn_rule import (SFNNRule, ROLE_DEFAULTS, ROLE_ALGORITHM)
    from utopia_client.local_compute import _BASIC_SFNN_TISSUES
    for role in _BASIC_SFNN_TISSUES:
        rule = SFNNRule.for_role(role)
        tau, r_imm, r_med, r_long, td, eta = ROLE_DEFAULTS[role]
        assert rule.tau == pytest.approx(tau)
        assert rule.r_imm_weight == pytest.approx(r_imm)
        assert rule.r_med_weight == pytest.approx(r_med)
        assert rule.r_long_weight == pytest.approx(r_long)
        assert rule.td_coupling == pytest.approx(td)
        assert rule.algorithm == ROLE_ALGORITHM[role]
        for c in rule.coeffs.values():
            assert c.eta == pytest.approx(eta)
            assert c.A == pytest.approx(1.0)  # дефолт A=1
            assert c.B == 0.0
            assert c.C == 0.0
            assert c.D == 0.0
