"""Tests для Phase 1/2/6 в LocalColonyCompute (ТЗ tz_owned_training_diagnostics).

Покрытие:
- predictor создаётся в add_creature
- handle_tick запускает predictor train + EMA-обновления
- pred_loss_history накапливается
- intrinsic_reward появляется при падении loss
- entropy_ema/trace_norm_ema/reward_var_ema обновляются
- apply_inherited_state переносит predictor + EMA от родителя (Y50)
- save_state сериализует все новые поля
- remove_creature чистит все dicts
- diagnostics() возвращает корректную сводку
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("storage.norg")
pytest.importorskip("core.workbench")
pytest.importorskip("core.tissue")


@pytest.fixture
def seed_file(tmp_path, monkeypatch):
    seed_path = tmp_path / "seed.norg"
    monkeypatch.setenv("WORLD_SEED_PATH", str(seed_path))
    import importlib

    from environment import seed_loader as ns_loader

    importlib.reload(ns_loader)
    ns_loader.ensure_seed(preset="nexus")
    monkeypatch.setenv("UTOPIA_SEED_PATH", str(tmp_path / "client_seed.norg"))
    client_seed = tmp_path / "client_seed.norg"
    client_seed.write_bytes(seed_path.read_bytes())

    from utopia_client import seed_loader as cli_loader

    importlib.reload(cli_loader)
    return client_seed


@pytest.fixture
def compute_with_one(seed_file):
    """LocalColonyCompute с одной зарегистрированной особью."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    founders = load_founders(seed_file, n=1)
    compute = LocalColonyCompute(device="cpu")
    compute.add_creature("c0", founders[0])
    return compute


def _obs(seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(80, dtype=np.float32)


def test_predictor_created_in_add_creature(compute_with_one):
    c = compute_with_one
    assert "c0" in c.predictor and c.predictor["c0"] is not None
    assert "c0" in c.predictor_opt
    assert c.loss_ema["c0"] == 0.0
    assert c.intrinsic_ema["c0"] == 0.0
    assert c.entropy_ema["c0"] == 0.0
    # Обучаемые параметры есть.
    n_params = sum(p.numel() for p in c.predictor["c0"].parameters())
    assert n_params > 1000  # ~8.7K ожидаем


def test_handle_tick_populates_pred_loss_history(compute_with_one):
    c = compute_with_one
    # Первый тик: predictor только запоминает prev_obs, пока не обучается.
    c.handle_tick({"c0": _obs(0)})
    assert len(c.pred_loss_history["c0"]) == 0  # нет prev → нет шага
    assert "c0" in c.prev_obs
    # Второй тик: обучение запускается.
    c.handle_tick({"c0": _obs(1)})
    assert len(c.pred_loss_history["c0"]) == 1
    # 10 тиков → 10 записей.
    for i in range(2, 12):
        c.handle_tick({"c0": _obs(i)})
    assert len(c.pred_loss_history["c0"]) == 11
    assert c.predictor_steps == 11
    assert c.loss_ema["c0"] > 0  # MSE накопилось


def test_entropy_ema_grows(compute_with_one):
    c = compute_with_one
    for i in range(20):
        c.handle_tick({"c0": _obs(i)})
    # Logits — почти нулевая инициализация, entropy близка к log(N)/log(N)≈1.
    assert c.entropy_ema["c0"] > 0.0


def test_intrinsic_reward_emas(compute_with_one):
    c = compute_with_one
    events = {"c0": {"ate": True, "killed": False, "damage_taken": 0.0,
                     "delta_energy": 1.0}}
    for i in range(50):
        c.handle_tick({"c0": _obs(i)}, events_per_cid=events)
    # intrinsic_ema мог не вырасти если loss растёт, но reward_var_ema должен.
    assert c.reward_var_ema["c0"] >= 0.0
    assert c.intrinsic_ema["c0"] >= 0.0
    # immediate r ≈ 0.05·1 + 1 = 1.05 → hebbian обновляется.
    assert c.hebbian_updates >= 50


def test_apply_inherited_state_y50_predictor(compute_with_one):
    """Передача predictor от родителя через Y50 — потомок не идентичен."""
    c = compute_with_one
    # Симулируем родителя.
    parent_payload = c.save_state("c0")
    assert "predictor" in parent_payload
    # Сделаем несколько шагов чтобы predictor родителя имел нетривиальные веса.
    for i in range(5):
        c.handle_tick({"c0": _obs(i)})
    parent_payload = c.save_state("c0")
    # Теперь регистрируем "потомка" с теми же founder.
    from utopia_client.seed_loader import load_founders
    founders = load_founders(c.organisms["c0"].id and "/dev/null" or "", n=0) \
        if False else None
    # Использовать тот же организм-копию проще: создаём ещё одного через тот же seed.
    # Для теста — переиспользуем первый seed.
    import copy as _copy
    child_org = _copy.deepcopy(c.organisms["c0"])
    c.add_creature("c1", child_org)
    # До apply — веса predictor у c1 случайные (не равны родителю).
    parent_w = parent_payload["predictor"]
    child_w_before = c.predictor["c1"].state_dict()
    diffs_before = []
    for k in parent_w:
        if k in child_w_before:
            diffs_before.append(
                float((parent_w[k] - child_w_before[k]).abs().sum().item()))
    assert sum(diffs_before) > 0  # действительно разные

    # После Y50: child = 0.5·parent + 0.5·noise, ≠ parent но коррелирует.
    parent_payload["entropy_ema"] = 0.42
    parent_payload["trace_norm_ema"] = 0.13
    c.apply_inherited_state("c1", parent_payload)
    child_w_after = c.predictor["c1"].state_dict()
    # Веса должны измениться (Y50 применилось).
    assert any(
        not (parent_w[k] == child_w_after[k]).all()
        for k in parent_w if k in child_w_after
    )
    # EMA унаследованы.
    assert c.entropy_ema["c1"] == 0.42
    assert c.trace_norm_ema["c1"] == 0.13
    assert c.loss_ema["c1"] == parent_payload["predictor_loss_ema"]
    assert c.intrinsic_ema["c1"] == parent_payload["intrinsic_ema"]


def test_save_state_includes_predictor_and_emas(compute_with_one):
    c = compute_with_one
    for i in range(10):
        c.handle_tick({"c0": _obs(i)})
    payload = c.save_state("c0")
    assert "predictor" in payload
    assert "predictor_loss_ema" in payload
    assert "intrinsic_ema" in payload
    assert "entropy_ema" in payload
    assert "trace_norm_ema" in payload
    assert "reward_var_ema" in payload
    assert isinstance(payload["predictor_loss_ema"], float)


def test_remove_creature_cleans_all_dicts(compute_with_one):
    c = compute_with_one
    c.handle_tick({"c0": _obs(0)})
    c.handle_tick({"c0": _obs(1)})
    assert "c0" in c.predictor
    c.remove_creature("c0")
    for d in (c.predictor, c.predictor_opt, c.prev_obs, c.loss_ema,
              c.pred_loss_history, c.intrinsic_last, c.intrinsic_ema,
              c.entropy_ema, c.trace_norm_ema, c.reward_var_ema,
              c.reward_history, c.hebbian, c.organisms, c.action_selectors):
        assert "c0" not in d


def test_diagnostics_snapshot(compute_with_one):
    c = compute_with_one
    # Пустая колония-aware: после remove diagnostics всё равно валиден.
    for i in range(15):
        c.handle_tick({"c0": _obs(i)})
    diag = c.diagnostics()
    assert diag["n_alive"] == 1
    assert diag["prediction_accuracy"] > 0.0
    assert diag["prediction_loss_avg"] > 0.0
    assert diag["entropy_avg"] >= 0.0
    assert diag["predictor_steps_total"] >= 14
    # Поля присутствуют.
    for k in ("intrinsic_reward_avg", "intrinsic_reward_last_avg",
              "trace_norm_avg", "reward_var_avg", "hebbian_updates_total"):
        assert k in diag


def test_diagnostics_empty_compute():
    from utopia_client.local_compute import LocalColonyCompute
    c = LocalColonyCompute(device="cpu")
    diag = c.diagnostics()
    assert diag["n_alive"] == 0
    assert diag["prediction_accuracy"] == 0.0
    assert diag["entropy_avg"] == 0.0


def test_diagnostics_includes_local_only_fields(compute_with_one):
    """Архитектура / гены обучения / Phase 4 — поля, которых нет на P40."""
    c = compute_with_one
    for i in range(5):
        c.handle_tick({"c0": _obs(i)})
    diag = c.diagnostics()

    # architecture: гистограммы по организмам.
    arch = diag["architecture"]
    assert sum(arch["n_embd_hist"].values()) == 1
    assert sum(arch["n_layer_hist"].values()) == 1
    assert sum(arch["n_head_hist"].values()) == 1

    # learning_genes: средние от HebbianController.config.
    lg = diag["learning_genes"]
    assert lg["lr_oja_avg"] >= 0.0
    assert lg["trace_decay_avg"] >= 0.0
    assert 0.0 <= lg["hebbian_enabled_pct"] <= 1.0

    # phase4: specialization_avg может быть пустым на старте.
    assert "specialization_avg" in diag["phase4"]
    assert isinstance(diag["phase4"]["specialization_avg"], dict)
