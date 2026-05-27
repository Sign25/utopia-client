"""Phase 4 этап B integration: persist_all_memory + restore через add_creature.

Сценарий «restart»:
  1. LocalColonyCompute A: add_creature(cid) → train (модифицируем weights) →
     persist_all_memory()
  2. LocalColonyCompute B: add_creature(cid, fresh organism) → memory
     автоматически восстанавливается из файла

Требует neurocore[client] (core.workbench + storage.norg) — поэтому
pytest.importorskip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")
pytest.importorskip("core.workbench")
pytest.importorskip("storage.norg")

# Real production seed (read-only) — без необходимости fastapi для ensure_seed.
# Скипуем если seed отсутствует (CI без production state).
_PROD_SEED = Path.home() / ".utopia-client" / "seed.norg"
if not _PROD_SEED.exists():
    pytest.skip(
        f"production seed not present at {_PROD_SEED} — integration tests skip",
        allow_module_level=True,
    )


@pytest.fixture
def patched_colonies_dir(tmp_path, monkeypatch):
    """Перенаправить colonies_dir() на tmp_path для теста."""
    from utopia_client import config as cli_config
    monkeypatch.setattr(cli_config, "colonies_dir",
                        lambda: tmp_path / "colonies")
    return tmp_path / "colonies"


@pytest.fixture
def seed_path():
    """Используем production seed read-only — изоляция через patched_colonies_dir."""
    return _PROD_SEED


def _load_organism(seed):
    from utopia_client import seed_loader as cli_loader
    return cli_loader.load_founders(seed, n=1)[0]


def test_persist_then_restore_episodic_weights(
        patched_colonies_dir, seed_path):
    """End-to-end: персистим episodic, создаём fresh compute, weights
    восстанавливаются автоматически на add_creature."""
    import torch
    from utopia_client.local_compute import LocalColonyCompute

    cid = "test-cid-persist"

    # ── Phase A: создаём, портим weights, персистим ──
    compute_a = LocalColonyCompute(device="cpu")
    org_a = _load_organism(seed_path)
    compute_a.add_creature(cid, org_a, lineage="zodchiy")

    epi_a = compute_a.episodic.get(cid)
    assert epi_a is not None, "episodic ткань не создана для zodchiy"

    # «Обучаем»: добавляем шум к одному weight tensor чтобы изменить state
    sd_a = epi_a.state_dict()
    target_key = next(iter(sd_a.keys()))
    learned_value = sd_a[target_key].clone() + torch.randn_like(sd_a[target_key]) * 0.5
    sd_a_modified = {**sd_a, target_key: learned_value}
    epi_a.load_state_dict(sd_a_modified)

    # Также добавим last_episodic_recall
    compute_a.last_episodic_recall[cid] = torch.tensor([0.42] * 64)

    # Persist
    n_saved = compute_a.persist_all_memory()
    assert n_saved >= 1

    # ── Phase B: fresh compute, тот же cid ──
    compute_b = LocalColonyCompute(device="cpu")
    org_b = _load_organism(seed_path)
    compute_b.add_creature(cid, org_b, lineage="zodchiy")

    epi_b = compute_b.episodic.get(cid)
    assert epi_b is not None

    sd_b = epi_b.state_dict()
    # learned_value должен быть восстановлен (1e-6 не нужен — torch.save
    # сохраняет точно)
    assert torch.allclose(sd_b[target_key], learned_value, atol=1e-6), \
        f"learned weight не восстановлен после restart"

    # last_episodic_recall тоже восстановлен
    recall_b = compute_b.last_episodic_recall.get(cid)
    assert recall_b is not None
    assert torch.allclose(recall_b, torch.tensor([0.42] * 64))


def test_no_memory_file_no_crash(patched_colonies_dir, seed_path):
    """Fresh start без файлов памяти — add_creature нормально работает."""
    from utopia_client.local_compute import LocalColonyCompute

    compute = LocalColonyCompute(device="cpu")
    org = _load_organism(seed_path)
    # Не должно крашнуться даже если memory/ ещё нет
    compute.add_creature("first-creature", org, lineage="zodchiy")
    assert "first-creature" in compute.episodic


def test_reset_all_persists_before_clearing(
        patched_colonies_dir, seed_path):
    """reset_all() сохраняет episodic state перед pop'ом организмов."""
    import torch
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.memory_store import load_memory_state, memory_file

    cid = "test-reset-persist"
    compute = LocalColonyCompute(device="cpu")
    org = _load_organism(seed_path)
    compute.add_creature(cid, org, lineage="zodchiy")

    # Изменяем episodic state
    epi = compute.episodic[cid]
    sd = epi.state_dict()
    k = next(iter(sd.keys()))
    sd[k] = sd[k].clone() + 1.0
    epi.load_state_dict(sd)

    # reset_all = orderly shutdown → должно persist
    compute.reset_all()
    assert memory_file(cid, base=patched_colonies_dir).exists()
    payload = load_memory_state(cid, base=patched_colonies_dir)
    assert payload is not None
    assert payload["cid"] == cid


def test_remove_creature_does_NOT_persist(
        patched_colonies_dir, seed_path):
    """remove_creature (= смерть) НЕ должен сохранять — это permanent
    cid disposal. Если организм умер, его memory не нужно хранить."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.memory_store import memory_file

    cid = "test-death"
    compute = LocalColonyCompute(device="cpu")
    org = _load_organism(seed_path)
    compute.add_creature(cid, org, lineage="zodchiy")

    # Death — remove_creature
    compute.remove_creature(cid)

    # Файл памяти НЕ должен появиться (death = no persist)
    assert not memory_file(cid, base=patched_colonies_dir).exists(), \
        "remove_creature ошибочно сохранил memory мёртвого организма"
