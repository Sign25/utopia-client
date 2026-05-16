"""SFNN S6.6 — диспетчер handle_tick + расширение skip_roles.

Что проверяем:
  - genome.basic_tissue_sfnn_enabled=False → счётчики базовых тканей не растут
  - genome.basic_tissue_sfnn_enabled=True → счётчики базовых тканей растут
  - skip_roles при флаге ON содержит все 10 базовых ролей (классика молчит)
  - skip_roles при флаге OFF не содержит базовых ролей (классика пишет)
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
pytest.importorskip("torch")


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


def _tick(seed_file, *, basic_on: bool):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)
    # Включаем/выключаем флаг S6.6.
    org.genome.basic_tissue_sfnn_enabled = basic_on
    rng = np.random.default_rng(0)
    compute.handle_tick(
        {"c1": rng.normal(size=80).astype(np.float32)},
        events_per_cid={"c1": {"ate": 1, "killed": 0,
                                "damage_taken": 0, "delta_energy": 0.1}},
    )
    return compute


def test_handle_tick_flag_off_no_basic_sfnn_steps(seed_file):
    """genome.basic_tissue_sfnn_enabled=False → счётчики базовых не растут."""
    compute = _tick(seed_file, basic_on=False)
    for steps in compute.basic_tissue_sfnn_steps.values():
        assert steps == 0


def test_handle_tick_flag_on_increments_basic_sfnn(seed_file):
    """genome.basic_tissue_sfnn_enabled=True → счётчики растут."""
    compute = _tick(seed_file, basic_on=True)
    incremented = [r for r, s in compute.basic_tissue_sfnn_steps.items()
                    if s > 0]
    assert incremented, (
        f"флаг ON, но ни одна базовая ткань не обновилась: "
        f"{compute.basic_tissue_sfnn_steps}"
    )


def test_handle_tick_flag_on_hebbian_skip_includes_basic(seed_file):
    """При флаге ON heb.update получает skip_roles ⊇ _BASIC_SFNN_TISSUES.

    Перехватываем heb.update через monkeypatch — фиксируем skip_roles, что
    реально пришёл из handle_tick.
    """
    from utopia_client.local_compute import (
        LocalColonyCompute, _BASIC_SFNN_TISSUES, _SFNN_MIGRATED_ROLES,
    )
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)
    org.genome.basic_tissue_sfnn_enabled = True

    captured = {}
    heb = compute.hebbian["c1"]
    real_update = heb.update

    def spy_update(*args, **kwargs):
        captured["skip_roles"] = kwargs.get("skip_roles")
        return real_update(*args, **kwargs)

    heb.update = spy_update
    rng = np.random.default_rng(7)
    compute.handle_tick(
        {"c1": rng.normal(size=80).astype(np.float32)},
        events_per_cid={"c1": {"ate": 1, "killed": 0,
                                "damage_taken": 0, "delta_energy": 0.1}},
    )
    assert captured.get("skip_roles") is not None
    expected = _SFNN_MIGRATED_ROLES | set(_BASIC_SFNN_TISSUES)
    assert captured["skip_roles"] == expected


def test_handle_tick_flag_off_hebbian_skip_excludes_basic(seed_file):
    """При флаге OFF heb.update получает skip_roles = _SFNN_MIGRATED_ROLES.

    То есть _BASIC_SFNN_TISSUES НЕ в skip_roles → классика свободно пишет
    в input_proj/output_proj базовых тканей (поведение до S6).
    """
    from utopia_client.local_compute import (
        LocalColonyCompute, _BASIC_SFNN_TISSUES, _SFNN_MIGRATED_ROLES,
    )
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)
    org.genome.basic_tissue_sfnn_enabled = False

    captured = {}
    heb = compute.hebbian["c1"]
    real_update = heb.update

    def spy_update(*args, **kwargs):
        captured["skip_roles"] = kwargs.get("skip_roles")
        return real_update(*args, **kwargs)

    heb.update = spy_update
    rng = np.random.default_rng(7)
    compute.handle_tick(
        {"c1": rng.normal(size=80).astype(np.float32)},
        events_per_cid={"c1": {"ate": 1, "killed": 0,
                                "damage_taken": 0, "delta_energy": 0.1}},
    )
    assert captured.get("skip_roles") == _SFNN_MIGRATED_ROLES
    # Проверяем что НЕмигрированные базовые роли отсутствуют в skip.
    # "motor" есть и в _BASIC_SFNN_TISSUES, и в _SFNN_MIGRATED_ROLES — это
    # ок (legacy motor_policy migration с S4).
    non_migrated_basic = set(_BASIC_SFNN_TISSUES) - _SFNN_MIGRATED_ROLES
    assert not (non_migrated_basic & captured["skip_roles"])


def test_handle_tick_no_events_skips_basic_sfnn(seed_file):
    """events_per_cid=None → ни heb.update, ни basic_sfnn_update_step
    (по аналогии с Phase F3.2.b)."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)
    org.genome.basic_tissue_sfnn_enabled = True
    rng = np.random.default_rng(0)
    # events_per_cid=None → блок hebbian не запускается.
    compute.handle_tick({"c1": rng.normal(size=80).astype(np.float32)})
    assert all(s == 0 for s in compute.basic_tissue_sfnn_steps.values())
