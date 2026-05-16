"""SFNN S6.9 — admin toggle set_basic_sfnn.

Что проверяем:
  - set_basic_sfnn(True) патчит genome.basic_tissue_sfnn_enabled у всех owned
  - дефолт колонии (`_basic_sfnn_default`) обновляется → новые особи
    в add_creature получают тот же флаг
  - возвращаемое n_changed = реальное число изменённых геномов
  - повторный вызов с тем же значением → n_changed = 0
  - toggle ON→OFF возвращает геном к False
  - дефолт инициализирован в False (как раньше — обратная совместимость)
"""
from __future__ import annotations

import sys
from pathlib import Path

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


def test_default_basic_sfnn_is_on(seed_file):
    """S6.11: _basic_sfnn_default стартует True — миграция завершена."""
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    assert compute._basic_sfnn_default is True


def test_set_basic_sfnn_flips_all_genomes(seed_file):
    """set_basic_sfnn(False) → set_basic_sfnn(True) патчит флаг у всех owned."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    compute.set_basic_sfnn(False)  # сбросим дефолт перед заселением
    for i, org in enumerate(load_founders(seed_file, 3)):
        compute.add_creature(f"c{i}", org, hebbian_enabled=True)
    for org in compute.organisms.values():
        assert org.genome.basic_tissue_sfnn_enabled is False
    n = compute.set_basic_sfnn(True)
    assert n == 3
    for org in compute.organisms.values():
        assert org.genome.basic_tissue_sfnn_enabled is True


def test_set_basic_sfnn_idempotent(seed_file):
    """Повторный вызов с тем же значением → n_changed = 0."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    compute.set_basic_sfnn(False)  # S6.11: дефолт True, сбрасываем для теста
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)
    assert compute.set_basic_sfnn(True) == 1
    assert compute.set_basic_sfnn(True) == 0  # уже True
    assert compute.set_basic_sfnn(False) == 1
    assert compute.set_basic_sfnn(False) == 0  # уже False


def test_set_basic_sfnn_updates_default_for_new_creatures(seed_file):
    """После set_basic_sfnn(True) новые особи рождаются с флагом True."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    compute.set_basic_sfnn(True)
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c_new", org, hebbian_enabled=True)
    assert compute.organisms["c_new"].genome.basic_tissue_sfnn_enabled is True


def test_set_basic_sfnn_off_for_new_creatures(seed_file):
    """После set_basic_sfnn(False) новые особи рождаются с флагом False."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    compute.set_basic_sfnn(True)
    compute.set_basic_sfnn(False)
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c_new", org, hebbian_enabled=True)
    assert compute.organisms["c_new"].genome.basic_tissue_sfnn_enabled is False


def test_set_basic_sfnn_preserves_higher_motor_flags(seed_file):
    """Переключение basic не трогает higher_sfnn и motor sfnn флаги."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)
    higher_before = bool(org.genome.higher_tissue_sfnn_enabled)
    motor_before = bool(org.genome.sfnn_enabled)
    compute.set_basic_sfnn(True)
    assert bool(org.genome.higher_tissue_sfnn_enabled) == higher_before
    assert bool(org.genome.sfnn_enabled) == motor_before
