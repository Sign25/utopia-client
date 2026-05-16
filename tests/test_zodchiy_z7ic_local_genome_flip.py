"""Z7.i.c — клиентский Genome-флип lineage_upgrade_to_zodchiy.

Покрытие:
  Z7ic.1  set_lineage_upgrade_pending(True) патчит флаг у всех owned особей.
  Z7ic.2  set_lineage_upgrade_pending(False) гасит флаг у всех owned особей.
  Z7ic.3  set_lineage_upgrade_pending идемпотентен (повтор → 0).
  Z7ic.4  Не обновляет дефолт колонии — новые особи рождаются с флагом False
          независимо от последнего вызова (это разовый триггер, а не
          постоянный признак, в отличие от set_*_sfnn).
  Z7ic.5  Не трогает SFNN-флаги (higher/basic/motor) и наоборот —
          set_*_sfnn не трогают lineage_upgrade_to_zodchiy.
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


# ── Z7ic.1 ────────────────────────────────────────────────────────────────

def test_set_lineage_upgrade_pending_flips_all_genomes(seed_file):
    """set_lineage_upgrade_pending(True) патчит флаг у всех owned особей."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    for i, org in enumerate(load_founders(seed_file, 3)):
        compute.add_creature(f"c{i}", org, hebbian_enabled=True)
    for org in compute.organisms.values():
        assert bool(getattr(org.genome,
                             "lineage_upgrade_to_zodchiy", False)) is False
    n = compute.set_lineage_upgrade_pending(True)
    assert n == 3
    for org in compute.organisms.values():
        assert org.genome.lineage_upgrade_to_zodchiy is True


# ── Z7ic.2 ────────────────────────────────────────────────────────────────

def test_set_lineage_upgrade_pending_clear(seed_file):
    """set_lineage_upgrade_pending(False) сбрасывает флаг у всех owned."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    for i, org in enumerate(load_founders(seed_file, 2)):
        compute.add_creature(f"c{i}", org, hebbian_enabled=True)
    compute.set_lineage_upgrade_pending(True)
    for org in compute.organisms.values():
        assert org.genome.lineage_upgrade_to_zodchiy is True
    n = compute.set_lineage_upgrade_pending(False)
    assert n == 2
    for org in compute.organisms.values():
        assert org.genome.lineage_upgrade_to_zodchiy is False


# ── Z7ic.3 ────────────────────────────────────────────────────────────────

def test_set_lineage_upgrade_pending_idempotent(seed_file):
    """Повторный вызов с тем же значением → n_changed = 0."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)
    assert compute.set_lineage_upgrade_pending(True) == 1
    assert compute.set_lineage_upgrade_pending(True) == 0   # уже True
    assert compute.set_lineage_upgrade_pending(False) == 1
    assert compute.set_lineage_upgrade_pending(False) == 0  # уже False


# ── Z7ic.4 ────────────────────────────────────────────────────────────────

def test_set_lineage_upgrade_pending_no_default_persistence(seed_file):
    """После set_lineage_upgrade_pending(True) новые особи рождаются с False.

    В отличие от set_*_sfnn это разовый триггер, а не сохраняемый дефолт
    колонии. Z7.c consume происходит при ближайшей репродукции — пытаться
    выставить флаг «на будущее» новым особям бессмысленно.
    """
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    compute.set_lineage_upgrade_pending(True)
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c_new", org, hebbian_enabled=True)
    assert bool(getattr(compute.organisms["c_new"].genome,
                         "lineage_upgrade_to_zodchiy", False)) is False


# ── Z7ic.5 ────────────────────────────────────────────────────────────────

def test_set_lineage_upgrade_pending_preserves_sfnn_flags(seed_file):
    """Переключение lineage-флага не трогает SFNN-флаги и vice versa."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)
    higher_before = bool(org.genome.higher_tissue_sfnn_enabled)
    motor_before = bool(org.genome.sfnn_enabled)
    basic_before = bool(org.genome.basic_tissue_sfnn_enabled)

    compute.set_lineage_upgrade_pending(True)
    assert bool(org.genome.higher_tissue_sfnn_enabled) == higher_before
    assert bool(org.genome.sfnn_enabled) == motor_before
    assert bool(org.genome.basic_tissue_sfnn_enabled) == basic_before
    assert org.genome.lineage_upgrade_to_zodchiy is True

    # Обратное: set_*_sfnn не трогают lineage-флаг.
    compute.set_higher_sfnn(not higher_before)
    compute.set_basic_sfnn(not basic_before)
    compute.set_motor_sfnn(not motor_before)
    assert org.genome.lineage_upgrade_to_zodchiy is True
