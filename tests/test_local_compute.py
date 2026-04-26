"""Smoke-тест клиентского compute (Phase F3.4).

Цель: 5 founder из seed.norg, 100 тиков синтетических obs, без смертей,
все actions ∈ [0..15]. Защищает от регрессии в API neurocore[client].

Зависит от neurocore (установлен в .venv P40 для тестового прогона) и
Утопия-API. fetch_seed мокается через MagicMock — реального VPS не требуется.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# В dev-окружении тесты прогоняются из корня utopia-client. Добавляем в path.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Pre-import neurocore: если не установлен — пропускаем тест (release).
pytest.importorskip("storage.norg")
pytest.importorskip("core.workbench")


@pytest.fixture
def seed_file(tmp_path, monkeypatch):
    """Сгенерировать seed.norg через neurocore.environment.seed_loader (P40-side)."""
    seed_path = tmp_path / "seed.norg"
    monkeypatch.setenv("WORLD_SEED_PATH", str(seed_path))

    import importlib

    from environment import seed_loader as ns_loader

    importlib.reload(ns_loader)
    ns_loader.ensure_seed(preset="tardigrade")
    monkeypatch.setenv("UTOPIA_SEED_PATH", str(tmp_path / "client_seed.norg"))
    # Скопируем в client cache, чтобы клиент-loader не лез к API.
    client_seed = tmp_path / "client_seed.norg"
    client_seed.write_bytes(seed_path.read_bytes())

    from utopia_client import seed_loader as cli_loader

    importlib.reload(cli_loader)
    return client_seed


def test_load_founders_returns_n_unique(seed_file):
    from utopia_client.seed_loader import load_founders

    founders = load_founders(seed_file, n=5)
    assert len(founders) == 5
    ids = {o.id for o in founders}
    assert len(ids) == 5  # уникальные id
    # Архитектура одинаковая
    n_tissues = founders[0].n_tissues
    for o in founders[1:]:
        assert o.n_tissues == n_tissues


def test_local_compute_smoke_100_ticks(seed_file):
    """Сценарий из ТЗ: 5 особей, 100 тиков, без смертей."""
    from utopia_client.local_compute import LocalColonyCompute, N_ACTIONS
    from utopia_client.seed_loader import load_founders

    founders = load_founders(seed_file, n=5)
    cids = [f"cid_{i}" for i in range(5)]

    compute = LocalColonyCompute(device="cpu")
    for cid, org in zip(cids, founders):
        compute.add_creature(cid, org)
    assert compute.n_alive == 5

    rng = np.random.default_rng(42)
    for _t in range(100):
        obs_batch = {cid: rng.normal(size=80).astype(np.float32) for cid in cids}
        actions = compute.handle_tick(obs_batch)

        assert set(actions.keys()) == set(cids)
        for cid, a in actions.items():
            assert isinstance(a["action"], int)
            assert 0 <= a["action"] < N_ACTIONS
            assert a["target_id"] is None or isinstance(a["target_id"], str)


def test_local_compute_missing_obs_returns_stay(seed_file):
    from utopia_client.local_compute import STAY, LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    founders = load_founders(seed_file, n=2)
    compute = LocalColonyCompute(device="cpu")
    compute.add_creature("c1", founders[0])
    compute.add_creature("c2", founders[1])

    actions = compute.handle_tick({"c1": np.zeros(80, dtype=np.float32)})
    assert actions["c2"]["action"] == STAY


def test_local_compute_remove_creature(seed_file):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    org = load_founders(seed_file, n=1)[0]
    compute = LocalColonyCompute(device="cpu")
    compute.add_creature("c1", org)
    assert compute.n_alive == 1
    compute.remove_creature("c1")
    assert compute.n_alive == 0
    out = compute.handle_tick({"c1": np.zeros(80, dtype=np.float32)})
    assert out == {}


def test_ensure_seed_uses_cache(seed_file, tmp_path, monkeypatch):
    """Если файл уже есть — fetch_seed не вызывается."""
    from utopia_client import seed_loader as cli_loader

    api = MagicMock()
    api.fetch_seed.return_value = True
    path = cli_loader.ensure_seed(api)
    assert path == seed_file
    api.fetch_seed.assert_not_called()


def test_ensure_seed_fetches_when_missing(tmp_path, monkeypatch):
    """Если файла нет — вызываем api.fetch_seed."""
    cache = tmp_path / "missing_seed.norg"
    monkeypatch.setenv("UTOPIA_SEED_PATH", str(cache))

    import importlib

    from utopia_client import seed_loader as cli_loader

    importlib.reload(cli_loader)

    def _fake_fetch(dest_path: str) -> bool:
        Path(dest_path).write_bytes(b"fake-seed-bytes")
        return True

    api = MagicMock()
    api.fetch_seed.side_effect = _fake_fetch
    path = cli_loader.ensure_seed(api)
    assert path == cache
    assert cache.exists()
    api.fetch_seed.assert_called_once_with(str(cache))
