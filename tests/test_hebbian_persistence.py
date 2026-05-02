"""Phase F3.2.c — персистенция Hebbian-state особей в локальный кеш колонии."""
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


@pytest.fixture
def seed_file(tmp_path, monkeypatch):
    seed_path = tmp_path / "seed.norg"
    monkeypatch.setenv("WORLD_SEED_PATH", str(seed_path))

    import importlib
    from environment import seed_loader as ns_loader
    importlib.reload(ns_loader)
    ns_loader.ensure_seed(preset="nexus")

    monkeypatch.setenv("UTOPIA_SEED_PATH", str(tmp_path / "client_seed.norg"))
    monkeypatch.setenv("UTOPIA_COLONIES_DIR", str(tmp_path / "colonies"))
    client_seed = tmp_path / "client_seed.norg"
    client_seed.write_bytes(seed_path.read_bytes())

    from utopia_client import seed_loader as cli_loader
    importlib.reload(cli_loader)
    return client_seed


# ── Пути к локальному кешу: env override + структура каталогов ──────

def test_colony_state_dir_uses_env(tmp_path, monkeypatch):
    monkeypatch.setenv("UTOPIA_COLONIES_DIR", str(tmp_path / "X"))
    import importlib
    from utopia_client import seed_loader as cli_loader
    importlib.reload(cli_loader)
    d = cli_loader.colony_state_dir("test_col")
    assert d == tmp_path / "X" / "test_col"


def test_creature_state_path_format(tmp_path, monkeypatch):
    monkeypatch.setenv("UTOPIA_COLONIES_DIR", str(tmp_path / "Y"))
    import importlib
    from utopia_client import seed_loader as cli_loader
    importlib.reload(cli_loader)
    p = cli_loader.creature_state_path("col1", "c_42")
    assert p == tmp_path / "Y" / "col1" / "c_42.pt"


# ── save_state / save_all_states ────────────────────────────────────

def test_save_state_returns_payload(seed_file):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)

    payload = compute.save_state("c1")
    assert payload is not None
    assert "tissues_by_role" in payload
    assert isinstance(payload["tissues_by_role"], dict)
    assert len(payload["tissues_by_role"]) > 0
    # Hebbian state включается, если HebbianController есть
    assert "hebbian" in payload


def test_save_state_unknown_cid_returns_none(seed_file):
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    assert compute.save_state("nope") is None


def test_save_all_states_creates_files(tmp_path, seed_file):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    compute = LocalColonyCompute(device="cpu")
    for cid in ("ca", "cb", "cc"):
        org = load_founders(seed_file, 1)[0]
        compute.add_creature(cid, org, hebbian_enabled=True)

    out_dir = tmp_path / "save"
    n = compute.save_all_states(out_dir)
    assert n == 3
    assert (out_dir / "ca.pt").exists()
    assert (out_dir / "cb.pt").exists()
    assert (out_dir / "cc.pt").exists()


def test_save_all_states_empty_compute(tmp_path):
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    n = compute.save_all_states(tmp_path / "empty")
    assert n == 0


# ── Roundtrip: save → load → веса совпадают ─────────────────────────

def test_save_load_roundtrip_preserves_tissues(tmp_path, seed_file):
    """Сохранённый payload загружается через organism_from_weights, веса tissues
    совпадают побитово."""
    import io
    import torch
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders, organism_from_weights

    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("c1", org, hebbian_enabled=True)

    # Прогон Hebbian update — веса сдвинутся.
    obs = {"c1": np.random.randn(80).astype(np.float32) * 0.1}
    events = {"c1": {"ate": True, "delta_energy": 2.0}}
    for _ in range(5):
        compute.handle_tick(obs, events_per_cid=events)

    out_dir = tmp_path / "save"
    compute.save_all_states(out_dir)
    saved_bytes = (out_dir / "c1.pt").read_bytes()

    # Load обратно через тот же путь, что и в _finalize_seed_creature
    org2, payload = organism_from_weights(saved_bytes, seed_file)
    first_tid = next(iter(org.tissues.keys()))
    sd_orig = org.tissues[first_tid].state_dict()
    sd_loaded = org2.tissues[first_tid].state_dict()
    for k in sd_orig:
        assert torch.allclose(sd_orig[k], sd_loaded[k]), \
            f"tissue {first_tid} param {k} mismatch"


# ── Приоритет local-cache над seed-bytes (smoke через обе ветки) ────

def test_local_cache_path_used_when_present(tmp_path, seed_file, monkeypatch):
    """Файл `{cid}.pt` существует → ws_client._finalize_seed_creature берёт
    его, а не присланные с P40 chunks."""
    import asyncio
    import base64
    import io
    import torch

    from utopia_client.seed_loader import (
        creature_state_path, colony_state_dir, load_founders)
    from utopia_client.ws_client import ColonyWSClient

    # Готовим локальный файл с заведомо «обученными» весами (модифицируем).
    org_local = load_founders(seed_file, 1)[0]
    first_tid = next(iter(org_local.tissues.keys()))
    sd = org_local.tissues[first_tid].state_dict()
    marker_key = next(iter(sd.keys()))
    sd[marker_key] = sd[marker_key] + 0.5  # сдвиг для отличия
    org_local.tissues[first_tid].load_state_dict(sd)
    payload_local = {
        "tissues_by_role": {(getattr(t,'role','') or '_'+tid): t.state_dict()
                                for tid, t in org_local.tissues.items()},
    }
    colony = "tcol"
    p = creature_state_path(colony, "c1")
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload_local, p)

    # Готовим P40-bytes из чистого seed (без сдвига).
    org_seed = load_founders(seed_file, 1)[0]
    p40_payload = {
        "tissues_by_role": {(getattr(t,'role','') or '_'+tid): t.state_dict()
                                for tid, t in org_seed.tissues.items()},
    }
    buf = io.BytesIO()
    torch.save(p40_payload, buf)
    p40_bytes = buf.getvalue()

    # Имитируем seed_chunk → finalize.
    ws = ColonyWSClient(server="https://x", token="t",
                         colony_name=colony, client_version="test")
    ws._seed_buffers[(1, "c1")] = [p40_bytes]
    ws._seed_meta[(1, "c1")] = {}
    assert ws._ensure_compute()
    ws._finalize_seed_creature((1, "c1"))

    org_loaded = ws.compute.organisms["c1"]
    sd_loaded = org_loaded.tissues[first_tid].state_dict()
    # Если использован local-cache, веса = local (со сдвигом), не P40-чистые.
    # Compute может уехать на CUDA — сравниваем на CPU.
    assert torch.allclose(sd_loaded[marker_key].cpu(),
                          sd[marker_key].cpu()), "local-cache был проигнорирован"


def test_seed_used_when_no_local_cache(tmp_path, seed_file):
    """Файла local-cache нет → используются присланные chunks."""
    import io
    import torch

    from utopia_client.seed_loader import (
        creature_state_path, load_founders)
    from utopia_client.ws_client import ColonyWSClient

    colony = "fresh_col"
    p = creature_state_path(colony, "c2")
    assert not p.exists()

    org_seed = load_founders(seed_file, 1)[0]
    payload = {
        "tissues_by_role": {(getattr(t,'role','') or '_'+tid): t.state_dict()
                                for tid, t in org_seed.tissues.items()},
    }
    buf = io.BytesIO()
    torch.save(payload, buf)

    ws = ColonyWSClient(server="https://x", token="t",
                         colony_name=colony, client_version="test")
    ws._seed_buffers[(1, "c2")] = [buf.getvalue()]
    ws._seed_meta[(1, "c2")] = {}
    assert ws._ensure_compute()
    ws._finalize_seed_creature((1, "c2"))

    assert "c2" in ws.compute.organisms
    assert ws._seed_accepted == 1


def test_corrupt_local_cache_falls_back_to_seed(tmp_path, seed_file):
    """Битый файл в local-cache → НЕ роняем загрузку, fallback на P40-bytes."""
    import io
    import torch

    from utopia_client.seed_loader import (
        creature_state_path, load_founders)
    from utopia_client.ws_client import ColonyWSClient

    colony = "broken_col"
    p = creature_state_path(colony, "c3")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"not a valid torch file")

    org_seed = load_founders(seed_file, 1)[0]
    payload = {
        "tissues_by_role": {(getattr(t,'role','') or '_'+tid): t.state_dict()
                                for tid, t in org_seed.tissues.items()},
    }
    buf = io.BytesIO()
    torch.save(payload, buf)

    ws = ColonyWSClient(server="https://x", token="t",
                         colony_name=colony, client_version="test")
    ws._seed_buffers[(1, "c3")] = [buf.getvalue()]
    ws._seed_meta[(1, "c3")] = {}
    assert ws._ensure_compute()
    ws._finalize_seed_creature((1, "c3"))

    # P40-bytes валидны → организм загружен. Битый local-cache не помешал.
    assert "c3" in ws.compute.organisms
    assert ws._seed_accepted == 1


# ── stop() сохраняет state ──────────────────────────────────────────

def test_stop_saves_local_state(tmp_path, seed_file):
    """ColonyWSClient.stop() после регистрации особи → файлы появляются на диске."""
    from utopia_client.seed_loader import (
        load_founders, colony_state_dir)
    from utopia_client.ws_client import ColonyWSClient

    colony = "stop_col"
    ws = ColonyWSClient(server="https://x", token="t",
                         colony_name=colony, client_version="test")
    assert ws._ensure_compute()
    org = load_founders(seed_file, 1)[0]
    ws.compute.add_creature("c_stop", org, hebbian_enabled=True)

    # Поток не запущен — stop() пройдёт сразу к save-блоку.
    ws.stop()

    saved = colony_state_dir(colony) / "c_stop.pt"
    assert saved.exists() and saved.stat().st_size > 0


def test_periodic_save_loop_writes_local_state(seed_file, monkeypatch):
    """`_save_loop` периодически вызывает `compute.save_all_states`.

    Без него local cache наполняется только на `stop()`, а если клиент
    падает без чистой остановки — обучение теряется. Тест: укорачиваем
    интервал до 0.05с и проверяем, что файлы появляются после нескольких
    итераций.
    """
    import asyncio

    from utopia_client import ws_client as wsmod
    from utopia_client.seed_loader import colony_state_dir, load_founders
    from utopia_client.ws_client import ColonyWSClient

    monkeypatch.setattr(wsmod, "LOCAL_SAVE_INTERVAL_SEC", 0.05)

    colony = "periodic_col"
    ws = ColonyWSClient(server="https://x", token="t",
                         colony_name=colony, client_version="test")
    assert ws._ensure_compute()
    org = load_founders(seed_file, 1)[0]
    ws.compute.add_creature("c_p1", org, hebbian_enabled=True)

    async def _run():
        task = asyncio.create_task(ws._save_loop())
        await asyncio.sleep(0.2)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(_run())

    saved = colony_state_dir(colony) / "c_p1.pt"
    assert saved.exists() and saved.stat().st_size > 0
