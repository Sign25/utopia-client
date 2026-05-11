"""Brain migration Etap 3.1 — Y50 наследование predictor + S2.E/G/A/F.

После mate-pair клиент:
  - кеширует child_org в `_pending_newborn_orgs[mother_cid]`
  - на следующем obs_batch с новым cid и parent_id==mother_cid:
    регистрирует ребёнка в compute + Y50-наследует мозг от матери.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
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


class _CapturingWS:
    def __init__(self):
        self.sent: list = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def _ws_with_compute(colony="c"):
    from utopia_client.ws_client import ColonyWSClient
    from utopia_client.local_compute import LocalColonyCompute
    ws = ColonyWSClient(server="https://x", token="t",
                         colony_name=colony, client_version="test")
    # CPU-only — иначе на сервере с CUDA tissue.device=cuda и обзоры из теста
    # (cpu numpy) не совпадут. CI обычно без GPU, локально с P40 — наоборот.
    ws.compute = LocalColonyCompute(device="cpu")
    ws._ws = _CapturingWS()
    return ws


def _serialize_father(seed_file):
    from utopia_client.seed_loader import load_founders
    org = load_founders(seed_file, 1)[0]
    payload = {"tissues_by_role":
               {tid: t.state_dict() for tid, t in org.tissues.items()}}
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def _snapshot_predictor_w(tissue) -> dict:
    """Снимок state_dict с клонированием тензоров (защита от in-place mutate)."""
    return {k: v.detach().clone() for k, v in tissue.state_dict().items()}


# ── 1. mate_request caches child_org ────────────────────────────────

def test_mate_request_caches_child_org(seed_file):
    async def _run():
        from utopia_client.seed_loader import load_founders
        ws = _ws_with_compute()
        mother_org = load_founders(seed_file, 1)[0]
        ws.compute.add_creature("m1", mother_org, hebbian_enabled=True)
        father_bytes = _serialize_father(seed_file)
        await ws._handle_mate_request({
            "request_id": "r1",
            "mother_cid": "m1",
            "father_cid": "f1",
            "father_blob_b64": base64.b64encode(father_bytes).decode("ascii"),
            "sigma_scale": 1.0,
        })
        assert ws._mate_newborns_sent == 1
        assert "m1" in ws._pending_newborn_orgs
        child_org, ts = ws._pending_newborn_orgs["m1"]
        assert hasattr(child_org, "tissues")
    asyncio.run(_run())


# ── 2. obs_batch with parent_id registers + Y50 ─────────────────────

def test_obs_batch_attaches_newborn_and_inherits_brain(seed_file):
    async def _run():
        import numpy as np
        from utopia_client.seed_loader import load_founders
        ws = _ws_with_compute()
        mother_org = load_founders(seed_file, 1)[0]
        ws.compute.add_creature("m1", mother_org, hebbian_enabled=True)
        # До mate-pair: обучим predictor родителя пару шагов, чтобы веса были
        # не identical к рандомной инициализации (иначе Y50 от random == random).
        m_pred = ws.compute.predictor["m1"]
        opt = ws.compute.predictor_opt["m1"]
        import torch.nn.functional as F
        for _ in range(3):
            inp = torch.randn(1, 64)
            tgt = torch.randn(1, 64)
            out = m_pred({"input": inp})["output"]
            loss = F.mse_loss(out, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
        parent_pred_snap = _snapshot_predictor_w(m_pred)
        parent_dop_snap = _snapshot_predictor_w(ws.compute.dopamine["m1"])

        father_bytes = _serialize_father(seed_file)
        await ws._handle_mate_request({
            "request_id": "r1", "mother_cid": "m1", "father_cid": "f1",
            "father_blob_b64": base64.b64encode(father_bytes).decode("ascii"),
            "sigma_scale": 1.0,
        })
        assert "m1" in ws._pending_newborn_orgs

        # P40 присылает обзор с новой особью c_child, parent_id=m1.
        obs_64 = [0.0] * 64
        await ws._handle_obs_batch({
            "world_tick": 100, "ts_p40_ns": 0,
            "creatures": [
                {"cid": "m1", "obs": obs_64, "parent_id": ""},
                {"cid": "c_child", "obs": obs_64, "parent_id": "m1"},
            ],
        })
        assert "c_child" in ws.compute.organisms
        assert ws._newborn_attached == 1
        assert ws._newborn_brain_inherited == 1
        # Cache очищен.
        assert "m1" not in ws._pending_newborn_orgs

        child_pred_snap = _snapshot_predictor_w(ws.compute.predictor["c_child"])
        child_dop_snap = _snapshot_predictor_w(ws.compute.dopamine["c_child"])

        # Y50: child = 0.5·parent + 0.5·noise(σ·std). Веса 2D — отличаются от
        # родителя, но коррелируют (0.5·parent доминирует при σ=0.0902).
        diffs = []
        for k, pv in parent_pred_snap.items():
            cv = child_pred_snap[k]
            if pv.dim() >= 2 and "weight" in k:
                diff = (pv - cv).abs().mean().item()
                diffs.append(diff)
        assert diffs, "predictor должен иметь хотя бы один 2D weight"
        assert all(d > 0 for d in diffs), "Y50 noise должен изменить веса"

        diffs_dop = []
        for k, pv in parent_dop_snap.items():
            cv = child_dop_snap[k]
            if pv.dim() >= 2 and "weight" in k:
                diff = (pv - cv).abs().mean().item()
                diffs_dop.append(diff)
        assert diffs_dop and all(d > 0 for d in diffs_dop)
    asyncio.run(_run())


# ── 3. obs_batch without parent_id → no attach (safe fallback) ──────

def test_obs_batch_without_parent_id_silent(seed_file):
    async def _run():
        from utopia_client.seed_loader import load_founders
        ws = _ws_with_compute()
        mother_org = load_founders(seed_file, 1)[0]
        ws.compute.add_creature("m1", mother_org, hebbian_enabled=True)
        father_bytes = _serialize_father(seed_file)
        await ws._handle_mate_request({
            "request_id": "r", "mother_cid": "m1", "father_cid": "f1",
            "father_blob_b64": base64.b64encode(father_bytes).decode("ascii"),
        })
        # Старый сервер не шлёт parent_id — никаких regression.
        obs_64 = [0.0] * 64
        await ws._handle_obs_batch({
            "world_tick": 100, "ts_p40_ns": 0,
            "creatures": [
                {"cid": "c_orphan", "obs": obs_64},
            ],
        })
        assert "c_orphan" not in ws.compute.organisms
        assert ws._newborn_attached == 0
        # Cache остаётся (TTL не истёк).
        assert "m1" in ws._pending_newborn_orgs
    asyncio.run(_run())


# ── 4. unknown parent_id in obs → no attach ─────────────────────────

def test_obs_batch_unknown_parent_silent(seed_file):
    async def _run():
        from utopia_client.seed_loader import load_founders
        ws = _ws_with_compute()
        mother_org = load_founders(seed_file, 1)[0]
        ws.compute.add_creature("m1", mother_org, hebbian_enabled=True)
        # Никакого mate_request — кеш пуст.
        obs_64 = [0.0] * 64
        await ws._handle_obs_batch({
            "world_tick": 50, "ts_p40_ns": 0,
            "creatures": [
                {"cid": "c_ghost", "obs": obs_64, "parent_id": "unknown_dad"},
            ],
        })
        assert "c_ghost" not in ws.compute.organisms
        assert ws._newborn_attached == 0
    asyncio.run(_run())


# ── 5. cache TTL expiry ─────────────────────────────────────────────

def test_pending_newborn_cache_expires(seed_file, monkeypatch):
    async def _run():
        import time as _time
        from utopia_client.seed_loader import load_founders
        ws = _ws_with_compute()
        mother_org = load_founders(seed_file, 1)[0]
        ws.compute.add_creature("m1", mother_org, hebbian_enabled=True)
        father_bytes = _serialize_father(seed_file)
        await ws._handle_mate_request({
            "request_id": "r", "mother_cid": "m1", "father_cid": "f1",
            "father_blob_b64": base64.b64encode(father_bytes).decode("ascii"),
        })
        # Перетравим timestamp: на 200с в прошлом → TTL должен expirе-нуть.
        ws._pending_newborn_orgs["m1"] = (
            ws._pending_newborn_orgs["m1"][0],
            _time.time() - 200.0,
        )
        obs_64 = [0.0] * 64
        await ws._handle_obs_batch({
            "world_tick": 1, "ts_p40_ns": 0,
            "creatures": [
                {"cid": "c_late", "obs": obs_64, "parent_id": "m1"},
            ],
        })
        # Запись истекла → attach не произошёл.
        assert "c_late" not in ws.compute.organisms
        assert "m1" not in ws._pending_newborn_orgs
        assert ws._newborn_attached == 0
    asyncio.run(_run())


# ── 6. inherit_brain_y50 unit: unknown cids → no-op ─────────────────

def test_inherit_brain_y50_unknown_cids(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("m1", org, hebbian_enabled=True)
    # parent unknown:
    assert compute.inherit_brain_y50("ghost", "m1") is False
    # child unknown:
    assert compute.inherit_brain_y50("m1", "ghost") is False
    # parent == child:
    assert compute.inherit_brain_y50("m1", "m1") is False
