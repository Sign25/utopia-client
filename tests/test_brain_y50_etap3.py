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


# ── 7. selector наследуется через inherit_brain_y50 ─────────────────


def test_inherit_brain_y50_carries_selector_bias(seed_file):
    """SFNN-стиль reward-modulated bias на ActionSelector должен наследоваться
    через Y50 рядом с predictor/dopamine/.../selector apply уже есть в
    apply_inherited_state, но extract/inherit раньше не клали selector в
    payload → child стартовал с нулевым bias.
    """
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    parents = load_founders(seed_file, 2)
    compute = LocalColonyCompute(device="cpu")
    compute.add_creature("parent", parents[0], hebbian_enabled=True)
    compute.add_creature("child", parents[1], hebbian_enabled=True)
    # Накопить ярко выраженный bias у parent.
    sel_parent = compute.action_selectors["parent"]
    for _ in range(30):
        sel_parent.reinforce(1.0, action_idx=8)  # SHARE
        sel_parent.reinforce(-0.5, action_idx=5)  # ATK
    parent_bias = sel_parent._action_bias.clone()
    assert float(parent_bias.abs().max().item()) > 0.1
    # До inherit — у child нулевой bias.
    sel_child = compute.action_selectors["child"]
    assert float(sel_child._action_bias.abs().max().item()) == 0.0
    # Y50 наследование.
    assert compute.inherit_brain_y50("parent", "child") is True
    # После inherit — child получил bias parent'а.
    inherited_max = float(sel_child._action_bias.abs().max().item())
    assert inherited_max > 0.1, (
        f"selector bias не передался: child_max={inherited_max}, "
        f"parent_max={float(parent_bias.abs().max().item())}"
    )
    # Достаточная близость к parent (load_state_dict без Y50-шума для bias).
    diff = (sel_child._action_bias - parent_bias).abs().max().item()
    assert diff < 1e-5, f"bias child != bias parent: max_diff={diff}"


def test_extract_brain_state_dicts_includes_selector(seed_file):
    """`extract_brain_state_dicts` должен класть selector в payload рядом с
    predictor — иначе asexual envelope (P40-mediated reproduce) не передаст
    bias ребёнку."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("m1", org, hebbian_enabled=True)
    sel = compute.action_selectors["m1"]
    for _ in range(20):
        sel.reinforce(0.5, action_idx=14)  # EAT
    brain, _ = compute.extract_brain_state_dicts("m1")
    assert "selector" in brain
    sd = brain["selector"]
    assert "action_bias" in sd
    assert max(abs(v) for v in sd["action_bias"]) > 0.05


# ── 8. SFNN S1.1: motor_sfnn_rule storage + наследование ────────────


def test_add_creature_initializes_default_sfnn_rule(seed_file):
    """Каждая особь с motor_policy получает дефолтное SFNN-правило (Phase 5d-эквивалент)."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    from core.sfnn_rule import SYNAPSE_TYPES
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("m1", org, hebbian_enabled=True)
    rule = compute.motor_sfnn_rule.get("m1")
    assert rule is not None, "motor_sfnn_rule не создан в add_creature"
    assert set(rule.coeffs.keys()) == set(SYNAPSE_TYPES)
    # Дефолт = Phase 5d (A=1, B=C=D=0, η=1e-3).
    for c in rule.coeffs.values():
        assert c.eta == 1e-3
        assert c.A == 1.0
        assert c.B == 0.0
        assert c.C == 0.0
        assert c.D == 0.0


def test_remove_creature_clears_sfnn_rule(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("m1", org, hebbian_enabled=True)
    assert "m1" in compute.motor_sfnn_rule
    compute.remove_creature("m1")
    assert "m1" not in compute.motor_sfnn_rule


def test_extract_brain_state_dicts_includes_motor_sfnn_rule(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    from core.sfnn_rule import SYNAPSE_TYPES
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("m1", org, hebbian_enabled=True)
    # Подкрутим коэффициенты у parent, чтобы было видно что наследуется
    # именно его значение, а не дефолт.
    compute.motor_sfnn_rule["m1"].coeffs["input_proj"].A = 0.5
    compute.motor_sfnn_rule["m1"].coeffs["mlp_fc1"].eta = 5e-3
    brain, _ = compute.extract_brain_state_dicts("m1")
    assert "motor_sfnn_rule" in brain
    rd = brain["motor_sfnn_rule"]
    assert set(rd.keys()) == set(SYNAPSE_TYPES)
    assert rd["input_proj"]["A"] == 0.5
    assert rd["mlp_fc1"]["eta"] == 5e-3


def test_inherit_brain_y50_carries_sfnn_rule_with_noise(seed_file):
    """Дочернее правило = parent.mutate(σ=0.1): коэффициенты близкие, но
    не идентичные (есть Y50-шум)."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    parents = load_founders(seed_file, 2)
    compute = LocalColonyCompute(device="cpu")
    compute.add_creature("parent", parents[0], hebbian_enabled=True)
    compute.add_creature("child", parents[1], hebbian_enabled=True)
    # Сдвинем parent rule подальше от дефолта, чтобы наследование было видно.
    p_rule = compute.motor_sfnn_rule["parent"]
    for c in p_rule.coeffs.values():
        c.eta = 5e-3
        c.A = 0.5
        c.B = 0.2
    # До inherit child = default.
    assert compute.motor_sfnn_rule["child"].coeffs["input_proj"].A == 1.0
    # Inherit.
    assert compute.inherit_brain_y50("parent", "child") is True
    child_rule = compute.motor_sfnn_rule["child"]
    # Child.coeff близок к parent (mutate σ=0.1, в среднем ±10%), но не равен дефолту.
    for c in child_rule.coeffs.values():
        # eta в окрестности 5e-3 (parent), не 1e-3 (default).
        assert abs(c.eta - 5e-3) < 4e-3
        # A в окрестности 0.5, не 1.0.
        assert abs(c.A - 0.5) < 0.4
        assert c.A != 1.0  # не дефолт


# ── SFNN S1.2a (14.05.2026) — флаг + skeleton _motor_sfnn_update_step ───


def _sfnn_setup_with_forward(seed_file):
    """Хелпер: создаёт compute с одной особью и прогоняет motor_forward
    с ненулевым obs, чтобы forward-hooks захватили pre≠0 / post≠0."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("m1", org, hebbian_enabled=True)
    torch.manual_seed(0)
    obs = torch.randn(1, 64)
    compute._motor_forward("m1", obs)
    return compute


def test_motor_sfnn_hooks_register_on_add_creature(seed_file):
    """add_creature вешает 6 forward-hooks на Linear motor_policy Tissue."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("m1", org, hebbian_enabled=True)
    handles = compute.motor_sfnn_hook_handles.get("m1", [])
    assert len(handles) == 6
    assert "m1" in compute.motor_sfnn_acts
    # acts пустые до первого forward.
    assert compute.motor_sfnn_acts["m1"] == {}


def test_motor_sfnn_acts_populated_by_forward(seed_file):
    """Forward через motor_policy наполняет motor_sfnn_acts всеми 6 типами."""
    from core.sfnn_rule import SYNAPSE_TYPES
    compute = _sfnn_setup_with_forward(seed_file)
    acts = compute.motor_sfnn_acts["m1"]
    for s in SYNAPSE_TYPES:
        assert s in acts, f"synapse {s} не захвачен hook'ом"
        pre, post = acts[s]
        assert pre.dim() == 1 and post.dim() == 1


def test_motor_sfnn_update_modifies_weights_when_reward_positive(seed_file):
    """С дефолтным правилом (A=1, η=1e-3) и положительным reward (ate=True
    даёт r_imm=5.0 → reward_gain=6) веса 6 Linear motor_policy сдвигаются."""
    compute = _sfnn_setup_with_forward(seed_file)
    motor = compute.motor_policy["m1"]
    weights_before = {n: p.detach().clone()
                       for n, p in motor.named_parameters()
                       if p.dim() == 2}
    # _compute_immediate_reward ожидает ключ "ate" (bool), не "event_type".
    events = {"m1": {"ate": True, "delta_energy": 1.0}}
    compute._motor_sfnn_update_step("m1", events, intrinsic_now=0.01)
    assert compute.motor_sfnn_steps == 1
    changed = sum(1 for n, w_before in weights_before.items()
                   if not torch.equal(dict(motor.named_parameters())[n].data,
                                        w_before))
    assert changed >= 1, "ни один weight не изменился под дефолтным правилом"


def test_motor_sfnn_update_noop_with_eta_zero(seed_file):
    """η=0 во всех 6 типах → ΔW = 0 → веса не меняются (sanity)."""
    from core.sfnn_rule import SFNNRule, SFNNSynapseCoeffs, SYNAPSE_TYPES
    compute = _sfnn_setup_with_forward(seed_file)
    # Подменяем правило на «всё нулевое».
    zero_coeffs = {k: SFNNSynapseCoeffs(eta=1e-5, A=0.0, B=0.0, C=0.0, D=0.0)
                    for k in SYNAPSE_TYPES}
    compute.motor_sfnn_rule["m1"] = SFNNRule(coeffs=zero_coeffs)
    motor = compute.motor_policy["m1"]
    weights_before = {n: p.detach().clone()
                       for n, p in motor.named_parameters()
                       if p.dim() == 2}
    events = {"m1": {"ate": True, "delta_energy": 1.0}}
    compute._motor_sfnn_update_step("m1", events, intrinsic_now=0.0)
    for n, w_before in weights_before.items():
        w_after = dict(motor.named_parameters())[n].data
        assert torch.allclose(w_after, w_before, atol=1e-10), \
            f"{n} изменился при A=B=C=D=0"


def test_motor_sfnn_update_clip_caps_huge_reward(seed_file):
    """Большой reward не должен сдвинуть ни один элемент W больше чем на clip (0.01)."""
    compute = _sfnn_setup_with_forward(seed_file)
    motor = compute.motor_policy["m1"]
    weights_before = {n: p.detach().clone()
                       for n, p in motor.named_parameters()
                       if p.dim() == 2}
    # Огромный reward через delta_energy (×0.5 в _compute_immediate_reward).
    events = {"m1": {"delta_energy": 1e6, "ate": True}}
    compute._motor_sfnn_update_step("m1", events, intrinsic_now=0.0)
    for n, w_before in weights_before.items():
        w_after = dict(motor.named_parameters())[n].data
        max_dw = float((w_after - w_before).abs().max().item())
        assert max_dw <= 0.01 + 1e-6, \
            f"{n}: max|ΔW|={max_dw} > clip 0.01"


def test_motor_sfnn_hooks_removed_on_remove_creature(seed_file):
    """remove_creature снимает все 6 forward-hooks и очищает кеши."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("m1", org, hebbian_enabled=True)
    assert len(compute.motor_sfnn_hook_handles["m1"]) == 6
    compute.remove_creature("m1")
    assert "m1" not in compute.motor_sfnn_hook_handles
    assert "m1" not in compute.motor_sfnn_acts


def test_genome_sfnn_enabled_default_false():
    """По умолчанию у любого Genome флаг выключен — legacy путь активен."""
    from core.organism import Genome
    g = Genome()
    assert g.sfnn_enabled is False
    g2 = Genome(sfnn_enabled=True)
    assert g2.sfnn_enabled is True


def test_motor_sfnn_update_step_clears_pending_log_prob(seed_file):
    """pending_log_prob/pending_action чистятся (как в REINFORCE-пути)."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("m1", org, hebbian_enabled=True)
    compute.pending_log_prob["m1"] = "stale"
    compute.pending_action["m1"] = 7
    events = {"m1": {"delta_energy": -0.1}}
    compute._motor_sfnn_update_step("m1", events, intrinsic_now=0.0)
    assert "m1" not in compute.pending_log_prob
    assert "m1" not in compute.pending_action
