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


# ── SFNN S3.0 (14.05.2026) — инфраструктура высших тканей ──────────────


_HIGHER_7 = (
    "dopamine", "imagination", "planner", "insula",
    "default_mode", "theory_of_mind", "language",
)


def test_higher_tissue_sfnn_rule_default_on_add_creature(seed_file):
    """add_creature создаёт дефолтное правило для каждой из 7 высших тканей."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("h1", org, hebbian_enabled=True)
    for t in _HIGHER_7:
        rule = compute.higher_tissue_sfnn_rule[t].get("h1")
        assert rule is not None, t
        # Дефолт = Phase 5d Hebb (A=1, B=C=D=0, η=1e-3).
        for c in rule.coeffs.values():
            assert c.A == 1.0, t
            assert c.eta == 1e-3, t


def test_higher_tissue_sfnn_rule_removed_on_remove_creature(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("h1", org, hebbian_enabled=True)
    compute.remove_creature("h1")
    for t in _HIGHER_7:
        assert "h1" not in compute.higher_tissue_sfnn_rule[t], t


def test_higher_tissue_sfnn_rule_steps_reset_in_reset_all(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("h1", org, hebbian_enabled=True)
    compute.higher_tissue_sfnn_steps["dopamine"] = 42
    compute.higher_tissue_sfnn_steps["imagination"] = 7
    compute.reset_all()
    for t in _HIGHER_7:
        assert compute.higher_tissue_sfnn_steps[t] == 0, t


def test_inherit_brain_y50_carries_higher_tissue_sfnn_rules(seed_file):
    """Все 7 правил родителя наследуются ребёнком с σ=0.1 mutate."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    parents = load_founders(seed_file, 2)
    compute = LocalColonyCompute(device="cpu")
    compute.add_creature("parent", parents[0], hebbian_enabled=True)
    compute.add_creature("child", parents[1], hebbian_enabled=True)
    # Сдвинем правила parent от дефолта по всем 7 тканям.
    for t in _HIGHER_7:
        p_rule = compute.higher_tissue_sfnn_rule[t]["parent"]
        for c in p_rule.coeffs.values():
            c.eta = 5e-3
            c.A = 0.5
    # До inherit child = default.
    for t in _HIGHER_7:
        assert compute.higher_tissue_sfnn_rule[t]["child"].coeffs["input_proj"].A == 1.0
    assert compute.inherit_brain_y50("parent", "child") is True
    # После inherit правила всех 7 тканей близки к parent (mutate σ=0.1).
    for t in _HIGHER_7:
        child_rule = compute.higher_tissue_sfnn_rule[t]["child"]
        for c in child_rule.coeffs.values():
            assert abs(c.eta - 5e-3) < 4e-3, t
            assert abs(c.A - 0.5) < 0.4, t
            assert c.A != 1.0, t  # не дефолт


def test_save_state_carries_higher_tissue_sfnn_rules(seed_file):
    """save_state кладёт 7 правил в payload['higher_tissue_sfnn_rules']."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("h1", org, hebbian_enabled=True)
    payload = compute.save_state("h1")
    assert "higher_tissue_sfnn_rules" in payload
    dumped = payload["higher_tissue_sfnn_rules"]
    for t in _HIGHER_7:
        assert t in dumped, t
        # 6 типов синапсов, дефолтные коэф.
        assert dumped[t]["input_proj"]["A"] == 1.0
        assert dumped[t]["input_proj"]["eta"] == 1e-3


# ── SFNN S3.1 (14.05.2026) — активация dopamine ───────────────────────────


def test_s3_1_dopamine_hooks_registered_on_add_creature(seed_file):
    """add_creature регистрирует 6 forward-hooks на dopamine Tissue."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("d1", org, hebbian_enabled=True)
    handles = compute.higher_tissue_sfnn_hook_handles["dopamine"].get("d1")
    assert handles is not None
    # 6 Linear модулей Tissue 21/3/1 = 6 хуков.
    assert len(handles) == 6
    # На момент S3.7 все 7 высших тканей активны (hooks регистрируются
    # для каждой при наличии).


def test_s3_1_dopamine_hooks_capture_pre_post_on_forward(seed_file):
    """После forward dopamine — higher_tissue_sfnn_acts заполнен 6 парами."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import torch
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("d1", org, hebbian_enabled=True)
    obs = torch.zeros(1, 64)
    compute._compute_higher_tissues("d1", obs, intero_tensor=None)
    acts = compute.higher_tissue_sfnn_acts["dopamine"]["d1"]
    assert len(acts) == 6
    for syn, (pre, post) in acts.items():
        assert pre.dim() == 1, syn
        assert post.dim() == 1, syn


def test_s3_1_dopamine_hooks_removed_on_remove_creature(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("d1", org, hebbian_enabled=True)
    compute.remove_creature("d1")
    assert "d1" not in compute.higher_tissue_sfnn_hook_handles["dopamine"]
    assert "d1" not in compute.higher_tissue_sfnn_acts["dopamine"]


def test_s3_1_higher_tissue_sfnn_update_step_changes_weights(seed_file):
    """update_step реально обновляет веса dopamine Tissue (после forward)."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import torch
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("d1", org, hebbian_enabled=True)
    # Forward dopamine → активации лежат в acts.
    obs = torch.randn(1, 64)
    compute._compute_higher_tissues("d1", obs, intero_tensor=None)
    # Усилим η чтобы дельта была заметной за один шаг.
    rule = compute.higher_tissue_sfnn_rule["dopamine"]["d1"]
    for c in rule.coeffs.values():
        c.eta = 0.005
    # Снимок весов до апдейта.
    tissue = compute.dopamine["d1"]
    w_before = {n: p.detach().clone()
                 for n, p in tissue.named_parameters()
                 if p.dim() == 2}
    steps_before = compute.higher_tissue_sfnn_steps["dopamine"]
    compute._higher_tissue_sfnn_update_step("dopamine", "d1", r=0.0)
    # Хотя бы один вес изменился.
    changed = False
    for n, p in tissue.named_parameters():
        if p.dim() == 2 and not torch.equal(w_before[n], p.detach()):
            changed = True
            break
    assert changed, "ни один weight dopamine не изменился"
    assert compute.higher_tissue_sfnn_steps["dopamine"] == steps_before + 1


def test_s3_1_update_step_noop_without_acts(seed_file):
    """Без forward (нет acts) update_step не падает и не считает шаг."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("d1", org, hebbian_enabled=True)
    # acts пуст — forward не вызывали.
    steps_before = compute.higher_tissue_sfnn_steps["dopamine"]
    compute._higher_tissue_sfnn_update_step("dopamine", "d1", r=0.0)
    assert compute.higher_tissue_sfnn_steps["dopamine"] == steps_before


def test_s3_1_update_step_noop_for_unknown_tissue(seed_file):
    """Защита от опечатки в имени ткани — no-op, не KeyError."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("d1", org, hebbian_enabled=True)
    compute._higher_tissue_sfnn_update_step("nonexistent", "d1", r=0.0)
    compute._higher_tissue_sfnn_update_step("dopamine", "unknown_cid", r=0.0)


# ── SFNN S3.2 (14.05.2026) — активация imagination ────────────────────────


def test_s3_2_imagination_hooks_registered_on_add_creature(seed_file):
    """add_creature регистрирует 6 forward-hooks на imagination Tissue."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("i1", org, hebbian_enabled=True)
    handles = compute.higher_tissue_sfnn_hook_handles["imagination"].get("i1")
    assert handles is not None
    assert len(handles) == 6
    # На момент S3.7 все 7 высших тканей активны.


def test_s3_2_imagination_hooks_capture_pre_post_on_forward(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import torch
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("i1", org, hebbian_enabled=True)
    obs = torch.zeros(1, 64)
    compute._compute_higher_tissues("i1", obs, intero_tensor=None)
    acts = compute.higher_tissue_sfnn_acts["imagination"]["i1"]
    assert len(acts) == 6


def test_s3_2_imagination_update_step_changes_weights(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import torch
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("i1", org, hebbian_enabled=True)
    obs = torch.randn(1, 64)
    compute._compute_higher_tissues("i1", obs, intero_tensor=None)
    rule = compute.higher_tissue_sfnn_rule["imagination"]["i1"]
    for c in rule.coeffs.values():
        c.eta = 0.005
    tissue = compute.imagination["i1"]
    w_before = {n: p.detach().clone()
                 for n, p in tissue.named_parameters() if p.dim() == 2}
    steps_before = compute.higher_tissue_sfnn_steps["imagination"]
    compute._higher_tissue_sfnn_update_step("imagination", "i1", r=0.0)
    changed = any(
        not torch.equal(w_before[n], p.detach())
        for n, p in tissue.named_parameters()
        if p.dim() == 2 and n in w_before)
    assert changed, "ни один weight imagination не изменился"
    assert (compute.higher_tissue_sfnn_steps["imagination"]
            == steps_before + 1)


# ── SFNN S3.3 (14.05.2026) — активация planner ────────────────────────────


def test_s3_3_planner_hooks_registered(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("p1", org, hebbian_enabled=True)
    handles = compute.higher_tissue_sfnn_hook_handles["planner"].get("p1")
    assert handles is not None and len(handles) == 6


def test_s3_3_planner_hooks_capture(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import torch
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("p1", org, hebbian_enabled=True)
    compute._compute_higher_tissues(
        "p1", torch.zeros(1, 64), intero_tensor=None)
    assert len(compute.higher_tissue_sfnn_acts["planner"]["p1"]) == 6


def test_s3_3_planner_update_changes_weights(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import torch
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("p1", org, hebbian_enabled=True)
    compute._compute_higher_tissues(
        "p1", torch.randn(1, 64), intero_tensor=None)
    rule = compute.higher_tissue_sfnn_rule["planner"]["p1"]
    for c in rule.coeffs.values():
        c.eta = 0.005
    tissue = compute.planner["p1"]
    w_before = {n: p.detach().clone()
                 for n, p in tissue.named_parameters() if p.dim() == 2}
    steps_before = compute.higher_tissue_sfnn_steps["planner"]
    compute._higher_tissue_sfnn_update_step("planner", "p1", r=0.0)
    changed = any(
        not torch.equal(w_before[n], p.detach())
        for n, p in tissue.named_parameters()
        if p.dim() == 2 and n in w_before)
    assert changed
    assert (compute.higher_tissue_sfnn_steps["planner"]
            == steps_before + 1)


# ── SFNN S3.4 (14.05.2026) — активация insula (data_dim=71) ───────────────


def test_s3_4_insula_hooks_registered(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("s1", org, hebbian_enabled=True)
    handles = compute.higher_tissue_sfnn_hook_handles["insula"].get("s1")
    assert handles is not None and len(handles) == 6


def test_s3_4_insula_hooks_capture_with_intero(seed_file):
    """Insula forward требует intero_tensor [1, 7]; hooks ловят 6 пар."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import torch
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("s1", org, hebbian_enabled=True)
    compute._compute_higher_tissues(
        "s1", torch.zeros(1, 64), intero_tensor=torch.zeros(1, 7))
    assert len(compute.higher_tissue_sfnn_acts["insula"]["s1"]) == 6


def test_s3_4_insula_noop_without_intero(seed_file):
    """Без intero_tensor insula forward пропускается → acts пуст → no-op."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import torch
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("s1", org, hebbian_enabled=True)
    compute._compute_higher_tissues(
        "s1", torch.zeros(1, 64), intero_tensor=None)
    assert len(compute.higher_tissue_sfnn_acts["insula"]["s1"]) == 0
    steps_before = compute.higher_tissue_sfnn_steps["insula"]
    compute._higher_tissue_sfnn_update_step("insula", "s1", r=0.0)
    assert compute.higher_tissue_sfnn_steps["insula"] == steps_before


def test_s3_4_insula_update_changes_weights(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import torch
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("s1", org, hebbian_enabled=True)
    compute._compute_higher_tissues(
        "s1", torch.randn(1, 64), intero_tensor=torch.randn(1, 7))
    rule = compute.higher_tissue_sfnn_rule["insula"]["s1"]
    for c in rule.coeffs.values():
        c.eta = 0.005
    tissue = compute.insula["s1"]
    w_before = {n: p.detach().clone()
                 for n, p in tissue.named_parameters() if p.dim() == 2}
    steps_before = compute.higher_tissue_sfnn_steps["insula"]
    compute._higher_tissue_sfnn_update_step("insula", "s1", r=0.0)
    changed = any(
        not torch.equal(w_before[n], p.detach())
        for n, p in tissue.named_parameters()
        if p.dim() == 2 and n in w_before)
    assert changed
    assert (compute.higher_tissue_sfnn_steps["insula"]
            == steps_before + 1)


# ── SFNN S3.5 (14.05.2026) — активация default_mode ───────────────────────


def test_s3_5_default_mode_hooks_registered(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("dm1", org, hebbian_enabled=True)
    handles = compute.higher_tissue_sfnn_hook_handles["default_mode"].get("dm1")
    assert handles is not None and len(handles) == 6


def test_s3_5_default_mode_hooks_capture(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import torch
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("dm1", org, hebbian_enabled=True)
    compute._compute_higher_tissues(
        "dm1", torch.zeros(1, 64), intero_tensor=None)
    assert len(compute.higher_tissue_sfnn_acts["default_mode"]["dm1"]) == 6


def test_s3_5_default_mode_update_changes_weights(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import torch
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("dm1", org, hebbian_enabled=True)
    compute._compute_higher_tissues(
        "dm1", torch.randn(1, 64), intero_tensor=None)
    rule = compute.higher_tissue_sfnn_rule["default_mode"]["dm1"]
    for c in rule.coeffs.values():
        c.eta = 0.005
    tissue = compute.default_mode["dm1"]
    w_before = {n: p.detach().clone()
                 for n, p in tissue.named_parameters() if p.dim() == 2}
    steps_before = compute.higher_tissue_sfnn_steps["default_mode"]
    compute._higher_tissue_sfnn_update_step("default_mode", "dm1", r=0.0)
    changed = any(
        not torch.equal(w_before[n], p.detach())
        for n, p in tissue.named_parameters()
        if p.dim() == 2 and n in w_before)
    assert changed
    assert (compute.higher_tissue_sfnn_steps["default_mode"]
            == steps_before + 1)


# ── SFNN S3.6 (14.05.2026) — активация theory_of_mind ──────────────────────


def test_s3_6_tom_hooks_registered(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("tm1", org, hebbian_enabled=True)
    handles = compute.higher_tissue_sfnn_hook_handles["theory_of_mind"].get("tm1")
    assert handles is not None and len(handles) == 6


def test_s3_6_tom_hooks_capture_on_forward(seed_file):
    """Прямой forward через tissue() заполняет acts (6 синапсов)."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import torch
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("tm1", org, hebbian_enabled=True)
    tissue = compute.theory_of_mind["tm1"]
    # data_dim = 4 соседа × 13 = 52.
    feat = torch.randn(1, tissue.input_proj.in_features)
    with torch.no_grad():
        tissue({"input": feat})
    acts = compute.higher_tissue_sfnn_acts["theory_of_mind"]["tm1"]
    assert len(acts) == 6


def test_s3_6_tom_update_step_changes_weights(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import torch
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("tm1", org, hebbian_enabled=True)
    tissue = compute.theory_of_mind["tm1"]
    feat = torch.randn(1, tissue.input_proj.in_features)
    with torch.no_grad():
        tissue({"input": feat})
    rule = compute.higher_tissue_sfnn_rule["theory_of_mind"]["tm1"]
    for c in rule.coeffs.values():
        c.eta = 0.005
    w_before = {n: p.detach().clone()
                 for n, p in tissue.named_parameters() if p.dim() == 2}
    steps_before = compute.higher_tissue_sfnn_steps["theory_of_mind"]
    compute._higher_tissue_sfnn_update_step("theory_of_mind", "tm1", r=0.0)
    changed = any(
        not torch.equal(w_before[n], p.detach())
        for n, p in tissue.named_parameters()
        if p.dim() == 2 and n in w_before)
    assert changed
    assert (compute.higher_tissue_sfnn_steps["theory_of_mind"]
            == steps_before + 1)


def test_s3_6_tom_adam_skipped_under_sfnn_flag(seed_file):
    """Под higher_tissue_sfnn_enabled=True _compute_theory_of_mind делает
    forward (hooks стреляют) но НЕ вызывает opt.step()."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    from tests.test_theory_of_mind_s2b import _FakeCache
    import torch
    import types
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    # CompositeOrganism не имеет .genome — впрыскиваем SimpleNamespace
    # с нужным флагом. Production-код читает через getattr-цепочку.
    org.genome = types.SimpleNamespace(higher_tissue_sfnn_enabled=True)
    compute.add_creature("tm1", org, hebbian_enabled=True)
    # Поставим world_cache с двумя особями (self + один сосед).
    wc = _FakeCache(size=32)
    wc.set_creature("tm1", 10, 10)
    wc.set_creature("n1", 11, 10)
    compute.world_cache = wc
    # На первом тике prev_focus is None → Adam-путь в legacy всё равно
    # не делает step. Но forward в SFNN-режиме должен случиться.
    # Очистить acts (после add_creature их нет).
    compute.higher_tissue_sfnn_acts["theory_of_mind"]["tm1"] = {}
    compute._compute_theory_of_mind("tm1")
    acts1 = compute.higher_tissue_sfnn_acts["theory_of_mind"]["tm1"]
    assert len(acts1) == 6, "forward должен сработать → 6 синапсов"
    # tom_steps НЕ растёт (Adam-путь пропущен).
    assert compute.tom_steps == 0
    # Adam optimizer.state пуст (нет .step() → нет moment buffers).
    opt = compute.theory_of_mind_opt["tm1"]
    for p in opt.param_groups[0]["params"]:
        assert p not in opt.state or "step" not in opt.state.get(p, {})
    # Второй тик — теперь есть prev_focus. Под флагом снова Adam пропускается.
    wc.set_creature("n1", 11, 11)  # focus двинулся.
    compute._compute_theory_of_mind("tm1")
    assert compute.tom_steps == 0  # всё ещё ноль.
    assert compute.last_tom_acc.get("tm1", 0.0) == 0.0  # EMA не обновлялась.


# ── SFNN S3.7 (14.05.2026) — активация language ────────────────────────────


def test_s3_7_lang_hooks_registered(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("lg1", org, hebbian_enabled=True)
    handles = compute.higher_tissue_sfnn_hook_handles["language"].get("lg1")
    assert handles is not None and len(handles) == 6


def test_s3_7_lang_hooks_capture_on_forward(seed_file):
    """Прямой forward через tissue() заполняет acts (6 синапсов)."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import torch
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("lg1", org, hebbian_enabled=True)
    tissue = compute.language["lg1"]
    feat = torch.randn(1, tissue.input_proj.in_features)
    with torch.no_grad():
        tissue({"input": feat})
    acts = compute.higher_tissue_sfnn_acts["language"]["lg1"]
    assert len(acts) == 6


def test_s3_7_lang_update_step_changes_weights(seed_file):
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import torch
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature("lg1", org, hebbian_enabled=True)
    tissue = compute.language["lg1"]
    feat = torch.randn(1, tissue.input_proj.in_features)
    with torch.no_grad():
        tissue({"input": feat})
    rule = compute.higher_tissue_sfnn_rule["language"]["lg1"]
    for c in rule.coeffs.values():
        c.eta = 0.005
    w_before = {n: p.detach().clone()
                 for n, p in tissue.named_parameters() if p.dim() == 2}
    steps_before = compute.higher_tissue_sfnn_steps["language"]
    compute._higher_tissue_sfnn_update_step("language", "lg1", r=0.0)
    changed = any(
        not torch.equal(w_before[n], p.detach())
        for n, p in tissue.named_parameters()
        if p.dim() == 2 and n in w_before)
    assert changed
    assert (compute.higher_tissue_sfnn_steps["language"]
            == steps_before + 1)


def test_s3_7_lang_adam_skipped_under_sfnn_flag(seed_file):
    """Под higher_tissue_sfnn_enabled=True _compute_language делает forward
    (hooks стреляют) но НЕ вызывает opt.step()."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    from tests.test_theory_of_mind_s2b import _FakeCache
    import torch
    import types
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    org.genome = types.SimpleNamespace(higher_tissue_sfnn_enabled=True)
    compute.add_creature("lg1", org, hebbian_enabled=True)
    # _build_lang_features требует world_cache.creature_tom + neighbors.
    wc = _FakeCache(size=32)
    wc.set_creature("lg1", 10, 10, sig=2)
    wc.set_creature("n1", 11, 10, sig=5)
    # _build_lang_features читает wc.creature_tom (dict cid → (lin,e,max,sig)).
    wc.creature_tom = wc._tom  # alias
    compute.world_cache = wc
    compute.higher_tissue_sfnn_acts["language"]["lg1"] = {}
    # Event ate=True должен был бы триггерить supervised step в legacy
    # пути (если бы был prev_context). Под флагом — всегда пропускается.
    compute._compute_language("lg1", {"ate": True})
    acts1 = compute.higher_tissue_sfnn_acts["language"]["lg1"]
    assert len(acts1) == 6, "forward должен сработать → 6 синапсов"
    assert compute.lang_steps == 0
    opt = compute.language_opt["lg1"]
    for p in opt.param_groups[0]["params"]:
        assert p not in opt.state or "step" not in opt.state.get(p, {})
    # Второй тик: prev_context уже есть, event=damage — Adam-путь в legacy
    # сделал бы step. Под флагом — снова пропуск.
    compute._compute_language("lg1", {"damage_taken": 1.0})
    assert compute.lang_steps == 0
    assert compute.last_lang_acc.get("lg1", 0.0) == 0.0


# ── SFNN S3.diag (14.05.2026) — per-tissue диагностика ─────────────────────


def test_diagnostics_higher_sfnn_empty_when_no_creatures(seed_file):
    """Без особей блок higher_sfnn присутствует с нулями для всех 7 тканей."""
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    diag = compute.diagnostics()
    assert "higher_sfnn" in diag
    block = diag["higher_sfnn"]
    assert block["enabled_pct"] == 0.0
    for t in ("dopamine", "imagination", "planner", "insula",
                "default_mode", "theory_of_mind", "language"):
        assert t in block, t
        assert block[t]["steps_total"] == 0
        assert block[t]["eta_avg"] == 0.0
        assert block[t]["A_avg"] == 0.0


def test_diagnostics_higher_sfnn_aggregates_per_tissue(seed_file):
    """С 2 особями: средние η/A агрегируются, steps_total отражают счётчики."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    orgs = load_founders(seed_file, 2)
    compute.add_creature("c1", orgs[0], hebbian_enabled=True)
    compute.add_creature("c2", orgs[1], hebbian_enabled=True)
    # Сдвинем dopamine у двух особей: η=0.005, A=0.5 у первой и
    # η=0.001, A=0.8 у второй → eta_avg=0.003, A_avg=0.65.
    r1 = compute.higher_tissue_sfnn_rule["dopamine"]["c1"]
    for c in r1.coeffs.values():
        c.eta = 0.005
        c.A = 0.5
    r2 = compute.higher_tissue_sfnn_rule["dopamine"]["c2"]
    for c in r2.coeffs.values():
        c.eta = 0.001
        c.A = 0.8
    # Симулируем счётчик апдейтов dopamine.
    compute.higher_tissue_sfnn_steps["dopamine"] = 42
    diag = compute.diagnostics()
    dop = diag["higher_sfnn"]["dopamine"]
    assert dop["steps_total"] == 42
    assert abs(dop["eta_avg"] - 0.003) < 1e-5
    assert abs(dop["A_avg"] - 0.65) < 1e-3
    # Остальные ткани — дефолт (η=1e-3, A=1.0).
    for t in ("imagination", "planner", "insula", "default_mode",
                "theory_of_mind", "language"):
        bl = diag["higher_sfnn"][t]
        assert abs(bl["eta_avg"] - 1e-3) < 1e-6, t
        assert abs(bl["A_avg"] - 1.0) < 1e-3, t


def test_diagnostics_higher_sfnn_enabled_pct(seed_file):
    """enabled_pct = доля особей с higher_tissue_sfnn_enabled=True."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    import types
    compute = LocalColonyCompute(device="cpu")
    orgs = load_founders(seed_file, 3)
    compute.add_creature("c1", orgs[0], hebbian_enabled=True)
    compute.add_creature("c2", orgs[1], hebbian_enabled=True)
    compute.add_creature("c3", orgs[2], hebbian_enabled=True)
    # 2/3 особей с флагом on.
    orgs[0].genome = types.SimpleNamespace(higher_tissue_sfnn_enabled=True)
    orgs[1].genome = types.SimpleNamespace(higher_tissue_sfnn_enabled=True)
    orgs[2].genome = types.SimpleNamespace(higher_tissue_sfnn_enabled=False)
    diag = compute.diagnostics()
    assert abs(diag["higher_sfnn"]["enabled_pct"] - 0.667) < 1e-3


# ─────────────────────────────────────────────────────────────────────
# SFNN S3.activate (14.05.2026) — manual endpoint + set_higher_sfnn
# ─────────────────────────────────────────────────────────────────────

def test_s3_activate_default_genome_attached_off(seed_file):
    """add_creature клеит genome.higher_tissue_sfnn_enabled=False по дефолту."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    orgs = load_founders(seed_file, 1)
    compute.add_creature("c1", orgs[0], hebbian_enabled=True)
    org = compute.organisms["c1"]
    assert hasattr(org, "genome")
    assert org.genome.higher_tissue_sfnn_enabled is False


def test_s3_activate_set_higher_sfnn_flips_existing(seed_file):
    """set_higher_sfnn(True) переписывает флаг у всех существующих особей."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    orgs = load_founders(seed_file, 3)
    for i, o in enumerate(orgs):
        compute.add_creature(f"c{i}", o, hebbian_enabled=True)
    n = compute.set_higher_sfnn(True)
    assert n == 3
    for cid in ("c0", "c1", "c2"):
        assert compute.organisms[cid].genome.higher_tissue_sfnn_enabled is True


def test_s3_activate_default_propagates_to_new_creatures(seed_file):
    """После set_higher_sfnn(True) новые особи из add_creature тоже True."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    compute.set_higher_sfnn(True)
    orgs = load_founders(seed_file, 1)
    compute.add_creature("c1", orgs[0], hebbian_enabled=True)
    assert compute.organisms["c1"].genome.higher_tissue_sfnn_enabled is True


def test_s3_activate_toggle_off_idempotent(seed_file):
    """Повторный set_higher_sfnn(False) — n_changed=0 после первого вызова."""
    from utopia_client.seed_loader import load_founders
    from utopia_client.local_compute import LocalColonyCompute
    compute = LocalColonyCompute(device="cpu")
    orgs = load_founders(seed_file, 2)
    for i, o in enumerate(orgs):
        compute.add_creature(f"c{i}", o, hebbian_enabled=True)
    # дефолт = False, повторный False не меняет ничего
    n1 = compute.set_higher_sfnn(False)
    assert n1 == 0
    n2 = compute.set_higher_sfnn(True)
    assert n2 == 2
    n3 = compute.set_higher_sfnn(True)
    assert n3 == 0
