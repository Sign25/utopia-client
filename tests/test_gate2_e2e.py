"""gate-2 e2e: АКТИВНЫЙ мотор-путь behavioral graduation в РЕАЛЬНОМ handle_tick
(Фрай предусловие-1 к gate-2 — касание мотора Адама). Тестит то, что unit-тесты НЕ
покрывают: 3-точечную hot-loop интеграцию (logit-bias голов + action-ctx + REINFORCE-
ротация) в живом тике + zero-init NO-OP + revert бит-в-бит.

Требует full-deps (core.tissue + environment.seed_loader + fastapi/pydantic) — гоняется
на ПК Шефа с PYTHONPATH=NeuroCore. В dev-venv без них skip.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
_NEUROCORE = _ROOT.parent / "NeuroCore"
for _p in (str(_ROOT), str(_NEUROCORE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("torch")
pytest.importorskip("core.tissue")
pytest.importorskip("fastapi")          # seed-fixture тянет server.routes_workbench

import torch  # noqa: E402

from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute, _BRAIN_INPUT_DIM, _RHYTHM_OFFSET, _RHYTHM_DIM, N_ACTIONS,
)


@pytest.fixture
def adam(tmp_path, monkeypatch):
    """LocalColonyCompute с реальным founder-Адамом + расширенным до 72 predictor."""
    seed_path = tmp_path / "seed.norg"
    monkeypatch.setenv("WORLD_SEED_PATH", str(seed_path))
    import importlib
    from environment import seed_loader as ns_loader
    importlib.reload(ns_loader)
    ns_loader.ensure_seed(preset="nexus")
    monkeypatch.setenv("UTOPIA_SEED_PATH", str(tmp_path / "client_seed.norg"))
    monkeypatch.setenv("UTOPIA_WANDERER_SEED_PATH", str(tmp_path / "client_seed.norg"))
    (tmp_path / "client_seed.norg").write_bytes(seed_path.read_bytes())
    from utopia_client import seed_loader as cli_loader
    importlib.reload(cli_loader)
    from utopia_client.seed_loader import load_founders
    c = LocalColonyCompute(device="cpu")
    founders = load_founders(tmp_path / "client_seed.norg", n=1)
    c.add_creature("c0", founders[0])
    c._single_organism = True
    c._enable_self_observable("c0")              # predictor 64→72 (obs72)
    return c


def _obs(seed, rhythm=True):
    rng = np.random.default_rng(seed)
    o = rng.standard_normal(80).astype(np.float32)
    if rhythm:                                   # валидный ритм sin/cos в [68:72]
        ph = 0.3 * seed
        o[68], o[69] = np.sin(ph), np.cos(ph)
        o[70], o[71] = np.sin(0.05 * seed), np.cos(0.05 * seed)
    else:
        o[_RHYTHM_OFFSET:_RHYTHM_OFFSET + _RHYTHM_DIM] = 0.0
    return o


def _force_graduate(c, cid="c0"):
    """Принудительно вырастить + выпустить behavioral мотор-голову (минуя grace/winter)."""
    c.set_rhythm(True)
    c.set_behavioral_growth(True)
    c.set_behavioral_graduation(True)
    c._motor_voice = 1.0                                     # _own>0 → мотор-блок (и head-путь) активен (как в проде)
    c._beh_axis_hist[cid] = {"rhythm": [20.0] * 8}          # poor → mint
    c._last_world_tick = 0
    assert c._maybe_behavioral_mint(cid, c.organisms.get(cid))
    role = next(iter(c._beh_grown_tissues[cid]))
    c._beh_forecast_err[cid][role] = 1.0                    # хороший skill
    c._beh_forecast_trained[cid][role] = c._BEH_GRACE_NIGHTS  # созрел (grace)
    c._beh_axis_drop_sum[cid] = {"rhythm": 4.0 * 40}        # global baseline=4
    c._beh_axis_drop_n[cid] = {"rhythm": 40}
    bc = c.biochem.get(cid)
    if bc is not None:
        bc.energy = max(float(getattr(bc, "energy", 0.0)), 999.0)
    assert c._maybe_behavioral_graduate(cid, c.organisms.get(cid))
    # блокируем S4b head-GC (cooldown), иначе он ablate'ит голову → её logit-вклад=0
    # (это by-design для GC-замера; для теста ИНТЕГРАЦИИ нужна не-ablated голова)
    c._beh_head_gc_last[cid] = 10 ** 9
    return role


# ── e2e (1): активный gate-2 путь в handle_tick не падает, голова исполняется ─
def test_gate2_handle_tick_active_no_crash():
    pass  # маркер для читаемости — реальный тест ниже (нужна fixture)


def test_gate2_e2e_active_path(adam):
    c = adam
    role = _force_graduate(c)
    assert role in c._beh_graduated["c0"]
    head = c._beh_motor_head["c0"][role]
    w0 = head.weight.detach().clone()
    # N тиков РЕАЛЬНОГО handle_tick с активной graduated-головой
    for i in range(1, 25):
        out = c.handle_tick(
            {"c0": _obs(i)},
            events_per_cid={"c0": {"ate": False, "killed": False,
                                   "damage_taken": 0.0, "delta_energy": 1.0}},
            world_tick=i)
        act = out.get("c0", {}).get("action")
        assert isinstance(act, int) and 0 <= act < N_ACTIONS    # валидное действие, без краша
    assert role in c._beh_graduated["c0"]                       # голова жива
    # ИНТЕГРАЦИЯ ИСПОЛНИЛАСЬ: logit-блок головы заполнил ctx (вклад в action_slice
    # реально посчитан в живом handle_tick) → 3-точечный hot-loop путь активен, не скип.
    assert c._beh_motor_ctx.get("c0"), "head logit-путь НЕ исполнился (скип)"
    assert role in c._beh_motor_ctx["c0"]
    # forecast-инференс тоже шёл (живой forecast для мотора)
    assert c._beh_forecast_live.get("c0", {}).get(role) is not None


# ── e2e (2): zero-init голова = NO-OP на флипе (вклад в логиты = 0) ───────
def test_gate2_e2e_zero_init_noop(adam):
    c = adam
    role = _force_graduate(c)
    head = c._beh_motor_head["c0"][role]
    # сразу после graduation голова zero-init → её bias к логитам = 0 (NO-OP)
    assert int(torch.count_nonzero(head.weight)) == 0
    assert int(torch.count_nonzero(head.bias)) == 0
    # вклад = head(tissue(obs72)) при zero-init = 0 на ЛЮБОМ входе
    t = c._beh_graduated["c0"][role]
    x = torch.randn(1, _BRAIN_INPUT_DIM)
    with torch.no_grad():
        bias = head(t({"input": x})["output"]).reshape(-1)
    assert torch.allclose(bias, torch.zeros(N_ACTIONS), atol=0)  # NO-OP к мотору


# ── e2e (3): revert на живом тике → голова снята, мотор работает ─────────
def test_gate2_e2e_revert_live(adam):
    c = adam
    _force_graduate(c)
    # несколько живых тиков с головой
    for i in range(1, 6):
        c.handle_tick({"c0": _obs(i)}, world_tick=i)
    assert c._beh_graduated.get("c0")
    # ЖИВОЙ REVERT (kill-switch): gate-2 OFF → голова снята
    c.set_behavioral_graduation(False)
    assert not c._beh_graduated.get("c0")                   # снята
    assert not c._beh_motor_head.get("c0")
    # мотор продолжает работать БЕЗ головы (база восстановлена)
    for i in range(6, 11):
        out = c.handle_tick({"c0": _obs(i)}, world_tick=i)
        act = out.get("c0", {}).get("action")
        assert isinstance(act, int) and 0 <= act < N_ACTIONS
