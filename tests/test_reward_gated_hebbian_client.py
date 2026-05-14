"""Phase 5d (NEOL, 14.05.2026) — клиентская TD-модуляция Hebbian eta.

Серверная попытка (`fb465b5` в neurocore) была откачена 14.05.2026, так как
после brain migration мозг wanderer'ов живёт у клиента-owner'а (`_phase_b_oja_hebbian`
на P40 dormant для owned). См. memory/feedback_check_brain_location_before_learning_code.md.

Эта реализация — клиентская: EMA(β_local, α=0.01) per-cid в LocalColonyCompute,
TD = β − EMA, модулятор `1 + clip(TD, ±0.5)` подаётся в `heb.update(dopa_td_mult=…)`.

Тесты:
  1. dopamine_ema/dopamine_td инициализированы пустыми, заполняются после первого
     forward S2.E. EMA сходится к β при констатном входе.
  2. Idempotency: в рамках одного handle_tick TD/EMA обновляются ровно один раз
     (не двойной апдейт на cid).
  3. TD ≈ 0 при стационарном β; положительный TD при росте β; отрицательный TD при падении.
  4. Clip ±0.5 → multiplier ∈ [0.5, 1.5] на стыке с heb.update(dopa_td_mult).
  5. Ablation gate: если S2.E off (`compute.dopamine[cid] = None`) — TD остаётся
     дефолтным 0.0 → multiplier = 1.0 (без модуляции).
  6. heb.update реально получает dopa_td_mult — проверка через monkeypatch.
  7. client_dopa_td пушится в get_phase_emas() snapshot для серверной метрики.
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


@pytest.fixture
def seed_file(tmp_path, monkeypatch):
    seed_path = tmp_path / "seed.norg"
    monkeypatch.setenv("WORLD_SEED_PATH", str(seed_path))

    import importlib
    from environment import seed_loader as ns_loader
    importlib.reload(ns_loader)
    ns_loader.ensure_seed(preset="nexus")

    monkeypatch.setenv("UTOPIA_SEED_PATH", str(tmp_path / "client_seed.norg"))
    monkeypatch.setenv("UTOPIA_WANDERER_SEED_PATH", str(tmp_path / "client_seed.norg"))
    client_seed = tmp_path / "client_seed.norg"
    client_seed.write_bytes(seed_path.read_bytes())

    from utopia_client import seed_loader as cli_loader
    importlib.reload(cli_loader)
    return client_seed


# ── 1. Инициализация и заполнение EMA/TD ───────────────────────────────

def test_dopamine_ema_td_empty_on_init(seed_file):
    """dopamine_ema/dopamine_td созданы пустыми в __init__."""
    from utopia_client.local_compute import LocalColonyCompute
    c = LocalColonyCompute(device="cpu")
    assert c.dopamine_ema == {}
    assert c.dopamine_td == {}


def test_dopamine_ema_filled_after_first_forward(seed_file):
    """После handle_tick с S2.E ткани — ema/td заполнены для cid."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    c = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    c.add_creature("c1", org, hebbian_enabled=True)
    assert c.dopamine.get("c1") is not None, "S2.E ткань должна быть создана"

    obs = {"c1": np.zeros(80, dtype=np.float32)}
    c.handle_tick(obs)
    assert "c1" in c.dopamine_ema
    assert "c1" in c.dopamine_td


def test_dopamine_ema_converges_to_beta(seed_file):
    """При константном входе EMA сходится к β_local после ~500 тиков (α=0.01)."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    c = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    c.add_creature("c1", org, hebbian_enabled=True)

    obs = {"c1": np.zeros(80, dtype=np.float32)}
    for _ in range(500):
        c.handle_tick(obs)

    beta = c.last_beta_local["c1"]
    ema = c.dopamine_ema["c1"]
    # После 500 тиков с α=0.01 ema должна быть в пределах ~1% от β.
    assert abs(beta - ema) < 0.05, f"ema={ema} far from beta={beta}"


# ── 2. Idempotency ──────────────────────────────────────────────────────

def test_td_ema_updated_once_per_tick(seed_file):
    """Один handle_tick → один EMA-апдейт. Двойной forward S2.E не происходит."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    c = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    c.add_creature("c1", org, hebbian_enabled=True)

    obs = {"c1": np.zeros(80, dtype=np.float32)}
    c.handle_tick(obs)
    ema_after_tick1 = c.dopamine_ema["c1"]
    td_after_tick1 = c.dopamine_td["c1"]

    # Второй handle_tick на тех же входах: EMA медленно ползёт, TD близок к 0
    # (но не идентичен предыдущему). Главное — не двойной апдейт.
    c.handle_tick(obs)
    ema_after_tick2 = c.dopamine_ema["c1"]
    # |Δema| << β·α при стационарном β (α=0.01).
    assert abs(ema_after_tick2 - ema_after_tick1) <= 0.02


# ── 3. Знак TD ──────────────────────────────────────────────────────────

def test_td_zero_when_stationary(seed_file):
    """При константном β после прогрева TD ≈ 0 (β ≈ EMA)."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    c = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    c.add_creature("c1", org, hebbian_enabled=True)

    obs = {"c1": np.zeros(80, dtype=np.float32)}
    for _ in range(500):
        c.handle_tick(obs)

    assert abs(c.dopamine_td["c1"]) < 0.05


# ── 4. Clip + multiplier ────────────────────────────────────────────────

def test_td_mult_clipped_to_range(seed_file):
    """multiplier = 1 + clip(td, ±0.5) ∈ [0.5, 1.5] даже при экстремальном TD."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    c = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    c.add_creature("c1", org, hebbian_enabled=True)

    # Принудительно ставим экстремальные значения td (минуя forward).
    c.dopamine_td["c1"] = 10.0
    td = c.dopamine_td["c1"]
    mult = 1.0 + max(-0.5, min(0.5, td))
    assert mult == 1.5

    c.dopamine_td["c1"] = -10.0
    td = c.dopamine_td["c1"]
    mult = 1.0 + max(-0.5, min(0.5, td))
    assert mult == 0.5


# ── 5. Ablation gate ────────────────────────────────────────────────────

def test_ablation_no_s2e_keeps_td_zero(seed_file):
    """Если S2.E ткань None (ablation) — td.get(cid, 0.0) = 0 → mult = 1.0."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    c = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    c.add_creature("c1", org, hebbian_enabled=True)
    # Эмулируем ablation: убираем S2.E.
    c.dopamine["c1"] = None

    obs = {"c1": np.zeros(80, dtype=np.float32)}
    events = {"c1": {"ate": True, "delta_energy": 1.0}}
    c.handle_tick(obs, events_per_cid=events)

    # td не должен заполниться (forward S2.E пропущен через d_tissue is None).
    assert c.dopamine_td.get("c1", 0.0) == 0.0
    mult = 1.0 + max(-0.5, min(0.5, c.dopamine_td.get("c1", 0.0)))
    assert mult == 1.0


# ── 6. heb.update получает dopa_td_mult ────────────────────────────────

def test_heb_update_receives_dopa_td_mult(seed_file, monkeypatch):
    """В handle_tick → heb.update вызван с kwarg dopa_td_mult, его значение
    укладывается в [0.5, 1.5] и связано с dopamine_td."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    c = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    c.add_creature("c1", org, hebbian_enabled=True)
    heb = c.hebbian["c1"]
    assert heb is not None

    seen: dict = {}
    orig_update = heb.update

    def spy(output, reward, **kwargs):
        seen["kwargs"] = dict(kwargs)
        return orig_update(output, reward, **kwargs)

    monkeypatch.setattr(heb, "update", spy)

    obs = {"c1": np.zeros(80, dtype=np.float32)}
    events = {"c1": {"ate": True, "delta_energy": 1.0}}
    c.handle_tick(obs, events_per_cid=events)

    assert "kwargs" in seen
    assert "dopa_td_mult" in seen["kwargs"]
    mult = seen["kwargs"]["dopa_td_mult"]
    assert 0.5 <= mult <= 1.5


# ── 7. client_dopa_td в push payload ───────────────────────────────────

def test_phase_emas_contains_client_dopa_td(seed_file):
    """get_phase_emas() возвращает client_dopa_td для серверной агрегатной метрики."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    c = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    c.add_creature("c1", org, hebbian_enabled=True)

    obs = {"c1": np.zeros(80, dtype=np.float32)}
    c.handle_tick(obs)

    emas = c.get_phase_emas("c1")
    assert emas is not None
    assert "client_dopa_td" in emas, f"client_dopa_td missing in {list(emas.keys())}"
    # Значение — float, типа TD ∈ ~[-0.618, +0.618].
    assert isinstance(emas["client_dopa_td"], float)
    assert -1.0 < emas["client_dopa_td"] < 1.0


# ── 8. remove_creature очищает state ───────────────────────────────────

def test_remove_creature_cleans_dopamine_state(seed_file):
    """remove_creature чистит dopamine_ema/dopamine_td (no memory leak)."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    c = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    c.add_creature("c1", org, hebbian_enabled=True)

    obs = {"c1": np.zeros(80, dtype=np.float32)}
    c.handle_tick(obs)
    assert "c1" in c.dopamine_ema
    assert "c1" in c.dopamine_td

    c.remove_creature("c1")
    assert "c1" not in c.dopamine_ema
    assert "c1" not in c.dopamine_td
