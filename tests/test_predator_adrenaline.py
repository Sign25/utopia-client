"""Predator-аффорданс v0.1 (Фрай 11.06): pred_prox (obs[61]) → adrenaline спайк
→ оживляет мёртвую ось. ТРАНЗИЕНТ: decay_step −2/тик гасит после побега.
"""
from __future__ import annotations
import sys, types
from pathlib import Path
import pytest
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
pytest.importorskip("torch")
from utopia_client.local_compute import LocalColonyCompute  # noqa: E402
from utopia_client.biochemistry import make_default  # noqa: E402


def _c():
    c = LocalColonyCompute(device="cpu")
    c.organisms["a"] = types.SimpleNamespace(generation=0)
    c.biochem["a"] = make_default()
    return c


def test_pred_prox_spikes_adrenaline():
    c = _c()
    c.biochem["a"].adrenaline = 0.0       # мёртвая ось
    c._last_pred_prox["a"] = 1.0          # хищник вплотную
    c._apply_biochem_decay("a")
    assert c.biochem["a"].adrenaline >= 50.0   # ожила (pred_prox·80, после decay)


def test_no_spike_below_gate():
    c = _c()
    c.biochem["a"].adrenaline = 0.0
    c._last_pred_prox["a"] = 0.05         # < gate 0.15 (шум)
    c._apply_biochem_decay("a")
    assert c.biochem["a"].adrenaline == 0.0    # не спайкаем


def test_adrenaline_transient_decays_when_predator_gone():
    c = _c()
    c.biochem["a"].adrenaline = 80.0      # был спайк
    c._last_pred_prox["a"] = 0.0          # хищник ушёл
    c._apply_biochem_decay("a")
    assert c.biochem["a"].adrenaline < 80.0    # decay гасит (транзиент)


def test_spike_scales_with_proximity():
    c = _c()
    c.biochem["a"].adrenaline = 0.0
    c._last_pred_prox["a"] = 0.5          # средняя угроза
    c._apply_biochem_decay("a")
    a_mid = c.biochem["a"].adrenaline
    assert 20.0 <= a_mid <= 45.0          # ~0.5·80 (после decay) — слабее чем вплотную


# ── Часть 2: adrenaline → flee speed_boost (контракт Хьюберта) ──────────

def _ws(adr=None):
    import utopia_client.ws_client as wsm
    ws = wsm.ColonyWSClient(server="https://e.com", token="t",
                            colony_name="cheef", client_version="0.0.0",
                            estimated_population=0)
    if adr is not None:
        c = _c(); c.biochem["a"].adrenaline = adr
        ws.compute = c
    return ws


def test_flee_boost_scales_with_adrenaline():
    assert _ws(80.0)._flee_speed_boost("a") == 3   # сильная реакция → отрыв
    assert _ws(50.0)._flee_speed_boost("a") == 2   # средняя → паритет
    assert _ws(20.0)._flee_speed_boost("a") == 1   # слабая → хищник догоняет


def test_flee_boost_no_compute():
    ws = _ws()
    ws.compute = None
    assert ws._flee_speed_boost("a") == 0
