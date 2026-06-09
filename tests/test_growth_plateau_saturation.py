"""§10.8 (Фрай 09.06) — fix ROOT1 (noise-robust плато) + ROOT2 (граф-derived
насыщение). Погода даёт всплески intrinsic → 55-подряд near-floor никогда не
набиралось (75.5% near-floor, но 25% всплесков сбивали consecutive-счётчик) →
плато не объявлялось → рост не стартовал. Фикс: плато = ДОЛЯ near-floor ≥ φ⁻¹
(устойчиво к всплескам). + насыщение связей выводим ИЗ ГРАФА (restart-robust, не
stale-флаг, который сбрасывался рестартами A/B → дедлок tissue-петли).
"""
from __future__ import annotations

import sys
import types
from collections import deque
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402


def _c():
    return LocalColonyCompute(device="cpu")


def _gene(src, enabled=True, tgt="cerebellum"):
    return types.SimpleNamespace(source_role=src, target_role=tgt, enabled=enabled)


# ── ROOT 1: noise-robust плато (доля near-floor) ────────────────────────

def test_plateau_fires_at_convergence_despite_bursts():
    # Сошёлся: ~75% у floor + 25% всплесков (как live погода) → плато ВСЁ РАВНО
    # объявляется (доля 0.755 ≥ φ⁻¹ 0.618), хотя 55-подряд бы не набралось.
    c = _c()
    floor = 0.0064
    hist = deque(maxlen=c._growth_intr_window)
    # 233 сэмпла: 76% near-floor (≤ floor*1.236=0.0079), 24% всплески 0.0089
    for i in range(233):
        hist.append(floor if i % 100 < 76 else 0.0089)
    c._growth_intr_hist["a"] = hist
    assert c._intrinsic_plateaued("a") is True   # доля near-floor проходит порог


def test_plateau_not_fires_during_learning():
    # Активное обучение: intrinsic высокий/переменный, мало near-floor → НЕ плато.
    c = _c()
    floor = 0.005
    hist = deque(maxlen=c._growth_intr_window)
    for i in range(233):
        hist.append(floor if i % 100 < 30 else 0.05)   # лишь 30% near-floor
    c._growth_intr_hist["a"] = hist
    assert c._intrinsic_plateaued("a") is False


def test_plateau_needs_full_window():
    c = _c()
    hist = deque([0.005] * 10, maxlen=c._growth_intr_window)   # < 55 сэмплов
    c._growth_intr_hist["a"] = hist
    assert c._intrinsic_plateaued("a") is False   # мало истории


def test_plateau_frac_phi_inv():
    c = _c()
    assert abs(c._growth_plateau_frac - (1.618033988749895 ** -1)) < 1e-9  # φ⁻¹


# ── ROOT 2: граф-derived насыщение связей ───────────────────────────────

def test_connections_saturated_all_connected():
    c = _c()
    org = types.SimpleNamespace(
        tissues={"t1": types.SimpleNamespace(role="cerebellum"),
                 "t2": types.SimpleNamespace(role="motor"),
                 "t3": types.SimpleNamespace(role="insula")},
        tissue_topology_genes=[_gene("motor"), _gene("insula")])
    assert c._connections_saturated(org) is True   # все не-cerebellum → cerebellum


def test_connections_not_saturated_missing_edge():
    c = _c()
    org = types.SimpleNamespace(
        tissues={"t1": types.SimpleNamespace(role="cerebellum"),
                 "t2": types.SimpleNamespace(role="motor"),
                 "t3": types.SimpleNamespace(role="insula")},
        tissue_topology_genes=[_gene("motor")])   # insula НЕ подключена
    assert c._connections_saturated(org) is False


def test_connections_saturated_disabled_gene_doesnt_count():
    c = _c()
    org = types.SimpleNamespace(
        tissues={"t1": types.SimpleNamespace(role="cerebellum"),
                 "t2": types.SimpleNamespace(role="motor")},
        tissue_topology_genes=[_gene("motor", enabled=False)])   # disabled
    assert c._connections_saturated(org) is False


def test_connections_saturated_no_cerebellum():
    c = _c()
    org = types.SimpleNamespace(
        tissues={"t2": types.SimpleNamespace(role="motor")},
        tissue_topology_genes=[])
    assert c._connections_saturated(org) is False   # нет cerebellum → не насыщено