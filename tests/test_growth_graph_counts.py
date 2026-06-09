"""§6 рост (Шеф 09.06): KPI «связей закреплено» из ГРАФА, не из волатильного
счётчика. Persistence-фикс (0.13.41) защитил будущее, но историческое значение
_growth_kept потерялось на первом рестарте до фикса → /stats показывал «0», хотя
связи живы (topo_active=19). Корень: счётчик event-based, обнуляется. Фикс: Адам
стартует topo=0 → ВСЕ topology_genes выращены петлёй → enabled=закреплено,
disabled=отвергнуто, ИЗ графа (durable, переживает рестарт, всегда = реальность).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402


def _gene(enabled):
    return types.SimpleNamespace(enabled=enabled, target_role="cerebellum")


def _org(enabled_n, disabled_n):
    genes = [_gene(True)] * enabled_n + [_gene(False)] * disabled_n
    return types.SimpleNamespace(tissue_topology_genes=genes, generation=0)


def test_graph_counts_single_org():
    c = LocalColonyCompute(device="cpu")
    c.organisms["a"] = _org(enabled_n=19, disabled_n=51)   # реальный Адам: 70 генов
    kept, reverted = c._growth_graph_counts(c.organisms["a"])
    assert kept == 19 and reverted == 51                   # закреплено / отвергнуто


def test_graph_counts_aggregate():
    c = LocalColonyCompute(device="cpu")
    c.organisms["a"] = _org(19, 51)
    kept, reverted = c._growth_graph_counts()              # org=None → агрегат
    assert kept == 19 and reverted == 51


def test_graph_counts_survives_counter_reset():
    # Главный кейс: волатильный _growth_kept=0 (сброшен рестартом), но граф жив →
    # из графа всё равно 19 закреплено (не врёт «0»).
    c = LocalColonyCompute(device="cpu")
    c._growth_kept = 0          # «сброшен рестартом»
    c._growth_reverted = 0
    c.organisms["a"] = _org(19, 51)
    kept, reverted = c._growth_graph_counts()
    assert kept == 19 and reverted == 51   # граф != счётчик


def test_graph_counts_empty():
    c = LocalColonyCompute(device="cpu")
    c.organisms["a"] = types.SimpleNamespace(tissue_topology_genes=[], generation=0)
    assert c._growth_graph_counts() == (0, 0)
    c.organisms["b"] = types.SimpleNamespace(generation=0)   # нет атрибута
    assert c._growth_graph_counts(c.organisms["b"]) == (0, 0)
