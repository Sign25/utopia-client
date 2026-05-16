"""Z7.i.d (Зодчий, 16.05.2026) — клиентский plumbing уникальных тканей Зодчего.

Что проверяем (без forward/SFNN apply — это Z3 hookup в следующих подфазах):
  - `add_creature(lineage="zodchiy")` создаёт 3 sidecar-ткани
    (cerebellum/amygdala/episodic).
  - `add_creature(lineage="wanderer")` (default) НЕ создаёт ни одной из
    Зодчий-тканей — обратная совместимость не нарушена.
  - SFNNRule.default() инициализируется в `zodchiy_extra_sfnn_rule` для
    каждой из трёх тканей.
  - `remove_creature` чистит все три store'а + sfnn_rule.
  - `reset_all` обнуляет step-counter и зачищает sfnn_rule по всем тканям.

Это plumbing-уровень: forward/apply-step не дёргаются, smoke-тиков нет.
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
    monkeypatch.setenv("UTOPIA_WANDERER_SEED_PATH",
                       str(tmp_path / "client_seed.norg"))
    monkeypatch.setenv("UTOPIA_COLONIES_DIR", str(tmp_path / "colonies"))
    client_seed = tmp_path / "client_seed.norg"
    client_seed.write_bytes(seed_path.read_bytes())
    from utopia_client import seed_loader as cli_loader
    importlib.reload(cli_loader)
    return client_seed


def _make_compute(seed_file):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    return compute, org


def test_wanderer_default_has_no_zodchiy_tissues(seed_file):
    """Default lineage="wanderer" → ни одной Зодчий-ткани."""
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org)  # default
    assert compute.cerebellum.get("c1") is None
    assert compute.amygdala.get("c1") is None
    assert compute.episodic.get("c1") is None


def test_zodchiy_lineage_creates_three_tissues(seed_file):
    """lineage="zodchiy" → cerebellum/amygdala/episodic созданы."""
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    assert compute.cerebellum.get("c1") is not None
    assert compute.amygdala.get("c1") is not None
    assert compute.episodic.get("c1") is not None


def test_zodchiy_sfnn_rule_populated(seed_file):
    """SFNNRule.default() записан для каждой из трёх Зодчий-тканей."""
    from utopia_client.local_compute import _ZODCHIY_EXTRA_TISSUES
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    for t in _ZODCHIY_EXTRA_TISSUES:
        assert "c1" in compute.zodchiy_extra_sfnn_rule[t]
        rule = compute.zodchiy_extra_sfnn_rule[t]["c1"]
        # SFNNRule.default() — у него `coeffs: dict[synapse → SFNNSynapseCoeffs]`
        # + общие τ/R3/td_coupling. Проверим хотя бы один coeff-блок.
        assert hasattr(rule, "coeffs") and rule.coeffs
        any_syn = next(iter(rule.coeffs.values()))
        for attr in ("A", "B", "C", "D", "eta"):
            assert hasattr(any_syn, attr), f"{t}: SFNNSynapseCoeffs missing {attr}"
        for attr in ("tau", "td_coupling"):
            assert hasattr(rule, attr), f"{t}: SFNNRule missing {attr}"


def test_zodchiy_sfnn_rule_not_created_for_wanderer(seed_file):
    """Wanderer не получает Zodchiy-sfnn-rule (но dict-ключи tissues живут)."""
    from utopia_client.local_compute import _ZODCHIY_EXTRA_TISSUES
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org)
    for t in _ZODCHIY_EXTRA_TISSUES:
        # Хранилище инициализировано ({}) в __init__, но cid в нём нет.
        assert "c1" not in compute.zodchiy_extra_sfnn_rule[t]


def test_remove_creature_clears_zodchiy_state(seed_file):
    """remove_creature вычищает 3 sidecar + sfnn_rule по всем тканям."""
    from utopia_client.local_compute import _ZODCHIY_EXTRA_TISSUES
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    compute.remove_creature("c1")
    assert compute.cerebellum.get("c1") is None
    assert compute.amygdala.get("c1") is None
    assert compute.episodic.get("c1") is None
    for t in _ZODCHIY_EXTRA_TISSUES:
        assert "c1" not in compute.zodchiy_extra_sfnn_rule[t]


def test_reset_all_clears_zodchiy_counters(seed_file):
    """reset_all обнуляет step-counter и sfnn_rule-dict по всем Зодчий-тканям."""
    from utopia_client.local_compute import _ZODCHIY_EXTRA_TISSUES
    compute, org = _make_compute(seed_file)
    compute.add_creature("c1", org, lineage="zodchiy")
    # Имитируем "наработку" step-counter, чтобы убедиться что reset обнуляет.
    for t in _ZODCHIY_EXTRA_TISSUES:
        compute.zodchiy_extra_sfnn_steps[t] = 42
    compute.reset_all()
    for t in _ZODCHIY_EXTRA_TISSUES:
        assert compute.zodchiy_extra_sfnn_steps[t] == 0
        assert compute.zodchiy_extra_sfnn_rule[t] == {}


def test_zodchiy_extra_tissues_const_is_three(seed_file):
    """Хранители-constant: 3 ткани, имена точны (исключает опечатки)."""
    from utopia_client.local_compute import _ZODCHIY_EXTRA_TISSUES
    assert _ZODCHIY_EXTRA_TISSUES == ("cerebellum", "amygdala", "episodic")
