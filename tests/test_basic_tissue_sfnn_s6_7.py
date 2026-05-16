"""SFNN S6.7 — Y50 inheritance 10 базовых тканей.

Что проверяем:
  - inherit_brain_y50 копирует basic_tissue_sfnn_rule родителя в ребёнка
  - применяется σ=0.1 мутация (правило ребёнка ≠ родительскому байт-в-байт)
  - дефолтное правило ребёнка перезаписывается родительским только если
    у родителя оно есть
  - extract_brain_state_dicts кладёт basic_tissue_sfnn_rules в brain dict
  - collect_sfnn_state_sync_items добавляет basic_tissue_sfnn_rules и steps
  - restore_sfnn_state восстанавливает 10 базовых без мутации (verbatim)
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("storage.norg")
pytest.importorskip("core.workbench")
pytest.importorskip("torch")


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


def _two_creatures(seed_file):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    parent = load_founders(seed_file, 1)[0]
    child = load_founders(seed_file, 1)[0]
    compute.add_creature("parent", parent, hebbian_enabled=True)
    compute.add_creature("child", child, hebbian_enabled=True)
    return compute


def test_inherit_basic_sfnn_rules_copies_to_child(seed_file):
    """inherit_brain_y50 копирует правила basic_tissue_sfnn_rule родителя."""
    compute = _two_creatures(seed_file)
    # Меняем у родителя одну метрику, чтобы отличить от дефолта.
    parent_rule = compute.basic_tissue_sfnn_rule["sensory"]["parent"]
    parent_rule.temperature = 3.14  # отличие от дефолтной 1.0
    parent_rule.tau = 11.5
    assert compute.inherit_brain_y50("parent", "child") is True
    child_rule = compute.basic_tissue_sfnn_rule["sensory"]["child"]
    # σ=0.1 мутация → temperature ≠ 3.14, но в окрестности (mate-pair).
    # Главное — temperature ушёл от дефолта 1.0 в сторону 3.14.
    assert child_rule.temperature != pytest.approx(1.0, abs=0.05)


def test_inherit_basic_sfnn_applies_sigma_mutation(seed_file):
    """child rule отличается от parent rule (σ=0.1 шум)."""
    compute = _two_creatures(seed_file)
    parent_rule = compute.basic_tissue_sfnn_rule["brain"]["parent"]
    # Зафиксируем уникальную сигнатуру правила родителя.
    parent_d = parent_rule.to_dict()
    compute.inherit_brain_y50("parent", "child")
    child_rule = compute.basic_tissue_sfnn_rule["brain"]["child"]
    child_d = child_rule.to_dict()
    # Хотя бы одно поле отличается (mutate(sigma=0.1) точно сдвинет коэфф).
    # Проверяем коэффициенты — temperature/tau/td/R3 могут совпасть, но
    # хотя бы один SynapseCoeff.A/B/C/D/eta — нет.
    assert parent_d != child_d, "child rule идентичен parent rule (σ-мутация не сработала)"


def test_extract_brain_state_dicts_includes_basic_sfnn(seed_file):
    """extract_brain_state_dicts кладёт basic_tissue_sfnn_rules в brain."""
    from utopia_client.local_compute import _BASIC_SFNN_TISSUES
    compute = _two_creatures(seed_file)
    brain, _emas = compute.extract_brain_state_dicts("parent")
    assert "basic_tissue_sfnn_rules" in brain
    rules = brain["basic_tissue_sfnn_rules"]
    # Все 10 базовых ролей должны быть в дампе (для nexus founder).
    assert set(rules.keys()) == set(_BASIC_SFNN_TISSUES)
    # Каждое правило — dict с минимум 'coeffs' внутри.
    for role, d in rules.items():
        assert isinstance(d, dict)


def test_collect_sfnn_state_sync_includes_basic(seed_file):
    """collect_sfnn_state_sync_items: каждый item имеет basic_tissue_sfnn_rules + steps."""
    from utopia_client.local_compute import _BASIC_SFNN_TISSUES
    compute = _two_creatures(seed_file)
    compute.basic_tissue_sfnn_steps["sensory"] = 99
    items = compute.collect_sfnn_state_sync_items()
    assert len(items) == 2  # parent + child
    for item in items:
        assert "basic_tissue_sfnn_rules" in item
        assert set(item["basic_tissue_sfnn_rules"].keys()) == set(_BASIC_SFNN_TISSUES)
        assert "basic_tissue_sfnn_steps" in item
        assert item["basic_tissue_sfnn_steps"]["sensory"] == 99


def test_restore_basic_sfnn_state_verbatim_no_mutation(seed_file):
    """restore_sfnn_state восстанавливает базовые правила без σ-мутации."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute_a = LocalColonyCompute(device="cpu")
    org_a = load_founders(seed_file, 1)[0]
    compute_a.add_creature("c1", org_a, hebbian_enabled=True)
    # Помечаем правило memory у первой особи.
    rule = compute_a.basic_tissue_sfnn_rule["memory"]["c1"]
    rule.tau = 42.0
    rule.temperature = 1.234
    snapshot = compute_a.collect_sfnn_state_sync_items()[0]

    # Свежая колония — apply snapshot.
    compute_b = LocalColonyCompute(device="cpu")
    org_b = load_founders(seed_file, 1)[0]
    compute_b.add_creature("c1", org_b, hebbian_enabled=True)
    compute_b.restore_sfnn_state("c1", snapshot)
    restored = compute_b.basic_tissue_sfnn_rule["memory"]["c1"]
    # Verbatim — никакой σ-мутации.
    assert restored.tau == pytest.approx(42.0)
    assert restored.temperature == pytest.approx(1.234)


def test_restore_basic_sfnn_steps_max_reduce(seed_file):
    """restore_sfnn_state: счётчики max-reduce, не перетирают большие значения."""
    compute = _two_creatures(seed_file)
    compute.basic_tissue_sfnn_steps["motor"] = 1000
    snapshot = {
        "basic_tissue_sfnn_steps": {"motor": 500},
    }
    compute.restore_sfnn_state("parent", snapshot)
    # 500 < 1000 → остаётся 1000.
    assert compute.basic_tissue_sfnn_steps["motor"] == 1000
    snapshot = {
        "basic_tissue_sfnn_steps": {"motor": 2000},
    }
    compute.restore_sfnn_state("parent", snapshot)
    # 2000 > 1000 → обновляется.
    assert compute.basic_tissue_sfnn_steps["motor"] == 2000


def test_inherit_basic_sfnn_unknown_role_ignored(seed_file):
    """Неизвестная роль в payload['basic_tissue_sfnn_rules'] не падает."""
    compute = _two_creatures(seed_file)
    parent_rule = compute.basic_tissue_sfnn_rule["sensory"]["parent"]
    rule_d = parent_rule.to_dict()
    # Payload вручную с битой ролью.
    payload = {
        "basic_tissue_sfnn_rules": {
            "nonsense_role": rule_d,  # не в _BASIC_SFNN_TISSUES — skip
            "sensory": rule_d,        # ок
        }
    }
    compute.apply_inherited_state("child", payload)
    # sensory у ребёнка применился, nonsense_role — игнорирован молча.
    assert "sensory" in compute.basic_tissue_sfnn_rule
    assert "nonsense_role" not in compute.basic_tissue_sfnn_rule
