"""SFNN S6.4 — LocalCompute init для 10 базовых тканей organism graph.

Что проверяем:
  - add_creature создаёт SFNNRule для всех 10 базовых тканей
  - дефолт без genome-поля = SFNNRule.for_role(role) (ROLE_DEFAULTS)
  - genome.<role>_sfnn_rule (dict) → SFNNRule.from_dict
  - повторный add_creature идемпотентен (setdefault, broker re-announce)
  - remove_creature очищает basic_tissue_sfnn_rule
  - reset_all обнуляет basic_tissue_sfnn_steps
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


_BASIC_TISSUES = (
    "sensory", "attention", "brain", "memory", "consciousness",
    "communication", "motor", "manipulator", "digestive", "immune",
)


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


def _compute(seed_file, cid="c1"):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature(cid, org, hebbian_enabled=True)
    return compute, org


def test_basic_sfnn_storage_keys(seed_file):
    """Все 10 базовых тканей зарегистрированы в storage."""
    compute, _ = _compute(seed_file)
    for role in _BASIC_TISSUES:
        assert role in compute.basic_tissue_sfnn_rule
        assert "c1" in compute.basic_tissue_sfnn_rule[role], \
            f"role={role} cid c1 not initialised"


def test_basic_sfnn_uses_role_defaults_when_genome_field_none(seed_file):
    """Без genome.<role>_sfnn_rule — берётся ROLE_DEFAULTS.for_role(role)."""
    from core.sfnn_rule import SFNNRule, ROLE_DEFAULTS, ROLE_ALGORITHM
    compute, _ = _compute(seed_file)
    for role in _BASIC_TISSUES:
        rule = compute.basic_tissue_sfnn_rule[role]["c1"]
        defaults = ROLE_DEFAULTS[role]
        assert rule.tau == pytest.approx(defaults[0])
        assert rule.r_imm_weight == pytest.approx(defaults[1])
        assert rule.r_med_weight == pytest.approx(defaults[2])
        assert rule.r_long_weight == pytest.approx(defaults[3])
        assert rule.td_coupling == pytest.approx(defaults[4])
        assert rule.mean_eta() == pytest.approx(defaults[5])
        assert rule.algorithm == ROLE_ALGORITHM[role]


def test_basic_sfnn_reads_genome_dict(seed_file):
    """genome.sensory_sfnn_rule = dict → SFNNRule.from_dict, не дефолт."""
    from core.sfnn_rule import SFNNRule
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    # Genome у CompositeOrganism может отсутствовать — мокаем SimpleNamespace
    # с заполненным sensory_sfnn_rule.
    custom = SFNNRule.for_role("sensory")
    # Сдвинем η, чтобы отличить от дефолта.
    custom_d = custom.to_dict()
    custom_d["temperature"] = 2.3   # отличие от 1.0 дефолта (clip [0.5, 5])
    custom_d["tau"] = 7.0           # отличие от 1.0 sensory-дефолта
    # Прикручиваем поле к genome (организм не имел `sensory_sfnn_rule` атрибута).
    import types
    if not hasattr(org, "genome"):
        org.genome = types.SimpleNamespace()
    org.genome.sensory_sfnn_rule = custom_d
    compute.add_creature("c1", org)
    rule = compute.basic_tissue_sfnn_rule["sensory"]["c1"]
    assert rule.temperature == pytest.approx(2.3)
    assert rule.tau == pytest.approx(7.0)


def test_basic_sfnn_invalid_dict_falls_back_to_defaults(seed_file):
    """genome.<role>_sfnn_rule с битыми данными → SFNNRule.for_role(role)."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    import types
    if not hasattr(org, "genome"):
        org.genome = types.SimpleNamespace()
    org.genome.attention_sfnn_rule = {"garbage": "not_a_rule"}
    # add_creature не должен упасть, fallback на defaults.
    compute.add_creature("c1", org)
    from core.sfnn_rule import ROLE_DEFAULTS
    rule = compute.basic_tissue_sfnn_rule["attention"]["c1"]
    defaults = ROLE_DEFAULTS["attention"]
    # tau из дефолтов (21.0) — значит fallback сработал.
    assert rule.tau == pytest.approx(defaults[0])


def test_basic_sfnn_add_creature_idempotent(seed_file):
    """Повторный add_creature для того же cid не перетирает правило."""
    compute, org = _compute(seed_file)
    # Сохраняем ссылку до повторного add_creature.
    rule_before = compute.basic_tissue_sfnn_rule["memory"]["c1"]
    rule_before.tau = 99.0  # эмулируем "эволюционировавшее" правило
    compute.add_creature("c1", org, hebbian_enabled=True)
    rule_after = compute.basic_tissue_sfnn_rule["memory"]["c1"]
    assert rule_after is rule_before, "setdefault должен сохранить старое правило"
    assert rule_after.tau == pytest.approx(99.0)


def test_basic_sfnn_remove_creature_clears(seed_file):
    """remove_creature вычищает все 10 ролей."""
    compute, _ = _compute(seed_file)
    for role in _BASIC_TISSUES:
        assert "c1" in compute.basic_tissue_sfnn_rule[role]
    compute.remove_creature("c1")
    for role in _BASIC_TISSUES:
        assert "c1" not in compute.basic_tissue_sfnn_rule[role], \
            f"role={role} not cleaned"


def test_basic_sfnn_steps_counter_starts_at_zero(seed_file):
    """basic_tissue_sfnn_steps[role] = 0 для всех ролей сразу после init."""
    compute, _ = _compute(seed_file)
    for role in _BASIC_TISSUES:
        assert compute.basic_tissue_sfnn_steps[role] == 0


def test_basic_sfnn_reset_all_zeros_steps(seed_file):
    """reset_all сбрасывает счётчики 10 базовых тканей."""
    compute, _ = _compute(seed_file)
    for role in _BASIC_TISSUES:
        compute.basic_tissue_sfnn_steps[role] = 42
    compute.reset_all()
    for role in _BASIC_TISSUES:
        assert compute.basic_tissue_sfnn_steps[role] == 0


def test_basic_sfnn_default_flag_on(seed_file):
    """S6.11: _basic_sfnn_default стартует включённым — миграция завершена."""
    compute, _ = _compute(seed_file)
    assert compute._basic_sfnn_default is True


def test_basic_sfnn_genome_attr_synth_when_missing(seed_file):
    """add_creature ставит basic_tissue_sfnn_enabled на synth-genome."""
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    # У CompositeOrganism нет genome — local_compute приклеит SimpleNamespace.
    if hasattr(org, "genome"):
        del org.genome
    compute.add_creature("c1", org)
    assert hasattr(org.genome, "basic_tissue_sfnn_enabled")
    # S6.11: дефолт True.
    assert org.genome.basic_tissue_sfnn_enabled is True
