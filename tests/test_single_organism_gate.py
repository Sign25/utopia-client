"""Single-organism pivot (01.06.2026, ТЗ e3cc81b §1, этап 2).

Флаг single_organism гейтит КОЛОНИАЛЬНЫЕ механики — код сохранён (Зоопарк
Эпохи 2), но под флагом не исполняется. Тестируем:
  - set_single_organism: идемпотентность + возврат значения + дефолт False
  - детект пары при флаге → empty list, нет add_creature, нет emit
  - _assign_species при флаге → species_id не назначается
  - выключенный флаг (колониальный режим) — репродукция работает как раньше
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")
pytest.importorskip("core.workbench")
pytest.importorskip("storage.norg")

_PROD_SEED = Path.home() / ".utopia-client" / "seed.norg"
if not _PROD_SEED.exists():
    pytest.skip(f"production seed not present at {_PROD_SEED}",
                allow_module_level=True)


class _MockEmbodiedClient:
    def __init__(self, send_success: bool = True):
        self.send_success = send_success
        self.sent_payloads: list[dict] = []

    def send_state(self, payload: dict) -> bool:
        self.sent_payloads.append(payload)
        return self.send_success


@pytest.fixture
def compute_with_two_zodchiy(tmp_path, monkeypatch):
    """LocalColonyCompute с двумя готовыми к репродукции зодчими (energy>порог)."""
    from utopia_client import config as cli_config
    monkeypatch.setattr(cli_config, "colonies_dir",
                        lambda: tmp_path / "colonies")

    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    c = LocalColonyCompute(device="cpu")
    orgs = load_founders(_PROD_SEED, 2)
    for i, o in enumerate(orgs):
        c.add_creature(f"parent-{i}", o, lineage="zodchiy")
    for cid in c.biochem:
        c.biochem[cid].energy = 600.0  # > MIN_ENERGY_FOR_REPRO ≈ 500
    return c


# ── set_single_organism ──────────────────────────────────────────────

def test_default_is_colony_mode(compute_with_two_zodchiy):
    """Дефолт — колониальный режим (флаг False)."""
    c = compute_with_two_zodchiy
    assert c._single_organism is False


def test_set_single_organism_returns_and_toggles(compute_with_two_zodchiy):
    c = compute_with_two_zodchiy
    assert c.set_single_organism(True) is True
    assert c._single_organism is True
    # Идемпотентность — повторный вызов с тем же значением безопасен.
    assert c.set_single_organism(True) is True
    assert c._single_organism is True
    assert c.set_single_organism(False) is False
    assert c._single_organism is False


# ── Гейт репродукции ─────────────────────────────────────────────────

def test_single_organism_blocks_reproduction(compute_with_two_zodchiy):
    """Флаг ВКЛ → готовая пара не размножается, child не добавлен, emit не идёт."""
    c = compute_with_two_zodchiy
    c.set_single_organism(True)
    mock = _MockEmbodiedClient()
    n_before = len(c.organisms)
    born = c.detect_and_emit_mate_pairs(world_tick=10000, embodied_client=mock)
    assert born == []
    assert len(c.organisms) == n_before      # нет нового ребёнка
    assert mock.sent_payloads == []          # нет newborn_announce
    assert c._pending_newborn_envelopes == {}


def test_colony_mode_still_reproduces(compute_with_two_zodchiy):
    """Контроль: при выключенном флаге репродукция работает (механика цела)."""
    c = compute_with_two_zodchiy
    assert c._single_organism is False
    mock = _MockEmbodiedClient()
    born = c.detect_and_emit_mate_pairs(world_tick=10000, embodied_client=mock)
    assert len(born) == 1
    assert len(mock.sent_payloads) == 1


# ── Гейт speciation ──────────────────────────────────────────────────

def test_single_organism_skips_speciation(tmp_path, monkeypatch):
    """Флаг ВКЛ до add_creature → species_id особи не назначается."""
    from utopia_client import config as cli_config
    monkeypatch.setattr(cli_config, "colonies_dir",
                        lambda: tmp_path / "colonies")
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    c = LocalColonyCompute(device="cpu")
    c.set_single_organism(True)
    org = load_founders(_PROD_SEED, 1)[0]
    c.add_creature("adam", org, lineage="zodchiy")
    assert "adam" not in c.species_id
