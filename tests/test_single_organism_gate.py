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


# ── bias_scale (срез 2) ──────────────────────────────────────────────

def test_single_organism_freezes_bias_scale():
    """set_single_organism(True) → bias_scale=0 (автономный мотор)."""
    from utopia_client.local_compute import LocalColonyCompute
    c = LocalColonyCompute(device="cpu")
    c._bias_scale = 0.8  # как будто untrained-колония
    c.set_single_organism(True)
    assert c._bias_scale == 0.0


def test_single_organism_skips_bias_curriculum():
    """Под флагом популяционный annealing не двигает bias_scale."""
    from utopia_client.local_compute import LocalColonyCompute
    c = LocalColonyCompute(device="cpu")
    c.set_single_organism(True)
    c._bias_scale = 0.7                     # вручную, после флага
    c._last_window = {"ratio": 0.0}         # health<1 → обычно bias += 0.1
    c._bias_last_update_tick = 0
    c._update_bias_curriculum(world_tick=5000)
    assert c._bias_scale == 0.7             # annealing заглушён, не дрейфует


def test_colony_mode_bias_curriculum_runs():
    """Контроль: без флага annealing двигает bias_scale (механика цела)."""
    from utopia_client.local_compute import LocalColonyCompute
    c = LocalColonyCompute(device="cpu")
    assert c._single_organism is False
    c._bias_scale = 0.5
    c._last_window = {"ratio": 0.0}         # health<1 → bias += 0.1
    c._bias_last_update_tick = 0
    c._update_bias_curriculum(world_tick=5000)
    assert c._bias_scale > 0.5              # сдвинулся вверх


# ── newborn-instinct (срез 2) ────────────────────────────────────────

def test_single_organism_instinct_noop():
    """Под флагом _apply_newborn_instinct не трогает logits."""
    from utopia_client.local_compute import LocalColonyCompute
    c = LocalColonyCompute(device="cpu")
    c.set_single_organism(True)
    c._birth_tick["adam"] = 1000            # был бы свежим (instinct=1)
    logits = [0.0] * 16
    c._apply_newborn_instinct("adam", logits, world_tick=1000,
                              on_flora=True, carried_food=3)
    assert logits == [0.0] * 16             # ноль изменений


def test_colony_mode_instinct_boosts():
    """Контроль: без флага свежий newborn получает GATHER-boost."""
    from utopia_client.local_compute import LocalColonyCompute
    c = LocalColonyCompute(device="cpu")
    assert c._single_organism is False
    c._birth_tick["baby"] = 1000            # age=0 → instinct=1
    logits = [0.0] * 16
    c._apply_newborn_instinct("baby", logits, world_tick=1000,
                              on_flora=True, carried_food=3)
    assert logits[13] > 0.0                 # GATHER подкручен


# ── snapshot_elite durability порог (срез 2, Фрай ОК#1) ───────────────

def test_snapshot_elite_min_alive_threshold(compute_with_two_zodchiy, tmp_path):
    """min_alive=1 снимает elite при одном живом; min_alive=4 — нет (n=2<4)."""
    c = compute_with_two_zodchiy
    elite_dir = tmp_path / "elite"
    # n=2 живых: порог 4 → не снимает (колониальное допущение)
    assert c.snapshot_elite(elite_dir, min_alive=4) == 0
    # порог 1 (single-режим) → снимает → durability восстановлена
    n = c.snapshot_elite(elite_dir, min_alive=1)
    assert n >= 1


# ── §3 paralysis вместо death-spiral (этап 3) ────────────────────────

def test_paralysis_instead_of_death(compute_with_two_zodchiy):
    """single_organism + energy≤0 → паралич, НЕ смерть."""
    c = compute_with_two_zodchiy
    c.set_single_organism(True)
    cid = next(iter(c.biochem))
    c.biochem[cid].energy = 0.0
    c._apply_metabolism(cid, {"step_cost_per_sec": 0.0})
    assert cid in c._paralysis_until          # парализован
    assert cid not in c._dead_cids            # НЕ мёртв
    assert c._deaths_by_cause.get("starvation", 0) == 0


def test_colony_mode_still_dies(compute_with_two_zodchiy):
    """Контроль: без флага energy≤0 → смерть (механика цела)."""
    c = compute_with_two_zodchiy
    assert c._single_organism is False
    cid = next(iter(c.biochem))
    c.biochem[cid].energy = 0.0
    c._apply_metabolism(cid, {"step_cost_per_sec": 0.0})
    assert cid in c._dead_cids                # умер
    assert cid not in c._paralysis_until


def test_paralysis_forces_stay(compute_with_two_zodchiy):
    """Пока паралич не снят — motor=STAY (не движется)."""
    import time
    c = compute_with_two_zodchiy
    c.set_single_organism(True)
    cid = next(iter(c.biochem))
    c._paralysis_until[cid] = time.monotonic() + 10.0   # активный паралич
    out = {cid: {"action": 13, "target_id": None}}      # хотел GATHER
    c._maybe_force_stay(cid, out)
    from utopia_client.local_compute import STAY
    assert out[cid]["action"] == STAY


def test_paralysis_recovery_grants_energy(compute_with_two_zodchiy):
    """После N паралич снимается + recovery-грант энергии."""
    import time
    c = compute_with_two_zodchiy
    c.set_single_organism(True)
    cid = next(iter(c.biochem))
    c.biochem[cid].energy = 0.0
    c._paralysis_until[cid] = time.monotonic() - 1.0    # срок истёк
    c._apply_metabolism(cid, {"step_cost_per_sec": 0.0})
    assert cid not in c._paralysis_until                # снят
    assert c.biochem[cid].energy == c._recovery_energy  # +45 (max/φ⁷)
    assert cid not in c._dead_cids
