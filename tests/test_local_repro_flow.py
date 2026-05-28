"""Phase 4 F integration: detect_and_emit_mate_pairs + handle_newborn_announce_ack.

Mock embodied_client. Тестируем:
  - detect 0 pairs (нет ready) → empty list, нет add_creature
  - detect 1 pair → child added + emit + pending registered
  - ack accepted → traits applied + pending cleared
  - ack rejected → remove_creature + pending cleared
  - ack unknown cid → no-op
  - cooldown enforced — повторный detect blocked
  - emit failed → rollback child creation
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
    """Mock EmbodiedWSClient с управляемым send_state."""

    def __init__(self, send_success: bool = True):
        self.send_success = send_success
        self.sent_payloads: list[dict] = []

    def send_state(self, payload: dict) -> bool:
        self.sent_payloads.append(payload)
        return self.send_success


@pytest.fixture
def compute_with_two_zodchiy(tmp_path, monkeypatch):
    """LocalColonyCompute с двумя готовыми к репродукции зодчими."""
    # isolate memory dir в tmp_path
    from utopia_client import config as cli_config
    monkeypatch.setattr(cli_config, "colonies_dir",
                        lambda: tmp_path / "colonies")

    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders

    c = LocalColonyCompute(device="cpu")
    orgs = load_founders(_PROD_SEED, 2)
    for i, o in enumerate(orgs):
        c.add_creature(f"parent-{i}", o, lineage="zodchiy")

    # Post-hotfix 28.05: energy-only gate. Default Адам energy=100,
    # ниже server threshold ≈500 → надо поднять.
    for cid in c.biochem:
        bc = c.biochem[cid]
        bc.energy = 600.0  # > MIN_ENERGY_FOR_REPRO ≈ 500

    return c


# ────────────────────────────────────────────────────────────────────
# detect_and_emit_mate_pairs
# ────────────────────────────────────────────────────────────────────

def test_no_ready_no_emit(compute_with_two_zodchiy):
    """Если none ready — empty list, нет добавления child."""
    c = compute_with_two_zodchiy
    # Make оба unready (energy < 500)
    for cid in c.biochem:
        c.biochem[cid].energy = 100.0
    mock = _MockEmbodiedClient()
    born = c.detect_and_emit_mate_pairs(world_tick=10000, embodied_client=mock)
    assert born == []
    assert mock.sent_payloads == []


def test_ready_pair_emits_and_registers_pending(compute_with_two_zodchiy):
    c = compute_with_two_zodchiy
    mock = _MockEmbodiedClient(send_success=True)
    n_before = len(c.organisms)
    born = c.detect_and_emit_mate_pairs(world_tick=10000, embodied_client=mock)

    assert len(born) == 1
    child_cid = born[0]
    # Child добавлен в colony
    assert child_cid in c.organisms
    assert len(c.organisms) == n_before + 1
    # Envelope отправлен
    assert len(mock.sent_payloads) == 1
    env = mock.sent_payloads[0]
    assert env["type"] == "newborn_announce"
    assert env["cid"] == child_cid
    assert set(env["parent_cids"]) == {"parent-0", "parent-1"}
    assert env["lineage"] == "zodchiy"
    # Pending зарегистрирован
    assert child_cid in c._pending_newborn_envelopes
    # Cooldown started для обоих parents
    assert "parent-0" in c._last_mate_tick
    assert "parent-1" in c._last_mate_tick


def test_emit_failed_rollbacks_child(compute_with_two_zodchiy):
    """Если embodied_client.send_state вернул False — child удаляется."""
    c = compute_with_two_zodchiy
    mock = _MockEmbodiedClient(send_success=False)
    n_before = len(c.organisms)
    born = c.detect_and_emit_mate_pairs(world_tick=10000, embodied_client=mock)
    assert born == []
    # Не оставили orphan child
    assert len(c.organisms) == n_before
    # Pending пуст
    assert c._pending_newborn_envelopes == {}


def test_no_embodied_client_no_emit(compute_with_two_zodchiy):
    """embodied_client=None — нет emit + rollback."""
    c = compute_with_two_zodchiy
    n_before = len(c.organisms)
    born = c.detect_and_emit_mate_pairs(world_tick=10000, embodied_client=None)
    assert born == []
    assert len(c.organisms) == n_before


def test_cooldown_blocks_second_detect(compute_with_two_zodchiy):
    """После успешного emit — повторный detect blocked cooldown=89."""
    c = compute_with_two_zodchiy
    mock = _MockEmbodiedClient()

    # First detect
    born1 = c.detect_and_emit_mate_pairs(world_tick=10000, embodied_client=mock)
    assert len(born1) == 1

    # Second detect через 50 ticks (< 89 cooldown)
    born2 = c.detect_and_emit_mate_pairs(world_tick=10050, embodied_client=mock)
    assert born2 == []


def test_cooldown_lifted_after_threshold(compute_with_two_zodchiy):
    """Через cooldown_ticks+ — пара снова может репродуцировать."""
    c = compute_with_two_zodchiy
    mock = _MockEmbodiedClient()
    c.detect_and_emit_mate_pairs(world_tick=10000, embodied_client=mock)
    # Через 100 ticks (> 89 cooldown)
    born2 = c.detect_and_emit_mate_pairs(world_tick=10100, embodied_client=mock)
    assert len(born2) == 1


# ────────────────────────────────────────────────────────────────────
# handle_newborn_announce_ack
# ────────────────────────────────────────────────────────────────────

def test_ack_accepted_applies_traits(compute_with_two_zodchiy):
    c = compute_with_two_zodchiy
    mock = _MockEmbodiedClient()
    born = c.detect_and_emit_mate_pairs(world_tick=10000, embodied_client=mock)
    child_cid = born[0]
    assert child_cid in c._pending_newborn_envelopes

    ack = {
        "type": "newborn_announce_ack",
        "cid": child_cid,
        "accepted": True,
        "reason": None,
        "traits": {
            "vision_radius": 12, "smell_radius": 22, "attack_radius": 2,
            "move_speed": 1, "attack_power": 3, "armor": 1,
            "efficiency": 6, "camel": 11, "diet_gene": 0.45,
        },
        "ts_server": 1000.0,
    }
    ok = c.handle_newborn_announce_ack(ack)
    assert ok is True
    # Pending очищен
    assert child_cid not in c._pending_newborn_envelopes
    # Traits применены к organism (setattr)
    child = c.organisms[child_cid]
    assert child.vision_radius == 12
    assert child.smell_radius == 22
    assert child.diet_gene == 0.45


def test_ack_rejected_removes_creature(compute_with_two_zodchiy):
    c = compute_with_two_zodchiy
    mock = _MockEmbodiedClient()
    born = c.detect_and_emit_mate_pairs(world_tick=10000, embodied_client=mock)
    child_cid = born[0]
    assert child_cid in c.organisms

    ack = {
        "type": "newborn_announce_ack",
        "cid": child_cid,
        "accepted": False,
        "reason": "parent_not_alive",
        "traits": None,
        "ts_server": 1000.0,
    }
    ok = c.handle_newborn_announce_ack(ack)
    assert ok is True
    # Cleanup произошёл
    assert child_cid not in c.organisms
    assert child_cid not in c._pending_newborn_envelopes


def test_ack_unknown_cid_returns_false(compute_with_two_zodchiy):
    """Ack для unknown cid (не в pending) — no-op."""
    c = compute_with_two_zodchiy
    ack = {
        "type": "newborn_announce_ack",
        "cid": "never-seen-uuid",
        "accepted": True,
        "traits": {"vision_radius": 5},
        "ts_server": 1.0,
    }
    ok = c.handle_newborn_announce_ack(ack)
    assert ok is False  # not in pending


def test_ack_missing_cid_returns_false(compute_with_two_zodchiy):
    c = compute_with_two_zodchiy
    ok = c.handle_newborn_announce_ack({"type": "newborn_announce_ack"})
    assert ok is False


def test_ack_non_dict_returns_false(compute_with_two_zodchiy):
    c = compute_with_two_zodchiy
    ok = c.handle_newborn_announce_ack("not a dict")  # type: ignore
    assert ok is False


# ────────────────────────────────────────────────────────────────────
# End-to-end cycle
# ────────────────────────────────────────────────────────────────────

def test_full_cycle_detect_emit_ack_accepted(compute_with_two_zodchiy):
    """Полный цикл: detect → emit → P40 ack → traits applied → child remains."""
    c = compute_with_two_zodchiy
    mock = _MockEmbodiedClient()
    born = c.detect_and_emit_mate_pairs(world_tick=10000, embodied_client=mock)
    assert len(born) == 1
    child_cid = born[0]

    # Mock P40 ack
    ack = {
        "type": "newborn_announce_ack",
        "cid": child_cid,
        "accepted": True,
        "traits": {"vision_radius": 7, "smell_radius": 20,
                   "attack_radius": 1, "move_speed": 1, "attack_power": 1,
                   "armor": 0, "efficiency": 5, "camel": 10, "diet_gene": 0.5},
        "ts_server": 1.0,
    }
    c.handle_newborn_announce_ack(ack)
    # Child осталась в colony
    assert child_cid in c.organisms
    assert child_cid not in c._pending_newborn_envelopes


def test_full_cycle_detect_emit_ack_rejected(compute_with_two_zodchiy):
    """Полный цикл: detect → emit → P40 reject → child removed."""
    c = compute_with_two_zodchiy
    mock = _MockEmbodiedClient()
    born = c.detect_and_emit_mate_pairs(world_tick=10000, embodied_client=mock)
    child_cid = born[0]

    ack = {
        "type": "newborn_announce_ack",
        "cid": child_cid,
        "accepted": False,
        "reason": "cid_collision",
        "traits": None,
        "ts_server": 1.0,
    }
    c.handle_newborn_announce_ack(ack)
    # Cleanup
    assert child_cid not in c.organisms
