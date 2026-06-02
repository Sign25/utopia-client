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

# ── Track 2: self-observable obs (расширение восприятия) ─────────────

def test_build_self_observable():
    """4 сигнала в контрактном порядке: entropy/trace/reward/paralyzed."""
    import time
    from utopia_client.local_compute import LocalColonyCompute
    c = LocalColonyCompute(device="cpu")
    cid = "adam"
    c.entropy_ema[cid] = 0.9
    c.trace_norm_ema[cid] = 0.01
    c.reward_var_ema[cid] = 0.03
    c._paralysis_until[cid] = time.monotonic() + 5.0    # парализован
    so = c._build_self_observable(cid)
    assert len(so) == 4
    assert abs(float(so[0]) - 0.9) < 1e-6      # entropy
    assert abs(float(so[1]) - 0.01) < 1e-6     # trace_norm
    assert abs(float(so[2]) - 0.03) < 1e-6     # reward_var
    assert abs(float(so[3]) - 1.0) < 1e-6      # paralyzed


def test_upgrade_tissue_input_dim_math_equivalence():
    """[I|0]-init: input_proj(obs68) == obs64 → founding-мозг НЕ дисраптится."""
    import torch
    from utopia_client.local_compute import LocalColonyCompute
    c = LocalColonyCompute(device="cpu")
    pred = c._make_predictor_tissue()
    assert pred is not None
    assert c._upgrade_tissue_input_dim(pred, 68) is True
    assert c._upgrade_tissue_input_dim(pred, 68) is False   # идемпотентно
    assert int(pred.data_dim) == 68
    obs64 = torch.randn(1, 64)
    self4 = torch.randn(1, 4)                  # любые self-observable
    obs68 = torch.cat([obs64, self4], dim=-1)
    out = pred.input_proj(obs68)
    # passthrough первых 64 + ноль на новых 4 → math-equivalence
    assert torch.allclose(out, obs64, atol=1e-5)


def test_self_observable_predictor_integration():
    """Интеграция Track 2: enable → predictor train с obs68/target-obs64 dim-корректен."""
    import numpy as np
    import torch
    from utopia_client.local_compute import LocalColonyCompute
    c = LocalColonyCompute(device="cpu")
    cid = "adam"
    c.predictor[cid] = c._make_predictor_tissue()
    c.predictor_opt[cid] = torch.optim.Adam(
        c.predictor[cid].parameters(), lr=1e-3)
    c.entropy_ema[cid] = 0.0
    c.trace_norm_ema[cid] = 0.0
    c.reward_var_ema[cid] = 0.0
    assert c._enable_self_observable(cid) is True
    assert int(c.predictor[cid].data_dim) == 68
    obs64 = torch.randn(1, 64)
    so = c._build_self_observable(cid)                    # [4]
    obs68 = torch.from_numpy(
        np.concatenate([obs64.numpy().reshape(-1), so]).astype(np.float32)
    ).unsqueeze(0)
    assert obs68.shape == (1, 68)
    c._predictor_train_step(cid, obs64, obs68)            # шаг 1: сохранит prev
    intr = c._predictor_train_step(cid, obs64, obs68)     # шаг 2: forward(obs68)→64
    assert isinstance(intr, float)                        # dim-согласовано, не упало


def test_default_is_colony_mode(compute_with_two_zodchiy):
    """Дефолт — колониальный режим (флаг False)."""
    c = compute_with_two_zodchiy
    assert c._single_organism is False


def test_set_single_organism_revives_dead_marked(compute_with_two_zodchiy):
    """Bootstrap-race fix: включение флага оживляет dead-marked особей
    (умерли в колониальном окне до применения флага → иначе frozen)."""
    c = compute_with_two_zodchiy
    cid = next(iter(c.biochem))
    c._dead_cids.add(cid)              # «умер» в pre-flag окне
    c._paralysis_until[cid] = 999.0
    c.biochem[cid].energy = 0.0
    c.biochem[cid].cortisol = 99.5     # bug-накопленный мусорный стресс
    c.biochem[cid].serotonin = 0.0
    c.biochem[cid].mental_break = "catatonic"
    c.set_single_organism(True)
    assert cid not in c._dead_cids      # оживлён
    assert cid not in c._paralysis_until
    assert c.biochem[cid].energy == c._recovery_energy  # стартовая энергия
    assert c.biochem[cid].cortisol < 80.0   # стресс очищен → не catatonic
    assert c.biochem[cid].mental_break == ""


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
    assert c.biochem[cid].energy == c._recovery_energy  # +73 (max/φ⁶)
    assert cid not in c._dead_cids


def test_paralysis_recovery_relieves_catatonic_stress(compute_with_two_zodchiy):
    """Фрай-инвариант: recovery размыкает absorbing-catatonic (cortisol-relief)."""
    import time
    c = compute_with_two_zodchiy
    c.set_single_organism(True)
    cid = next(iter(c.biochem))
    bc = c.biochem[cid]
    bc.energy = 0.0
    bc.cortisol = 99.5                  # застрявший стресс → catatonic
    bc.serotonin = 0.0
    bc.mental_break = "catatonic"
    c._paralysis_until[cid] = time.monotonic() - 1.0    # срок истёк → recovery
    c._apply_metabolism(cid, {"step_cost_per_sec": 0.0})
    assert bc.cortisol < 80.0           # стресс облегчён → не catatonic
    assert bc.mental_break == ""        # залипание снято
    assert bc.energy == c._recovery_energy


# ── §3 mirror-контракт: projection-поля + триггер 2 (Хьюберт 2b0f3a2) ──

def test_projection_includes_paralysis_fields(compute_with_two_zodchiy):
    """build_projection_batch несёт paralyzed + paralysis_ticks_remaining."""
    import time
    c = compute_with_two_zodchiy
    c.set_single_organism(True)
    cid = next(iter(c.organisms))
    c._paralysis_until[cid] = time.monotonic() + 2.0   # активный паралич
    projs = {p["cid"]: p for p in c.build_projection_batch()}
    assert projs[cid]["paralyzed"] is True
    assert projs[cid]["paralysis_ticks_remaining"] > 0
    # Не-парализованный сосед → False/0.
    other = [x for x in c.organisms if x != cid][0]
    assert projs[other]["paralyzed"] is False
    assert projs[other]["paralysis_ticks_remaining"] == 0


def test_death_suppressed_triggers_paralysis_energy_independent(
        compute_with_two_zodchiy):
    """Триггер 2: death_suppressed → paralysis ДАЖЕ при energy>0 (Фрай)."""
    c = compute_with_two_zodchiy
    c.set_single_organism(True)
    cid = next(iter(c.biochem))
    c.biochem[cid].energy = 600.0          # здоров, НЕ голодает
    c._enter_paralysis(cid, "pvp_kill")    # как из handle_tick на death_suppressed
    assert cid in c._paralysis_until        # парализован независимо от energy
    assert cid not in c._dead_cids


def test_enter_paralysis_idempotent(compute_with_two_zodchiy):
    """Повторный _enter_paralysis в активном параличе не продлевает дедлайн."""
    c = compute_with_two_zodchiy
    c.set_single_organism(True)
    cid = next(iter(c.biochem))
    c._enter_paralysis(cid, "starved")
    d1 = c._paralysis_until[cid]
    c._enter_paralysis(cid, "pvp_kill")    # поток событий не должен «вечнить»
    assert c._paralysis_until[cid] == d1


def test_enter_paralysis_no_rearm_when_expired(compute_with_two_zodchiy):
    """Баг live 01.06: _enter_paralysis НЕ ре-армит при ИСТЁКШЕМ дедлайне —
    иначе поток death_suppressed глушил recovery (419 start / 1 recovery)."""
    import time
    c = compute_with_two_zodchiy
    c.set_single_organism(True)
    cid = next(iter(c.biochem))
    c._paralysis_until[cid] = time.monotonic() - 1.0   # ИСТЁКШИЙ дедлайн
    d_expired = c._paralysis_until[cid]
    c._enter_paralysis(cid, "starved")                 # не должен ре-армить
    assert c._paralysis_until[cid] == d_expired         # дедлайн НЕ обновлён →
    #                                                     recovery в metab снимет


# ── Гейт колониального mental_break (Фрай watch-item, §1) ─────────────

def _set_loner_biochem(bc):
    bc.oxytocin = 0.0; bc.cortisol = 5.0; bc.serotonin = 60.0
    bc.dopamine = 0.0; bc.adrenaline = 0.0; bc.fatigue = 0.0
    bc.histamine = 0.0; bc.last_social_tick = 0
    bc.mental_break = ""; bc.mental_break_ticks = 0


def test_single_organism_gates_loner(compute_with_two_zodchiy):
    """Колониальный loner гейтится под флагом (для одиночки always-true)."""
    c = compute_with_two_zodchiy
    c.set_single_organism(True)
    cid = next(iter(c.biochem))
    _set_loner_biochem(c.biochem[cid])
    c._apply_biochem_mental_break(cid, world_tick=2000)   # gap 2000>500
    assert c.biochem[cid].mental_break != "loner"          # загейчен


def test_colony_mode_allows_loner(compute_with_two_zodchiy):
    """Без флага loner ставится (механика цела для Зоопарка)."""
    c = compute_with_two_zodchiy
    assert c._single_organism is False
    cid = next(iter(c.biochem))
    _set_loner_biochem(c.biochem[cid])
    c._apply_biochem_mental_break(cid, world_tick=2000)
    assert c.biochem[cid].mental_break == "loner"


def test_single_organism_keeps_stress_mental_break(compute_with_two_zodchiy):
    """Стресс-состояния (catatonic) НЕ гейтятся — не колониальный артефакт."""
    c = compute_with_two_zodchiy
    c.set_single_organism(True)
    cid = next(iter(c.biochem))
    bc = c.biochem[cid]
    bc.cortisol = 90.0; bc.serotonin = 10.0                # >80 / <20 → catatonic
    bc.dopamine = 0.0; bc.adrenaline = 0.0; bc.fatigue = 0.0
    bc.histamine = 0.0; bc.oxytocin = 0.0
    bc.mental_break = ""; bc.mental_break_ticks = 0
    c._apply_biochem_mental_break(cid, world_tick=2000)
    assert bc.mental_break == "catatonic"                  # стресс не маскируем
