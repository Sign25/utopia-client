"""Phase 4 этап E — mate-pair detection client-side.

ТЗ Body Migration v1.6 §Phase 4 + vision §3.2:
  Размножение между собственными Зодчими ОДНОЙ колонии.
  Cross-owner mate ЗАПРЕЩЁН (vision §3.2 — Фраева ошибка #28).

**MVP detection criteria** (упрощённый под client state — server имеет
больше context: row/col, frozen, wants_reproduce, last_reproduce_tick):

- alive (biochem.energy > MIN_ENERGY)
- not catastrophic mental_break (catatonic/depression/panic — force_STAY)
- past cooldown (cooldown_ticks с last_mate_tick)
- sex drive proxy: oxytocin > MIN_OXYTOCIN
- one organism per pair per tick (used set)
- **Implicit co-location:** все own organisms одной колонии — в одной
  client-side compute, ассумирующее «рядом». Real distance check
  делает P40 при handle_newborn_announce (reject parent_not_alive).

**Скипуем (server-side ground truth):**
- Manhattan distance — сервер reject если parent умер / spawn fail
- Close-kin (parent-child / siblings) — client не tracket pedigree,
  random pairing acceptable для Phase 4 MVP
- Cross-species — все founder Адамы species=0 (пусто), эволюции genes
  ещё нет; gate включится когда mutate_topology=True заработает

Pure function — не мутирует input. Caller хранит `last_mate_tick`
state в LocalColonyCompute и обновляет после успешного newborn_announce_ack.
"""
from __future__ import annotations

from typing import Optional

# Thresholds — могут быть пере-tuned через config
MIN_ENERGY: float = 30.0           # ниже — слабый, не готов к репродукции
MIN_OXYTOCIN: float = 30.0         # sex drive proxy (Z7 биохимия)
MIN_SEROTONIN: float = 30.0        # social readiness
DEFAULT_COOLDOWN_TICKS: int = 500  # ~16 сек @ 30 TPS
DEFAULT_MAX_PAIRS_PER_TICK: int = 1  # natural rate-limit

# Mental break states которые блокируют размножение (force_STAY territory)
BLOCKING_MENTAL_BREAK: set[str] = {"catatonic", "depression", "panic"}


def is_reproduction_ready(
    biochem,
    *,
    min_energy: float = MIN_ENERGY,
    min_oxytocin: float = MIN_OXYTOCIN,
    min_serotonin: float = MIN_SEROTONIN,
) -> bool:
    """Проверка одного organism: готов ли к репродукции по биохимии.

    Не учитывает cooldown — это делает caller через `_last_mate_tick`.
    """
    energy = float(getattr(biochem, "energy", 0.0))
    if energy < min_energy:
        return False
    oxytocin = float(getattr(biochem, "oxytocin", 0.0))
    if oxytocin < min_oxytocin:
        return False
    serotonin = float(getattr(biochem, "serotonin", 0.0))
    if serotonin < min_serotonin:
        return False
    mb_state = str(getattr(biochem, "mental_break", "") or "").lower()
    if mb_state in BLOCKING_MENTAL_BREAK:
        return False
    return True


def detect_mate_pairs(
    biochems_dict: dict,
    last_mate_tick: dict[str, int],
    world_tick: int,
    *,
    cooldown_ticks: int = DEFAULT_COOLDOWN_TICKS,
    max_pairs: int = DEFAULT_MAX_PAIRS_PER_TICK,
    min_energy: float = MIN_ENERGY,
    min_oxytocin: float = MIN_OXYTOCIN,
    min_serotonin: float = MIN_SEROTONIN,
) -> list[tuple[str, str]]:
    """Найти mate-pairs среди own colony organisms.

    Args:
        biochems_dict: {cid: ClientCreatureBiochem} — all own organisms
        last_mate_tick: {cid: world_tick_of_last_repro} — caller state
        world_tick: current tick
        cooldown_ticks: минимум тиков с last mate
        max_pairs: лимит pairs за один tick (rate-limit)
        min_*: тhreshold override

    Returns:
        List of (mother_cid, father_cid) tuples. Mother = алфавитно меньший cid.
        Каждый organism в максимум одной паре. Empty list если нет ready candidates.
    """
    # Step 1: фильтр готовых
    candidates: list[str] = []
    for cid, bc in biochems_dict.items():
        if not is_reproduction_ready(
            bc, min_energy=min_energy,
            min_oxytocin=min_oxytocin, min_serotonin=min_serotonin,
        ):
            continue
        # Cooldown check
        last_tick = int(last_mate_tick.get(cid, 0))
        if world_tick - last_tick < cooldown_ticks:
            continue
        candidates.append(cid)

    if len(candidates) < 2:
        return []

    # Step 2: deterministic pairing (сортировка для repeatability)
    candidates.sort()
    pairs: list[tuple[str, str]] = []
    used: set[str] = set()

    # Greedy pairing — берём pairs последовательно, без cross-species/kin
    # check (см. module docstring — server reject если что не так).
    for cid_a in candidates:
        if cid_a in used or len(pairs) >= max_pairs:
            continue
        for cid_b in candidates:
            if cid_b <= cid_a or cid_b in used:
                continue
            # mother = алфавитно меньший cid (для stability)
            pairs.append((cid_a, cid_b))
            used.add(cid_a)
            used.add(cid_b)
            break

    return pairs


def detect_asexual_candidates(
    biochems_dict: dict,
    last_mate_tick: dict[str, int],
    world_tick: int,
    *,
    cooldown_ticks: int = DEFAULT_COOLDOWN_TICKS,
    max_births: int = DEFAULT_MAX_PAIRS_PER_TICK,
    min_energy: float = MIN_ENERGY + 20.0,   # для asexual нужно больше энергии
    min_oxytocin: float = MIN_OXYTOCIN,
    min_serotonin: float = MIN_SEROTONIN,
) -> list[str]:
    """Asexual reproduction candidates (через REPRODUCE-action).

    Caller вызывает только если organism сам выбрал REPRODUCE через
    motor sampling — это его волевое действие. Здесь мы лишь подтверждаем
    что биохимия позволяет.
    """
    candidates: list[str] = []
    for cid, bc in biochems_dict.items():
        if not is_reproduction_ready(
            bc, min_energy=min_energy,
            min_oxytocin=min_oxytocin, min_serotonin=min_serotonin,
        ):
            continue
        last_tick = int(last_mate_tick.get(cid, 0))
        if world_tick - last_tick < cooldown_ticks:
            continue
        candidates.append(cid)
        if len(candidates) >= max_births:
            break
    return candidates


def mark_mate_event(
    last_mate_tick: dict[str, int],
    cids: list[str],
    world_tick: int,
) -> None:
    """Обновить cooldown trackers после успешного reproduce.

    Caller (LocalColonyCompute) вызывает после получения successful
    `newborn_announce_ack`.
    """
    for cid in cids:
        last_mate_tick[cid] = int(world_tick)
