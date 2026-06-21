"""Client-side Z7 biochemistry layer (Body Migration Этап 3a Phase 2).

**Архитектурный принцип:** прямо re-use чистых функций из
`environment.biochemistry` (server-side, NeuroCore package). Этот модуль
содержит ТОЛЬКО:
  - `ClientCreatureBiochem` — dataclass со всеми атрибутами `CreatureState`
    которых касается биохимия (duck-typing). Позволяет вызывать
    `environment.biochemistry.apply_*` / `decay_step` / `compute_mental_break`
    напрямую без модификации.
  - `BiochemTickContext` — минимальный duck-type `world.config` для
    `decay_step` (`max_energy`, `max_hydration`,
    `biochem_dopamine_decay_per_tick`).
  - `_FakeWorld` — обёртка под `world` параметр `decay_step`.
  - `make_default(genome=None)` — фабрика инициализации новой особи
    (Адам-Зодчий или потомок).

**Math equivalence:** автоматическая — тот же код P40 и client.

**Phase 2 §4 ТЗ Body Migration (commit 8238b06 в DiviSci/NeuroCore):**
8 веществ + 7 baseline-генов темперамента + mental_break + hysteresis.
В Phase 3+ добавятся ткани, Hebbian, NEAT-мутации.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("utopia_client.biochemistry")


# ---------------------------------------------------------------------------
# Server-side defaults (re-export через DEFAULT_BASELINES если доступен)
# ---------------------------------------------------------------------------

# Локальная копия — на случай отсутствия `environment.biochemistry` в среде.
# При нормальной работе берётся свежее из server module (см. _import_server).
_LOCAL_DEFAULT_BASELINES: dict[str, float] = {
    "cortisol": 0.0,
    "dopamine": 0.0,
    "serotonin": 50.0,
    "oxytocin": 0.0,
    "adrenaline": 0.0,
    "glucose": 50.0,
    "fatigue": 0.0,
}


def _import_server_defaults() -> dict[str, float]:
    """Подтянуть DEFAULT_BASELINES из server biochemistry если доступен.

    Возвращает локальную копию при отсутствии neurocore-package — это
    позволяет dev-venv'у без neurocore работать (например для импорта
    тестов которые мокают всё).
    """
    try:
        from environment.biochemistry import DEFAULT_BASELINES  # type: ignore
        return dict(DEFAULT_BASELINES)
    except Exception:
        return dict(_LOCAL_DEFAULT_BASELINES)


# ---------------------------------------------------------------------------
# Client-side state holder — duck-typed под CreatureState
# ---------------------------------------------------------------------------


@dataclass
class ClientCreatureBiochem:
    """Per-cid биохимическое состояние на клиенте.

    Дублирует поля `environment.world.CreatureState` которые касаются
    биохимии Z7 (и зависимостей biochemistry-функций). Не наследует и
    не импортирует `CreatureState` — это server-side dataclass с кучей
    лишних полей (positions, tissue activations, kills и т.д.).

    Используется как duck-type для прямых вызовов
    `environment.biochemistry.apply_*` / `decay_step` /
    `compute_mental_break` / `should_force_stay`.

    Не thread-safe. Caller (LocalColonyCompute.handle_tick) держит
    `_world_lock`-эквивалент на client-side.
    """

    # ── 8 эфемерных веществ (clip [0, 100]) ───────────────────────────
    cortisol: float = 0.0
    dopamine: float = 0.0
    serotonin: float = 50.0
    oxytocin: float = 0.0
    adrenaline: float = 0.0
    glucose: float = 50.0
    fatigue: float = 0.0
    histamine: float = 0.0

    # ── 7 baseline-генов темперамента ─────────────────────────────────
    baseline_cortisol: float = 0.0
    baseline_dopamine: float = 0.0
    baseline_serotonin: float = 50.0
    baseline_oxytocin: float = 0.0
    baseline_adrenaline: float = 0.0
    baseline_glucose: float = 50.0
    baseline_fatigue: float = 0.0

    # ── Mental break state ────────────────────────────────────────────
    mental_break: str = ""  # "" / "catatonic" / "berserk" / ...
    mental_break_ticks: int = 0  # hold-counter (через MENTAL_BREAK_DURATIONS)

    # ── Зависимости от мира (читаются biochemistry функциями) ─────────
    # Эти поля синхронизируются caller'ом из obs_batch перед apply'ями.
    # energy=500: matched genesis P40 WorldConfig.initial_energy. Раньше
    # было 100 (default) → reproduce_threshold=500 недостижим. Phase 4 fix 0.11.2.
    energy: float = 500.0
    # stamina 4-шкальная модель, шаг 1a (Фрай/Хьюберт §15, lockstep с server
    # CreatureState.hp/max_hp): HP-бак. На 1a hp = energy ЗЕРКАЛО (dormant,
    # бит-в-бит; разъезд в 1b: death/damage пишут hp). server decay_step
    # (environment/biochemistry.py:459+) ставит creature.hp=energy всегда +
    # creature.max_hp=max_energy ТОЛЬКО если max_hp<=0. Поля ОБЯЗАТЕЛЬНЫ — иначе
    # новый decay_step упадёт на dataclass. max_hp дефолт=1309 (НЕ 0): client
    # decay-ctx _FakeWorld max_energy=100 (менять нельзя — сломает бит-в-бит
    # decay-нормировку), потому дефолт 1309 → guard max_hp<=0 ложен → decay_step
    # НЕ перетрёт → паритет с server max_hp=1309 (=_CLIENT_MAX_ENERGY).
    hp: float = 500.0
    max_hp: float = 1309.0
    # stamina 1b.1 lockstep (Фрай/Хьюберт §18.6): общая decay_step снимает зеркало
    # creature.hp=energy ПЕР-CREATURE по is_adam (для Адама hp authoritative; system
    # одношкальны → зеркало остаётся). Client тикает ТОЛЬКО Адама → is_adam=True.
    is_adam: bool = True
    hydration: float = 100.0
    # §energy-Scaling-Law (Фрай/Шеф 20.06, satiety=(b), lockstep CreatureState.satiety
    # cfcc0aa): сытость 0-100 — нужда (провал s=1−satiety/100 в compute_energy_from_needs).
    # ON _scaling_energy: eat/kill→satiety, метаболизм→satiety, energy=f(нужды). Дефолт
    # 100 (сыт при рождении) = зеркало server. OFF dormant → не используется.
    satiety: float = 100.0
    max_satiety: float = 100.0
    infected: bool = False
    infection_severity: float = 0.0
    pair_bond_strength: float = 0.5
    last_social_tick: int = 0

    # ── Temp-attrs (выставляются caller'ом для одного тика) ───────────
    _biochem_close_kin: int = 0
    _biochem_lone: bool = False
    _biochem_last_action: int = 4  # STAY по умолчанию

    def as_snapshot(self) -> dict[str, float]:
        """Лёгкий snapshot для diagnostics/observability.

        Возвращает только 8 веществ + mental_break (то что нужно для UI
        StatsPage и Хьюбертовой merge-логики через diag push).
        """
        return {
            "cortisol": round(self.cortisol, 2),
            "dopamine": round(self.dopamine, 2),
            "serotonin": round(self.serotonin, 2),
            "oxytocin": round(self.oxytocin, 2),
            "adrenaline": round(self.adrenaline, 2),
            "glucose": round(self.glucose, 2),
            "fatigue": round(self.fatigue, 2),
            "histamine": round(self.histamine, 2),
            "hp": round(self.hp, 2),          # stamina 1a (зеркало energy)
            "max_hp": round(self.max_hp, 2),  # stamina 1a (=1309, паритет server)
            "mental_break": self.mental_break,
        }


# ---------------------------------------------------------------------------
# Duck-type для `world.config` (нужен для decay_step)
# ---------------------------------------------------------------------------


@dataclass
class BiochemTickContext:
    """Минимальный subset `WorldConfig` для `decay_step`.

    Используется как `world.config` в вызове
    `environment.biochemistry.decay_step(creature, world)`.

    Дефолты соответствуют production значениям. Если будут нужны
    динамические — caller обновляет поля.
    """
    max_energy: float = 100.0
    max_hydration: float = 100.0
    # `DOPAMINE_DECAY_DEFAULT = 0.2` из environment/biochemistry.py.
    biochem_dopamine_decay_per_tick: float = 0.2


class _FakeWorld:
    """Duck-type под `world` параметр `decay_step`.

    `decay_step(creature, world)` обращается только к `world.config.*`
    полям выше. Этот тонкий wrapper позволяет передать `BiochemTickContext`
    под видом `world`.
    """

    def __init__(self, ctx: Optional[BiochemTickContext] = None) -> None:
        self.config = ctx if ctx is not None else BiochemTickContext()


# ---------------------------------------------------------------------------
# Фабрика — init для новой особи (Адам или потомок)
# ---------------------------------------------------------------------------


def make_default() -> ClientCreatureBiochem:
    """Дефолтная биохимия Адам-Зодчего (genesis founder).

    Baseline-гены = `DEFAULT_BASELINES` (sustains через
    `inherit_baselines` при mate-pair с шумом σ=4.0). Ephemeral веще-
    ства = baseline (через `reset_ephemeral_to_baseline` логика).

    После init у Адама нейтральный темперамент: cortisol=0, dopamine=0,
    serotonin=50, glucose=50, и т.д. Эмерджентное поведение возникает
    через дрейф baselines при наследовании.
    """
    defaults = _import_server_defaults()
    bc = ClientCreatureBiochem()
    for chem, value in defaults.items():
        setattr(bc, chem, value)
        setattr(bc, f"baseline_{chem}", value)
    # histamine init=0 (computed-mirror от infection_severity, без baseline)
    bc.histamine = 0.0
    return bc


def make_from_inheritance(
    mother: ClientCreatureBiochem,
    father: Optional[ClientCreatureBiochem] = None,
    *,
    rng=None,
) -> ClientCreatureBiochem:
    """Создать потомка с наследованием baseline-темперамента.

    Если `father` дан — половое наследование (mean ± noise).
    Если `father` None — бесполое (mean = parent ± noise).

    Использует `environment.biochemistry.inherit_baselines{,_asexual}`
    напрямую (math equivalence с server). Ephemeral веществ
    инициализирует baseline-уровнями.

    `rng` — `random.Random` (или None → global random).
    """
    try:
        from environment.biochemistry import (  # type: ignore
            inherit_baselines,
            inherit_baselines_asexual,
            reset_ephemeral_to_baseline,
        )
    except Exception as e:
        logger.warning(
            "biochemistry import failed (using local defaults): %s", e)
        return make_default()

    child = ClientCreatureBiochem()
    if father is None:
        inherit_baselines_asexual(child, mother, rng=rng)
    else:
        inherit_baselines(child, mother, father, rng=rng)
    reset_ephemeral_to_baseline(child)
    return child
