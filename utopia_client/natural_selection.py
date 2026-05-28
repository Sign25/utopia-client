"""Phase 4 этап G — естественный отбор tracking на клиенте.

ТЗ Body Migration v1.6 §Phase 4 + vision §3.1 (вариант D):
  Колония Зодчих регулируется через **естественный отбор**, а не cap.
  Размножение продолжается на пределе, server-side давление
  (`flora_density`, `TelomerePhase decay`, `safe_grove_mult`) убивает
  слабых первыми. Client просто **наблюдает** — не вмешивается в жизнь:
  не убивает явно, не замедляет tick, не шлёт hint серверу.

**Этот модуль — pure tracking + diag (вариант A confirmed Фраем 28.05).**

Что считаем (минимальный MVP — только данные что точно есть на client):
  - **energy** (низкая → слабый)
  - **cortisol** (хронически высокий → слабый, хронический стресс)
  - **fatigue** (высокая → слабый)
  - **glucose** (низкая → слабый, голод)
  - **infection** (infected или infection_severity → слабый)
  - **mental_break** (catatonic / depression → слабый)

Чего пока нет на client (skipped в MVP):
  - age (P40-side через CreatureState.age)
  - adaptability_score (P40-side если есть)
  - behavioral_repertoire (можно вычислить через action history, todo)
  - fitness kills/mates (P40-side)

Расширим scope когда метрики появятся в client state.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# ────────────────────────────────────────────────────────────────────
# Weights — vision §3.1 «вариант D» комплексные критерии
# ────────────────────────────────────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "energy_low": 1.0,        # критично для жизни
    "cortisol_chronic": 0.6,  # хронический стресс
    "fatigue_high": 0.4,      # переутомление
    "glucose_low": 0.8,       # голод
    "infection": 1.2,         # острая болезнь
    "mental_break": 1.5,      # катастрофическое состояние
}

# Пороги "слабого" по веществам (vision §3.1 baseline 50, exceptional)
LOW_ENERGY_THRESHOLD: float = 30.0       # < 30 / 100 → слабый
HIGH_CORTISOL_THRESHOLD: float = 70.0    # > 70 / 100 → хронический стресс
HIGH_FATIGUE_THRESHOLD: float = 70.0     # > 70 / 100 → переутомление
LOW_GLUCOSE_THRESHOLD: float = 25.0      # < 25 / 100 → голодный
INFECTION_SEVERITY_FLOOR: float = 0.1    # severity > 0.1 → counted


# ────────────────────────────────────────────────────────────────────
# Per-cid weakness factors
# ────────────────────────────────────────────────────────────────────

@dataclass
class WeaknessFactors:
    """Компоненты weakness_score одного организма.

    Все компоненты в [0, 1]: 0 = идеально / 1 = критично.
    `weakness_score` — взвешенная сумма (НЕ нормированная в [0,1] —
    в крайнем плохом случае может > 5).
    """
    cid: str
    energy: float           # 0..1 (1 = umirayushij)
    cortisol_chronic: float # 0..1
    fatigue: float          # 0..1
    glucose_low: float      # 0..1
    infection: float        # 0..1
    mental_break: float     # 0 / 0.5 / 1.0 в зависимости от state
    weakness_score: float = 0.0


# ────────────────────────────────────────────────────────────────────
# Pure scoring functions
# ────────────────────────────────────────────────────────────────────

def _norm_low(value: float, threshold: float, floor: float = 0.0) -> float:
    """Нормировать «низкое значение плохо»: если value <= floor → 1,
    если value >= threshold → 0, между — линейно."""
    if value <= floor:
        return 1.0
    if value >= threshold:
        return 0.0
    return (threshold - value) / (threshold - floor)


def _norm_high(value: float, threshold: float, ceil: float = 100.0) -> float:
    """Нормировать «высокое значение плохо»: если value >= ceil → 1,
    если value <= threshold → 0."""
    if value >= ceil:
        return 1.0
    if value <= threshold:
        return 0.0
    return (value - threshold) / (ceil - threshold)


def _mental_break_severity(state: str) -> float:
    """Тяжесть mental_break state. Catatonic — наихудший
    (force_STAY = смертельно опасно). Остальные — medium."""
    state = (state or "").lower()
    if state == "catatonic":
        return 1.0
    if state in ("depression", "panic"):
        return 0.7
    if state in ("berserk", "mania", "obsession", "withdrawal", "loner"):
        return 0.4
    if not state:
        return 0.0
    return 0.3  # unknown state → medium


def score_organism(
    biochem,
    *,
    weights: Optional[dict[str, float]] = None,
    cid: str = "",
) -> WeaknessFactors:
    """Compute weakness_score для одного организма.

    Args:
        biochem: `ClientCreatureBiochem` или duck-type с теми же полями
        weights: переопределить дефолтные веса
        cid: ID организма (для удобства trace)

    Returns:
        WeaknessFactors с пер-компонентным breakdown + total score.
    """
    w = weights or DEFAULT_WEIGHTS

    energy_val = float(getattr(biochem, "energy", 100.0))
    cortisol_val = float(getattr(biochem, "cortisol", 0.0))
    fatigue_val = float(getattr(biochem, "fatigue", 0.0))
    glucose_val = float(getattr(biochem, "glucose", 100.0))
    infected = bool(getattr(biochem, "infected", False))
    inf_severity = float(getattr(biochem, "infection_severity", 0.0))
    mb_state = str(getattr(biochem, "mental_break", "") or "")

    energy_n = _norm_low(energy_val, LOW_ENERGY_THRESHOLD)
    cortisol_n = _norm_high(cortisol_val, HIGH_CORTISOL_THRESHOLD)
    fatigue_n = _norm_high(fatigue_val, HIGH_FATIGUE_THRESHOLD)
    glucose_n = _norm_low(glucose_val, LOW_GLUCOSE_THRESHOLD)
    infection_n = (
        min(1.0, inf_severity / 1.0) if (infected or inf_severity > INFECTION_SEVERITY_FLOOR)
        else 0.0
    )
    mb_n = _mental_break_severity(mb_state)

    score = (
        energy_n * w["energy_low"]
        + cortisol_n * w["cortisol_chronic"]
        + fatigue_n * w["fatigue_high"]
        + glucose_n * w["glucose_low"]
        + infection_n * w["infection"]
        + mb_n * w["mental_break"]
    )

    return WeaknessFactors(
        cid=cid,
        energy=round(energy_n, 4),
        cortisol_chronic=round(cortisol_n, 4),
        fatigue=round(fatigue_n, 4),
        glucose_low=round(glucose_n, 4),
        infection=round(infection_n, 4),
        mental_break=round(mb_n, 4),
        weakness_score=round(score, 4),
    )


def rank_organisms(
    biochems_dict: dict,
    *,
    weights: Optional[dict[str, float]] = None,
) -> list[WeaknessFactors]:
    """Ранжировать всех организмов по weakness_score (descending: слабые первые).

    Args:
        biochems_dict: {cid: ClientCreatureBiochem}
        weights: optional override

    Returns:
        List[WeaknessFactors] sorted: weakest first.
    """
    scored = [
        score_organism(bc, weights=weights, cid=cid)
        for cid, bc in biochems_dict.items()
    ]
    scored.sort(key=lambda f: f.weakness_score, reverse=True)
    return scored


def weakest_n(
    biochems_dict: dict,
    n: int = 3,
    *,
    weights: Optional[dict[str, float]] = None,
) -> list[str]:
    """Top-N weakest cids (для emit в diag, e.g., observability)."""
    ranked = rank_organisms(biochems_dict, weights=weights)
    return [f.cid for f in ranked[:n]]


# ────────────────────────────────────────────────────────────────────
# Diag snapshot — for push_diagnostics
# ────────────────────────────────────────────────────────────────────

def natural_selection_snapshot(
    biochems_dict: dict,
    *,
    capacity: Optional[int] = None,
    top_n_to_emit: int = 3,
    weights: Optional[dict[str, float]] = None,
) -> dict:
    """Snapshot для эмита в diag["natural_selection"].

    Args:
        biochems_dict: {cid: ClientCreatureBiochem}
        capacity: estimate_population() result (railroad cap), для context
        top_n_to_emit: сколько слабых cid'ов включить в payload
        weights: override

    Returns:
        {
            "n_organisms": int,
            "capacity": int | None,
            "weakest_cids": [str, ...],     # top-N weakest
            "scores": {cid: weakness_score}, # per-cid float
            "mean_score": float,
            "max_score": float,
        }
    """
    if not biochems_dict:
        return {
            "n_organisms": 0,
            "capacity": capacity,
            "weakest_cids": [],
            "scores": {},
            "mean_score": 0.0,
            "max_score": 0.0,
        }

    ranked = rank_organisms(biochems_dict, weights=weights)
    scores = {f.cid: f.weakness_score for f in ranked}
    weakest = [f.cid for f in ranked[:top_n_to_emit]]
    mean_s = sum(scores.values()) / len(scores)
    max_s = max(scores.values())

    return {
        "n_organisms": len(ranked),
        "capacity": capacity,
        "weakest_cids": weakest,
        "scores": {k: round(v, 3) for k, v in scores.items()},
        "mean_score": round(mean_s, 3),
        "max_score": round(max_s, 3),
    }
