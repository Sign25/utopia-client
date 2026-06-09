"""Локальный compute-движок колонии (Phase F3.4).

Один экземпляр на клиента. Хранит `dict[cid → CompositeOrganism]` — личный
зоопарк особей, обновляемый по сообщениям от P40:
  - tick      → forward + Hebbian capture + ActionSelector → action
              + Phase 1 predictor train + Phase 2 intrinsic + Phase 6 self-obs EMA
  - newborn   → создаёт нового CompositeOrganism (F3.5)
  - death     → выгружает org (F3.6)

handle_tick — чистая функция: берёт `{cid: obs[80]}`, отдаёт `{cid: {action, target_id}}`.
Нет сетевых вызовов. WS-обвязка снаружи (ws_client.py F3.x).
"""

from __future__ import annotations

import copy
import logging
import math
import random
import time
import types
from collections import deque
from typing import Optional

import numpy as np

logger = logging.getLogger("utopia_client.compute")

# Локальный action enum — синхронен с environment.world.Action на P40.
# Дублируем константы здесь, чтобы handle_tick работал даже если
# environment.world не импортируется (защита от dep-цикла).
STAY = 4
N_ACTIONS = 16

# §3 paralysis (ТЗ e3cc81b, single-organism, контракт Фрая 01.06.2026):
# owned-смерть client-authoritative. Под single_organism energy≤0 — НЕ смерть,
# а паралич: осознаёт, но не движется (motor=STAY), обучающий сигнал «плохо».
# N ≈ 3с (= 89 P40-тиков @30TPS; wall-clock → client-TPS-агностик, НЕ копируем
# «89» вслепую). После N — recovery-грант энергии (НЕ времени).
# ── Self-observable obs contract (Track 2 / этап 4, Фрай 01.06.2026) ──────
# Phase 6 self-observable = интероцепция ВЫСШЕГО порядка: организм ощущает не
# тело, а свой УМ. Дом — STATE_DIM[64:80] (P40 шлёт zeros, designated internal
# регион; client [64:80] свободен), путь через insula (ткань самоощущения) +
# predictor, позже DMN (S2.D). НЕ компромисс — концептуально верный дом.
# ПОСТОЯННЫЙ КОНТРАКТ obs (порядок зафиксирован, не менять — Фрай):
#   obs[64] = entropy_ema     — энтропия выбора действия («не уверен»)
#   obs[65] = trace_norm_ema  — норма Hebbian-следов («учусь»)
#   obs[66] = reward_var_ema  — разброс награды («среда изменилась»)
#   obs[67] = paralyzed       — §3 паралич 0/1 («осознаёт паралич», §3 learning-half)
# Мозг расширяет read-окно 64→68 (predictor+insula): новые input-веса init ≈0,
# чтобы обученный founding-мозг c103927 не сломался (Hebbian подхватит). env[:64]
# НЕ трогаем. Реализация read-expansion + weight-init — отдельным шагом.
_SELF_OBS_OFFSET = 64
_SELF_OBS_DIM = 4
_BRAIN_INPUT_DIM = _SELF_OBS_OFFSET + _SELF_OBS_DIM  # 68 — окно чтения мозга

# Track 2 / направление (б) (Фрай 02.06.2026): insula-стресс → LEARNED
# temperature-модуляция СУЩЕСТВУЮЩЕЙ motor-политики. НЕ отдельный actor.
# Temperature только sharpen/flatten распределение действий → НЕ может толкнуть
# неверное НАПРАВЛЕНИЕ → структурно не повторит провал action-head (1-dim policy,
# низкая variance vs 16-dim). Вход — интероцептивный self-observable 4-вектор
# (entropy/trace/reward_var/paralyzed = состояние «ума/стресса», субсумирует
# мёртвый insula.last_stress). T_mod = exp(mu + σ·ε), mu = head(so4), zero-init →
# mu=0 → T_mod≈1 (near-identity старт). σ мал → temperature-only jitter.
# Clamp log T ∈ ±ln2 → T_mod ∈ [0.5, 2.0] (не может задавить/взорвать действие).
_INSULA_TEMP_LOG_CLAMP = math.log(2.0)   # T_mod ∈ [0.5, 2.0]
_INSULA_TEMP_SIGMA = 0.08                # σ exploration на log-T (мал → near-identity)
_INSULA_TEMP_LR = 3e-4                   # << self_obs_head 1e-3: медленный, аккуратный

# Колониально-специфичные mental_break (Фрай 01.06.2026): требуют колонию,
# под single_organism для одиночки always-true/бессодержательны → гейтятся.
#   loner    = oxytocin<10 + social_gap>500 — у одиночки oxytocin всегда→0
#              (растёт только от clan-proximity/mating) → loner всегда (sabotage).
#   pair-bond = oxytocin>75 — mate-связь (только от спаривания).
# Стресс-состояния (catatonic/exhaustion/wander/berserk/hunt-mode/inflammation)
# НЕ гейтим — валидны для solo, маскировать нельзя (это не колониальные артефакты,
# а реальные сигналы; foraging Адам учит сам, давление не снимаем).
_COLONIAL_MENTAL_BREAKS = frozenset({"loner", "pair-bond"})

_PARALYSIS_SEC = 3.0
# Tier 1 (Хьюберт 01.06.2026): φ⁷≈45 (2.8с окно) → φ⁶≈73 (4.6с, ~9 тайлов
# навигации @ move_speed=2 → шанс найти flora ~25%→~80%). Окно НАВИГАЦИИ до
# первого укуса, не отмена голода. Если on_flora_ticks=0 при 73 → policy gap
# (моя зона); если растёт но не доедает → Tier 2 φ⁵≈118. max_energy=1309.
_RECOVERY_ENERGY_DEFAULT = 73.0  # max_energy/φ⁶

# Dehydration-модель (mirror environment/world.py:909/926 — источник истины).
# Стадии по hydration ratio → energy_drain грызёт ЭНЕРГИЮ (не отдельная
# hydration-смерть): смерть через energy<=0 (starvation), единая ось.
# РЕКАЛИБРОВКА под client-tick (0.11.37, Шеф «давление смерти, мягче»): сервер
# φ²≈2.618/φ³≈4.236 ПЕР-СЕРВЕР-ТИК; клиент применяет ~5Гц → полный φ² = ~13
# энергии/сек → выжигал запас за <1мин (0.11.34 вайп). ×0.1 → ≈0.26/0.42 за
# тик: мягкое давление отбора (застрявший без воды гибнет за ~4мин, сытый у
# воды не страдает), не вайп. Тюнить scale если 50→16 не идёт.
_PHI_CONST = 1.618033988749895
_DEHYDRATION_DRAIN_SCALE = 0.1
_DEHYDRATION_DRAIN = {2: _PHI_CONST ** 2 * _DEHYDRATION_DRAIN_SCALE,
                      3: _PHI_CONST ** 3 * _DEHYDRATION_DRAIN_SCALE}

# Метаболизм per-sec (01.06.2026, Хьюберт): rate × wall-clock dt. Клемп dt —
# чтобы reconnect-разрыв (организм отсутствовал минуты) не списал разом.
_MAX_METAB_DT = 3.0          # сек; типичный интервал handle_tick ~2.4с
# Infection client-side per-sec: server world.py 0.005/sim-тик при sim 30 Гц
# (Хьюберт) → 0.15/сек прогресс; energy-drain 2.0/sim-тик → 60/сек × severity.
_SIM_TPS = 30.0
_INFECTION_SEVERITY_PER_SEC = 0.005 * _SIM_TPS   # 0.15/сек
_INFECTION_DRAIN_PER_SEC = 2.0 * _SIM_TPS        # 60/сек × severity


def _dehydration_stage(hydration: float, max_hydration: float) -> int:
    """0=норма, 1=жажда, 2=обезвоживание, 3=критическое (world.py:909)."""
    ratio = hydration / max(1.0, max_hydration)
    if ratio > 0.5:
        return 0
    if ratio > 0.25:
        return 1
    if ratio > 0.0:
        return 2
    return 3

# Phase 2 — intrinsic reward coefficient (1/φ²). Идентично P40.
_BETA_INTRINSIC = 0.3819660112501051
# Y50 noise scale для predictor наследования (1/φ⁵ ≈ 0.0902). Идентично P40.
_PREDICTOR_Y50_SCALE = 0.0902
# EMA-коэффициент для Phase 1/2/6.
_EMA_ALPHA = 0.01

# Brain migration (10.05.2026): высшие ткани S2.E/G/A/F на клиенте.
# Биты ablation_mask (синхронны с P40 routes_world.py):
_DOPAMINE_BIT = 14      # S2.E
_PLANNER_BIT = 10       # S2.A
_TOM_BIT = 11           # S2.B (theory_of_mind) — добавлен 13.05.2026
_DEFAULT_MODE_BIT = 13  # S2.D
_INSULA_BIT = 15        # S2.F
_IMAGINATION_BIT = 16   # S2.G
_MOTOR_POLICY_BIT = 17  # Phase 7 — перенумерован с 11 на 17 (13.05.2026),
                         # чтобы освободить канонический бит 11 для S2.B.
# Phase S2.A planner — N_ACTIONS из routes_world.
_PLANNER_N_ACTIONS = 16
_PLANNER_SCALE = 1.0
# motor_policy: delta = scale·tanh(out[:16]) ∈ [-1, 1]^16.
_MOTOR_POLICY_N_ACTIONS = 16
_MOTOR_POLICY_SCALE = 1.0
# S2.B theory_of_mind (13.05.2026): supervised predict 8 motor-actions для соседа.
# 4 соседа × 13 признаков = 52 data_dim. target = направление Δposition.
_TOM_DATA_DIM = 52
_TOM_N_NEIGHBORS = 4
_TOM_FEATURES_PER_NEIGHBOR = 13
_TOM_N_ACTIONS = 8
_TOM_LR = 1e-4
_TOM_ACC_EMA_ALPHA = 0.02      # быстрый EMA по hit/miss
_TOM_DELTA_NORM = 32.0         # шкала нормализации dx/dy ∈ [-1, 1] для радиуса ~32
_TOM_N_LINEAGES = 2            # one-hot: [elder, wanderer]
_TOM_N_SIGNALS = 8             # one-hot: sig ∈ [0, 7]
# S2.C language (13.05.2026): supervised decode чужого сигнала → собственный
# исход на следующий тик. data_dim=16 (own_sig one_hot[8] + neighbor counts/k[8]).
# 4 класса события: 0=ate, 1=damage, 2=killed, 3=idle (idle пропускаем).
_LANG_BIT = 12
_LANG_DATA_DIM = 16
_LANG_N_SIGNALS = 8
_LANG_N_CLASSES = 4
_LANG_K_NEIGHBORS = 8         # топ-K ближайших соседей для counts
_LANG_LR = 1e-4
_LANG_ACC_EMA_ALPHA = 0.02
_LANG_EVT_ATE = 0
_LANG_EVT_DAMAGE = 1
_LANG_EVT_KILLED = 2
_LANG_EVT_IDLE = 3
# Action codes для target classification (синхронно с environment.world.Action).
_TOM_ACT_NORTH = 0
_TOM_ACT_SOUTH = 1
_TOM_ACT_EAST = 2
_TOM_ACT_WEST = 3
_TOM_ACT_STAY = 4
# Phase S2.E dopamine: β_local ∈ [0, 1/φ ≈ 0.618].
_PHI = 1.6180339887498949
# Phase 4 fix 0.11.6: energy-порог для glucose floor. Organism с energy выше
# этого считается "сытым" → glucose поддерживается из энергозапасов
# (гомеостаз), не падает ниже baseline. 200 = заметно выше голодного (~0),
# но ниже reproduce_threshold (500) — floor работает и для не-готовых к
# размножению, но живых особей.
_GLUCOSE_FLOOR_ENERGY_THRESHOLD = 200.0
# Phase 4 fix 0.11.7: cortisol baseline-clearance (гомеостаз). 0.995 →
# half-life ~144 тика (F12). Equilibrium при continuous stress ~40 (< 80
# catatonic threshold). Старт; 0.99 (F11, eq~20) если recover медленный.
_CORTISOL_HOMEOSTASIS_DECAY = 0.995
# Phase S2.F insula: 64 obs + 7 интероцепции = 71.
_INSULA_DATA_DIM = 71
# §3.2 (Фрай 09.06.2026): client-authoritative интероцепция. Строим 7-вектор
# ЛОКАЛЬНО из self.biochem (energy/hydration/cortisol/serotonin/infection —
# точно из биохимии Адама; age/valence — carry последнего P40-intero). Снимает
# P40-blind risk: insula не слепнет, если P40 перестанет слать intero.
# Зеркало server-side _gather_interoception (NeuroCore/server/tick/sidecars.py:48)
# — нормировки совпадают ТОЧЬ-В-ТОЧЬ (incl. raw-camel quirk slot[1]), иначе insula
# получит сдвиг распределения входа.
_CLIENT_MAX_ENERGY = 1309.0      # mirror cfg.max_energy default (_gather_interoception)
_CLIENT_MAX_AGE = 17711.0        # mirror base_max_age (Fib)
_CLIENT_DEFAULT_CAMEL = 10.0     # mirror server getattr(creature,'camel',10)
# Phase S2.D default_mode: floor ∈ [0, 0.01] — мягкая добавка к Δsurprise.
# Совпадает с _DEFAULT_MODE_FLOOR_MAX на P40 (routes_world.py).
_DEFAULT_MODE_FLOOR_MAX = 0.01
# Y50 для высших тканей (тот же scale, что у predictor).
_HIGHER_TISSUE_Y50_SCALE = 0.0902

# SFNN S3.0 (14.05.2026): 7 высших тканей, у которых будет своё эволюционирующее
# правило локальной пластичности (ΔW = η·(A·post·preᵀ + B·post + C·preᵀ + D),
# r=0 — чистая Hebb с эволюционируемыми коэффициентами). predictor НЕ входит:
# он остаётся на Adam (supervised, минимизация prediction error). Активируется
# флагом genome.higher_tissue_sfnn_enabled (S3.1+); в S3.0 — только storage +
# наследование через single-parent Y50.
_HIGHER_SFNN_TISSUES: tuple[str, ...] = (
    "dopamine",
    "imagination",
    "planner",
    "insula",
    "default_mode",
    "theory_of_mind",
    "language",
)

# S6.0 (16.05.2026): Роли, для которых веса обновляются SFNN-правилом,
# а НЕ классическим HebbianController. Передаются как `skip_roles` в
# `heb.update(...)` — иначе один и тот же тик апдейтит веса дважды
# (классика + SFNN), и эволюционируемое правило тонет в шуме базовой
# Phase 5d-механики. "motor" — это базовая ткань в organism graph
# (motor_policy на клиенте — отдельный sidecar, не в graph). 7 высших
# тоже формально в TISSUE_ROLES, поэтому попадают в _tissue_info heb'а.
_SFNN_MIGRATED_ROLES: set[str] = {
    "motor",
    "dopamine",
    "imagination",
    "planner",
    "insula",
    "default_mode",
    "theory_of_mind",
    "language",
}

# S6.4 (16.05.2026): 10 базовых тканей organism graph, которые в S6
# мигрируют на унифицированное SFNN apply-step (eligibility trace + R3
# + TD-coupling). Веса каждой роли живут в HebbianController._tissue_info
# (один heb на cid). Под флагом `genome.basic_tissue_sfnn_enabled=True`
# классический Hebbian для этих ролей пропускается через
# `heb.update(skip_roles=…)` (S6.6), а SFNN apply-step применяет ΔW из
# pre/post, накопленных самим heb'ом (S6.5).
_BASIC_SFNN_TISSUES: tuple[str, ...] = (
    "sensory",
    "attention",
    "brain",
    "memory",
    "consciousness",
    "communication",
    "motor",
    "manipulator",
    "digestive",
    "immune",
)

# Z7.i.d (16.05.2026, Зодчий): три уникальные высшие ткани третьей линии.
# Создаются client-side в add_creature только для lineage="zodchiy". Sidecar-
# storage, как dopamine/imagination — НЕ часть organism graph (на P40 живёт
# CreatureState с lineage, тканей нет — все ткани клиентские). Аналог
# `_build_zodchiy_connections` (workbench), но без полного rebuild графа:
# Tissue 21/3/1 на cid → forward в _compute_higher_tissues + SFNN-rule
# storage. Z2.b overlay подключён (01.06.2026) для ролей в org.tissues; эти
# z-ткани — sidecar (не в org.tissues), потому genes на их роли overlay
# игнорирует (apply_topology_overlay_by_role: unknown role → skip) — divergence
# по genes идёт, но рёбра на sidecar не материализуются (как на сервере).
_ZODCHIY_EXTRA_TISSUES: tuple[str, ...] = (
    "cerebellum",
    "amygdala",
    "episodic",
)

# Маппинг "роль в organism graph" → "поле в Genome" с сериализованным
# SFNNRule (см. core/organism.py:Genome). "motor" в graph ≠ motor_policy
# sidecar, поэтому поле зовётся `motor_low_sfnn_rule`.
_BASIC_SFNN_GENOME_FIELD: dict[str, str] = {
    "sensory":        "sensory_sfnn_rule",
    "attention":      "attention_sfnn_rule",
    "brain":          "brain_sfnn_rule",
    "memory":         "memory_sfnn_rule",
    "consciousness":  "consciousness_sfnn_rule",
    "communication":  "communication_sfnn_rule",
    "motor":          "motor_low_sfnn_rule",
    "manipulator":    "manipulator_sfnn_rule",
    "digestive":      "digestive_sfnn_rule",
    "immune":         "immune_sfnn_rule",
}

# TZ B Phase 2 (26.05.2026, Бендер): per-role Hebbian metrics tracker для
# observability обучения в Mode E (cheef-PC online). 20 ролей = 10 basic +
# 7 higher + 3 zodchiy sidecar. Schema: Option B (extend diagnostics),
# согласовано с Хьюбертом, ТЗ §3.2 commit 8238b06.
_HEB_PT_ALL_ROLES: tuple[str, ...] = (
    _BASIC_SFNN_TISSUES + _HIGHER_SFNN_TISSUES + _ZODCHIY_EXTRA_TISSUES
)
# Порог "учится vs не учится". |delta| > epsilon → n_learning++. Epsilon
# мал (1e-9): SFNN traces после первого update имеют norm порядка lr_oja
# (~1e-4), zero только при reset/первом тике. Reasonable separation.
_HEB_PT_EPSILON: float = 1e-9


class LocalColonyCompute:
    """Локальная колония: forward + Hebbian + ActionSelector per-creature.

    Все Tensor живут на `device` (по умолчанию CPU; с CUDA — autodetect).

    Использование:
        from utopia_client.seed_loader import ensure_seed, load_founders
        founders = load_founders(seed_path, n=5)
        compute = LocalColonyCompute()
        cids = [f"c_{i}" for i in range(5)]
        for cid, org in zip(cids, founders):
            compute.add_creature(cid, org)
        # На каждый tick от P40:
        actions = compute.handle_tick({cid: obs_array_80 for cid, obs_array_80 in obs.items()})
        # actions = {cid: {"action": int, "target_id": Optional[str]}}
    """

    def __init__(self, *, device: Optional[str] = None) -> None:
        import torch

        self.device = torch.device(device) if device else (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        # Поздний импорт — neurocore[client] нужен только если этим пользуются.
        self._torch = torch

        self.organisms: dict = {}            # cid → CompositeOrganism
        self.action_selectors: dict = {}     # cid → ActionSelector
        self.hebbian: dict = {}              # cid → HebbianController | None
        # Phase F3.2.b: метрика — сколько раз обновили Hebbian.
        self.hebbian_updates: int = 0

        # Phase 1 — Forward Model (sidecar Tissue, supervised MSE).
        self.predictor: dict = {}            # cid → Tissue
        self.predictor_opt: dict = {}        # cid → torch.optim.Adam
        self.prev_obs: dict = {}             # cid → torch.Tensor [1, 64]
        self.loss_ema: dict = {}             # cid → float (running MSE)
        self.pred_loss_history: dict = {}    # cid → deque[float] maxlen=100
        # Phase 2 — Intrinsic reward (Δsurprise).
        self.intrinsic_last: dict = {}       # cid → float (текущий тик)
        self.intrinsic_ema: dict = {}        # cid → float (baseline)
        # Phase 6 — Self-observable states (entropy/trace/reward var).
        self.entropy_ema: dict = {}          # cid → float ∈ [0, 1]
        self.trace_norm_ema: dict = {}       # cid → float ∈ [0, 1)
        self.reward_var_ema: dict = {}       # cid → float
        self.reward_history: dict = {}       # cid → deque[float] maxlen=10
        # Метрики счётчиков (для diagnostics endpoint).
        self.predictor_steps: int = 0
        # Вариант A (08.06.2026, Фрай/Хьюберт): рост мозга при жизни. Predictor —
        # узел графа ЧЕРЕЗ cerebellum (бит 18, TISSUE_ROLES_ZODCHIY). forward-hook
        # ловит выход cerebellum-ткани → он становится входом обученного
        # predictor'а. Связи {ткань}→cerebellum двигают вход прогноза → Δloss_ema
        # реагирует напрямую = драйвер для propose/keep петли роста. Только
        # single-Adam; граф client-local, в P40 не уходит (Хьюберт verified).
        # _predictor_from_cerebellum — kill-switch (False → predictor на raw obs).
        self._predictor_from_cerebellum: bool = True
        self._cerebellum_tid: dict = {}      # cid → tissue_id | None (cache)
        self._cerebellum_out: dict = {}      # cid → torch.Tensor [1,64] выход/тик
        self._cerebellum_hooked: set = set() # cid с зарегистрированным hook
        # Шаг 2 — петля роста связей (08.06, Фрай go): propose→dwell→Δloss_ema→
        # keep/backoff, ОДНА связь {роль}→cerebellum за раз. Триггер — плато
        # intrinsic (мозгу нечему учиться весами → менять структуру). keep на
        # ЗНАЧИМОМ сошедшемся Δloss_ema (Фрай: не любой Δ>0); backoff revert'ит
        # ребро (enabled=False) + защищает net/§3. Gated OFF (ship dormant) —
        # включить после live-проверки встраивания (cerebellum_out≠0, loss не взлетел).
        self._growth_enabled: bool = True             # kill-switch (флип 08.06: рост ВКЛ для Адама)
        self._growth_tracker = None                   # client-local innovation tracker (lazy)
        self._growth_rng = random.Random(0)           # детерминируемый выбор src-роли
        self._growth_state: dict = {}                 # cid → {gene, loss_before, ticks, par_before, energy_before}
        # ОТНОСИТЕЛЬНЫЙ триггер (Фрай 08.06): embodied intrinsic floor не ноль и
        # дрейфует (0.005-0.009) → абсолютный порог хрупок. Плато = intrinsic у
        # СВОЕГО трейлинг-floor (min за окно) + стагнация N тиков (перестал падать).
        # Self-referencing, drift-robust. Floor — референс, не абсолютный уровень.
        self._growth_intr_hist: dict = {}             # cid → deque[float] intrinsic_ema (трейлинг-floor)
        self._growth_stagnation_n: dict = {}          # cid → тиков «у floor» подряд
        self._growth_intr_window: int = 233           # Fib — окно трейлинг-floor
        self._growth_near_floor_margin: float = _PHI_CONST ** -3  # φ⁻³≈0.236 «у floor»
        self._growth_plateau_ticks: int = 55          # Fib — стагнация у floor до propose
        self._growth_dwell_ticks: int = 89            # Fib — окно пере-сходимости predictor'а
        self._growth_min_delta_frac: float = _PHI_CONST ** -5  # φ⁻⁵≈0.09 относит. улучшение
        self._growth_kept: int = 0                    # принятых связей (метрика)
        self._growth_reverted: int = 0                # откатов (метрика)
        # tried-set с COOLDOWN (Фрай/Шеф 08.06): не ретраить отвергнутое ребро
        # _growth_retry_cooldown проб подряд (поиск покрывает новые рёбра). НЕ
        # permanent (эпистаз: ребро, бесполезное сейчас, может помочь после смены
        # графа) → через cooldown снова eligible. Часы — счётчик проб (монотонный).
        self._growth_propose_count: int = 0           # монотонные часы проб
        self._growth_rejected: dict = {}              # cid → {source_role: propose_count при отказе}
        self._growth_retry_cooldown: int = 13         # Fib — проб до повторной попытки отвергнутого
        # SATURATION-guard (Шеф 08.06): когда полезные {роль}→cerebellum рёбра
        # исчерпаны (все свежие кандидаты в cooldown → fallback гоняет оставшиеся
        # бесполезные по кругу) — встать на ПАУЗУ, не churn'ить. Сигнал «связи
        # насыщены → готов к фазе тканей §3.2». Снимается reset_growth_saturation()
        # (вызвать при добавлении тканей — новые роли = новые кандидаты) или KEEP.
        self._growth_fallback_streak: int = 0         # подряд fallback-backoff'ов
        self._growth_saturated: bool = False          # пауза петли связей
        self._growth_saturation_threshold: int = 5    # fallback-backoff'ов до насыщения
        # §10.8 РОСТ ТКАНЕЙ (Фрай 09.06): рост УЗЛАМИ (не рёбрами). После
        # насыщения связей (19/19→cerebellum) петля растит НОВУЮ ТКАНЬ кандидатом:
        # минт (_make_higher_tissue) + вставка в org.tissues + проводка {роль}→
        # cerebellum → dwell → Δloss_ema → keep/backoff (как рёбра). Узел без
        # входящего ребра = СЕНСОР (читает obs напрямую, workbench.forward:97-126).
        # Драйвер prediction: keep-путь СПИТ, пока мир (погода/аффордансы Хьюберта,
        # §10.9) не вернёт prediction-давление (loss 0.046 = пол на статичном мире).
        # Дисциплина та же: одна за раз, dwell, keep/backoff, durable-персист,
        # kill-switch. Client-local. Спек гибкий (читает obs, расширяемо под новый
        # obs-канал погоды — координация с Хьюбертом).
        self._tissue_growth_enabled: bool = False     # kill-switch (дефолт OFF, dormant)
        self._tissue_growth_state: dict = {}          # cid → in-flight {role,tid,loss_before,ticks,...}
        self._tissue_kept: int = 0                    # принятых тканей (метрика)
        self._tissue_reverted: int = 0                # откатанных тканей
        self._tissue_propose_count: int = 0           # монотонный счётчик (имена ролей grownN)
        self._tissue_grown_specs: dict = {}           # cid → [{role,data_dim,n_embd}] KEEP'нутых (персист+recreate)
        self._TISSUE_GROWTH_DATA_DIM: int = 64        # читает obs64 (сенсор); под погоду расширяемо
        self._TISSUE_GROWTH_N_EMBD: int = 21          # sidecar 21/3/1 (как высшие)

        # Brain migration (10.05.2026): высшие ткани S2.E/G/A/F per cid.
        # Forward-only (MVP-lite, без supervised), Y50 наследование от родителя.
        self.dopamine: dict = {}        # cid → Tissue (S2.E)
        self.imagination: dict = {}     # cid → Tissue (S2.G)
        self.planner: dict = {}         # cid → Tissue (S2.A)
        self.insula: dict = {}          # cid → Tissue (S2.F, data_dim=71)
        # 13.05.2026: S2.D default_mode мигрирован с P40 на клиент. Раньше
        # жил только как server-side sidecar и каждое register_newborn
        # рождался со случайными весами (P40 потерял default_mode в
        # envelope_brain). Теперь — owned-only, как dopamine/imagination.
        self.default_mode: dict = {}    # cid → Tissue (S2.D)
        # motor_policy — Tissue 21/3/1, обучается SFNN-правилом (S4).
        self.motor_policy: dict = {}        # cid → Tissue
        # SFNN S1.1 (14.05.2026) — per-tissue SFNN-правило обучения для
        # motor_policy. Per-synapse-type коэффициенты A/B/C/D/η.
        # Дефолтное правило (A=1, B=C=D=0, η=1e-3) эквивалентно Phase 5d Hebb.
        self.motor_sfnn_rule: dict = {}     # cid → SFNNRule
        # SFNN S1.2b (14.05.2026) — forward-hooks pre/post активаций для
        # 6 Linear в motor_policy Tissue. Захватываются на каждом forward
        # через _motor_forward; используются на следующем тике в
        # _motor_sfnn_update_step для ΔW. Очищаются в remove_creature.
        self.motor_sfnn_hook_handles: dict = {}   # cid → list[hook handle]
        self.motor_sfnn_acts: dict = {}            # cid → {synapse_type → (pre, post)}
        # SFNN S1.2c (14.05.2026) — baseline L2-нормы строк 6 weight matrices
        # motor_policy, снятые при add_creature. После каждого ΔW row-wise
        # renorm: W[i] ← W[i] · (baseline‖W[i]‖) / ‖W[i]‖. Защита от взрыва
        # весов / tanh-saturation при долгой SFNN-тренировке. Per-row, чтобы
        # сохранить относительный масштаб output-нейронов друг к другу.
        self.motor_sfnn_row_norms: dict = {}      # cid → {synapse → Tensor[rows]}
        # SFNN S3.0 (14.05.2026) — те же SFNN-правила, но для 7 высших тканей.
        # Внутренний dict per tissue: cid → SFNNRule. Под флагом
        # genome.higher_tissue_sfnn_enabled (дефолт False) сейчас только
        # storage + Y50-наследование, активное применение — S3.1+.
        # predictor сюда не входит — остаётся на Adam (supervised).
        # Z7.i.e (16.05.2026): расширен на 3 Зодчий-ткани (cerebellum/
        # amygdala/episodic) — apply-step `_higher_tissue_sfnn_update_step`
        # полностью generic (getattr(self, tissue_name)), переиспользуется.
        _ALL_HIGHER = _HIGHER_SFNN_TISSUES + _ZODCHIY_EXTRA_TISSUES
        self.higher_tissue_sfnn_rule: dict[str, dict] = {
            t: {} for t in _ALL_HIGHER
        }
        # Per-tissue счётчик SFNN-апдейтов (заполняется в S3.1+ при включении
        # дисптача). Используется в diagnostics для отслеживания активности.
        self.higher_tissue_sfnn_steps: dict[str, int] = {
            t: 0 for t in _ALL_HIGHER
        }
        # SFNN S3.1 (14.05.2026) — forward-hooks pre/post активаций 6 Linear
        # для каждой из 7 высших тканей. Регистрируются в add_creature только
        # для активных-в-S3.x тканей; cleanup в remove_creature. Per tissue:
        # {cid → list[hook handle]} и {cid → {synapse_type → (pre, post)}}.
        self.higher_tissue_sfnn_hook_handles: dict[str, dict] = {
            t: {} for t in _ALL_HIGHER
        }
        self.higher_tissue_sfnn_acts: dict[str, dict] = {
            t: {} for t in _ALL_HIGHER
        }
        # SFNN S1.2c (14.05.2026) — baseline row-norms 6 weight matrices
        # каждой из 7 высших тканей. Та же защита от взрыва весов, что и
        # для motor_policy. {tissue → {cid → {synapse → Tensor[rows]}}}.
        self.higher_tissue_sfnn_row_norms: dict[str, dict] = {
            t: {} for t in _ALL_HIGHER
        }
        # Z1 (16.05.2026, Зодчий) — eligibility trace per (tissue, cid) для
        # унифицированной S6.5-формулы на 7 высших тканях. Tensor формы W
        # каждой Linear-матрицы внутри ткани (под ключом synapse_type).
        # Эфемерное — ресет при рестарте клиента и при reset_all().
        self.higher_tissue_sfnn_trace: dict[str, dict] = {
            t: {} for t in _ALL_HIGHER
        }
        # S2.B (13.05.2026) — theory_of_mind: supervised predict 8 motor-action
        # ближайшего соседа по Δposition (data_dim=52, 4 neighbors × 13 feat).
        self.theory_of_mind: dict = {}      # cid → Tissue (S2.B)
        self.theory_of_mind_opt: dict = {}  # cid → torch.optim.Adam
        # last_tom_acc[cid] — running accuracy ∈ [0, 1], EMA по hit/miss
        # последних supervised step'ов. Пушим в actions_batch.phase_emas.
        self.last_tom_acc: dict = {}        # cid → float
        # _tom_prev_focus[cid] = (neighbor_cid, x, y) — кого предсказывали
        # на прошлом тике и где он был. На текущем — ищем того же соседа,
        # считаем Δposition → target action, cross-entropy step.
        self._tom_prev_focus: dict = {}     # cid → (str, int, int) | None
        self.tom_steps: int = 0
        # S2.C (13.05.2026) — language: supervised decode чужого сигнала →
        # собственное событие на следующем тике. data_dim=16 (own_sig
        # one_hot[8] + neighbor signal counts/k[8]). target ∈ {ate, damage,
        # killed, idle}, idle skip (class imbalance).
        self.language: dict = {}              # cid → Tissue (S2.C)
        self.language_opt: dict = {}          # cid → torch.optim.Adam
        self.last_lang_acc: dict = {}         # cid → float ∈ [0, 1]
        # _lang_prev_context[cid] = torch.Tensor [1, 16] — контекст прошлого
        # тика (на t слышали сигналы, на t+1 получаем event для t→t+1).
        self._lang_prev_context: dict = {}    # cid → torch.Tensor [1, 16] | None
        self.lang_steps: int = 0
        # Связь с WorldStateCache (для tom_neighbors_view). Прокидывается
        # из ws_client после bootstrap'а кеша. None → ToM no-op.
        self.world_cache = None  # type: Optional[object]
        # SFNN S1.2 (14.05.2026) — счётчик SFNN-апдейтов motor_policy.
        # ΔW = η·(1+r)·(A·hebb_A + B·post + C·preᵀ + D), где
        # hebb_A = outer(post_c, pre_c) − post_c²·W (S1.2c Oja + mean-center).
        self.motor_sfnn_steps: int = 0
        # Last-snapshot для get_phase_emas → push в actions_batch.
        self.last_beta_local: dict = {}    # cid → float ∈ [0, 1/φ]
        self.last_imag_mult: dict = {}     # cid → float ∈ [1, 2]
        self.last_planner_delta: dict = {} # cid → torch.Tensor [16]
        self.last_motor_delta: dict = {}   # cid → torch.Tensor [16] (Phase 7)
        # REINFORCE baseline (Phase 4 #1, 01.06.2026): бегущее среднее reward
        # на cid → advantage = r − baseline. Без него gain=1+r всегда >0 →
        # политика только reinforce, не дискриминирует. Server intent:
        # "REINFORCE по log_prob[a]·(r_imm_total − baseline)".
        self._motor_reward_baseline: dict = {}  # cid → float (EMA reward)
        # MOTOR_LEARN телеметрия (03.06, full-world Ступень 2): диагностика
        # доставки reward→policy (Фрай: исключить delivery-bug ПЕРВЫМ).
        # adv_ema = EMA |advantage| (reward дискриминирует?), dw_ema = EMA ‖ΔW‖
        # применённого (policy движется?). Вместе с motor_sfnn_steps + flip_rate
        # разводят (i) delivery-bug / (ii) credit-fail / (iii) slow-learn.
        self._motor_adv_ema: dict = {}   # cid → EMA |advantage|
        self._motor_dw_ema: dict = {}    # cid → EMA ‖ΔW‖ (до renorm)
        # ΔW-инструментирование output_proj (Фрай 04.06): разводит (a) renorm
        # проецирует радиальную компоненту вон vs (b) incoherent Hebbian под
        # exploration. dw_radial_ema высокий (→1) → ΔW вдоль W → renorm режет = (a).
        # dw_cos_ema осциллирует (→0/<0) → ΔW-направление скачет tick-to-tick = (b).
        self._motor_dw_last: dict = {}       # cid → flat ΔW output_proj (пред. тик)
        self._motor_dw_radial_ema: dict = {}  # cid → EMA доли ΔW вдоль W (renorm-cut)
        self._motor_dw_cos_ema: dict = {}     # cid → EMA cos(ΔW_t, ΔW_t-1)
        # MOTOR renorm growth-cap (03.06, Ступень 2, Фрай): рекалибровка row-L2-
        # renorm, который пиннит магнитуду весов motor_policy → не даёт заостриться
        # → flip 0.99. cap=1.0 → текущий пин (no-op default); cap>1 → строка может
        # вырасти до target×cap (заострение возможно), но не взорваться (cap bounds).
        # Управляется client_flags motor_renorm_cap (мгновенно, без рестарта).
        # Тест: cap↑ → flip падает = renorm был супрессором (починка, не fundamental).
        self._motor_renorm_growth_cap: float = 1.0
        # MOTOR Oja-scale (03.06, Ступень 2, Фрай (a)): множитель Oja-вычитающего
        # члена (−post²·W) в hebb_A. 1.0 → текущая полная Oja-стабилизация;
        # <1.0 → Oja слабее → ΔW не сжимается при росте W → policy может реально
        # заостриться (вместе с renorm_cap>1 «освобождает магнитуду»). Тест Фрая:
        # в спокойствии (грасс вверх) свободная магнитуда → flip падает? Следить
        # за tanh-saturation (motor_norm→4.0). client_flags motor_oja_scale.
        self._motor_oja_scale: float = 1.0
        # MOTOR output_proj-specific развязка (Фрай 04.06, верифицировано dw_radial≈1):
        # Oja-член (−post²·W) РАДИАЛЕН по построению → свампит reward-направление
        # (тангенциальное, к forage) на output_proj → ΔW радиален → renorm режет →
        # policy залочена. Oja+renorm НОРМАЛЬНЫ на input/attn/mlp (представленческие),
        # но НЕВЕРНЫ на policy-output. _out-версии применяются ТОЛЬКО к output_proj:
        # снизить Oja (первично) → ΔW тангенциальнее → reward шейпит направление →
        # разлочить forage-learning. <1.0 = слабее. Дефолт 1.0 = как глобальный.
        self._motor_oja_scale_out: float = 1.0
        self._motor_renorm_cap_out: float = 1.0   # вторично: дать магнитуде survive
        # Policy-gradient на output_proj (Фрай 04.06, rule-upgrade): тангенциальный
        # REINFORCE credit к СЭМПЛИРОВАННОМУ действию (∇log π·advantage) вместо
        # correlational Hebbian (radial=1 → залочен на вестигиальной КОЛОНИАЛЬНОЙ
        # политике SHARE_FOOD/FLEE Адама-Зодчего). Может разучить колонию + выучить
        # forage. _on>0 → заменяет Hebbian ТОЛЬКО на output_proj (прочие синапсы —
        # Hebbian-представленческое). client_flags motor_pg + motor_pg_lr. Дефолт off.
        self._motor_pg_on: float = 0.0
        self._motor_pg_lr: float = 1.618   # φ (Шеф: φ-коэффициенты)
        self._motor_pg_ctx: dict = {}      # cid → (sampled_action:int, π:Tensor[16])
        self._motor_pg_steps: int = 0      # верификация: сколько раз PG реально бежал
        # МЕДЛЕННЫЙ канал — порт серверного WorldTrainer (Фрай 05.06, MIGRATION GAP):
        # batch-REINFORCE по буферу опыта (norm-advantage + adaptive-entropy +
        # batch-стабильность) = учитель политики, которого не было на клиенте.
        # _on>0 → вкл (заменяет per-tick PG; быстрый Hebbian остаётся). Per-cid
        # MotorSlowTrainer + pending (action_t→reward_t+1 credit через P40-lag).
        self._motor_slow_on: float = 0.0
        self.motor_slow_trainer: dict = {}   # cid → MotorSlowTrainer
        self._slow_pending: dict = {}        # cid → (obs[D], action, base_logits[16])
        # «Доедай, не убегай» (Фрай 06.06): пока текущий тайл ещё отдаёт (ate/
        # delta_energy>0 в прошлый тик) прайор эмитит STAY/GATHER на месте, а НЕ
        # «беги к ближайшей флоре» — иначе Адам бросает кормящий тайл и бегает
        # каждый тик (cost > income). φ-гистерезис: память _TILE_YIELD_MEM тиков
        # (Fib 2) чтобы один пустой тик не срывал в бегство; исчерпание → быстрый
        # релиз (защита от idle-голода, критерий Хьюберта eats/sec≥1.62).
        self._tile_yield_mem: dict = {}      # cid → остаток тиков «тайл кормит»
        # ARRIVAL STUCK-DETECTION (Фрай 07.06): nearest_flora недостижима (через
        # воду/препятствие) → Адам коммитит но не движется (pos застрял, onf=0).
        # Трекаем вектор (dr,dc,dist); не меняется N тиков при dist>0 → абандон+обход.
        self._arrival_last_nf: dict = {}     # cid → последний (dr,dc,dist)
        self._arrival_stuck_n: dict = {}     # cid → тиков без прогресса к флоре
        # MOTOR de-saturation (04.06, Фрай): tanh-голова бистабильна — pre-tanh
        # `out` большой → tanh(out/T) залипает ±0.99, градиент≈0 → залипает в
        # экстремуме (наблюдалось: alignment флипнулся +0.99→-0.91 и не вернулся).
        # _motor_temp: override T в _motor_forward (tanh(out/T)). >1 делит pre-tanh
        # → дельты в отзывчивый варьирующий диапазон → градиент работает → REINFORCE
        # выбивает из экстремума (а) И держит отзывчивой (б). 0.0 = use rule.temperature
        # (текущее поведение). client_flags motor_temp. Реверс — 0.0.
        self._motor_temp: float = 0.0
        # _motor_lr_scale: множитель eta в _motor_sfnn_update_step (анти-saddle-flip,
        # Фрай (б)). <1 → медленнее REINFORCE → меньше крупных перекидов через седло.
        # 1.0 = текущее. client_flags motor_lr_scale. Реверс — 1.0.
        self._motor_lr_scale: float = 1.0
        # Reward-баланс forage/hunt (Фрай 04.06): порт серверного энергобаланса
        # вместо плоских равных +5·ate/+5·killed (корень бистабильности мотора —
        # два равных reward-аттрактора). _reward_balance_on>0 → энерго-диффер.
        # формула (hunt φ⁷ vs forage φ⁴, risk=damage). Веса tunable. Дефолт 0 =
        # старый плоский reward (колония + safety). Реверс — 0.
        self._reward_balance_on: float = 0.0
        self._reward_forage_w: float = 1.0   # ×φ⁴≈6.85 на ate
        self._reward_kill_w: float = 1.0     # ×φ⁷≈29 на killed
        self._reward_risk_w: float = 1.0     # ×damage_taken (риск hunt)
        # LOGIT_DEBUG (03.06, Фрай): локализация uniformity policy-выхода —
        # base (organism+shaping, до motor) vs final (action_slice, после motor).
        # Где equiprobable: base flat (shaping не пикует?) или motor_delta смазывает
        # (uniform blob, shift-инвариант)? cid → (base[16], final[16]) detached.
        self._logit_dbg: dict = {}
        # Инстинкт-развязка (03.06, Фрай): food/prey/predator DIRECTION-градиенты
        # в _shape_action_logits были ×bias_scale, а set_single_organism морозит
        # bias_scale=0 → направление ЗАНУЛЕНО (прекондишн навыка снят, приняв за
        # крутилку). Развязка: под single_organism direction = ×_instinct_dir_strength
        # (всегда on, инстинкт/перцептивный приор), а НЕ ×bias_scale. УМЕРЕННАЯ
        # сила → ориентир не диктат (мотор SFNN + insula-мост модулируют поверх).
        # context-boosts (ATTACK/FLEE/BUILD) + φ остаются curriculum под bias_scale.
        # Дефолт 0.0 = текущее зануление (деплой нейтрален); активирую/тюню flag'ом
        # client_flags instinct_dir_strength после ОК Шефа+Фрая (стратег. сдвиг).
        self._instinct_dir_strength: float = 0.0
        # Голос мотора (Фрай 03.06 curriculum): множитель motor_delta (_own) под
        # single_organism. Фаза 1 — убавить (~0.2, прайор ведёт → плотный reward →
        # мотор учит food-alignment, переучивает ATTACK-override); Фаза 2 — fade-up
        # (0.2→1.0, тест адекватности SFNN-модулятора: держит alignment или срыв?).
        # Дефолт 1.0 = полный мотор (текущее _own=1 при bias_scale=0). client_flags.
        self._motor_voice: float = 1.0
        # ИЗОЛИРУЮЩИЙ ТЕСТ override-мотора (Фрай 06.06): на on-flora тиках STAY
        # выигрывает безусловно (паркуем), мотор обычно ставят voice=0. Цель:
        # если паркуется + net flip → виноват был override мотора (не фундамент);
        # не паркуется даже при voice=0 → прайор сам не держит hold на dist=0.
        # Обратимо через client_flags (motor_park_test). 0=off (штатно).
        self._motor_park_test: float = 0.0
        # STAY-исполнение контроль (Фрай 06.06): >0 → эмитить STAY БЕЗУСЛОВНО
        # каждый тик (не только on-flora). Чистый тест протокола: Хьюберт смотрит
        # pos-delta — 0=STAY honored, ≠0=протокол-баг (P40 игнорит STAY). Обратимо.
        self._motor_stay_force: float = 0.0
        # DAMAGE-канал (Фрай 07.06): predator damage_per_tick от сервера →
        # авторитетный energy-ledger per-client-tick (§3.5-симметрично step_cost,
        # БЕЗ drop на пропущенных тиках). _damage_factor = калибровка (мал→расти,
        # обучающий сигнал не инстакил). 0=off (default, безопасно). §3 = защита.
        self._damage_factor: float = 0.0
        self._dmg_sum: float = 0.0           # Σ применённого урона (DAMAGE_DIAG)
        self._dmg_rate_sum: float = 0.0      # Σ raw damage_per_tick (давление)
        self._dmg_apply_n: int = 0           # тиков с урон>0
        # EEG tissue-activation ring (#2, 01.06.2026): нормированная [0,1]
        # активность тканей per snapshot для осциллографа /stats
        # (TissueActivityPanel). P40 world_meta.ring пуст для owned (тикаются
        # на клиенте) → клиент шлёт свой. Нормировка per-role runmax (Δ-нормы
        # разномасштабны: motor 0.17 vs amygdala 172k).
        self._tissue_activation_ring = deque(maxlen=60)  # [[v per role], ...]
        self._tissue_act_runmax: dict = {}               # role → running max
        # Lamarckian skill-growth (F5, 01.06.2026, Фрай): owned skill-growth
        # ведёт КЛИЕНТ (P40 phase-out). Окно 200 тиков: eat/kill/move counts →
        # efficiency/attack_power/move_speed растут/падают (mirror world.py:4595).
        # Внутрижизненная эволюция тела → наследуется через crossover.
        self._skill_eat: dict = {}          # cid → eat count (окно 200т)
        self._skill_kill: dict = {}         # cid → kill count
        self._skill_atk: dict = {}          # cid → melee-ATTACK count (§6 atk-growth)
        self._skill_move: dict = {}         # cid → move count
        self._skill_window_tick: dict = {}  # cid → world_tick последнего окна
        self._skill_changed_cids: set = set()  # cid с изменёнными traits → re-announce
        # Newborn-инстинкт (01.06.2026, Фрай): client-рождённые особи тянутся к
        # GATHER/EAT первые 500 тиков (scaffold, затухает) → motor_policy
        # учится есть на eat-reward, прежде чем голод. birth_tick — только для
        # mate-рождённых (старые/restored не трекаются → инстинкт=0). carried_
        # food — клиентское зеркало серверного инвентаря (P40 его не шлёт).
        self._birth_tick: dict = {}         # cid → world_tick рождения (newborn)
        self._carried_food: dict = {}       # cid → int 0..5 (зеркало world.py)
        # Bootstrap (Фрай): restored-особи омолаживаются (birth_tick=now) →
        # инстинкт активен → учатся есть тем же механизмом, что newborn (учит,
        # не дарит энергию). Self-decay: как пойдут natural-роды и колония
        # станет self-sustaining, restore станет редким → bootstrap сойдёт сам.
        self._bootstrap_pending: set = set()  # cid restored → омолодить в handle_tick
        self._n_bootstrap_rejuv: int = 0      # счётчик омоложённых (verify ratio)
        self._n_natural_newborn: int = 0      # счётчик natural-родов (verify ratio)
        # Stats colony_summary (01.06.2026, для UI /stats): агрегаты выживания/
        # эволюции/обучения → push в public_meta. last_world_tick для age,
        # deaths-by-cause кумулятивно, history — downsampled ring (~окно 2ч).
        self._last_world_tick: int = 0
        self._deaths_by_cause: dict = {"starvation": 0, "telomere": 0,
                                       "infection": 0}
        self._summary_history: deque = deque(maxlen=120)
        self._last_window: Optional[dict] = None  # последняя 300-тик энерго-точка
        # NAV_DIAG (01.06.2026, Фрай): диагностика навигации к еде ПЕРЕД портом
        # bias_scale. Корень голода — доходят ли до флоры. onf=on_flora-rate,
        # gather/gather_onf — GATHER issued + на флоре (успех), flip — shaping-
        # argmax != финал (motor_policy перебил выбор), mnorm — ||motor_delta||.
        self._nav: dict = {"ticks": 0, "onf": 0, "gather": 0, "gather_onf": 0,
                           "eat": 0, "flip": 0, "mnorm": 0.0, "p40_ate": 0,
                           "yield_fire": 0, "move": 0, "stay": 0, "cf_last": 0,
                           "cf_p40_seen": 0, "nav_hit": 0, "nav_moves": 0, "attack": 0, "flee": 0, "atk_pp_sum": 0.0, "atk_contact": 0, "flee_pp_sum": 0.0}
        # Contract per-sec (01.06.2026, Хьюберт): server чист (0.272), двоение
        # у нас — obs 6Hz vs sim 30Hz, применяли rate лишний раз. Решение: P40
        # шлёт rate в energy/СЕК, client интегрирует energy -= rate × dt_wallclock
        # между applies. Убирает tick-mismatch НАВСЕГДА, независимо от client TPS
        # (dworld был ~36 — handle_tick тяжёлый). Все 4 оси в _apply_metabolism.
        self._last_metab_wall: dict = {}     # cid → wall-clock последнего apply
        self._metab_applies: int = 0          # для METAB_DIAG
        self._metab_dt_sum: float = 0.0       # Σ dt_seconds — средний интервал
        self._metab_sc_sum: float = 0.0       # Σ step_cost_now (per-sec rate)
        # bias_scale curriculum (01.06.2026, порт server routes_world/loop.py:
        # 600-636 — Фрай/Хьюберт). Кроссфейд shaping↔motor: own_contribution =
        # max(0, 1-bias_scale) масштабирует motor_delta. Старт 1.0 (untrained →
        # shaping ведёт, motor подавлен) → decay по ratio n_alive/target →
        # motor автономен. NAV-данные подтвердили: motor перебивал shaping
        # (flip_rate 0.6, motor_norm≈shaping) → голод. Каждые 1000 world-тиков.
        self._bias_scale: float = 1.0
        self._bias_last_update_tick: int = 0
        self.last_stress: dict = {}        # cid → float ∈ [0, 1]
        self.last_dmn_floor: dict = {}     # cid → float ∈ [0, _DEFAULT_MODE_FLOOR_MAX]
        # §3.2 (Фрай 09.06.2026): последний P40-intero[7] per-cid — для carry
        # slots 2/6 (age/valence) в client-built intero + fallback, если
        # биохимии нет. См. _build_client_intero.
        self._last_p40_intero: dict = {}   # cid → np.ndarray[7]
        # Phase 5d (NEOL, 14.05.2026) — reward-gated Hebbian.
        # TD = β_local − EMA(β_local, α=0.01) per-cid, обновляется один раз за
        # тик в _compute_higher_tissues. effective_eta_reward = lr_reward ·
        # (1 + clip(td, ±0.5)). Применяется ТОЛЬКО к reward_output (motor/
        # brain/manipulator/...) через dopa_td_mult kwarg в hebbian.update.
        # Эфемерное состояние (ресет при рестарте клиента), α=0.01 сходится
        # за ~100 тиков. Если S2.E off (ablation бит 14) — beta не пишется,
        # td/ema остаются на дефолте 0.0 → mult=1.0 (без модуляции).
        self.dopamine_ema: dict = {}       # cid → float (running EMA(β_local))
        self.dopamine_td: dict = {}        # cid → float TD ∈ ~[-0.618, +0.618]

        # Brain migration v0.9.25: TPS-метрика клиента (handle_tick rate).
        self.tick_ts: deque = deque(maxlen=200)

        # SFNN S4 (14.05.2026): дефолты после удаления legacy REINFORCE+Adam.
        # set_higher_sfnn/set_motor_sfnn(False) всё ещё работают — без legacy
        # пути это "заморозить веса" (forward без update_step).
        self._higher_sfnn_default: bool = True
        self._motor_sfnn_default: bool = True

        # Single-organism pivot (01.06.2026, ТЗ e3cc81b): режим одного
        # самообучающегося Адама вместо колонии. Под флагом гейтятся
        # КОЛОНИАЛЬНЫЕ механики (репродукция/speciation/...) — код НЕ удаляется,
        # пригодится для Зоопарка Эпохи 2. Дефолт False (колониальный режим);
        # включается через cmd["flags"]["single_organism"] → set_single_organism.
        self._single_organism: bool = False
        # §3 paralysis: cid → monotonic-дедлайн снятия паралича (energy≤0).
        self._paralysis_until: dict[str, float] = {}
        # Recovery-грант энергии после паралича. Тюнится мной (Фрай): если
        # grass-only зацикливает — поднимаю (max/φ⁶≈73 / max/φ⁵≈118).
        self._recovery_energy: float = _RECOVERY_ENERGY_DEFAULT
        # Glucose→energy конверсия (Фрай 04.06, экономика): энергия=P40 delta_e
        # (yield) − step_cost (drain); glucose отдельно (apply_feed), конверсии НЕТ
        # → излишек glucose maxится и ПРОПАДАЕТ, energy net-negative даже на макс.
        # еде (glucose 99.7, energy↓). Фикс: излишек glucose (>baseline 50) → energy
        # (per-sec rate × surplus × dt, glucose потребляется но не ниже baseline).
        # Делает «плотная еда (glucose↑) → net-positive, бедная → net-negative»
        # (skill важен/достижим). Тюнится client_flags glucose_energy_rate (дефолт
        # 0=нейтрально, калибрую чтобы плотная еда давала net-positive).
        self._glucose_energy_rate: float = 0.0
        # Track 2 (этап 4): self-obs→action REINFORCE-голова. Linear(4→N_ACTIONS),
        # zero-init (non-destructive старт): мапит self-observable (голод/
        # неуверенность/паралич) в bias логитов, учится REINFORCE-наградой. Прямой
        # обучаемый путь self-observable → ДЕЙСТВИЕ (predictor его лишь моделирует;
        # базовые ткани hebbian/insula-стресс этот путь не несут — investigation).
        self.self_obs_head: dict = {}        # cid → nn.Linear(4, N_ACTIONS)
        self.self_obs_head_opt: dict = {}    # cid → Adam
        self._so_head_ctx: dict = {}         # cid → (so4, action, base_logits) prev-тик
        self._so_head_baseline: dict = {}    # cid → бегущая средняя reward (REINFORCE baseline)
        # ОТКЛЮЧЕНА 02.06.2026: live-эксперимент показал, что self-obs→action
        # голова ДЕГРАДИРУЕТ foraging (income 123→45 за 1ч, 2 точки) — высоко-
        # вариативный REINFORCE поверх работающей motor-политики добавлял шум.
        # Код сохранён для rework (lower lr / variance reduction / иная интеграция).
        # Predictor self-observable (самомодель) остаётся включён — он non-destructive.
        self._self_obs_head_enabled: bool = False

        # Track 2 / направление (б) (Фрай 02.06.2026): insula-стресс → LEARNED
        # temperature-модуляция motor-политики (НЕ actor). Безопасная форма:
        # temperature только sharpen/flatten, не меняет НАПРАВЛЕНИЕ → не повторит
        # action-head corruption. head: Linear(_SELF_OBS_DIM→1) zero-init → mu=0 →
        # T_mod≈1 (near-identity старт). 1-dim Gaussian policy REINFORCE (низкая
        # variance). Под флагом, дефолт OFF — live не трогаем до закрепления
        # viability + ОК Фрая (его порядок: viability → build dev-side → запуск).
        self.insula_temp_head: dict = {}      # cid → nn.Linear(_SELF_OBS_DIM, 1)
        self.insula_temp_head_opt: dict = {}  # cid → Adam (lr=_INSULA_TEMP_LR)
        self._it_ctx: dict = {}               # cid → (so4_tensor, log_tmod) prev-тик
        self._it_baseline: dict = {}          # cid → бегущая средняя advantage (variance-reduction)
        self._it_last_tmod: dict = {}         # cid → последний T_mod (телеметрия обучения моста)
        self._insula_temp_enabled: bool = False
        # §3.2 (Фрай 09.06.2026): felt-thirst gradual drive. False (дефолт) →
        # текущий бинарный 30% water-seek (прод не меняется до go). True →
        # градуальный felt-drive (intero[1]-афферент масштабирует приоритет
        # рефлекса A, φ-onset 0.382). Мгновенный on/off через client_flags
        # (felt_thirst_drive); kill-switch = false → откат к бинарному. См.
        # ws_client._apply_water_seek.
        self._felt_thirst_drive_enabled: bool = False

        # SFNN S6.4 (16.05.2026): per-cid SFNN-правила для 10 базовых тканей
        # organism graph. Веса самих тканей живут в HebbianController; здесь
        # хранится только evolvable rule (A/B/C/D/η + τ + R3 + td_coupling +
        # algorithm). В S6.5 будет apply-step через eligibility trace, в
        # S6.6 — диспетчер под флагом `genome.basic_tissue_sfnn_enabled`.
        self.basic_tissue_sfnn_rule: dict[str, dict] = {
            t: {} for t in _BASIC_SFNN_TISSUES
        }
        # Per-tissue счётчик SFNN apply-step'ов (S6.5+). В S6.4 — нули.
        self.basic_tissue_sfnn_steps: dict[str, int] = {
            t: 0 for t in _BASIC_SFNN_TISSUES
        }
        # Дефолт флага `basic_tissue_sfnn_enabled` для новых особей.
        # S6.11 (16.05.2026): переключён на True — миграция завершена,
        # 10 базовых ролей по умолчанию на SFNN-плейстике. cabinet.sh
        # sfnn-basic off — kill-switch для regression-debug.
        self._basic_sfnn_default: bool = True
        # SFNN S6.5 (16.05.2026): eligibility trace per (role, cid). Tensor
        # формы W_sub (slice матрицы, к которой применяется ΔW). decay = exp(-1/τ)
        # из SFNNRule.tau. Эфемерное — ресет при рестарте клиента и при
        # `reset_all()`. Хранится отдельно от HebbianController._tissue_info[trace],
        # т.к. классика этих ролей не трогает (skip_roles).
        self.basic_tissue_sfnn_trace: dict[str, dict] = {
            t: {} for t in _BASIC_SFNN_TISSUES
        }

        # Z7.i.d (16.05.2026, Зодчий): sidecar Tissue-storage'ы для трёх
        # уникальных тканей Зодчего. Заполняются только в add_creature при
        # lineage="zodchiy"; для elder/wanderer остаются пустыми. Геометрия
        # та же, что у dopamine/imagination — Tissue 21/3/1, data_dim=64.
        # Z7.i.e (16.05.2026): forward + apply-step активированы;
        # SFNNRule / step-counters / hooks / acts / row_norms / trace —
        # все в общих `higher_tissue_sfnn_*` dict'ах (см. _ALL_HIGHER выше),
        # update_step переиспользует `_higher_tissue_sfnn_update_step`.
        self.cerebellum: dict = {}    # cid → Tissue (motor error-loop)
        self.amygdala: dict = {}      # cid → Tissue (valence)
        self.episodic: dict = {}      # cid → Tissue (long-term memory)
        # Last-snapshot Зодчий-ткани → push в actions_batch.phase_emas (Z3).
        self.last_cerebellum_delta: dict = {}  # cid → torch.Tensor [16]
        self.last_amygdala_valence: dict = {}  # cid → float ∈ [-1, 1]
        self.last_episodic_recall: dict = {}   # cid → torch.Tensor [64]

        # TZ B Phase 2 (26.05.2026, Бендер): per-role Hebbian-learning
        # accumulators для observability. Накапливаются между emit cycles
        # (30с push_diagnostics) и сбрасываются в _build_hebbian_per_tissue_
        # snapshot(). См. _record_hebbian_per_tissue_sample (hook в handle_tick).
        self._heb_pt_n_total: dict[str, int] = {
            r: 0 for r in _HEB_PT_ALL_ROLES}
        self._heb_pt_n_learning: dict[str, int] = {
            r: 0 for r in _HEB_PT_ALL_ROLES}
        self._heb_pt_delta_sum: dict[str, float] = {
            r: 0.0 for r in _HEB_PT_ALL_ROLES}
        self._heb_pt_samples: dict[str, int] = {
            r: 0 for r in _HEB_PT_ALL_ROLES}

        # Body Migration Этап 3a Phase 2 (27.05.2026, Бендер): client-side
        # биохимия Z7 на owned Зодчих. Per-cid `ClientCreatureBiochem` со
        # всеми 8 эфемерными веществами + 7 baseline-генами темперамента +
        # mental_break state. Apply* / decay_step / compute_mental_break
        # вызываются напрямую из `environment.biochemistry` через duck-type
        # (math equivalence с server). Заполняется в `add_creature` для
        # `lineage="zodchiy"`, чистится в `remove_creature` / `reset_all`.
        # Для elder/wanderer dict остаётся пустым — биохимия этих lineage
        # остаётся на P40 (vision §2.5).
        self.biochem: dict = {}  # cid → ClientCreatureBiochem

        # Phase 4 этап D (30.05.2026, Бендер): client-local speciation (ТЗ §3.7).
        # cid → species_id; реестр client-local (core.tissue_speciation), lazy
        # из colonies_dir/species_registry.json. До этого speciation.py был
        # написан, но НЕ подключён → species_id всегда None в projection.
        self.species_id: dict = {}      # cid → int
        self._species_registry = None   # SpeciesRegistry, lazy

        # Phase 4 этап G (28.05.2026): cache естественный отбор capacity.
        # estimate_population() запускает CPU benchmark (~секунды), не
        # должна вызываться каждые push_diagnostics. Init once on demand.
        self._natural_selection_capacity: int | None = None

        # Phase 4 этап E+F (28.05.2026): local-only reproduction state.
        # `_last_mate_tick`: cooldown per cid (mate_detection module).
        # `_pending_newborn_envelopes`: newborns awaiting newborn_announce_ack
        #   {cid: {"parent_cids": [...], "ts_emit": float}}.
        # При successful ack — traits применяются + cooldown updates.
        # При reject — `remove_creature(cid)` cleanup.
        self._last_mate_tick: dict[str, int] = {}
        self._pending_newborn_envelopes: dict[str, dict] = {}

        # Evolved-traits recovery (30.05.2026, Бендер; согласовано Фрай/Хьюберт).
        # Authoritative per-cid стор 9 body-traits (vision_radius, smell_radius,
        # attack_radius, move_speed, attack_power, armor, efficiency, camel,
        # diet_gene). Источники наполнения:
        #   - newborn-ack (handle_newborn_announce_ack) — для client-born;
        #   - owned-handoff от P40 (ingest_owned_traits) — для founders/не-своих,
        #     traits приходят в seed_pack.meta / hello.
        # Переживает client-restart через save_state/restore_persisted_state.
        # Принцип миграции: client держит ТЕЛО как мозг → при P40 self-heal
        # (baseline bootstrap) client re-announce'ит evolved через
        # `emit_traits_announce`; P40 принимает поверх baseline. До этого
        # traits жили как loose-атрибуты на CompositeOrganism (только
        # client-born, без persist) → терялись при рестарте.
        self.traits: dict[str, dict] = {}        # cid → {9 trait полей}
        # Body Migration метаболизм (31.05.2026): cid'ы с метаболической смертью
        # (energy<=0 голод / telomere AGONY старость / hydration<=0 жажда).
        # build_projection_batch шлёт для них alive=False → P40 убирает.
        # Чистится в remove_creature (после P40-removal client GC снимет cid).
        self._dead_cids: set = set()
        # Hydration-ось отбора (31.05.2026): cid'ы, для которых P40 шлёт
        # delta_hydration (income от питья) — присутствие ключа = «система
        # питья активна». thirst-декей + hydration<=0 death применяются ТОЛЬКО
        # для активных cid (deploy-order-safe: пока P40 не шлёт delta_hydration,
        # жажда не действует → нет коллапса от монотонного декея без income).
        self._hydration_active: set = set()
        # Carrying-capacity cap (31.05.2026): cid'ы, сброшенные restore-cap'ом
        # (persist bloat сверх ёмкости). ws_client читает → owned_bye на P40
        # (despawn осиротевших проекций) → чистит после отправки.
        self._cull_bye_cids: list = []
        # Water calibration (31.05.2026, Шеф: «раньше на сервере работала»).
        # Инструментация баланса income(drink)/cost(thirst): декей жажды ВКЛ
        # для наблюдения, смерть от жажды ОТКЛ (без риска падежа). Сравниваем
        # thirst_sum (P40 thirst_now) vs drink_sum (P40 delta_hydration) +
        # тренд hydration → находим перекос, калибруем с серверной моделью.
        # ENERGY_CALIB (02.06.2026, Фрай): замер income vs cost per-тик.
        self._e_income_sum: float = 0.0   # delta_energy (eat)
        self._e_cost_sum: float = 0.0     # step_cost
        self._e_infdrain_sum: float = 0.0  # infection-drain
        # §3.5-ПОЛНОТА ledger (Фрай 07.06): компонентный net (income-cost-infdrain)
        # недосчитывал метаб-цену berserk/атак/дегидрации → ложный net+ при energy,
        # пилящей §3. net_true = реальная Δenergy за окно (authoritative, не врёт);
        # residual = net_true − net_component = СУММА непосчитанных расходов/грантов.
        # +paralysis_n: окна с §3-recovery несустейнабельны (грант +73 искажает Δ).
        self._e_window_e0: float = -1.0   # energy на старте окна (для net_true)
        self._paralysis_window_n: int = 0  # §3-recovery за окно
        self._e_srv_cost_sum: float = 0.0  # серверная per-event цена (delta_e<0)
        self._hyd_thirst_sum: float = 0.0
        self._hyd_drink_sum: float = 0.0
        self._hyd_calib_ticks: int = 0
        # Death-урон от обезвоживания (01.06.2026, Шеф: «вода влияет на общее
        # состояние и гибель»). dh_stage>=2 → energy_drain → смерть через
        # energy<=0. ОТКАЧЕНО 0.11.38: дрейн (даже ×0.1) опрокидывал маргинальный
        # eat-income в минус → энергия падала НИЖЕ порога размножения (~309) →
        # рождения вставали, смерти шли → death-спираль → вымирание (3-й раз
        # после 0.11.24/0.11.34). Death-налог на ВСЕХ ≠ отбор слабых. Контроль
        # населения теперь через ПОПУЛЯЦИОННЫЙ КЭП в detect_and_emit_mate_pairs
        # (рождения до ёмкости, без энерго-налога) → bounded self-sustaining
        # цикл как на сервере (рождения заменяют смерти, не вымирает).
        self._dehydration_damage_enabled: bool = False
        # Pending re-announce: cid'ы, для которых traits_announce отправлен и
        # ждёт ack (зеркало _pending_newborn_envelopes). Очищается в
        # handle_traits_announce_ack.
        self._pending_traits_announce: dict[str, float] = {}

        logger.info("LocalColonyCompute device=%s", self.device)

    # ── Client-local speciation (ТЗ §3.7) ───────────────────────────────

    def _ensure_species_registry(self):
        """Lazy SpeciesRegistry из colonies_dir/species_registry.json.
        False-sentinel если core.tissue_speciation недоступен."""
        if self._species_registry is None:
            try:
                from .speciation import load_or_create_registry
                self._species_registry = load_or_create_registry()
            except Exception as e:
                logger.debug("species registry load failed: %s", e)
                self._species_registry = False
        return self._species_registry or None

    @staticmethod
    def _organism_topology_genes(organism) -> list:
        """Межтканевые NEAT-гены организма как list[dict] для assign_species.
        Z2.b подключён (01.06.2026): mate-flow наполняет genes через
        crossover_org_topology_for_zodchiy → дивергенция графа по поколениям.
        Founders/первое поколение ещё []; расходятся органично (p_add=0.02)."""
        try:
            genes = getattr(organism, "tissue_topology_genes", None)
            if genes:
                return [g.to_dict() if hasattr(g, "to_dict") else dict(g)
                        for g in genes]
        except Exception:
            pass
        return []

    def _assign_species(self, cid: str, organism, tick: int = 0) -> None:
        """Назначить species_id особи через client-local registry (идемпотентно)."""
        # Single-organism pivot (ТЗ e3cc81b §1): видообразование — колониальная
        # механика (нет популяции — нет видов). Под флагом не назначаем.
        if self._single_organism:
            return
        if cid in self.species_id:
            return
        reg = self._ensure_species_registry()
        if reg is None:
            return
        try:
            from .speciation import assign_species, save_registry
            topo = self._organism_topology_genes(organism)
            sid, is_new = assign_species(
                reg, topo, tick=int(tick), founder_cid=str(cid))
            self.species_id[cid] = int(sid)
            if is_new:
                try:
                    save_registry(reg)
                except Exception as e:
                    logger.debug("save_registry failed: %s", e)
        except Exception as e:
            logger.debug("assign_species cid=%s: %s", cid, e)

    # ── Регистрация особей ───────────────────────────────────────────────

    def add_creature(self, cid: str, organism, *, hebbian_enabled: bool = True,
                     learning_rate: float = 1e-4, trace_decay: float = 0.9,
                     lineage: str = "zodchiy") -> None:
        """Зарегистрировать особь. Organism — CompositeOrganism из seed_loader.

        `lineage` (Z7.i.d, 16.05.2026): "elder" | "wanderer" | "zodchiy".
        Z8 (17.05.2026): дефолт переключён "wanderer" → "zodchiy". Новые
        клиентские особи стартуют с 20 тканями (5 brain + motor + 7 высших
        + 3 sidecar) и telomere ×φ³. Странники, оставшиеся живыми, доживают
        естественно. Explicit lineage="wanderer" нужен только в тестах,
        которые проверяют старое поведение.
        """
        from core.action_selector import ActionSelector

        if hasattr(organism, "to"):
            organism.to(self.device)
        if hasattr(organism, "eval"):
            organism.eval()
        self.organisms[cid] = organism
        # Phase 4 этап D: client-local species_id (идемпотентно; topology []
        # пока Z2.b не подключён → founder-вид). build_projection_batch шлёт его.
        self._assign_species(cid, organism)
        # SFNN S3.activate: CompositeOrganism не несёт genome — приклеиваем
        # SimpleNamespace с текущим дефолтом флага. set_higher_sfnn() позже
        # переписывает атрибут у всех существующих особей.
        if not hasattr(organism, "genome"):
            organism.genome = types.SimpleNamespace(
                higher_tissue_sfnn_enabled=self._higher_sfnn_default,
                sfnn_enabled=self._motor_sfnn_default,
                basic_tissue_sfnn_enabled=self._basic_sfnn_default,
            )
        self.action_selectors[cid] = ActionSelector()
        self.hebbian[cid] = self._make_hebbian(organism, hebbian_enabled,
                                                learning_rate, trace_decay)
        # Phase 1 — Forward Model sidecar (Tissue 21/3/1, ~8.7K params).
        pred = self._make_predictor_tissue()
        if pred is not None:
            self.predictor[cid] = pred
            self.predictor_opt[cid] = self._torch.optim.Adam(
                pred.parameters(), lr=1e-3)
            self.loss_ema[cid] = 0.0
            self.pred_loss_history[cid] = deque(maxlen=100)
            self.intrinsic_last[cid] = 0.0
            self.intrinsic_ema[cid] = 0.0
            self.entropy_ema[cid] = 0.0
            self.trace_norm_ema[cid] = 0.0
            self.reward_var_ema[cid] = 0.0
            self.reward_history[cid] = deque(maxlen=10)
        # Brain migration (10.05.2026): 4 высшие ткани S2.E/G/A/F. Создаём
        # для всех owned-cid (P40 шлёт в obs_batch только owned, лineage-гейт
        # делает сервер на fastpath; client_* ткани нейтральны для elder'а).
        self.dopamine[cid] = self._make_higher_tissue("dopamine")
        self.imagination[cid] = self._make_higher_tissue("imagination")
        self.planner[cid] = self._make_higher_tissue("planner")
        self.insula[cid] = self._make_higher_tissue("insula",
                                                     data_dim=_INSULA_DATA_DIM)
        # 13.05.2026: default_mode (S2.D) — data_dim=64, как dopamine/imagination.
        self.default_mode[cid] = self._make_higher_tissue("default_mode")
        # motor_policy — Tissue 21/3/1, обучается SFNN-правилом (S4).
        motor = self._make_higher_tissue("motor_policy")
        if motor is not None:
            self.motor_policy[cid] = motor
            # SFNN S1.1: дефолтное правило (Phase 5d-эквивалент).
            # S6.0b-A (16.05.2026): setdefault вместо прямой записи —
            # add_creature идемпотентен по SFNN-правилам. При reseed после
            # broker re-announce сервер вызывает add_creature повторно для
            # того же cid; раньше это стирало эволюционировавшее правило
            # к дефолту. apply_inherited_state ниже / mate-pair / asexual
            # перезапишут правило явно от родителя (через .pop+set).
            from core.sfnn_rule import SFNNRule
            self.motor_sfnn_rule.setdefault(cid, SFNNRule.default())
            # SFNN S1.2b: hooks для захвата pre/post 6 Linear motor_policy.
            # Регистрируются всегда (overhead — копия (pre,post) на forward),
            # используются только при genome.sfnn_enabled=True.
            self._register_motor_sfnn_hooks(cid, motor)
            # SFNN S1.2c: baseline row-norms для renormalization после ΔW.
            self.motor_sfnn_row_norms[cid] = self._snapshot_row_norms(motor)
        # S2.B (13.05.2026) — theory_of_mind: Tissue 21/3/1 data_dim=52 + Adam.
        tom = self._make_higher_tissue("theory_of_mind", data_dim=_TOM_DATA_DIM)
        if tom is not None:
            self.theory_of_mind[cid] = tom
            self.theory_of_mind_opt[cid] = self._torch.optim.Adam(
                tom.parameters(), lr=_TOM_LR)
            self.last_tom_acc[cid] = 0.0
        # S2.C (13.05.2026) — language: Tissue 21/3/1 data_dim=16 + Adam.
        lang = self._make_higher_tissue("language", data_dim=_LANG_DATA_DIM)
        if lang is not None:
            self.language[cid] = lang
            self.language_opt[cid] = self._torch.optim.Adam(
                lang.parameters(), lr=_LANG_LR)
            self.last_lang_acc[cid] = 0.0
        # SFNN S3.0 (14.05.2026): дефолтное правило (A=1, B=C=D=0, η=1e-3)
        # для каждой существующей высшей ткани. Используется только при
        # genome.higher_tissue_sfnn_enabled=True (S3.1+); в S3.0 — storage
        # + наследование через single-parent Y50.
        try:
            from core.sfnn_rule import SFNNRule
            _tissue_stores = (
                ("dopamine", self.dopamine),
                ("imagination", self.imagination),
                ("planner", self.planner),
                ("insula", self.insula),
                ("default_mode", self.default_mode),
                ("theory_of_mind", self.theory_of_mind),
                ("language", self.language),
            )
            for _name, _store in _tissue_stores:
                if _store.get(cid) is not None:
                    # S6.0b-A: setdefault — идемпотентность по правилам.
                    self.higher_tissue_sfnn_rule[_name].setdefault(
                        cid, SFNNRule.default())
        except Exception as e:
            logger.debug("add_creature %s higher_tissue_sfnn_rule init: %s", cid, e)
        # SFNN S6.4 (16.05.2026): per-role SFNNRule для 10 базовых тканей
        # organism graph. Источник: серийный dict в genome.<role>_sfnn_rule
        # (если есть и валидный) → SFNNRule.from_dict; иначе — дефолты
        # ROLE_DEFAULTS через SFNNRule.for_role(role). Storage идемпотентен
        # к повторному add_creature (broker re-announce) — setdefault.
        try:
            from core.sfnn_rule import SFNNRule
            for _role, _field in _BASIC_SFNN_GENOME_FIELD.items():
                _rule_d = getattr(organism.genome, _field, None)
                if isinstance(_rule_d, dict):
                    try:
                        _rule = SFNNRule.from_dict(_rule_d)
                    except Exception as _e:
                        logger.debug(
                            "add_creature %s basic_sfnn %s decode: %s — using role defaults",
                            cid, _role, _e)
                        _rule = SFNNRule.for_role(_role)
                else:
                    _rule = SFNNRule.for_role(_role)
                self.basic_tissue_sfnn_rule[_role].setdefault(cid, _rule)
        except Exception as e:
            logger.debug("add_creature %s basic_tissue_sfnn_rule init: %s",
                          cid, e)
        # SFNN S3.1 (14.05.2026): hooks для dopamine — первой активной высшей
        # ткани под флагом higher_tissue_sfnn_enabled. Hooks дёшевы и
        # регистрируются всегда (при выключенном флаге активации просто
        # перезаписываются).
        if self.dopamine.get(cid) is not None:
            try:
                self._register_higher_tissue_sfnn_hooks(
                    "dopamine", cid, self.dopamine[cid])
            except Exception as e:
                logger.debug("add_creature %s dopamine sfnn hooks: %s",
                              cid, e)
        # SFNN S3.2 (14.05.2026): hooks для imagination — второй активной
        # высшей ткани. Та же логика, дёшево, под тем же общим флагом.
        if self.imagination.get(cid) is not None:
            try:
                self._register_higher_tissue_sfnn_hooks(
                    "imagination", cid, self.imagination[cid])
            except Exception as e:
                logger.debug("add_creature %s imagination sfnn hooks: %s",
                              cid, e)
        # SFNN S3.3 (14.05.2026): hooks для planner — третьей активной
        # высшей ткани.
        if self.planner.get(cid) is not None:
            try:
                self._register_higher_tissue_sfnn_hooks(
                    "planner", cid, self.planner[cid])
            except Exception as e:
                logger.debug("add_creature %s planner sfnn hooks: %s",
                              cid, e)
        # SFNN S3.4 (14.05.2026): hooks для insula — четвёртой активной.
        # Forward в _compute_higher_tissues идёт ТОЛЬКО при intero_tensor
        # is not None — если P40 не шлёт интероцепцию, acts не обновятся
        # и update_step станет no-op (что корректно).
        if self.insula.get(cid) is not None:
            try:
                self._register_higher_tissue_sfnn_hooks(
                    "insula", cid, self.insula[cid])
            except Exception as e:
                logger.debug("add_creature %s insula sfnn hooks: %s",
                              cid, e)
        # SFNN S3.5 (14.05.2026): hooks для default_mode — пятой активной.
        if self.default_mode.get(cid) is not None:
            try:
                self._register_higher_tissue_sfnn_hooks(
                    "default_mode", cid, self.default_mode[cid])
            except Exception as e:
                logger.debug("add_creature %s default_mode sfnn hooks: %s",
                              cid, e)
        # SFNN S3.6 (14.05.2026): hooks для theory_of_mind — шестой активной.
        # Особенность: ткань имеет Adam-оптимизатор (supervised по Δposition
        # focus-соседа). Под флагом higher_tissue_sfnn_enabled Adam-шаг в
        # _compute_theory_of_mind пропускается, forward остаётся — чтобы
        # SFNN-hooks словили активации.
        if self.theory_of_mind.get(cid) is not None:
            try:
                self._register_higher_tissue_sfnn_hooks(
                    "theory_of_mind", cid, self.theory_of_mind[cid])
            except Exception as e:
                logger.debug("add_creature %s theory_of_mind sfnn hooks: %s",
                              cid, e)
        # SFNN S3.7 (14.05.2026): hooks для language — седьмой и последней.
        # Та же особенность что и tom: Adam-оптимизатор (supervised по
        # prev_context → event_class). Под флагом Adam-шаг пропускается,
        # forward выполняется для активации SFNN-hooks.
        if self.language.get(cid) is not None:
            try:
                self._register_higher_tissue_sfnn_hooks(
                    "language", cid, self.language[cid])
            except Exception as e:
                logger.debug("add_creature %s language sfnn hooks: %s",
                              cid, e)
        # Z7.i.d (16.05.2026, Зодчий): для lineage="zodchiy" создаём три
        # уникальные ткани третьей линии — cerebellum/amygdala/episodic.
        # Z7.i.e (16.05.2026): forward + SFNN apply-step активны.
        # Tissue 21/3/1 data_dim=64, SFNN-правило берём из ROLE_DEFAULTS
        # через for_role (τ/R3/TD/algorithm), hooks регистрируются всегда —
        # apply-step гейтит флаг `genome.higher_tissue_sfnn_enabled`.
        if str(lineage) == "zodchiy":
            try:
                self.cerebellum[cid] = self._make_higher_tissue("cerebellum")
                self.amygdala[cid] = self._make_higher_tissue("amygdala")
                self.episodic[cid] = self._make_higher_tissue("episodic")
            except Exception as e:
                logger.warning("add_creature %s zodchiy tissues: %s", cid, e)
            try:
                from core.sfnn_rule import SFNNRule
                # Z7.i.d.1 (16.05.2026): ROLE_DEFAULTS для Зодчий-тканей несут
                # ненулевые R3/TD/τ (cerebellum τ=21/imm=0.7/td=1.0,
                # amygdala τ=233/med=0.6/td=1.0, episodic τ=233/long=0.6/td=0).
                # `default()` обнулил бы их → правило стартовало бы как
                # classical Hebbian и эволюция тратила бы время на дрейф к
                # роль-специфичным значениям. `for_role` поднимает их сразу.
                for _t in _ZODCHIY_EXTRA_TISSUES:
                    if getattr(self, _t).get(cid) is not None:
                        self.higher_tissue_sfnn_rule[_t].setdefault(
                            cid, SFNNRule.for_role(_t))
            except Exception as e:
                logger.debug(
                    "add_creature %s zodchiy higher_tissue_sfnn_rule init: %s",
                    cid, e)
            # Z7.i.e (16.05.2026): forward-hooks для 3 Зодчий-тканей. Та же
            # логика, что и у 7 высших Странника — hooks дёшевы, регистрируем
            # всегда; apply-step гейтит флаг higher_tissue_sfnn_enabled.
            for _t in _ZODCHIY_EXTRA_TISSUES:
                _tissue = getattr(self, _t).get(cid)
                if _tissue is None:
                    continue
                try:
                    self._register_higher_tissue_sfnn_hooks(_t, cid, _tissue)
                except Exception as e:
                    logger.debug(
                        "add_creature %s %s sfnn hooks: %s", cid, _t, e)
            # Phase 2 (27.05.2026, Бендер): инициализация client-side
            # биохимии Z7 для zodchiy. Идемпотентно — повторный add_creature
            # после reseed/respawn не сбрасывает существующее состояние.
            if cid not in self.biochem:
                try:
                    from .biochemistry import make_default
                    self.biochem[cid] = make_default()
                except Exception as e:
                    # Warning (не debug) чтобы видеть в production если
                    # neurocore[client] not available на embedded Python.
                    logger.warning(
                        "biochem init failed %s: %s", cid, e)
            # Phase 4 этап B (27.05.2026, Бендер): episodic memory
            # persistence через client restart. Если для этого cid
            # есть сохранённый state на диске — загружаем в свежесозданную
            # episodic ткань. Idempotent: повторный add_creature тоже
            # вызывает load, но overwrite допустимо (state файл — source
            # of truth, in-memory копия — производное).
            try:
                from .memory_store import (
                    load_memory_state, apply_memory_state_to_tissue,
                )
                payload = load_memory_state(cid)
                if payload is not None:
                    epi = self.episodic.get(cid)
                    ok = apply_memory_state_to_tissue(payload, epi)
                    if ok:
                        recall = payload.get("last_episodic_recall")
                        if recall is not None:
                            self.last_episodic_recall[cid] = recall
                        logger.info(
                            "memory restore %s: episodic loaded ts_saved=%s",
                            cid, payload.get("ts_saved"))
            except Exception as e:
                logger.warning("memory restore %s failed: %s", cid, e)
        logger.info(
            "add_creature %s lineage=%s n_tissues=%d predictor=%s S2=%s zodchiy=%s",
            cid, lineage, getattr(organism, "n_tissues", 0), pred is not None,
            all(self.dopamine.get(cid) is not None for _ in [0]),
            self.cerebellum.get(cid) is not None)

    def remove_creature(self, cid: str) -> None:
        self.organisms.pop(cid, None)
        self.action_selectors.pop(cid, None)
        self.hebbian.pop(cid, None)
        self.predictor.pop(cid, None)
        self.predictor_opt.pop(cid, None)
        self._cerebellum_tid.pop(cid, None)
        self._cerebellum_out.pop(cid, None)
        self._cerebellum_hooked.discard(cid)
        self._growth_state.pop(cid, None)
        self._growth_intr_hist.pop(cid, None)
        self._growth_stagnation_n.pop(cid, None)
        self._growth_rejected.pop(cid, None)
        self._tissue_growth_state.pop(cid, None)   # §10.8 рост тканей cleanup
        self._tissue_grown_specs.pop(cid, None)
        self.prev_obs.pop(cid, None)
        self.loss_ema.pop(cid, None)
        self.pred_loss_history.pop(cid, None)
        self.intrinsic_last.pop(cid, None)
        self.intrinsic_ema.pop(cid, None)
        self.entropy_ema.pop(cid, None)
        self.trace_norm_ema.pop(cid, None)
        self.reward_var_ema.pop(cid, None)
        self.reward_history.pop(cid, None)
        self._paralysis_until.pop(cid, None)  # §3 paralysis cleanup
        self.self_obs_head.pop(cid, None)     # Track 2 head cleanup
        self.self_obs_head_opt.pop(cid, None)
        self._so_head_ctx.pop(cid, None)
        self._so_head_baseline.pop(cid, None)
        self.insula_temp_head.pop(cid, None)  # Track 2 (б) insula-temp cleanup
        self.insula_temp_head_opt.pop(cid, None)
        self._it_ctx.pop(cid, None)
        self._it_baseline.pop(cid, None)
        # Brain migration (10.05.2026): высшие ткани S2.E/G/A/F.
        self.dopamine.pop(cid, None)
        self.imagination.pop(cid, None)
        self.planner.pop(cid, None)
        self.insula.pop(cid, None)
        self._last_p40_intero.pop(cid, None)   # §3.2 carry cleanup
        # 13.05.2026: S2.D default_mode.
        self.default_mode.pop(cid, None)
        self.last_beta_local.pop(cid, None)
        self.last_imag_mult.pop(cid, None)
        self.last_planner_delta.pop(cid, None)
        self.last_stress.pop(cid, None)
        self.last_dmn_floor.pop(cid, None)
        # Phase 5d — reward-gated Hebbian per-cid state.
        self.dopamine_ema.pop(cid, None)
        self.dopamine_td.pop(cid, None)
        # motor_policy + SFNN-state (S4).
        self.motor_policy.pop(cid, None)
        self.motor_sfnn_rule.pop(cid, None)
        # SFNN S1.2b: снять forward-hooks и забыть кешированные активации.
        for h in self.motor_sfnn_hook_handles.pop(cid, []) or []:
            try:
                h.remove()
            except Exception:
                pass
        self.motor_sfnn_acts.pop(cid, None)
        # SFNN S1.2c: forget baseline row-norms.
        self.motor_sfnn_row_norms.pop(cid, None)
        self.last_motor_delta.pop(cid, None)
        self._motor_reward_baseline.pop(cid, None)
        self._motor_adv_ema.pop(cid, None)
        self._motor_dw_ema.pop(cid, None)
        self._motor_dw_last.pop(cid, None)        # ΔW-инструментирование (Фрай 04.06)
        self._motor_dw_radial_ema.pop(cid, None)
        self._motor_dw_cos_ema.pop(cid, None)
        self._motor_pg_ctx.pop(cid, None)         # policy-gradient контекст (Фрай 04.06)
        self.motor_slow_trainer.pop(cid, None)    # медленный канал (Фрай 05.06)
        self._slow_pending.pop(cid, None)
        self._tile_yield_mem.pop(cid, None)       # «доедай» yield-память (Фрай 06.06)
        self._logit_dbg.pop(cid, None)
        self._skill_eat.pop(cid, None)
        self._skill_kill.pop(cid, None)
        self._skill_atk.pop(cid, None)
        self._skill_move.pop(cid, None)
        self._skill_window_tick.pop(cid, None)
        self._skill_changed_cids.discard(cid)
        # Newborn-инстинкт (Фрай): снять трекинг рождения + зеркало carried_food.
        self._birth_tick.pop(cid, None)
        self._carried_food.pop(cid, None)
        self._last_metab_wall.pop(cid, None)  # contract per-sec dt-интеграция
        # S2.B — theory_of_mind.
        self.theory_of_mind.pop(cid, None)
        self.theory_of_mind_opt.pop(cid, None)
        self.last_tom_acc.pop(cid, None)
        self._tom_prev_focus.pop(cid, None)
        # S2.C — language.
        self.language.pop(cid, None)
        self.language_opt.pop(cid, None)
        self.last_lang_acc.pop(cid, None)
        self._lang_prev_context.pop(cid, None)
        # SFNN S3.0: вычистить правила высших тканей.
        # Z7.i.e (16.05.2026): + 3 Зодчий-ткани (cerebellum/amygdala/episodic)
        # — их state теперь в общих higher_tissue_sfnn_* dict'ах.
        for _t in _HIGHER_SFNN_TISSUES + _ZODCHIY_EXTRA_TISSUES:
            self.higher_tissue_sfnn_rule.get(_t, {}).pop(cid, None)
        # SFNN S6.4: вычистить правила 10 базовых тканей.
        for _t in _BASIC_SFNN_TISSUES:
            self.basic_tissue_sfnn_rule.get(_t, {}).pop(cid, None)
        # SFNN S6.5: вычистить eligibility traces 10 базовых тканей.
        for _t in _BASIC_SFNN_TISSUES:
            self.basic_tissue_sfnn_trace.get(_t, {}).pop(cid, None)
        # SFNN S3.1: снять forward-hooks и забыть кешированные активации
        # всех зарегистрированных в S3.x тканей.
        # Z7.i.e (16.05.2026): + 3 Зодчий-ткани через _ALL_HIGHER.
        for _t in _HIGHER_SFNN_TISSUES + _ZODCHIY_EXTRA_TISSUES:
            for h in (self.higher_tissue_sfnn_hook_handles.get(_t, {})
                      .pop(cid, []) or []):
                try:
                    h.remove()
                except Exception:
                    pass
            self.higher_tissue_sfnn_acts.get(_t, {}).pop(cid, None)
            # SFNN S1.2c: cleanup baseline row-norms каждой высшей ткани.
            self.higher_tissue_sfnn_row_norms.get(_t, {}).pop(cid, None)
            # Z1 (Зодчий, 16.05.2026): eligibility trace тоже эфемерно.
            self.higher_tissue_sfnn_trace.get(_t, {}).pop(cid, None)
        # Z7.i.d (16.05.2026): три zodchiy-ткани (только Tissue + last-snap).
        # Memory persistence НЕ делаем здесь — remove_creature вызывается
        # на смерть (permanent, не вернётся), а сохранение нужно только
        # перед shutdown (см. `persist_all_memory` в reset_all и
        # explicit shutdown hook в main.py).
        self.cerebellum.pop(cid, None)
        self.amygdala.pop(cid, None)
        self.episodic.pop(cid, None)
        self.last_cerebellum_delta.pop(cid, None)
        self.last_amygdala_valence.pop(cid, None)
        self.last_episodic_recall.pop(cid, None)
        # Phase 2 (27.05.2026, Бендер): client-side биохимия Z7.
        self.biochem.pop(cid, None)
        self.species_id.pop(cid, None)
        # Evolved-traits recovery (30.05.2026): drop стор + pending re-announce.
        self.traits.pop(cid, None)
        self._pending_traits_announce.pop(cid, None)
        # Body Migration метаболизм (31.05.2026): drop death-mark + hydration.
        self._dead_cids.discard(cid)
        self._hydration_active.discard(cid)

    def persist_all_memory(self) -> int:
        """Phase 4 этап B: сохранить episodic state всех живых организмов.

        Вызывается перед shutdown (`reset_all`, main.py SIGTERM handler).
        Save на death (`remove_creature`) не нужен — мёртвый cid не
        вернётся.

        Returns:
            Число успешно сохранённых организмов.
        """
        try:
            from .memory_store import save_memory_state
        except Exception as e:
            logger.warning("persist_all_memory: import failed: %s", e)
            return 0
        n_saved = 0
        for cid, epi in list(self.episodic.items()):
            if epi is None:
                continue
            try:
                path = save_memory_state(
                    cid, epi, self.last_episodic_recall.get(cid))
                if path is not None:
                    n_saved += 1
            except Exception as e:
                logger.debug("persist_all_memory %s failed: %s", cid, e)
        if n_saved:
            logger.info("persist_all_memory: %d organisms saved", n_saved)
        return n_saved

    def reset_all(self) -> int:
        # Phase 4 этап B: сохранить episodic memory ДО pop организмов.
        # reset_all = orderly shutdown semantics (отличается от
        # remove_creature, который вызывается на смерть).
        try:
            self.persist_all_memory()
        except Exception as e:
            logger.warning("reset_all: persist_all_memory failed: %s", e)
        n = len(self.organisms)
        for cid in list(self.organisms.keys()):
            self.remove_creature(cid)
        self.hebbian_updates = 0
        self.predictor_steps = 0
        self.motor_sfnn_steps = 0
        # SFNN S3.0: сбросить счётчики апдейтов всех 7 высших тканей.
        # Z7.i.e (16.05.2026): + 3 Зодчий-ткани в общем _ALL_HIGHER.
        for _t in _HIGHER_SFNN_TISSUES + _ZODCHIY_EXTRA_TISSUES:
            self.higher_tissue_sfnn_steps[_t] = 0
        # SFNN S6.4: сбросить счётчики 10 базовых тканей.
        for _t in _BASIC_SFNN_TISSUES:
            self.basic_tissue_sfnn_steps[_t] = 0
        # SFNN S6.5: сбросить eligibility traces 10 базовых тканей.
        for _t in _BASIC_SFNN_TISSUES:
            self.basic_tissue_sfnn_trace[_t] = {}
        # Z1 (Зодчий, 16.05.2026): сбросить eligibility traces высших тканей.
        # Z7.i.e (16.05.2026): + 3 Зодчий-ткани в общем _ALL_HIGHER.
        for _t in _HIGHER_SFNN_TISSUES + _ZODCHIY_EXTRA_TISSUES:
            self.higher_tissue_sfnn_trace[_t] = {}
        # Body Migration Phase 2 (27.05.2026, Бендер): client-side биохимия.
        # remove_creature уже очистил per-cid записи, но clear на всякий
        # случай (defense-in-depth от stale records при partial reset).
        self.biochem.clear()
        self.species_id.clear()  # реестр видов на диске сохраняется
        # Evolved-traits recovery (30.05.2026): remove_creature уже очистил
        # per-cid, clear на всякий случай (defense-in-depth от stale records).
        self.traits.clear()
        self._pending_traits_announce.clear()
        # TZ B Phase 2 (26.05.2026, Бендер): сбросить per-role tracker.
        for _r in _HEB_PT_ALL_ROLES:
            self._heb_pt_n_total[_r] = 0
            self._heb_pt_n_learning[_r] = 0
            self._heb_pt_delta_sum[_r] = 0.0
            self._heb_pt_samples[_r] = 0
        return n

    def apply_inherited_state(self, cid: str, payload: dict) -> None:
        """Phase F3.1.b: накатить унаследованные state_dicts на уже
        зарегистрированную особь.

        Payload — то, что вернул `seed_loader.organism_from_weights`.
        Содержит ключи 'hebbian' / 'selector' / 'predictor' от родителя
        на P40. Что отсутствует — пропускается без ошибок.
        """
        if cid not in self.organisms:
            logger.warning("apply_inherited_state: cid=%s unknown (skip)", cid)
            return
        heb_sd = payload.get("hebbian")
        if heb_sd is not None and self.hebbian.get(cid) is not None:
            try:
                self.hebbian[cid].load_state_dict(heb_sd)
            except Exception as e:
                logger.warning("apply_inherited_state %s hebbian: %s", cid, e)
        sel_sd = payload.get("selector")
        sel = self.action_selectors.get(cid)
        if sel_sd is not None and sel is not None and hasattr(sel, "load_state_dict"):
            try:
                sel.load_state_dict(sel_sd)
            except Exception as e:
                logger.warning("apply_inherited_state %s selector: %s", cid, e)
        # SFNN S1.1: motor_sfnn_rule — наследуем правило родителя, мутируем
        # для diversity (σ=0.1). Если правила в payload нет — оставляем
        # дефолтное, созданное в add_creature.
        rule_d = payload.get("motor_sfnn_rule")
        if rule_d is not None and self.motor_sfnn_rule.get(cid) is not None:
            try:
                from core.sfnn_rule import SFNNRule
                parent_rule = SFNNRule.from_dict(rule_d)
                self.motor_sfnn_rule[cid] = parent_rule.mutate(sigma=0.1)
            except Exception as e:
                logger.warning("apply_inherited_state %s motor_sfnn_rule: %s",
                                cid, e)
        # SFNN S3.0 (14.05.2026): то же наследование для 7 высших тканей.
        # Под ключом "higher_tissue_sfnn_rules" в payload — словарь
        # {tissue_name: SFNNRule.to_dict()}. Каждое правило мутируется σ=0.1.
        higher_rules_d = payload.get("higher_tissue_sfnn_rules")
        if higher_rules_d:
            try:
                from core.sfnn_rule import SFNNRule
                for _t, _rule_d in higher_rules_d.items():
                    if _t not in self.higher_tissue_sfnn_rule:
                        continue
                    if self.higher_tissue_sfnn_rule[_t].get(cid) is None:
                        continue
                    parent_rule = SFNNRule.from_dict(_rule_d)
                    self.higher_tissue_sfnn_rule[_t][cid] = parent_rule.mutate(sigma=0.1)
            except Exception as e:
                logger.warning(
                    "apply_inherited_state %s higher_tissue_sfnn_rules: %s",
                    cid, e)
        # SFNN S6.7 (16.05.2026): аналогичное наследование 10 базовых тканей
        # (sensory/attention/brain/memory/consciousness/communication/motor/
        # manipulator/digestive/immune). Ключ "basic_tissue_sfnn_rules" в
        # payload — словарь {role: SFNNRule.to_dict()}, σ=0.1 мутация.
        basic_rules_d = payload.get("basic_tissue_sfnn_rules")
        if basic_rules_d:
            try:
                from core.sfnn_rule import SFNNRule
                for _role, _rule_d in basic_rules_d.items():
                    if _role not in self.basic_tissue_sfnn_rule:
                        continue
                    if self.basic_tissue_sfnn_rule[_role].get(cid) is None:
                        continue
                    parent_rule = SFNNRule.from_dict(_rule_d)
                    self.basic_tissue_sfnn_rule[_role][cid] = (
                        parent_rule.mutate(sigma=0.1))
            except Exception as e:
                logger.warning(
                    "apply_inherited_state %s basic_tissue_sfnn_rules: %s",
                    cid, e)
        # Phase 1 — Y50 наследование predictor от родителя.
        pred_sd = payload.get("predictor")
        if pred_sd is not None and self.predictor.get(cid) is not None:
            try:
                # Сначала просто загружаем веса родителя (robust к Track2 obs68).
                self._load_predictor_sd(cid, pred_sd)
                # Затем применяем Y50 noise: 0.5·parent + 0.5·noise(σ·std).
                self._apply_y50_to_predictor(self.predictor[cid])
                # Y50 поломал параметры — нужен свежий optimizer.
                self.predictor_opt[cid] = self._torch.optim.Adam(
                    self.predictor[cid].parameters(), lr=1e-3)
            except Exception as e:
                logger.warning("apply_inherited_state %s predictor: %s", cid, e)
        # Phase 1/2/6 — наследование EMA-агрегатов (как _inherit_member_emas на P40).
        if cid in self.loss_ema:
            for key, target in (
                ("predictor_loss_ema", "loss_ema"),
                ("intrinsic_ema", "intrinsic_ema"),
                ("entropy_ema", "entropy_ema"),
                ("trace_norm_ema", "trace_norm_ema"),
                ("reward_var_ema", "reward_var_ema"),
                ("tom_acc", "last_tom_acc"),
                ("lang_acc", "last_lang_acc"),
            ):
                if key in payload:
                    try:
                        getattr(self, target)[cid] = float(payload[key])
                    except Exception:
                        pass
        # Brain migration (10.05.2026): Y50 для высших тканей S2.E/G/A/F.
        # 13.05.2026: + default_mode (S2.D мигрирована с P40 на клиент).
        # Z7.i.e (16.05.2026): + 3 Зодчий-ткани (Y50 применяется только если
        # ткань уже создана у child'а через add_creature(lineage="zodchiy");
        # для wanderer/elder store.get(cid) is None → skip).
        for key, store in (
            ("dopamine", self.dopamine),
            ("imagination", self.imagination),
            ("planner", self.planner),
            ("insula", self.insula),
            ("default_mode", self.default_mode),
            ("cerebellum", self.cerebellum),
            ("amygdala", self.amygdala),
            ("episodic", self.episodic),
        ):
            sd = payload.get(key)
            tissue = store.get(cid)
            if sd is None or tissue is None:
                continue
            try:
                tissue.load_state_dict(sd)
                self._apply_y50_to_tissue(tissue)
            except Exception as e:
                logger.warning("apply_inherited_state %s %s: %s", cid, key, e)
        # motor_policy — Y50 (S4: без Adam, SFNN-правило применяет ΔW сам).
        m_sd = payload.get("motor_policy")
        motor = self.motor_policy.get(cid)
        if m_sd is not None and motor is not None:
            try:
                motor.load_state_dict(m_sd)
                self._apply_y50_to_tissue(motor)
            except Exception as e:
                logger.warning("apply_inherited_state %s motor_policy: %s",
                                cid, e)
        # S2.B — theory_of_mind: Y50 + свежий Adam.
        tom_sd = payload.get("theory_of_mind")
        tom_tissue = self.theory_of_mind.get(cid)
        if tom_sd is not None and tom_tissue is not None:
            try:
                tom_tissue.load_state_dict(tom_sd)
                self._apply_y50_to_tissue(tom_tissue)
                self.theory_of_mind_opt[cid] = self._torch.optim.Adam(
                    tom_tissue.parameters(), lr=_TOM_LR)
            except Exception as e:
                logger.warning("apply_inherited_state %s theory_of_mind: %s",
                                cid, e)
        # S2.C — language: Y50 + свежий Adam.
        lang_sd = payload.get("language")
        lang_tissue = self.language.get(cid)
        if lang_sd is not None and lang_tissue is not None:
            try:
                lang_tissue.load_state_dict(lang_sd)
                self._apply_y50_to_tissue(lang_tissue)
                self.language_opt[cid] = self._torch.optim.Adam(
                    lang_tissue.parameters(), lr=_LANG_LR)
            except Exception as e:
                logger.warning("apply_inherited_state %s language: %s",
                                cid, e)

    def restore_persisted_state(self, cid: str, payload: dict) -> None:
        """Colony Ownership Migration §5.1: bit-exact restore from local .pt.

        В отличие от `apply_inherited_state` (Y50 noise для дочернего
        organism), здесь restore-from-disk применяет state **as-is** —
        тот же organism, не ребёнок. Дополнительно восстанавливает
        biochem (8 эфемерид + 7 baseline + mental_break).

        Pre-requisite: `add_creature(cid, organism, lineage)` уже
        выполнен с tissues от родителя/seed. Здесь только load_state_dict
        на нейронные модули + biochem restore.
        """
        if cid not in self.organisms:
            logger.warning("restore_persisted_state: cid=%s unknown (skip)", cid)
            return
        org = self.organisms[cid]
        # §10.8 рост тканей: ПЕРЕСОЗДАТЬ выросшие узлы ДО загрузки tissues_by_role
        # (скелет фикс → роли grownN в нём нет → без recreate restore пропустит их
        # веса). Создаём ткань из спека → tissues_by_role накатит веса (по роли) →
        # overlay (ниже) проведёт {роль}→cerebellum (ген уже в topology_genes).
        # Durable-инвариант: рост узлов переживает рестарт/сбой питания.
        _gl0 = payload.get("growth_loop")
        if isinstance(_gl0, dict) and (_gl0.get("grown_tissues") and hasattr(org, "tissues")):
            _specs_ok = []
            for _sp in (_gl0.get("grown_tissues") or []):
                try:
                    _role = str(_sp.get("role"))
                    _dd = int(_sp.get("data_dim", 64))
                    _ne = int(_sp.get("n_embd", 21))
                    _t = self._make_higher_tissue(_role, data_dim=_dd, n_embd=_ne)
                    if _t is not None:
                        org.tissues[f"gt_{_role}"] = _t
                        _specs_ok.append({"role": _role, "data_dim": _dd, "n_embd": _ne})
                except Exception as e:
                    logger.debug("restore grown tissue %s: %s", cid, e)
            if _specs_ok:
                self._tissue_grown_specs[cid] = _specs_ok
                _maxn = self._tissue_propose_count
                for _s in _specs_ok:
                    try:
                        _maxn = max(_maxn, int(str(_s["role"]).replace("grown", "")))
                    except Exception:
                        pass
                self._tissue_propose_count = _maxn
        # tissues_by_role — bit-exact (no Y50)
        tbr = payload.get("tissues_by_role")
        if isinstance(tbr, dict) and tbr:
            role_to_tid = {
                getattr(t, "role", ""): tid for tid, t in org.tissues.items()
            }
            for role, sd in tbr.items():
                tid = role_to_tid.get(role)
                if tid is None:
                    continue
                try:
                    org.tissues[tid].load_state_dict(sd)
                except Exception as e:
                    logger.debug("restore_persisted_state %s tissue %s: %s",
                                  cid, role, e)
        # Z2.b (01.06.2026, Фрай): восстановить межтканевые topology genes и
        # пере-наложить overlay на org.connections (иначе resume/elite даёт
        # дефолтный граф, divergence теряется). add_creature уже назначил
        # species по ПУСТЫМ генам (founder) — после restore графа переназначаем.
        genes_dicts = payload.get("tissue_topology_genes")
        if genes_dicts:
            try:
                from core.tissue_topology import (
                    TissueConnectionGene, apply_topology_overlay_to_org)
                org.tissue_topology_genes = [
                    TissueConnectionGene.from_dict(d) for d in genes_dicts]
                apply_topology_overlay_to_org(org)
                self.species_id.pop(cid, None)
                self._assign_species(cid, org)
            except Exception as e:
                logger.debug("restore_persisted_state %s topology: %s", cid, e)
        # Hebbian / selector — direct load_state_dict
        heb_sd = payload.get("hebbian")
        if heb_sd is not None and self.hebbian.get(cid) is not None:
            try:
                self.hebbian[cid].load_state_dict(heb_sd)
            except Exception as e:
                logger.debug("restore_persisted_state %s hebbian: %s", cid, e)
        sel_sd = payload.get("selector")
        sel = self.action_selectors.get(cid)
        if sel_sd is not None and sel is not None and hasattr(sel, "load_state_dict"):
            try:
                sel.load_state_dict(sel_sd)
            except Exception as e:
                logger.debug("restore_persisted_state %s selector: %s", cid, e)
        # Predictor — без Y50, exact
        pred_sd = payload.get("predictor")
        if pred_sd is not None and self.predictor.get(cid) is not None:
            try:
                self._load_predictor_sd(cid, pred_sd)  # robust к Track2 obs68
                self.predictor_opt[cid] = self._torch.optim.Adam(
                    self.predictor[cid].parameters(), lr=1e-3)
            except Exception as e:
                logger.debug("restore_persisted_state %s predictor: %s", cid, e)
        # EMA aggregates
        for key, target in (
            ("predictor_loss_ema", "loss_ema"),
            ("intrinsic_ema", "intrinsic_ema"),
            ("entropy_ema", "entropy_ema"),
            ("trace_norm_ema", "trace_norm_ema"),
            ("reward_var_ema", "reward_var_ema"),
        ):
            if key in payload:
                try:
                    getattr(self, target)[cid] = float(payload[key])
                except Exception:
                    pass
        # §6 рост мозга durable (Фрай 09.06): восстановить ПРОГРЕСС петли роста
        # (kept/reverted KPI + трейлинг-floor deque + стагнация). predictor цел →
        # floor-история валидна → PROPOSE возобновляется без re-warm. См. save_state.
        _gl = payload.get("growth_loop")
        if isinstance(_gl, dict):
            try:
                self._growth_kept = int(_gl.get("kept", self._growth_kept))
                self._growth_reverted = int(_gl.get("reverted", self._growth_reverted))
                self._tissue_kept = int(_gl.get("tissue_kept", self._tissue_kept))
                self._tissue_reverted = int(
                    _gl.get("tissue_reverted", self._tissue_reverted))
                _hist = _gl.get("intr_hist") or []
                if _hist:
                    self._growth_intr_hist[cid] = deque(
                        (float(x) for x in _hist),
                        maxlen=self._growth_intr_window)
                self._growth_stagnation_n[cid] = int(_gl.get("stagnation_n", 0))
            except Exception as e:
                logger.debug("restore_persisted_state %s growth_loop: %s", cid, e)
        # Higher tissues (brain migration) — exact, без Y50
        for key, store in (
            ("dopamine", self.dopamine),
            ("imagination", self.imagination),
            ("planner", self.planner),
            ("insula", self.insula),
            ("motor_policy", self.motor_policy),
            ("theory_of_mind", self.theory_of_mind),
            ("language", self.language),
        ):
            sd = payload.get(key)
            tissue = store.get(cid)
            if sd is None or tissue is None:
                continue
            try:
                tissue.load_state_dict(sd)
            except Exception as e:
                logger.debug("restore_persisted_state %s %s: %s", cid, key, e)
        # motor_sfnn_rule / higher rules — direct apply
        rule_d = payload.get("motor_sfnn_rule")
        if rule_d is not None:
            try:
                from core.sfnn_rule import SFNNRule
                self.motor_sfnn_rule[cid] = SFNNRule.from_dict(rule_d)
            except Exception as e:
                logger.debug("restore_persisted_state %s motor_sfnn_rule: %s", cid, e)
        higher_rules_d = payload.get("higher_tissue_sfnn_rules")
        if higher_rules_d:
            try:
                from core.sfnn_rule import SFNNRule
                for _t, _rule_d in higher_rules_d.items():
                    if _t in self.higher_tissue_sfnn_rule:
                        self.higher_tissue_sfnn_rule[_t][cid] = SFNNRule.from_dict(_rule_d)
            except Exception as e:
                logger.debug(
                    "restore_persisted_state %s higher_rules: %s", cid, e)
        # Biochem (Phase 2 эфемериды + baselines + mental_break)
        bc_dict = payload.get("biochem")
        if isinstance(bc_dict, dict):
            try:
                from .biochemistry import ClientCreatureBiochem
                self.biochem[cid] = ClientCreatureBiochem(**bc_dict)
            except TypeError:
                # Schema mismatch — старый формат без новых полей; восстанавливаем
                # частично через setattr на default instance.
                try:
                    from .biochemistry import make_default
                    bc = make_default()
                    for k, v in bc_dict.items():
                        if hasattr(bc, k):
                            setattr(bc, k, v)
                    self.biochem[cid] = bc
                except Exception as e:
                    logger.warning("restore_persisted_state %s biochem fallback: %s", cid, e)
            except Exception as e:
                logger.warning("restore_persisted_state %s biochem: %s", cid, e)
            # ZOMBIFICATION-fix (31.05.2026, Бендер; диагноз Хьюберта): сброс
            # метаболических РЕСУРСОВ (energy/hydration) к свежим на restore.
            # .pt мог сохранить death-state (energy=0, когда организм умер от
            # метаболизма) → restore грузил energy=0 → мгновенная starvation-
            # смерть → alive=False → P40 kill → persist помнит → re-restore →
            # петля (zombification, created=20160+). P40 self-heal'ит cid fresh
            # (initial energy) — клиент матчит: ресурсы свежие, а brain/traits/
            # telomere (эволюция) сохраняются из .pt. Вариант B Хьюберта.
            _bc = self.biochem.get(cid)
            if _bc is not None:
                try:
                    _bc.energy = 500.0  # genesis initial_energy (fresh ресурс)
                    _bc.hydration = float(getattr(_bc, "max_hydration", 100.0))
                except Exception:
                    pass
        # Evolved-traits recovery (30.05.2026): restore 9 body-traits +
        # generation из .pt. Закрывает client-restart дыру — тело переживает
        # рестарт как мозг. ingest_owned_traits санитизирует/клампит и
        # дублирует на атрибуты организма (back-compat для getattr-чтения).
        traits_d = payload.get("traits")
        if isinstance(traits_d, dict) and traits_d:
            try:
                self.ingest_owned_traits(cid, traits_d)
            except Exception as e:
                logger.warning("restore_persisted_state %s traits: %s", cid, e)
        gen = payload.get("generation")
        if gen is not None:
            try:
                setattr(org, "generation", int(gen))
            except Exception:
                pass

    def restore_sfnn_state(self, cid: str, sfnn_state: dict) -> None:
        """S6.0b-B (16.05.2026): восстановить SFNN-правила из persistent-store.

        В отличие от `apply_inherited_state` (родительское правило + σ=0.1
        мутация), это «верни мне моё» — без мутации. Используется при
        legitimate reset (hello_recovery / admin_respawn): сервер
        подкладывает в seed_chunk[payload]['sfnn_state'] последний blob,
        который клиент сам отправил через `client_state_sync`.

        Структура sfnn_state:
          motor_sfnn_rule: dict          # SFNNRule.to_dict()
          higher_tissue_sfnn_rules: dict  # {tissue: SFNNRule.to_dict()}
          motor_sfnn_steps: int           # глобальный счётчик (не per-cid)
          higher_tissue_sfnn_steps: dict  # {tissue: int}

        Счётчики восстанавливаются на глобальном уровне (a max-reduce, если
        вызовов несколько за один restore: берём максимум).
        """
        if cid not in self.organisms or not isinstance(sfnn_state, dict):
            return
        try:
            from core.sfnn_rule import SFNNRule
        except Exception as e:
            logger.warning("restore_sfnn_state %s: import SFNNRule: %s", cid, e)
            return
        rule_d = sfnn_state.get("motor_sfnn_rule")
        if isinstance(rule_d, dict) and self.motor_sfnn_rule.get(cid) is not None:
            try:
                self.motor_sfnn_rule[cid] = SFNNRule.from_dict(rule_d)
            except Exception as e:
                logger.warning("restore_sfnn_state %s motor_sfnn_rule: %s",
                                cid, e)
        higher_d = sfnn_state.get("higher_tissue_sfnn_rules") or {}
        if isinstance(higher_d, dict):
            for _t, _rule_d in higher_d.items():
                if _t not in self.higher_tissue_sfnn_rule:
                    continue
                if self.higher_tissue_sfnn_rule[_t].get(cid) is None:
                    continue
                try:
                    self.higher_tissue_sfnn_rule[_t][cid] = SFNNRule.from_dict(_rule_d)
                except Exception as e:
                    logger.warning(
                        "restore_sfnn_state %s higher %s: %s", cid, _t, e)
        # SFNN S6.7: восстановить 10 базовых правил без мутации.
        basic_d = sfnn_state.get("basic_tissue_sfnn_rules") or {}
        if isinstance(basic_d, dict):
            for _role, _rule_d in basic_d.items():
                if _role not in self.basic_tissue_sfnn_rule:
                    continue
                if self.basic_tissue_sfnn_rule[_role].get(cid) is None:
                    continue
                try:
                    self.basic_tissue_sfnn_rule[_role][cid] = SFNNRule.from_dict(_rule_d)
                except Exception as e:
                    logger.warning(
                        "restore_sfnn_state %s basic %s: %s", cid, _role, e)
        steps = sfnn_state.get("motor_sfnn_steps")
        if isinstance(steps, int) and steps > self.motor_sfnn_steps:
            self.motor_sfnn_steps = steps
        higher_steps = sfnn_state.get("higher_tissue_sfnn_steps") or {}
        if isinstance(higher_steps, dict):
            # Z7.i.e (16.05.2026): _ALL_HIGHER принимает счётчики 3 Зодчий-тканей.
            _all_higher = _HIGHER_SFNN_TISSUES + _ZODCHIY_EXTRA_TISSUES
            for _t, _s in higher_steps.items():
                if _t in _all_higher and isinstance(_s, int):
                    prev = int(self.higher_tissue_sfnn_steps.get(_t, 0))
                    if _s > prev:
                        self.higher_tissue_sfnn_steps[_t] = _s
        # SFNN S6.7: max-reduce восстановление счётчиков 10 базовых.
        basic_steps = sfnn_state.get("basic_tissue_sfnn_steps") or {}
        if isinstance(basic_steps, dict):
            for _role, _s in basic_steps.items():
                if _role in _BASIC_SFNN_TISSUES and isinstance(_s, int):
                    prev = int(self.basic_tissue_sfnn_steps.get(_role, 0))
                    if _s > prev:
                        self.basic_tissue_sfnn_steps[_role] = _s

    def collect_sfnn_state_sync_items(self) -> list[dict]:
        """S6.0b-B: собрать snapshot SFNN-правил всех живых cid для
        отправки на P40 через `client_state_sync`.

        Возвращает список dict (по одному на cid):
          {
            "cid": str,
            "motor_sfnn_rule": dict,
            "higher_tissue_sfnn_rules": {tissue: dict},
            "motor_sfnn_steps": int,
            "higher_tissue_sfnn_steps": {tissue: int},
          }

        Счётчики на клиенте — глобальные (один на компьют, не per-cid).
        Чтобы все cid не «делили» один счётчик — пишем одинаковое значение
        каждому; сервер upsert per-cid сохранит как есть, а P40-сторона
        восстановит max-reduce из first cid в restore_sfnn_state.
        """
        items: list[dict] = []
        motor_steps = int(self.motor_sfnn_steps)
        # Z7.i.e (16.05.2026): + 3 Зодчий-ткани в общий dump.
        higher_steps = {t: int(self.higher_tissue_sfnn_steps.get(t, 0))
                         for t in _HIGHER_SFNN_TISSUES
                                  + _ZODCHIY_EXTRA_TISSUES}
        basic_steps = {t: int(self.basic_tissue_sfnn_steps.get(t, 0))
                        for t in _BASIC_SFNN_TISSUES}
        for cid in list(self.organisms.keys()):
            entry: dict = {"cid": cid}
            rule = self.motor_sfnn_rule.get(cid)
            if rule is not None:
                try:
                    entry["motor_sfnn_rule"] = rule.to_dict()
                except Exception:
                    pass
            higher: dict = {}
            for _t in _HIGHER_SFNN_TISSUES + _ZODCHIY_EXTRA_TISSUES:
                r = self.higher_tissue_sfnn_rule.get(_t, {}).get(cid)
                if r is None:
                    continue
                try:
                    higher[_t] = r.to_dict()
                except Exception:
                    pass
            if higher:
                entry["higher_tissue_sfnn_rules"] = higher
            basic: dict = {}
            for _role in _BASIC_SFNN_TISSUES:
                r = self.basic_tissue_sfnn_rule.get(_role, {}).get(cid)
                if r is None:
                    continue
                try:
                    basic[_role] = r.to_dict()
                except Exception:
                    pass
            if basic:
                entry["basic_tissue_sfnn_rules"] = basic
            entry["motor_sfnn_steps"] = motor_steps
            entry["higher_tissue_sfnn_steps"] = dict(higher_steps)
            entry["basic_tissue_sfnn_steps"] = dict(basic_steps)
            items.append(entry)
        return items

    def extract_brain_state_dicts(self, cid: str) -> tuple[dict, dict]:
        """Brain migration Etap 3.2 (11.05.2026): собрать state_dict мозга
        родителя для отправки на P40 в asexual reproduce-envelope.

        Возвращает `(brain_state_dicts, brain_emas)`:
          - `brain_state_dicts`: dict с ключами predictor/dopamine/imagination/
            planner/insula → state_dict (только те ткани, что есть у parent).
          - `brain_emas`: dict со скалярными EMA (predictor_loss_ema,
            intrinsic_ema, entropy_ema, trace_norm_ema, reward_var_ema).

        Y50 здесь НЕ применяется — это делает `build_reproduce_envelope`
        перед упаковкой. Метод чисто извлекает.
        """
        brain: dict = {}
        emas: dict = {}
        if cid not in self.organisms:
            return brain, emas
        pred = self.predictor.get(cid)
        if pred is not None:
            try:
                brain["predictor"] = pred.state_dict()
            except Exception as e:
                logger.debug("extract_brain_state_dicts predictor: %s", e)
        sel = self.action_selectors.get(cid)
        if sel is not None and hasattr(sel, "state_dict"):
            try:
                brain["selector"] = sel.state_dict()
            except Exception as e:
                logger.debug("extract_brain_state_dicts selector: %s", e)
        sfnn_rule = self.motor_sfnn_rule.get(cid)
        if sfnn_rule is not None and hasattr(sfnn_rule, "to_dict"):
            try:
                brain["motor_sfnn_rule"] = sfnn_rule.to_dict()
            except Exception as e:
                logger.debug("extract_brain_state_dicts motor_sfnn_rule: %s", e)
        # SFNN S3.0: 7 высших правил.
        # Z7.i.e (16.05.2026): + 3 Зодчий-правила в общем dict (для lineage
        # != "zodchiy" cid там не появится — пропадает естественно).
        higher_rules_out: dict = {}
        for _t in _HIGHER_SFNN_TISSUES + _ZODCHIY_EXTRA_TISSUES:
            _r = self.higher_tissue_sfnn_rule.get(_t, {}).get(cid)
            if _r is None or not hasattr(_r, "to_dict"):
                continue
            try:
                higher_rules_out[_t] = _r.to_dict()
            except Exception as e:
                logger.debug(
                    "extract_brain_state_dicts higher %s: %s", _t, e)
        if higher_rules_out:
            brain["higher_tissue_sfnn_rules"] = higher_rules_out
        # SFNN S6.7: 10 базовых правил.
        basic_rules_out: dict = {}
        for _role in _BASIC_SFNN_TISSUES:
            _r = self.basic_tissue_sfnn_rule.get(_role, {}).get(cid)
            if _r is None or not hasattr(_r, "to_dict"):
                continue
            try:
                basic_rules_out[_role] = _r.to_dict()
            except Exception as e:
                logger.debug(
                    "extract_brain_state_dicts basic %s: %s", _role, e)
        if basic_rules_out:
            brain["basic_tissue_sfnn_rules"] = basic_rules_out
        for key, store in (
            ("dopamine", self.dopamine),
            ("imagination", self.imagination),
            ("planner", self.planner),
            ("insula", self.insula),
            ("motor_policy", self.motor_policy),
            ("default_mode", self.default_mode),
            ("theory_of_mind", self.theory_of_mind),
            ("language", self.language),
            # Z7.i.e (16.05.2026): 3 Зодчий-ткани (для wanderer/elder
            # store.get(cid) is None — естественно skip).
            ("cerebellum", self.cerebellum),
            ("amygdala", self.amygdala),
            ("episodic", self.episodic),
        ):
            tissue = store.get(cid)
            if tissue is None:
                continue
            try:
                brain[key] = tissue.state_dict()
            except Exception as e:
                logger.debug("extract_brain_state_dicts %s: %s", key, e)
        # EMA-агрегаты — float, baseline для ребёнка.
        for ema_key, attr in (
            ("predictor_loss_ema", "loss_ema"),
            ("intrinsic_ema", "intrinsic_ema"),
            ("entropy_ema", "entropy_ema"),
            ("trace_norm_ema", "trace_norm_ema"),
            ("reward_var_ema", "reward_var_ema"),
            ("tom_acc", "last_tom_acc"),
            ("lang_acc", "last_lang_acc"),
        ):
            store = getattr(self, attr, None)
            if isinstance(store, dict) and cid in store:
                try:
                    emas[ema_key] = float(store[cid])
                except Exception:
                    pass
        return brain, emas

    def inherit_brain_y50(self, parent_cid: str, child_cid: str) -> bool:
        """Brain migration Etap 3.1 (11.05.2026): Y50-наследование мозга
        родителя в свежезарегистрированного потомка.

        Тело (tissues) ребёнок получает через mate-pair кроссинговер в
        `_handle_mate_request`. Сайдкары (predictor + S2.E/G/A/F высшие
        ткани) у новой особи в `add_creature` создаются random-init —
        этот метод накатывает Y50(parent) поверх.

        Возвращает True если хотя бы одна ткань была унаследована, False
        если parent или child неизвестен / pyторч state_dict собрать не вышло.
        """
        if parent_cid == child_cid:
            return False
        if child_cid not in self.organisms or parent_cid not in self.organisms:
            return False
        payload: dict = {}
        parent_pred = self.predictor.get(parent_cid)
        if parent_pred is not None:
            try:
                payload["predictor"] = parent_pred.state_dict()
                # EMA-агрегаты родителя → стартовые baseline для ребёнка.
                payload["predictor_loss_ema"] = float(
                    self.loss_ema.get(parent_cid, 0.0))
                payload["intrinsic_ema"] = float(
                    self.intrinsic_ema.get(parent_cid, 0.0))
                payload["tom_acc"] = float(
                    self.last_tom_acc.get(parent_cid, 0.0))
                payload["lang_acc"] = float(
                    self.last_lang_acc.get(parent_cid, 0.0))
            except Exception as e:
                logger.debug("inherit_brain_y50 parent predictor: %s", e)
        parent_sel = self.action_selectors.get(parent_cid)
        if parent_sel is not None and hasattr(parent_sel, "state_dict"):
            try:
                payload["selector"] = parent_sel.state_dict()
            except Exception as e:
                logger.debug("inherit_brain_y50 parent selector: %s", e)
        parent_rule = self.motor_sfnn_rule.get(parent_cid)
        if parent_rule is not None and hasattr(parent_rule, "to_dict"):
            try:
                payload["motor_sfnn_rule"] = parent_rule.to_dict()
            except Exception as e:
                logger.debug("inherit_brain_y50 parent motor_sfnn_rule: %s", e)
        # SFNN S3.0: 7 правил высших тканей родителя → дочерней особи (mutate
        # σ=0.1 уже в apply_inherited_state). Дефолтное правило ребёнка из
        # add_creature перезаписывается родительским только если оно есть.
        # Z7.i.e (16.05.2026): + 3 Зодчий-ткани (для wanderer/elder rule
        # is None → пропадает естественно).
        higher_rules_dump: dict = {}
        for _t in _HIGHER_SFNN_TISSUES + _ZODCHIY_EXTRA_TISSUES:
            _parent_rule = self.higher_tissue_sfnn_rule.get(_t, {}).get(parent_cid)
            if _parent_rule is None or not hasattr(_parent_rule, "to_dict"):
                continue
            try:
                higher_rules_dump[_t] = _parent_rule.to_dict()
            except Exception as e:
                logger.debug(
                    "inherit_brain_y50 parent higher_tissue_sfnn_rule[%s]: %s",
                    _t, e)
        if higher_rules_dump:
            payload["higher_tissue_sfnn_rules"] = higher_rules_dump
        # SFNN S6.7 (16.05.2026): 10 базовых правил родителя → ребёнку.
        basic_rules_dump: dict = {}
        for _role in _BASIC_SFNN_TISSUES:
            _parent_rule = self.basic_tissue_sfnn_rule.get(_role, {}).get(
                parent_cid)
            if _parent_rule is None or not hasattr(_parent_rule, "to_dict"):
                continue
            try:
                basic_rules_dump[_role] = _parent_rule.to_dict()
            except Exception as e:
                logger.debug(
                    "inherit_brain_y50 parent basic_tissue_sfnn_rule[%s]: %s",
                    _role, e)
        if basic_rules_dump:
            payload["basic_tissue_sfnn_rules"] = basic_rules_dump
        for key, store in (
            ("dopamine", self.dopamine),
            ("imagination", self.imagination),
            ("planner", self.planner),
            ("insula", self.insula),
            ("motor_policy", self.motor_policy),
            ("default_mode", self.default_mode),
            ("theory_of_mind", self.theory_of_mind),
            ("language", self.language),
            # Z7.i.e (16.05.2026): 3 Зодчий-ткани (wanderer/elder skip).
            ("cerebellum", self.cerebellum),
            ("amygdala", self.amygdala),
            ("episodic", self.episodic),
        ):
            tissue = store.get(parent_cid)
            if tissue is None:
                continue
            try:
                payload[key] = tissue.state_dict()
            except Exception as e:
                logger.debug("inherit_brain_y50 parent %s: %s", key, e)
        if not payload:
            return False
        self.apply_inherited_state(child_cid, payload)
        logger.info(
            "inherit_brain_y50 %s → %s (%s)",
            parent_cid, child_cid, ",".join(sorted(payload.keys())))
        return True

    # ── SFNN S3.activate (14.05.2026) ─────────────────────────────────────

    def set_higher_sfnn(self, on: bool) -> int:
        """Включить/выключить higher_tissue_sfnn_enabled у всех owned особей.

        Дефолт колонии обновляется → новые особи в `add_creature` родятся
        с этим значением. Существующие — патчатся in-place через
        `org.genome.higher_tissue_sfnn_enabled`. Возвращает число изменённых.
        """
        on = bool(on)
        self._higher_sfnn_default = on
        n_changed = 0
        for cid, org in list(self.organisms.items()):
            if not hasattr(org, "genome"):
                org.genome = types.SimpleNamespace(
                    higher_tissue_sfnn_enabled=on,
                    sfnn_enabled=False,
                )
                n_changed += 1
                continue
            prev = bool(getattr(org.genome,
                                 "higher_tissue_sfnn_enabled", False))
            if prev != on:
                org.genome.higher_tissue_sfnn_enabled = on
                n_changed += 1
        logger.info("set_higher_sfnn(%s) — changed %d / %d organisms",
                    on, n_changed, len(self.organisms))
        return n_changed

    def set_basic_sfnn(self, on: bool) -> int:
        """SFNN S6.9 (16.05.2026): включить/выключить basic_tissue_sfnn_enabled
        у всех owned особей.

        Зеркало `set_higher_sfnn` для 10 базовых тканей organism graph
        (sensory/attention/brain/memory/consciousness/communication/motor/
        manipulator/digestive/immune). Дефолт колонии обновляется → новые
        особи в `add_creature` родятся с этим значением. Существующие —
        патчатся in-place через `org.genome.basic_tissue_sfnn_enabled`.
        Возвращает число изменённых.
        """
        on = bool(on)
        self._basic_sfnn_default = on
        n_changed = 0
        for cid, org in list(self.organisms.items()):
            if not hasattr(org, "genome"):
                org.genome = types.SimpleNamespace(
                    higher_tissue_sfnn_enabled=self._higher_sfnn_default,
                    sfnn_enabled=False,
                    basic_tissue_sfnn_enabled=on,
                )
                n_changed += 1
                continue
            prev = bool(getattr(org.genome,
                                 "basic_tissue_sfnn_enabled", False))
            if prev != on:
                org.genome.basic_tissue_sfnn_enabled = on
                n_changed += 1
        logger.info("set_basic_sfnn(%s) — changed %d / %d organisms",
                    on, n_changed, len(self.organisms))
        return n_changed

    # ── Single-organism pivot (01.06.2026, ТЗ e3cc81b) ──────────────────

    def set_single_organism(self, on: bool) -> bool:
        """Включить/выключить режим одного организма (Адам).

        В отличие от set_*_sfnn это compute-level mode-флаг, а не per-organism
        genome-атрибут: под ним гейтятся КОЛОНИАЛЬНЫЕ механики (репродукция
        client/P40, speciation, Z2.b topology-crossover). Код механик НЕ
        удаляется — нужен для Зоопарка Эпохи 2. Возвращает применённое значение.

        Идемпотентно: повторный вызов с тем же значением — no-op (только лог при
        изменении). Сами особи не трогаются — выбор/эвакуация Адама это этап 1.
        """
        on = bool(on)
        if on != self._single_organism:
            logger.info("set_single_organism: %s → %s (n_organisms=%d)",
                        self._single_organism, on, len(self.organisms))
        self._single_organism = on
        if on:
            # bias_scale=0 → own_contribution=1 → мотор автономен, без
            # motor-scaffold (прямого подруливания в обход мозга). Адам стоит
            # на своём мозге; легитимная помощь — от среды (Storyteller §6), не
            # puppeting. СЕКВЕНС (Фрай): целевое состояние, но снимает выживальный
            # scaffold — автономного Адама в live только в связке с паралич-сеткой
            # (§3) + починкой income, не раньше. Сейчас живого Адама нет, код
            # стейджится; не tag'ать+гонять автономного Адама в зазоре до §3.
            self._bias_scale = 0.0
            # Bootstrap-race fix (баг live 01.06): на рестарте флаг применяется
            # на ~8с ПОЗЖЕ restore (нужен command-poll). В этом окне активен
            # КОЛОНИАЛЬНЫЙ death-check — если persisted energy=0 (голодавший
            # прогон), Адам умирает → _dead_cids → handle_tick его скипает →
            # ЗАМОРОЗКА навсегда (метаболизм/mental_break не тикают). Под
            # single_organism смерти нет → оживляем dead-marked: un-mark + снять
            # paralysis + дать стартовую энергию (recovery), чтобы не зависнуть
            # в параличе сразу. Идемпотентно.
            _revived = 0
            for _cid in list(self._dead_cids):
                self._dead_cids.discard(_cid)
                self._paralysis_until.pop(_cid, None)
                _bc = self.biochem.get(_cid)
                if _bc is not None:
                    if float(getattr(_bc, "energy", 0.0)) <= 0.0:
                        _bc.energy = float(self._recovery_energy)
                    # Очистка bug-corrupted стресс-биохимии: за часы голода
                    # (routing-баг) cortisol застревал на максимуме (99.5) →
                    # catatonic → force STAY → не может добывать → стресс-спираль
                    # не разрывается. Сброс к здоровому baseline — одноразовая
                    # очистка МУСОРА от бага (не маскировка реального стресса:
                    # это накопленный артефакт сломанного периода). Мозг (веса/
                    # Hebbian) не трогаем — он не corrupted.
                    try:
                        _bc.cortisol = float(getattr(_bc, "baseline_cortisol", 10.0))
                        _bc.serotonin = float(getattr(_bc, "baseline_serotonin", 50.0))
                        _bc.mental_break = ""
                        _bc.mental_break_ticks = 0
                    except Exception:
                        pass
                _revived += 1
            if _revived:
                logger.info(
                    "single_organism: revived %d dead-marked особей "
                    "(bootstrap race fix)", _revived)
            # Track 2 (этап 4): расширить восприятие — predictor читает
            # self-observable (obs68). Non-destructive ([I|0]-init, math-equivalence
            # доказана). Адам начинает моделировать своё мета-состояние → основа
            # для обучения active EAT. Идемпотентно.
            _so = 0
            for _cid in list(self.predictor.keys()):
                if self._enable_self_observable(_cid):
                    _so += 1
            if _so:
                logger.info("single_organism: self-observable enabled on "
                            "%d predictors (Track 2)", _so)
            # Track 2: self-obs→action голова — ОТКЛЮЧЕНА (деградировала foraging,
            # см. _self_obs_head_enabled). Код сохранён для rework.
            if self._self_obs_head_enabled:
                _sh = 0
                for _cid in list(self.organisms.keys()):
                    if self._enable_self_obs_action_head(_cid):
                        _sh += 1
                if _sh:
                    logger.info("single_organism: self-obs action head on "
                                "%d organisms (Track 2)", _sh)
            # Направление (б): insula-стресс → temperature-модуляция. За флагом,
            # дефолт OFF (live не трогаем до закрепления viability + ОК Фрая).
            # near-identity старт (T≈1), temperature-only → структурно безопасна.
            if self._insula_temp_enabled:
                _it = 0
                for _cid in list(self.organisms.keys()):
                    if self._enable_insula_temp(_cid):
                        _it += 1
                if _it:
                    logger.info("single_organism: insula-temp modulation on "
                                "%d organisms (Track 2 (б))", _it)
        return on

    def set_insula_temp(self, on: bool) -> bool:
        """Канал client_flags: вкл/выкл (б) insula-temp модуляцию МГНОВЕННО
        (без деплоя/рестарта). Запуск по go Фрая; tripwire-откат при деградации
        foraging = client_flags.insula_temp=false (мгновенно, без execv).

        on=True: ставит флаг + создаёт головы на всех organisms (идемпотентно),
        ЕСЛИ single_organism активен (иначе головы создаст set_single_organism
        при включении — порядок флагов не важен). on=False: флаг off →
        _apply_insula_temp становится полным no-op; головы СОХРАНЯЮТСЯ для
        мгновенного ре-enable (обучение продолжится с накопленного).
        """
        self._insula_temp_enabled = bool(on)
        if on and self._single_organism:
            cnt = 0
            for cid in list(self.organisms.keys()):
                if self._enable_insula_temp(cid):
                    cnt += 1
            if cnt:
                logger.info("set_insula_temp: insula-temp on %d organisms "
                            "(Track 2 (б))", cnt)
        elif on:
            logger.info("set_insula_temp: флаг ON, single_organism off — "
                        "головы создадутся при включении single_organism")
        else:
            logger.info("set_insula_temp: OFF (модуляция no-op, головы "
                        "сохранены для ре-enable)")
        return self._insula_temp_enabled

    def set_felt_thirst_drive(self, on: bool) -> bool:
        """Канал client_flags: вкл/выкл §3.2 felt-thirst gradual drive МГНОВЕННО
        (без деплоя/рестарта). on=True: рефлекс A (water-seek) масштабирует
        приоритет по градуальному felt-сигналу жажды (intero[1]-афферент,
        φ-onset 0.382) вместо бинарного порога 30%. on=False (kill-switch):
        откат к проверенному бинарному 30% (read-only флаг — поведение
        water-seek читает его per-тик; ткани/состояние не трогаются). Реверс
        мгновенный. backoff: VPS/кабинет ставит false, если выживание падает."""
        self._felt_thirst_drive_enabled = bool(on)
        logger.info("set_felt_thirst_drive: %s (%s)", on,
                    "градуальный felt-drive" if on else "бинарный 30% (kill-switch)")
        return self._felt_thirst_drive_enabled

    def set_motor_renorm_cap(self, cap: float) -> float:
        """Канал client_flags: рекалибровка motor renorm growth-cap (Ступень 2,
        Фрай). cap=1.0 → пин магнитуды к target (текущее); cap>1 → веса motor_policy
        могут вырасти до target×cap → policy может ЗАОСТРИТЬСЯ (тест: flip падает =
        renorm был супрессором). Мгновенно, без рестарта. Анти-взрыв сохранён (cap
        ограничивает рост). Реверс — cap=1.0."""
        try:
            v = float(cap)
        except (TypeError, ValueError):
            return self._motor_renorm_growth_cap
        v = max(1.0, min(5.0, v))   # кламп [1, 5] (5 — потолок безопасности)
        if v != self._motor_renorm_growth_cap:
            logger.info("set_motor_renorm_cap: %.2f → %.2f",
                        self._motor_renorm_growth_cap, v)
        self._motor_renorm_growth_cap = v
        return v

    def set_motor_oja_scale(self, scale: float) -> float:
        """Канал client_flags: множитель Oja-члена motor_policy (Ступень 2,
        Фрай (a)). 1.0 → полная Oja (текущее); <1.0 → Oja слабее → ΔW не
        самосжимается при росте W → policy может заостриться (с renorm_cap>1
        «освобождает магнитуду»). Тест в спокойствии: flip падает? Следить за
        tanh-saturation (motor_norm→4.0). Кламп [0, 1]. Реверс — scale=1.0."""
        try:
            v = float(scale)
        except (TypeError, ValueError):
            return self._motor_oja_scale
        v = max(0.0, min(1.0, v))
        if v != self._motor_oja_scale:
            logger.info("set_motor_oja_scale: %.2f → %.2f",
                        self._motor_oja_scale, v)
        self._motor_oja_scale = v
        return v

    def set_motor_oja_out(self, scale: float) -> float:
        """Канал client_flags: Oja-scale ТОЛЬКО на output_proj (Фрай 04.06,
        верифицировано dw_radial≈1). Снижение → ΔW на policy-выходе тангенциальнее
        (reward шейпит направление, не свампится радиальным Oja) → разлок forage-
        learning. На input/attn/mlp Oja остаётся глобальной (нормальной). Кламп
        [0, 1]. Реверс — 1.0. Минимально достаточно (передекрут → риск флипа)."""
        try:
            v = float(scale)
        except (TypeError, ValueError):
            return self._motor_oja_scale_out
        v = max(0.0, min(1.0, v))
        if v != self._motor_oja_scale_out:
            logger.info("set_motor_oja_out: %.2f → %.2f", self._motor_oja_scale_out, v)
        self._motor_oja_scale_out = v
        return v

    def set_motor_renorm_cap_out(self, cap: float) -> float:
        """Канал client_flags: renorm growth-cap ТОЛЬКО на output_proj (Фрай 04.06,
        вторично к Oja). >1 → магнитуда строки output_proj может расти до target×cap
        → тангенциальная компонента survive. На input/attn/mlp renorm нормальный.
        Кламп [1, 5]. Реверс — 1.0."""
        try:
            v = float(cap)
        except (TypeError, ValueError):
            return self._motor_renorm_cap_out
        v = max(1.0, min(5.0, v))
        if v != self._motor_renorm_cap_out:
            logger.info("set_motor_renorm_cap_out: %.2f → %.2f",
                        self._motor_renorm_cap_out, v)
        self._motor_renorm_cap_out = v
        return v

    def set_motor_slow(self, on: float) -> float:
        """Канал client_flags: МЕДЛЕННЫЙ batch-REINFORCE канал (порт серверного
        WorldTrainer, Фрай 05.06, MIGRATION GAP fix). >0 → вкл: buffer 200, batch 16,
        train_every 10, norm-advantage, adaptive-entropy 0.15→2.0 — учитель политики
        motor_policy. Гейчит per-tick Hebbian motor_policy + per-tick PG OFF (слабый
        канал единственный учитель; быстрый Hebbian на ДРУГИХ тканях остаётся, как
        сервер). Кламп [0, 1]. Реверс — 0."""
        try:
            v = float(on)
        except (TypeError, ValueError):
            return self._motor_slow_on
        v = max(0.0, min(1.0, v))
        if v != self._motor_slow_on:
            logger.info("set_motor_slow: %.1f → %.1f", self._motor_slow_on, v)
        self._motor_slow_on = v
        return v

    def set_motor_pg(self, on: float) -> float:
        """Канал client_flags: policy-gradient на output_proj (Фрай 04.06, rule-
        upgrade). >0 → тангенциальный REINFORCE credit к сэмплированному действию
        (∇log π·advantage) заменяет correlational Hebbian на output_proj → может
        разучить вестигиальную колониальную политику (SHARE_FOOD/FLEE) + выучить
        forage. Прочие синапсы — Hebbian. Кламп [0, 1]. Реверс — 0."""
        try:
            v = float(on)
        except (TypeError, ValueError):
            return self._motor_pg_on
        v = max(0.0, min(1.0, v))
        if v != self._motor_pg_on:
            logger.info("set_motor_pg: %.1f → %.1f", self._motor_pg_on, v)
        self._motor_pg_on = v
        return v

    def set_motor_pg_lr(self, lr: float) -> float:
        """Канал client_flags: learning-rate policy-gradient output_proj (Фрай).
        ΔW = pg_lr·advantage·outer(g, pre), бортуется clip ±0.01 + renorm_cap_out.
        Дефолт φ=1.618. Кламп [0, 5]. φ-выровненные значения (Шеф): 0.618/1.618/2.618."""
        try:
            v = float(lr)
        except (TypeError, ValueError):
            return self._motor_pg_lr
        v = max(0.0, min(5.0, v))
        if v != self._motor_pg_lr:
            logger.info("set_motor_pg_lr: %.3f → %.3f", self._motor_pg_lr, v)
        self._motor_pg_lr = v
        return v

    def set_motor_temp(self, temp: float) -> float:
        """Канал client_flags: де-сатурация tanh-головы motor_policy (Фрай 04.06).
        Override T в _motor_forward: delta = tanh(out/T)·SCALE. T>1 делит pre-tanh →
        залипшие ±0.99 дельты возвращаются в отзывчивый ВАРЬИРУЮЩИЙ диапазон (T-делёж
        сохраняет относительные различия выходов, в отличие от clip) → градиент tanh
        снова ≠0 → REINFORCE выбивает голову из застрявшего экстремума (а) и держит
        отзывчивой (б). 0.0 → use rule.temperature (текущее). Кламп [0, 16]. Реверс —
        0.0. Тест-вердикт Фрая: де-сатурир. голова сходится стабильно → SFNN+reg
        адекватен; всё равно флипает → SFNN-head неадекватен → RL-head."""
        try:
            v = float(temp)
        except (TypeError, ValueError):
            return self._motor_temp
        v = max(0.0, min(16.0, v))
        if v != self._motor_temp:
            logger.info("set_motor_temp: %.2f → %.2f", self._motor_temp, v)
        self._motor_temp = v
        return v

    def set_motor_lr_scale(self, scale: float) -> float:
        """Канал client_flags: множитель eta REINFORCE-апдейта motor_policy
        (анти-saddle-flip, Фрай 04.06 (б)). <1.0 → медленнее обучение → меньше
        крупных градиент-перекидов через седло (которые перебрасывают tanh-голову
        между насыщенными экстремумами). 1.0 → текущее. Кламп [0, 1]. Реверс — 1.0."""
        try:
            v = float(scale)
        except (TypeError, ValueError):
            return self._motor_lr_scale
        v = max(0.0, min(1.0, v))
        if v != self._motor_lr_scale:
            logger.info("set_motor_lr_scale: %.3f → %.3f", self._motor_lr_scale, v)
        self._motor_lr_scale = v
        return v

    def set_reward_balance(self, on: float) -> float:
        """Канал client_flags: вкл серверный энергобаланс reward (Фрай 04.06).
        >0 → differentiated value (hunt φ⁷≈29 yield + risk vs forage φ⁴≈6.85,
        без риска) вместо плоских равных +5·ate/+5·killed → НЕ равные аттракторы
        → контекстная policy, не бистабильность. 0 → старый плоский (колония +
        safety). Кламп [0, 1]. Реверс — 0."""
        try:
            v = float(on)
        except (TypeError, ValueError):
            return self._reward_balance_on
        v = max(0.0, min(1.0, v))
        if v != self._reward_balance_on:
            logger.info("set_reward_balance: %.1f → %.1f", self._reward_balance_on, v)
        self._reward_balance_on = v
        return v

    def set_reward_weights(self, forage_w=None, kill_w=None, risk_w=None) -> tuple:
        """Канал client_flags: tunable веса энергобаланс-reward (Фрай 04.06).
        forage_w ×φ⁴(6.85) на ate; kill_w ×φ⁷(29) на killed; risk_w ×damage.
        Дефолты 1.0 → forage 6.85, safe-kill 29, full-damage-kill ≈0. Кламп [0, 4]."""
        def _clamp(x, cur):
            if x is None:
                return cur
            try:
                return max(0.0, min(4.0, float(x)))
            except (TypeError, ValueError):
                return cur
        nf = _clamp(forage_w, self._reward_forage_w)
        nk = _clamp(kill_w, self._reward_kill_w)
        nr = _clamp(risk_w, self._reward_risk_w)
        if (nf, nk, nr) != (self._reward_forage_w, self._reward_kill_w,
                            self._reward_risk_w):
            logger.info("set_reward_weights: forage %.2f kill %.2f risk %.2f",
                        nf, nk, nr)
        self._reward_forage_w, self._reward_kill_w, self._reward_risk_w = nf, nk, nr
        return (nf, nk, nr)

    def set_instinct_dir_strength(self, strength: float) -> float:
        """Канал client_flags: сила инстинкт-направления (food/prey/predator)
        в _shape_action_logits под single_organism (Фрай 03.06). 0.0 → занулено
        (текущее, прекондишн снят); умеренное (~0.3-0.6) → ориентир-приор, не
        диктат (мотор+мост модулируют поверх). Развязано от bias_scale.
        Кламп [0, 4] (4 = исходная full-сила). Реверс — 0.0. Тюнится эмпирически."""
        try:
            v = float(strength)
        except (TypeError, ValueError):
            return self._instinct_dir_strength
        v = max(0.0, min(4.0, v))
        if v != self._instinct_dir_strength:
            logger.info("set_instinct_dir_strength: %.2f → %.2f",
                        self._instinct_dir_strength, v)
        self._instinct_dir_strength = v
        return v

    def set_motor_voice(self, voice: float) -> float:
        """Канал client_flags: голос мотора (_own множитель motor_delta) под
        single_organism (Фрай curriculum). Фаза 1 ~0.2 (прайор ведёт), Фаза 2
        fade-up к 1.0 (тест SFNN-модулятора). Кламп [0, 1]. Дефолт 1.0=полный."""
        try:
            v = float(voice)
        except (TypeError, ValueError):
            return self._motor_voice
        v = max(0.0, min(1.0, v))
        if v != self._motor_voice:
            logger.info("set_motor_voice: %.2f → %.2f", self._motor_voice, v)
        self._motor_voice = v
        return v

    def set_motor_park_test(self, on: float) -> float:
        """Канал client_flags: изолирующий тест override-мотора (Фрай 06.06).
        >0 → на on-flora тиках STAY выигрывает безусловно (паркуем). Обычно
        ставят с motor_voice=0 (мотор из контура). Кламп [0,1]. 0=off."""
        try:
            v = max(0.0, min(1.0, float(on)))
        except (TypeError, ValueError):
            return self._motor_park_test
        if v != self._motor_park_test:
            logger.info("set_motor_park_test: %.1f → %.1f",
                        self._motor_park_test, v)
        self._motor_park_test = v
        return v

    def set_motor_stay_force(self, on: float) -> float:
        """Канал client_flags: STAY-исполнение контроль (Фрай 06.06). >0 →
        STAY эмитится БЕЗУСЛОВНО каждый тик (не только on-flora). Чистый тест
        протокола: P40 honored STAY или нет (Хьюберт смотрит pos-delta). 0=off."""
        try:
            v = max(0.0, min(1.0, float(on)))
        except (TypeError, ValueError):
            return self._motor_stay_force
        if v != self._motor_stay_force:
            logger.info("set_motor_stay_force: %.1f → %.1f",
                        self._motor_stay_force, v)
        self._motor_stay_force = v
        return v

    def set_damage_factor(self, f: float) -> float:
        """Канал client_flags: калибровка damage-канала (Фрай 07.06). Множитель
        к damage_per_tick от сервера → energy-ledger per-client-tick. Сперва малый
        (обучающий сигнал, не инстакил), растить градуально. Кламп [0, 4]. 0=off."""
        try:
            v = max(0.0, min(4.0, float(f)))
        except (TypeError, ValueError):
            return self._damage_factor
        if v != self._damage_factor:
            logger.info("set_damage_factor: %.3f → %.3f", self._damage_factor, v)
        self._damage_factor = v
        return v

    def set_glucose_energy_rate(self, rate: float) -> float:
        """Канал client_flags: rate конверсии излишка glucose→energy в _apply_
        metabolism (Фрай экономика). 0=нет (текущее). Калибруется чтобы плотная
        еда (glucose↑) давала net-positive energy. Кламп [0, 0.2] (per-sec ×
        glucose-surplus). Реверс — 0.0."""
        try:
            v = float(rate)
        except (TypeError, ValueError):
            return self._glucose_energy_rate
        v = max(0.0, min(0.2, v))
        if v != self._glucose_energy_rate:
            logger.info("set_glucose_energy_rate: %.4f → %.4f",
                        self._glucose_energy_rate, v)
        self._glucose_energy_rate = v
        return v

    # ── Zodchiy Z7.i.c (16.05.2026) ─────────────────────────────────────

    def set_lineage_upgrade_pending(self, on: bool) -> int:
        """One-shot Genome-флип `lineage_upgrade_to_zodchiy` у owned особей.

        Z7.i.c — клиентская сторона апгрейда Странник→Зодчий.
        В отличие от `set_*_sfnn` дефолт колонии НЕ обновляется (это разовый
        триггер, а не постоянный признак). Просто патчит атрибут у всех
        существующих organisms через `org.genome.lineage_upgrade_to_zodchiy`.

        Если у организма нет `genome`, навешиваем SimpleNamespace по той же
        схеме, что и `add_creature` (с текущими SFNN-дефолтами).

        Z7.c.apply_lineage_upgrade при следующей репродукции гейтит по
        `parent.lineage == "wanderer"`, поэтому ставить флаг на elder/zodchiy
        безопасно — Z7.c сбросит флаг без апгрейда. Клиент не знает lineage
        своих особей (P40 не шлёт его в owned-payload), поэтому ставим на
        всех — а гейтинг делает pure-уровень.

        Возвращает число изменённых организмов.
        """
        on = bool(on)
        n_changed = 0
        for cid, org in list(self.organisms.items()):
            if not hasattr(org, "genome"):
                org.genome = types.SimpleNamespace(
                    higher_tissue_sfnn_enabled=self._higher_sfnn_default,
                    sfnn_enabled=self._motor_sfnn_default,
                    basic_tissue_sfnn_enabled=self._basic_sfnn_default,
                    lineage_upgrade_to_zodchiy=on,
                )
                n_changed += 1
                continue
            prev = bool(getattr(org.genome,
                                 "lineage_upgrade_to_zodchiy", False))
            if prev != on:
                org.genome.lineage_upgrade_to_zodchiy = on
                n_changed += 1
        logger.info(
            "set_lineage_upgrade_pending(%s) — changed %d / %d organisms",
            on, n_changed, len(self.organisms))
        return n_changed

    def set_motor_sfnn(self, on: bool) -> int:
        """Включить/выключить motor sfnn_enabled у всех owned особей.

        Зеркало `set_higher_sfnn` для motor_policy ветки. Дефолт колонии
        обновляется → новые особи в `add_creature` родятся с этим значением.
        Существующие — патчатся in-place через `org.genome.sfnn_enabled`.
        Возвращает число изменённых.
        """
        on = bool(on)
        self._motor_sfnn_default = on
        n_changed = 0
        for cid, org in list(self.organisms.items()):
            if not hasattr(org, "genome"):
                org.genome = types.SimpleNamespace(
                    higher_tissue_sfnn_enabled=self._higher_sfnn_default,
                    sfnn_enabled=on,
                )
                n_changed += 1
                continue
            prev = bool(getattr(org.genome, "sfnn_enabled", False))
            if prev != on:
                org.genome.sfnn_enabled = on
                n_changed += 1
        logger.info("set_motor_sfnn(%s) — changed %d / %d organisms",
                    on, n_changed, len(self.organisms))
        return n_changed

    @property
    def n_alive(self) -> int:
        return len(self.organisms)

    # ── Phase emas pushback (03.05.2026) ─────────────────────────────────

    def get_phase_emas(self, cid: str) -> Optional[dict]:
        """Снимок EMA для отправки в actions_batch.

        Поля идентичны server-side `phase_emas` receiver:
          - loss_ema  (Phase 1 predictor running MSE)
          - entropy_ema (Phase 6 action-distribution entropy)
          - trace_norm_ema (Phase 6 Hebbian eligibility trace L2)
          - intrinsic_ema (Phase 2 Δsurprise baseline)
          - specialization_ema (Phase 4 per-role tissue attribution, dict)

        Возвращает None для незарегистрированных cid (нечего слать).
        Все скалярные значения нормированы как float и проверены на finite.
        """
        if cid not in self.loss_ema:
            return None
        import math
        out: dict = {}
        for src, key in (
            (self.loss_ema, "loss_ema"),
            (self.entropy_ema, "entropy_ema"),
            (self.trace_norm_ema, "trace_norm_ema"),
            (self.intrinsic_ema, "intrinsic_ema"),
        ):
            v = src.get(cid)
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(vf):
                continue
            out[key] = vf
        # Phase 4 — specialization_ema (per-role attribution, dict role→share).
        ctrl = self.hebbian.get(cid)
        if ctrl is not None and hasattr(ctrl, "tissue_specialization"):
            try:
                shares = ctrl.tissue_specialization()
            except Exception:
                shares = None
            if isinstance(shares, dict) and shares:
                spec: dict = {}
                for role, share in shares.items():
                    try:
                        sf = float(share)
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(sf):
                        continue
                    spec[str(role)] = sf
                if spec:
                    out["specialization_ema"] = spec
        # Brain migration (10.05.2026): S2.E/G/A/F push.
        # 13.05.2026: + client_dmn_floor (S2.D) + client_tom_acc (S2.B).
        # 14.05.2026 (Phase 5d NEOL): + client_dopa_td — TD-error от S2.E
        # для серверной агрегатной метрики /api/world/hebbian_stats.reward_gated.
        # Модуляция η происходит локально в heb.update(dopa_td_mult=...),
        # сервер этот push НЕ применяет к мозгу (мозг у владельца).
        for src, key in (
            (self.last_beta_local, "client_beta_local"),
            (self.last_imag_mult, "client_imag_mult"),
            (self.last_stress, "client_stress"),
            (self.last_dmn_floor, "client_dmn_floor"),
            (self.last_tom_acc, "client_tom_acc"),
            (self.last_lang_acc, "client_lang_acc"),
            (self.dopamine_td, "client_dopa_td"),
        ):
            v = src.get(cid)
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(vf):
                continue
            out[key] = vf
        pd = self.last_planner_delta.get(cid)
        if pd is not None:
            try:
                vals = [float(x) for x in pd.reshape(-1).tolist()]
                if all(math.isfinite(x) for x in vals):
                    out["client_planner_delta"] = vals
            except Exception:
                pass
        # Phase 7 — motor_policy delta push (server fastpath override).
        md = self.last_motor_delta.get(cid)
        if md is not None:
            try:
                vals_m = [float(x) for x in md.reshape(-1).tolist()]
                if all(math.isfinite(x) and -1.0 <= x <= 1.0 for x in vals_m):
                    out["client_motor_delta"] = vals_m
            except Exception:
                pass
        # Z7.i.f (16.05.2026, Зодчий) — push Зодчий-снапшотов в phase_emas.
        # Заполняются только для lineage="zodchiy" (forward в _compute_higher_
        # tissues создаёт записи только когда соответствующая ткань создана
        # в add_creature). Для wanderer/elder dict'ы пустые → ключ не появится.
        # Cerebellum/episodic — векторы [16]/[64], отдаём L2-норму (скаляр).
        # Amygdala — уже скалярный valence ∈ [-1, 1].
        cer = self.last_cerebellum_delta.get(cid)
        if cer is not None:
            try:
                norm = float(cer.norm().item())
                if math.isfinite(norm):
                    out["client_cerebellum_delta_norm"] = norm
            except Exception:
                pass
        val = self.last_amygdala_valence.get(cid)
        if val is not None:
            try:
                vf = float(val)
                if math.isfinite(vf) and -1.0 <= vf <= 1.0:
                    out["client_amygdala_valence"] = vf
            except Exception:
                pass
        epi = self.last_episodic_recall.get(cid)
        if epi is not None:
            try:
                norm = float(epi.norm().item())
                if math.isfinite(norm):
                    out["client_episodic_recall_norm"] = norm
            except Exception:
                pass
        return out or None

    # ── Tick ─────────────────────────────────────────────────────────────

    def handle_tick(self, obs_per_cid: dict,
                    events_per_cid: Optional[dict] = None,
                    intero_per_cid: Optional[dict] = None,
                    world_tick: int = 0,
                    rates_per_cid: Optional[dict] = None,
                    on_flora_per_cid: Optional[dict] = None,
                    carried_food_per_cid: Optional[dict] = None,
                    nearest_flora_per_cid: Optional[dict] = None) -> dict:
        """Forward + ActionSelector + Hebbian update для всех cid.

        Args:
            obs_per_cid: {cid: np.ndarray[80] float32} — env-наблюдения от P40.
            events_per_cid: {cid: {ate, killed, damage_taken, delta_energy}} —
                события прошлого тика. Если None — Hebbian update пропускается.
            intero_per_cid: {cid: np.ndarray[7] float32} — Brain migration
                (10.05.2026) интероцепция от P40 для S2.F insula. None или
                отсутствие cid → insula пропускается, остальные ткани работают.

        Returns:
            {cid: {"action": int, "target_id": Optional[str]}} — готово к
            упаковке в `actions`-envelope для WS.

        Особи, не зарегистрированные локально, игнорируются. Особи без obs —
        получают STAY (защита от рассинхронизации).
        """
        self.tick_ts.append(time.time())
        self._last_world_tick = int(world_tick)  # для age/ts в colony_summary
        self._apply_bootstrap_pending(world_tick)
        self._update_bias_curriculum(world_tick)
        out: dict = {}
        torch = self._torch
        # Снапшот итерации: self.organisms мутируется из main-потока
        # (detect_and_emit_mate_pairs → add_creature) и ws-потока (GC orphan,
        # seed) параллельно с этим проходом. Без list() — "dictionary changed
        # size during iteration" под нагрузкой (133 орг + mate/GC churn).
        for cid, organism in list(self.organisms.items()):
            # Body Migration метаболизм (31.05.2026): мёртвый (energy<=0 /
            # telomere AGONY) не действует — ждёт removal P40 по alive=False.
            if cid in self._dead_cids:
                out[cid] = {"action": STAY, "target_id": None}
                continue
            obs = obs_per_cid.get(cid)
            if obs is None:
                out[cid] = {"action": STAY, "target_id": None}
                continue
            try:
                obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
                # P40 присылает 80 (DATA_DIM=64 + STATE_DIM=16). forward берёт первые 64.
                obs64 = obs_arr[:64]
                obs_tensor = torch.from_numpy(obs64).to(self.device).unsqueeze(0)

                # Track 2 (этап 4): вход predictor = obs68 (env64 + self-observable4)
                # ЕСЛИ predictor расширен (_enable_self_observable). target остаётся
                # obs64 — мозг СВОИМ состоянием моделирует мир (Phase 6 самосознание).
                # Прочие ткани (forward/hebbian/motor) пока на obs64. [I|0]-init →
                # на старте obs68→obs64 (math-equivalence), self доучивается.
                _pred = self.predictor.get(cid)
                pred_input = obs_tensor
                if _pred is not None and int(
                        getattr(_pred, "data_dim", _SELF_OBS_OFFSET)) == _BRAIN_INPUT_DIM:
                    so = self._build_self_observable(cid)
                    obs68 = np.concatenate([obs64, so]).astype(np.float32)
                    pred_input = torch.from_numpy(obs68).to(
                        self.device).unsqueeze(0)

                # Вариант A: hook на cerebellum (один раз/cid) — ДО forward,
                # чтобы organism.forward (ниже) наполнил _cerebellum_out[cid].
                if self._predictor_from_cerebellum and self._single_organism:
                    self._ensure_cerebellum_hook(cid, organism)

                heb = self.hebbian.get(cid)
                if heb is not None:
                    try:
                        heb.capture_activations(obs_tensor)
                    except Exception as e:
                        logger.debug("hebbian capture %s: %s", cid, e)

                with torch.no_grad():
                    logits = organism.forward(obs_tensor)
                # LOGIT_DEBUG: raw org.forward (ДО shaping) — разводит (A)
                # org.forward доминирует vs (B) obs-градиент слаб.
                _raw_dbg = logits[0, :N_ACTIONS].detach().clone()

                # Вариант A: вход predictor'а = выход cerebellum (связи кормят
                # прогноз). Берём ПОСЛЕ organism.forward (hook наполнил
                # _cerebellum_out). Только при совпадении размерности с predictor
                # (64) — иначе остаёмся на obs-пути (predictor-68 не трогаем).
                # target остаётся obs_{t+1} → loss_ema/intrinsic сопоставимы.
                if (self._predictor_from_cerebellum and self._single_organism
                        and _pred is not None):
                    _cer = self._cerebellum_out.get(cid)
                    if _cer is not None and int(getattr(
                            _pred, "data_dim", _SELF_OBS_OFFSET)) == int(_cer.shape[-1]):
                        pred_input = _cer

                # Phase 1 — predictor supervised step + Phase 2 intrinsic.
                # Идёт ДО motor REINFORCE update, чтобы intrinsic подмешать
                # в r_imm_total как baseline-сигнал.
                intrinsic_now = self._predictor_train_step(
                    cid, obs_tensor, pred_input)

                # Шаг 2 — петля роста связей (propose→dwell→Δloss_ema→keep/backoff).
                # После train-step (loss_ema свежий). Gated: только single-Adam + флаг.
                if self._growth_enabled and self._single_organism:
                    try:
                        self._brain_growth_step(cid)
                    except Exception as e:
                        logger.warning("brain-growth step %s: %s", cid, e)
                # §10.8 — петля роста ТКАНЕЙ (узлами). Gated отдельным флагом;
                # растит узел ТОЛЬКО когда связи насыщены (узлы после рёбер).
                if self._tissue_growth_enabled and self._single_organism:
                    try:
                        self._tissue_growth_step(cid)
                    except Exception as e:
                        logger.warning("tissue-growth step %s: %s", cid, e)

                # Phase 7 — REINFORCE update от прошлого тика. Сначала
                # SFNN S4 (14.05.2026): motor обучается локальным правилом
                # пластичности. Под genome.sfnn_enabled=False — forward без
                # update_step (заморозка весов).
                org = self.organisms.get(cid)
                sfnn_on = bool(getattr(getattr(org, "genome", None),
                                        "sfnn_enabled", True))
                # Гейт (Фрай 05.06): при медленном канале Hebbian на motor_policy
                # ВЫКЛ — renorm воевал бы с REINFORCE-градиентом; на сервере Hebbian
                # на memory/brain, REINFORCE на policy (разные ткани). Слабый канал =
                # единственный учитель motor_policy. Высшие/basic ткани — свой Hebbian.
                if sfnn_on and self._motor_slow_on <= 0.0:
                    self._motor_sfnn_update_step(cid, events_per_cid,
                                                   intrinsic_now)

                # motor_policy forward → motor_delta [16] (tanh). hooks
                # SFNN-стиля захватывают pre/post активации для следующего
                # update_step. Применяется к первым 16 элементам logits
                # (action-space). Если ткани нет — fallback к raw_logits.
                # Phase 4 (01.06.2026): контекстный шейпинг логитов до motor_delta
                # (порт _decide_action) — даёт сигнал «иди к еде/беги/атакуй»,
                # без которого ActionSelector коллапсирует в move-only → голод.
                # on_flora + carried_food (P40 authoritative > client зеркало).
                _onf = bool((on_flora_per_cid or {}).get(cid, False))
                _p40_cf = (carried_food_per_cid or {}).get(cid)  # None если нет
                try:
                    _diet = float((self.traits.get(cid) or {}).get(
                        "diet_gene", 0.5) or 0.5)
                    _bc2 = self.biochem.get(cid)
                    _er = (float(getattr(_bc2, "energy", 500.0)) / 1000.0
                           if _bc2 is not None else 0.5)
                    _nf_cid = ((nearest_flora_per_cid or {}).get(cid)
                               if nearest_flora_per_cid is not None else None)
                    # obs[62/63] инжект CLIENT-SIDE из nearest_flora-поля (Фрай 05.06):
                    # Хьюберт шлёт поле (для прайора), но obs[62/63] не заполнил
                    # server-side → инжектим сами (само-содержащееся). obs[62]=dist-
                    # сигнал 1/(1+dist) (1=на флоре, →0 далеко), obs[63]=kind/3. Мотор
                    # читает → медленный канал учит «GATHER↔высокий obs62» = контекст.
                    if _nf_cid is not None and len(obs_arr) > 63:
                        try:
                            _d = float(_nf_cid.get("dist", 0.0) or 0.0)
                            _k = float(_nf_cid.get("kind", 0) or 0)
                            _o62 = 1.0 / (1.0 + max(0.0, _d))
                            _o63 = _k / 3.0
                            obs_arr[62] = _o62
                            obs_arr[63] = _o63
                            if obs_tensor.shape[-1] > 63:
                                obs_tensor[0, 62] = _o62
                                obs_tensor[0, 63] = _o63
                        except Exception:
                            pass
                    # FLORA_NAV-диаг (Фрай 05.06): подтвердить приём сигнала +
                    # dist-тренд (arrival-прокси) + obs[62/63]. Rate-limit 1/50.
                    if _nf_cid is not None:
                        self._nf_diag_n = getattr(self, "_nf_diag_n", 0) + 1
                        if self._nf_diag_n % 50 == 0:
                            try:
                                logger.info(
                                    "FLORA_NAV_DIAG cid=%s dr=%s dc=%s dist=%s "
                                    "kind=%s obs62=%.3f obs63=%.3f", cid,
                                    _nf_cid.get("dr"), _nf_cid.get("dc"),
                                    _nf_cid.get("dist"), _nf_cid.get("kind"),
                                    float(obs_arr[62]) if len(obs_arr) > 63 else -1.0,
                                    float(obs_arr[63]) if len(obs_arr) > 63 else -1.0)
                            except Exception:
                                pass
                    # «Доедай, не убегай» (Фрай 06.06): ground-truth «поел в
                    # прошлый тик» (ate / delta_energy>0) → тайл ещё кормит →
                    # прайор STAY/GATHER на месте, гасим nav-толчок. φ-гистерезис
                    # _TILE_YIELD_MEM=2 (Fib): refill при отдаче, иначе decay.
                    _ev_cid = (events_per_cid.get(cid) or {}) if events_per_cid else {}
                    _de_cid = float(_ev_cid.get("delta_energy", 0.0) or 0.0)
                    if bool(_ev_cid.get("ate")) or _de_cid > 0.0:
                        self._tile_yield_mem[cid] = 2          # Fib refill
                    else:
                        self._tile_yield_mem[cid] = max(
                            0, self._tile_yield_mem.get(cid, 0) - 1)
                    _staying = self._tile_yield_mem.get(cid, 0) > 0
                    # YIELD_GATE-диаг (Фрай 06.06, read-only): частота срабатывания
                    # gate + carried_food snapshot — проверка гипотезы «собирает
                    # вместо ест» (gate ждёт delta_energy, а Адам GATHER'ит в склад).
                    if _staying:
                        self._nav["yield_fire"] += 1
                    # cf-валидация (Фрай 06.06): P40 реально шлёт carried_food?
                    # cf_p40_seen>0 → cf=значение валидно; =0 → cf=0 это «нет
                    # сигнала» (caveat подтверждён), а не «склад пуст».
                    if _p40_cf is not None:
                        self._nav["cf_p40_seen"] += 1
                    self._nav["cf_last"] = int(
                        _p40_cf if _p40_cf is not None
                        else self._carried_food.get(cid, 0))
                    # КОНТР-АТАКА (Фрай/Хьюберт 07.06): хищник УДАРИЛ Адама этот тик
                    # (damage_taken>0) = хищник ТОЧНО был в radius=1 → надёжнейший
                    # сигнал контакта (лучше obs[61], который лагает/не пикует). →
                    # сильный ATTACK-bias «бей в ответ, пока он рядом».
                    _just_hit = float(_ev_cid.get("damage_taken", 0.0) or 0.0) > 0.0
                    # ARRIVAL STUCK-DETECTION (Фрай 07.06 go): вектор (dr,dc,dist)
                    # к nearest_flora НЕ меняется N=13 тиков (φ-Fib) при dist>0 (не
                    # на флоре) → цель недостижима (P40 блокит ход через воду) →
                    # АБАНДОН: обход препятствия перпендикулярно (сторона чередуется
                    # каждые 13 тиков). pos сменится → nf пересчёт → счётчик сброс →
                    # фураж к достижимой флоре. _staying (ест) → не стак. server-незав.
                    _abandon_dir = None
                    if _nf_cid is not None and not _staying:
                        _nfk = (_nf_cid.get("dr"), _nf_cid.get("dc"),
                                _nf_cid.get("dist"))
                        _nfd = _nf_cid.get("dist")
                        if (_nfd is not None and float(_nfd) > 0.0
                                and _nfk == self._arrival_last_nf.get(cid)):
                            self._arrival_stuck_n[cid] = (
                                self._arrival_stuck_n.get(cid, 0) + 1)
                        else:
                            self._arrival_stuck_n[cid] = 0
                        self._arrival_last_nf[cid] = _nfk
                        _sn = self._arrival_stuck_n.get(cid, 0)
                        if _sn >= 13:
                            # ROBUST escape (0.13.29): цель недостижима → СКАН всех
                            # 4 кардинальных направлений (не только перпендикуляр —
                            # ловушка может блокить N/S И флора по диагонали, escape
                            # только E/W). Ротация каждые 5 тиков (sn//5)%4 → все 4
                            # за 20 тиков, любое открытое сменит pos → стак-сброс.
                            # Порядок [N,E,S,W] начинаем с ПЕРПЕНДИКУЛЯРА к флоре
                            # (вероятнее открыт, чем сторона флоры).
                            _dr0 = float(_nf_cid.get("dr", 0.0) or 0.0)
                            _dc0 = float(_nf_cid.get("dc", 0.0) or 0.0)
                            if abs(_dc0) >= abs(_dr0):   # флора E/W → старт с N/S
                                _ring = [0, 1, 2, 3]     # N, S, E, W
                            else:                         # флора N/S → старт с E/W
                                _ring = [2, 3, 0, 1]     # E, W, N, S
                            _abandon_dir = _ring[(_sn // 5) % 4]
                            if self._nf_diag_n % 50 == 0:
                                logger.info(
                                    "ARRIVAL_STUCK cid=%s nf=%s stuck=%d "
                                    "abandon_dir=%d", cid, _nfk, _sn, _abandon_dir)
                    else:
                        self._arrival_stuck_n[cid] = 0
                    self._shape_action_logits(logits[0], obs_arr, _diet, _er,
                                              nearest_flora=_nf_cid,
                                              recent_yield=_staying,
                                              on_flora=_onf,
                                              just_hit=_just_hit,
                                              flora_abandon_dir=_abandon_dir)
                    # Newborn-инстинкт (Фрай, порт phase_a.py:748-755): тяга к
                    # GATHER/EAT, затухает за 500 тиков. Только client-рождённые
                    # (birth_tick трекается в mate-flow). Даёт eat-reward →
                    # motor_policy доучивается есть до затухания инстинкта →
                    # Lamarckian (eat→eff↑→income↑). passive_eating — бэкстоп.
                    self._apply_newborn_instinct(
                        cid, logits[0], world_tick, _onf, _p40_cf)
                except Exception as e:
                    logger.debug("action shaping %s: %s", cid, e)

                motor_delta = self._motor_forward(cid, obs_tensor)
                selector = self.action_selectors[cid]
                # Кроссфейд (порт server loop.py:603): own_contribution =
                # max(0, 1-bias_scale) масштабирует motor_delta. bias_scale=1.0
                # (untrained) → motor×0 → shaping (флора-градиент) ведёт чисто
                # → доходят до еды; bias_scale→0 (обучена) → motor автономен.
                # Голос мотора (Фрай 03.06 curriculum): под single_organism
                # управляется _motor_voice (флаг) — Фаза 1 убавить (прайор ведёт →
                # плотный reward → мотор учит alignment), Фаза 2 fade-up (тест
                # адекватности SFNN-модулятора). Колония: старый bias_scale-кроссфейд.
                _own = (float(self._motor_voice) if self._single_organism
                        else max(0.0, 1.0 - float(self._bias_scale)))
                _so_this_ctx = None
                _it_this_ctx = None
                if motor_delta is not None and _own > 0.0:
                    _base_dbg = logits[0, :N_ACTIONS].detach().clone()  # LOGIT_DEBUG: post-shaping, pre-motor
                    action_slice = logits[0, :N_ACTIONS] + motor_delta * _own
                    # Track 2: self-obs→action голова — bias логитов от состояния
                    # (zero-init → старт без влияния, учится REINFORCE). Прямой
                    # путь self-observable (голод/неуверенность/паралич) → действие.
                    _so_head = (self.self_obs_head.get(cid)
                                if self._self_obs_head_enabled else None)
                    if _so_head is not None:
                        try:
                            _so4t = torch.from_numpy(
                                self._build_self_observable(cid)).to(self.device)
                            with torch.no_grad():
                                _so_bias = _so_head(_so4t)
                            _base = action_slice.detach()
                            action_slice = action_slice + _so_bias
                            _so_this_ctx = [_so4t, None, _base]  # action ниже
                        except Exception as e:
                            logger.debug("self_obs head bias %s: %s", cid, e)
                    # Направление (б): insula-temp модуляция (sharpen/flatten,
                    # НЕ меняет направление). action_slice / T_mod до select.
                    action_slice, _it_this_ctx = self._apply_insula_temp(
                        cid, action_slice)
                    self._logit_dbg[cid] = (
                        _raw_dbg, _base_dbg, action_slice.detach(), float(_own),
                        float(obs_arr[33]) if len(obs_arr) > 34 else 0.0,
                        float(obs_arr[34]) if len(obs_arr) > 34 else 0.0,
                        motor_delta.detach().clone(),  # LOGIT_DEBUG + motor_delta
                        float(obs_arr[58]) if len(obs_arr) > 61 else 0.0,  # prey-близость (Шеф 04.06)
                        float(obs_arr[61]) if len(obs_arr) > 61 else 0.0)  # predator-близость
                    action = int(selector.select(
                        action_slice, n_actions=N_ACTIONS))
                    # Policy-gradient контекст (Фрай 04.06): сохранить СЭМПЛИРОВАННОЕ
                    # действие + π=softmax(action_slice) для PG-апдейта output_proj
                    # (∇log π·advantage к выбранному действию) на следующем тике —
                    # тангенциальный credit, может разучить колонию + выучить forage.
                    if self._motor_pg_on > 0.0 and self._motor_slow_on <= 0.0:
                        self._motor_pg_ctx[cid] = (
                            int(action),
                            torch.softmax(action_slice.detach(), dim=-1))
                    # МЕДЛЕННЫЙ канал (Фрай 05.06, порт WorldTrainer): record
                    # (obs, action, reward, base) + train каждые train_every. Credit:
                    # pending action_t → reward из event ЭТОГО тика (P40-lag даёт
                    # outcome предыдущего действия). base = прайор (_base_dbg, detach).
                    if self._motor_slow_on > 0.0:
                        _tr = self.motor_slow_trainer.get(cid)
                        if _tr is None:
                            try:
                                from .slow_trainer import MotorSlowTrainer
                                _tr = MotorSlowTrainer(
                                    torch, self.motor_policy[cid],
                                    _MOTOR_POLICY_SCALE)
                                self.motor_slow_trainer[cid] = _tr
                            except Exception as _e:
                                logger.debug("slow_trainer init %s: %s", cid, _e)
                        if _tr is not None:
                            _ev = (events_per_cid.get(cid)
                                   if events_per_cid is not None else None)
                            _rew = (float(self._compute_immediate_reward(_ev))
                                    if _ev is not None else 0.0)
                            _rl = self.motor_sfnn_rule.get(cid)
                            _T_m = (self._motor_temp if self._motor_temp > 0.0
                                    else (_rl.temperature if _rl is not None else 1.0))
                            _prev = self._slow_pending.get(cid)
                            if _prev is not None:
                                try:
                                    _tr.record(_prev[0], _prev[1], _rew, _prev[2])
                                    if _tr.should_train():
                                        _tr.train_step(float(_own), float(_T_m))
                                except Exception as _e:
                                    logger.debug("slow_train %s: %s", cid, _e)
                            self._slow_pending[cid] = (
                                obs_tensor.detach().reshape(-1),
                                int(action), _base_dbg)
                else:
                    _base_dbg2 = logits[0, :N_ACTIONS].detach().clone()
                    logits_eff = logits[0, :N_ACTIONS]
                    logits_eff, _it_this_ctx = self._apply_insula_temp(
                        cid, logits_eff)
                    self._logit_dbg[cid] = (
                        _raw_dbg, _base_dbg2, logits_eff.detach(), float(_own),
                        float(obs_arr[33]) if len(obs_arr) > 34 else 0.0,
                        float(obs_arr[34]) if len(obs_arr) > 34 else 0.0,
                        None,  # LOGIT_DEBUG (_own=0, нет motor_delta)
                        float(obs_arr[58]) if len(obs_arr) > 61 else 0.0,  # prey-близость (Шеф 04.06)
                        float(obs_arr[61]) if len(obs_arr) > 61 else 0.0)  # predator-близость
                    action = int(selector.select(logits_eff, n_actions=N_ACTIONS))
                if _so_this_ctx is not None:
                    _so_this_ctx[1] = action
                out[cid] = {"action": action, "target_id": None}
                # STAY_PROBE (Фрай 06.06, совместная тик-в-тик проба с Хьюбертом):
                # за флагом park_test (контролируемое условие). По-тиковый лог,
                # выровнен по world_tick для кросс-сверки со steps/tick сервера.
                # Главный вопрос #1: совпадает ли мой STAY-тик с тем, что P40 НЕ
                # сдвинул Адама (Хьюберт видит steps/tick=1.0 при моём STAY →
                # подозрение STAY не исполняется). Гип-2: расходятся ли nf.dist==0
                # и _onf (доходит до nav-цели, а P40 не считает on-flora).
                if self._motor_park_test > 0.0 or self._motor_stay_force > 0.0:
                    try:
                        _nfd0 = (1 if (_nf_cid is not None
                                       and float(_nf_cid.get("dist", -1.0)) == 0.0)
                                 else 0)
                        logger.info(
                            "STAY_PROBE tick=%s cid=%s action=%d stay=%d "
                            "nf_dist0=%d onf=%d", world_tick, cid, int(action),
                            1 if action == 4 else 0, _nfd0, 1 if _onf else 0)
                    except Exception:
                        pass

                # carried_food зеркало — ТОЛЬКО fallback, если P40 не шлёт
                # authoritative carried_food (переходный период). Когда P40
                # шлёт (_p40_cf не None) — зеркало не нужно (P40 = истина).
                if _p40_cf is None and cid in self._birth_tick:
                    self._update_carried_food_mirror(cid, action, _onf)

                # NAV_DIAG (Фрай): измеряем навигацию к еде + кто доминирует.
                self._nav["ticks"] += 1
                if _onf:
                    self._nav["onf"] += 1
                # P40 ground-truth: реально ли P40 зарегистрировал eat (ate
                # событие прошлого тика). Сравнение с gather_onf (клиентское
                # зеркало) вскрывает рассинхрон client-выбор vs P40-резолв (#2).
                if events_per_cid and events_per_cid.get(cid, {}).get("ate"):
                    self._nav["p40_ate"] += 1
                if action == 13:        # GATHER
                    self._nav["gather"] += 1
                    if _onf:
                        self._nav["gather_onf"] += 1
                elif action == 14:      # EAT
                    self._nav["eat"] += 1
                if 0 <= action <= 3:    # move N/S/E/W (YIELD_GATE-диаг)
                    self._nav["move"] += 1
                    # NAV-HIT (Фрай 06.06): депишн-независимая метрика навигации —
                    # идёт ли move В СТОРОНУ nearest_flora? Высокий hit-rate →
                    # прайор навигирует; ~случайный (≤0.5) → бродит. dr<0→N(0),
                    # dr>0→S(1), dc>0→E(2), dc<0→W(3). Считаем только когда есть
                    # nf и dist>0 (есть куда идти).
                    if _nf_cid is not None:
                        try:
                            _ndr = float(_nf_cid.get("dr", 0.0))
                            _ndc = float(_nf_cid.get("dc", 0.0))
                            _nds = float(_nf_cid.get("dist", 0.0))
                            if _nds > 0.0:
                                self._nav["nav_moves"] += 1
                                _hit = ((action == 0 and _ndr < 0) or
                                        (action == 1 and _ndr > 0) or
                                        (action == 2 and _ndc > 0) or
                                        (action == 3 and _ndc < 0))
                                if _hit:
                                    self._nav["nav_hit"] += 1
                        except Exception:
                            pass
                elif action == 4:       # STAY
                    self._nav["stay"] += 1
                if action == 5:         # ATTACK (DAMAGE-канал: оборона-эмиссия)
                    self._nav["attack"] += 1
                    # ATTACK_REACH (Фрай 07.06): долетает ли ATTACK? pred_prox
                    # (obs[61]) в момент эмиссии — контакт (≥0.5) vs воздух.
                    _pp = float(obs_arr[61]) if len(obs_arr) > 61 else 0.0
                    self._nav["atk_pp_sum"] += _pp
                    if _pp >= 0.85:   # истинный melee radius=1 (obs[61]≈1, dist≈1)
                        self._nav["atk_contact"] += 1
                        # §6: attack_power растёт от БОЕВОГО ИСПОЛЬЗОВАНИЯ (atk в
                        # melee), клиент-надёжно — killed-события дропаются на
                        # пропущенных тиках (§3.5), kill=1 vs server 14.
                        self._skill_atk[cid] = self._skill_atk.get(cid, 0) + 1
                elif action == 10:      # FLEE
                    self._nav["flee"] += 1
                    # FLEE pred_prox — сравнение: бежит-при-контакте (avoid) vs
                    # ATTACK-при-контакте (engage). flee_pp>atk_pp = избегает.
                    _ppf = float(obs_arr[61]) if len(obs_arr) > 61 else 0.0
                    self._nav["flee_pp_sum"] += _ppf
                if motor_delta is not None:
                    try:
                        _sh_arg = int(torch.argmax(
                            logits[0, :N_ACTIONS]).item())  # shaping-выбор
                        if _sh_arg != action:               # motor перебил
                            self._nav["flip"] += 1
                        self._nav["mnorm"] += float(motor_delta.norm().item())
                    except Exception:
                        pass

                # F5 skill-growth (01.06.2026, Фрай): счётчик move (0-3) + окно
                # 200 тиков → _skill_growth_step. §6 predator_defense: FLEE(10) =
                # бегство-локомоция → тоже растит move_speed (×2: рывок 3 тайла,
                # интенсивнее обычного шага; «бегство рабочее при move_speed>3»).
                if 0 <= action <= 3:
                    self._skill_move[cid] = self._skill_move.get(cid, 0) + 1
                elif action == 10:      # FLEE — flight локомоция (§6)
                    self._skill_move[cid] = self._skill_move.get(cid, 0) + 2
                # §6/§3.5 (Фрай 07.06): окно по CLIENT-тикам, НЕ world_tick.
                # world_tick прыгает (~4-100/client-apply) → 200-world-окно
                # истекало за 1-2 client-тика → move=0-2, пороги недостижимы,
                # decay всегда → move_speed дренил к полу. Считаем client-applies.
                _sn = self._skill_window_tick.get(cid, 0) + 1
                if _sn >= 200:
                    self._skill_growth_step(cid)
                    self._skill_window_tick[cid] = 0
                else:
                    self._skill_window_tick[cid] = _sn

                # Brain migration (10.05.2026): forward S2.E/G/A/F (no_grad).
                # §3.2 (Фрай 09.06.2026): client-authoritative intero — строим
                # ЛОКАЛЬНО из self.biochem (primary), P40-intero храним для carry
                # slots 2/6 + fallback. Снимает P40-blind risk (insula не слепнет
                # без P40-поля). См. _build_client_intero.
                intero_tensor = None
                if intero_per_cid is not None:
                    _p40_intero = intero_per_cid.get(cid)
                    if _p40_intero is not None:
                        try:
                            self._last_p40_intero[cid] = np.asarray(
                                _p40_intero, dtype=np.float32).reshape(-1)
                        except Exception:
                            pass
                intero_arr = self._build_client_intero(cid)
                if intero_arr is None:
                    intero_arr = self._last_p40_intero.get(cid)  # fallback P40
                if intero_arr is not None:
                    try:
                        intero_tensor = torch.from_numpy(
                            np.asarray(intero_arr, dtype=np.float32).reshape(-1)
                        ).to(self.device).unsqueeze(0)
                    except Exception:
                        intero_tensor = None
                self._compute_higher_tissues(cid, obs_tensor, intero_tensor)

                # Z1 (16.05.2026, Зодчий) — apply-step 7 высших тканей
                # переехал в единый проход после heb.update (там доступны
                # r_imm_total и td_mult для полной S6.5-формулы). Здесь
                # только forward'ы theory_of_mind / language, чьи hooks
                # должны отстрелить ДО второго прохода.
                higher_sfnn_on = bool(getattr(
                    getattr(org, "genome", None),
                    "higher_tissue_sfnn_enabled", False))

                # S2.B (13.05.2026) — theory_of_mind supervised step.
                # Использует WorldStateCache.tom_neighbors_view; если кеша
                # нет / соседей нет — no-op без побочных эффектов.
                self._compute_theory_of_mind(cid)

                # S2.C (13.05.2026) — language supervised step.
                # prev_context (сигналы соседей с прошлого тика) → текущий
                # event_class (ate/damage/killed/idle, idle skip).
                lang_event = (events_per_cid.get(cid)
                              if events_per_cid is not None else None)
                self._compute_language(cid, lang_event)

                # Phase 6 — entropy EMA по action-distribution.
                self._update_entropy_ema(cid, logits)

                # Phase F3.2.b: Hebbian update по локальному R3 reward.
                if heb is not None and events_per_cid is not None:
                    event = events_per_cid.get(cid)
                    if event is not None:
                        r_imm = self._compute_immediate_reward(event)
                        # Phase 2 — подмешать intrinsic в immediate.
                        r_imm_total = r_imm + intrinsic_now
                        # Track 2: REINFORCE self-obs→action голова. events =
                        # прошлый тик → r_imm_total = награда за ПРОШЛОЕ действие →
                        # обновляем ПРЕДЫДУЩИЙ ctx. advantage = r − baseline (своя
                        # бегущая средняя). Потом ротация: this-tick ctx → prev.
                        if cid in self.self_obs_head:
                            _b = self._so_head_baseline.get(cid, 0.0)
                            _adv = r_imm_total - _b
                            self._so_head_baseline[cid] = (
                                0.99 * _b + 0.01 * r_imm_total)
                            self._self_obs_head_reinforce(
                                cid, self._so_head_ctx.get(cid), _adv)
                            self._so_head_ctx[cid] = _so_this_ctx
                        # Направление (б): REINFORCE insula-temp голова (1-dim
                        # Gaussian-policy). Та же ротация: награда r_imm_total за
                        # ПРОШЛОЕ действие → обновляем ПРЕДЫДУЩИЙ temp-ctx
                        # (baseline/variance-reduction внутри). this-tick → prev.
                        if cid in self.insula_temp_head:
                            self._insula_temp_reinforce(
                                cid, self._it_ctx.get(cid), r_imm_total)
                            self._it_ctx[cid] = _it_this_ctx
                        # Phase 5d (NEOL): TD-модулятор η для reward_output.
                        # td = β − EMA(β, α=0.01), mult = 1 + clip(td, ±0.5).
                        # Если S2.E off (нет beta → td=0.0) → mult=1.0 (no-op).
                        # Если HebbianController старой версии (без kwarg) —
                        # TypeError → fallback на безмодуляторный вызов.
                        td = self.dopamine_td.get(cid, 0.0)
                        td_mult = 1.0 + max(-0.5, min(0.5, td))
                        # SFNN S6.6 (16.05.2026): под флагом
                        # genome.basic_tissue_sfnn_enabled расширяем skip_roles
                        # на 10 базовых тканей — классика их больше не пишет.
                        basic_sfnn_on = bool(getattr(
                            getattr(org, "genome", None),
                            "basic_tissue_sfnn_enabled", False))
                        if basic_sfnn_on:
                            heb_skip = (_SFNN_MIGRATED_ROLES
                                         | set(_BASIC_SFNN_TISSUES))
                        else:
                            heb_skip = _SFNN_MIGRATED_ROLES
                        try:
                            heb.update(logits,
                                       {"immediate": r_imm_total,
                                        "medium": 0.0, "long": 0.0},
                                       dopa_td_mult=td_mult,
                                       skip_roles=heb_skip)
                            self.hebbian_updates += 1
                        except TypeError:
                            try:
                                heb.update(logits,
                                           {"immediate": r_imm_total,
                                            "medium": 0.0, "long": 0.0},
                                           skip_roles=heb_skip)
                                self.hebbian_updates += 1
                            except Exception as e:
                                logger.debug("hebbian update %s: %s", cid, e)
                        except Exception as e:
                            logger.debug("hebbian update %s: %s", cid, e)
                        # SFNN S6.6 (16.05.2026): apply-step 10 базовых ролей.
                        # Запускается после heb.update — _last_input/_last_output
                        # уже проставлены. r_imm_eff = r_imm_total как baseline-
                        # subtracted сигнал (heb внутри обновил _baseline_imm).
                        if basic_sfnn_on:
                            try:
                                self._basic_sfnn_update_step(
                                    cid, heb,
                                    dopa_td_mult=td_mult,
                                    r_imm_eff=r_imm_total,
                                    r_med_eff=0.0,
                                    r_long_eff=0.0,
                                )
                            except Exception as e:
                                logger.debug("basic_sfnn step %s: %s", cid, e)
                        # Z1 (Зодчий, 16.05.2026): apply-step 7 высших с
                        # полной S6.5-формулой (τ-trace + R3 + td_coupling).
                        # Запускается после heb.update, как basic_sfnn —
                        # к этому моменту все 7 высших уже отстреляли forward
                        # в _compute_higher_tissues / _compute_theory_of_mind /
                        # _compute_language, acts актуальны.
                        # Z7.i.e (16.05.2026): + 3 Зодчий-ткани. Для cid с
                        # lineage != "zodchiy" rule_store пустой по этим
                        # ключам → update_step делает early return на rule
                        # is None. Apply-step generic, переиспользуется.
                        if higher_sfnn_on:
                            for _t in (_HIGHER_SFNN_TISSUES
                                       + _ZODCHIY_EXTRA_TISSUES):
                                try:
                                    self._higher_tissue_sfnn_update_step(
                                        _t, cid,
                                        dopa_td_mult=td_mult,
                                        r_imm_eff=r_imm_total,
                                        r_med_eff=0.0,
                                        r_long_eff=0.0,
                                    )
                                except Exception as e:
                                    logger.debug(
                                        "higher_sfnn step %s/%s: %s",
                                        _t, cid, e)
                        # Phase 6 — reward_var_ema по последним 10 r_imm.
                        self._update_reward_var_ema(cid, r_imm_total)
                # Phase 6 — trace_norm_ema по Hebbian-traces.
                self._update_trace_norm_ema(cid, heb)
                # TZ B Phase 2 (Бендер, 26.05.2026): per-role tracker sample.
                # Накапливает n_learning/n_total/delta_sum для observability;
                # snapshot emit'ится в diagnostics() каждые 30с (DIAGNOSTICS_
                # PUSH_SEC из main.py). Source приоритет: SFNN → classic.
                self._record_hebbian_per_tissue_sample(cid)
                # Body Migration Phase 2 (Бендер, 27.05.2026): client-side
                # биохимия. Порядок: events → decay. apply_* применяет
                # deltas из event_dict (ate/killed/damage_taken), затем
                # decay_step обновляет 8 веществ passive growth/decay.
                # Math equivalence с server — environment.biochemistry.*
                # вызывается напрямую. Safety: try/except — не ломает
                # handle_tick.
                _bc_event = (events_per_cid.get(cid)
                             if events_per_cid is not None else None)
                self._apply_biochem_events(cid, _bc_event)
                self._apply_biochem_decay(cid)
                # §3 триггер 2 (Хьюберт 2b0f3a2 / Фрай): death_suppressed от P40
                # (chokepoint suppress'нул PvP/age/...) → paralysis, ДАЖЕ если
                # energy>0 (отдельный вход, не через energy≤0). Reason — для лога.
                if self._single_organism and _bc_event:
                    _ds = _bc_event.get("death_suppressed")
                    if _ds:
                        _reason = "suppressed"
                        try:
                            _reason = str((_ds[0] or {}).get("reason", "suppressed"))
                        except Exception:
                            pass
                        # Триггер 2 — ТОЛЬКО транзиентные внешние death-вектора
                        # (pvp/attack): паралич как обучающий сигнал «избегай».
                        # ПЕРМАНЕНТНЫЕ состояния Адама игнорим — для персистентного
                        # одиночки они always-true → паралич каждый тик (как loner):
                        #   starv — энерго-ось, домен триггера 1 (energy≤0);
                        #   age   — старения-смерти у Адама нет (§1), telomere
                        #           AGONY давится immortality, но P40 спамит
                        #           death_suppressed(aged) каждый тик.
                        _rl = _reason.lower()
                        if not any(k in _rl for k in ("starv", "age")):
                            self._enter_paralysis(cid, _reason)
                # Body Migration метаболизм (31.05.2026, Бендер; контракт
                # Хьюберт): client-authoritative энергозатрата/гидрация/теломеры.
                # P40 шлёт effective rates (step_cost_now/thirst_now/
                # telomere_decay_now), client интегрирует. income (delta_energy)
                # уже добавлен в _apply_biochem_events выше → здесь только cost
                # + death-check (energy<=0 голод / telomere AGONY старость).
                _rates = (rates_per_cid.get(cid)
                          if rates_per_cid is not None else None)
                self._apply_metabolism(cid, _rates)
                # Phase 2 этап 5 (27.05.2026): hysteresis-aware mental_break
                # update + force_STAY override action если catatonic/
                # exhaustion/glucose<5. Порядок: decay сначала (обновил
                # cortisol/serotonin/...), потом recompute mental_break,
                # потом override action для P40 actions_batch.
                self._apply_biochem_mental_break(cid, world_tick)
                self._maybe_force_stay(cid, out)
            except Exception as e:
                logger.warning("handle_tick %s failed: %s", cid, e)
                out[cid] = {"action": STAY, "target_id": None}
        return out

    # ── Phase F3.2.c: персистенция Hebbian-state на диск ────────────────

    def save_state(self, cid: str) -> Optional[dict]:
        """Собрать payload для torch.save: формат идентичен P40 `_save_member_pt`,
        чтобы при загрузке можно было прогнать через `organism_from_weights`.

        Возвращает None, если особь неизвестна.
        """
        org = self.organisms.get(cid)
        if org is None:
            return None
        payload: dict = {}
        if hasattr(org, "tissues"):
            try:
                payload["tissues_by_role"] = {
                    (getattr(t, "role", "") or f"_unknown_{tid}"): t.state_dict()
                    for tid, t in org.tissues.items()
                }
            except Exception as e:
                logger.warning("save_state %s tissues: %s", cid, e)
                return None
        # Z2.b (01.06.2026, Фрай): межтканевые topology genes (NEAT-overlay
        # Зодчего). Без них граф особи теряется на restart/resume/elite →
        # speciation схлопывается в founder. Сериализуем как list[dict]
        # (TissueConnectionGene.to_dict), пустые не пишем.
        genes = getattr(org, "tissue_topology_genes", None)
        if genes:
            try:
                payload["tissue_topology_genes"] = [
                    g.to_dict() if hasattr(g, "to_dict") else dict(g)
                    for g in genes
                ]
            except Exception as e:
                logger.debug("save_state %s topology genes: %s", cid, e)
        heb = self.hebbian.get(cid)
        if heb is not None and hasattr(heb, "state_dict"):
            try:
                payload["hebbian"] = heb.state_dict()
            except Exception as e:
                logger.debug("save_state %s hebbian: %s", cid, e)
        sel = self.action_selectors.get(cid)
        if sel is not None and hasattr(sel, "state_dict"):
            try:
                payload["selector"] = sel.state_dict()
            except Exception as e:
                logger.debug("save_state %s selector: %s", cid, e)
        # SFNN S1.1: motor_sfnn_rule в seed_pack для reconnect.
        sfnn_rule = self.motor_sfnn_rule.get(cid)
        if sfnn_rule is not None and hasattr(sfnn_rule, "to_dict"):
            try:
                payload["motor_sfnn_rule"] = sfnn_rule.to_dict()
            except Exception as e:
                logger.debug("save_state %s motor_sfnn_rule: %s", cid, e)
        # SFNN S3.0: 7 правил высших тканей в seed_pack для reconnect.
        # Z7.i.e (16.05.2026): + 3 Зодчий-правила (для elder/wanderer
        # rule is None → пропадают естественно).
        higher_rules_dump: dict = {}
        for _t in _HIGHER_SFNN_TISSUES + _ZODCHIY_EXTRA_TISSUES:
            _rule = self.higher_tissue_sfnn_rule.get(_t, {}).get(cid)
            if _rule is None or not hasattr(_rule, "to_dict"):
                continue
            try:
                higher_rules_dump[_t] = _rule.to_dict()
            except Exception as e:
                logger.debug(
                    "save_state %s higher_tissue_sfnn_rule[%s]: %s", cid, _t, e)
        if higher_rules_dump:
            payload["higher_tissue_sfnn_rules"] = higher_rules_dump
        # Phase 1 — predictor + EMA. Формат идентичен P40 _save_member_pt.
        pred = self.predictor.get(cid)
        if pred is not None:
            try:
                payload["predictor"] = pred.state_dict()
                payload["predictor_loss_ema"] = float(self.loss_ema.get(cid, 0.0))
                payload["intrinsic_ema"] = float(self.intrinsic_ema.get(cid, 0.0))
            except Exception as e:
                logger.debug("save_state %s predictor: %s", cid, e)
        # §6 рост мозга durable (Фрай 09.06): персистить ПРОГРЕСС петли роста.
        # predictor цел (loss_ema непрерывен) → трейлинг-floor история ВАЛИДНА →
        # PROPOSE возобновляется без ~233-тик re-warm после сбоя питания.
        # growth_kept/reverted — KPI «закреплено» больше не врёт «0» после рестарта;
        # intr_hist — трейлинг-floor deque; stagnation_n — счётчик «у floor».
        # _growth_saturated НЕ персистим: на рестарте сброс в False → петля заново
        # ищет (durable-инвариант роста, Фрай). См. restore_persisted_state.
        try:
            _gi_hist = self._growth_intr_hist.get(cid)
            payload["growth_loop"] = {
                "kept": int(self._growth_kept),
                "reverted": int(self._growth_reverted),
                "intr_hist": list(_gi_hist) if _gi_hist is not None else [],
                "stagnation_n": int(self._growth_stagnation_n.get(cid, 0)),
                # §10.8 рост тканей: счётчики + СПЕКИ выросших узлов для recreate-
                # on-restore (скелет фикс → новую роль restore не создаст без спека).
                # Веса узлов едут в tissues_by_role (org.tissues их уже включает).
                "tissue_kept": int(self._tissue_kept),
                "tissue_reverted": int(self._tissue_reverted),
                "grown_tissues": list(self._tissue_grown_specs.get(cid, [])),
            }
        except Exception as e:
            logger.debug("save_state %s growth_loop: %s", cid, e)
        # Phase 6 — self-observable EMAs.
        if cid in self.entropy_ema:
            payload["entropy_ema"] = float(self.entropy_ema.get(cid, 0.0))
            payload["trace_norm_ema"] = float(self.trace_norm_ema.get(cid, 0.0))
            payload["reward_var_ema"] = float(self.reward_var_ema.get(cid, 0.0))
        # Brain migration (10.05.2026): высшие ткани S2.E/G/A/F + Phase 7 motor.
        # 13.05.2026: + S2.B theory_of_mind.
        for key, store in (
            ("dopamine", self.dopamine),
            ("imagination", self.imagination),
            ("planner", self.planner),
            ("insula", self.insula),
            ("motor_policy", self.motor_policy),
            ("theory_of_mind", self.theory_of_mind),
            ("language", self.language),
        ):
            tissue = store.get(cid)
            if tissue is None:
                continue
            try:
                payload[key] = tissue.state_dict()
            except Exception as e:
                logger.debug("save_state %s %s: %s", cid, key, e)
        # Colony Ownership Migration §5.1 (28.05.2026, Бендер):
        # client-authoritative biochem persistence. Раньше Phase 2
        # биохимия жила in-memory и reset'илась на каждый restart
        # → root cause Phase 4 hotfix (oxytocin=0 → детекция спит).
        # Теперь сохраняется в payload[biochem] как dict от dataclass.
        bc = self.biochem.get(cid) if hasattr(self, "biochem") else None
        if bc is not None:
            try:
                from dataclasses import asdict, is_dataclass
                if is_dataclass(bc):
                    payload["biochem"] = asdict(bc)
            except Exception as e:
                logger.debug("save_state %s biochem: %s", cid, e)
        # Evolved-traits recovery (30.05.2026, Бендер): 9 body-traits +
        # generation в .pt. Раньше тело (traits) жило только loose-атрибутами
        # и НЕ персистилось → client-restart сбрасывал к baseline. Теперь —
        # bit-exact как мозг. Источник: authoritative стор self.traits.
        traits_d = self.traits.get(cid)
        if isinstance(traits_d, dict) and traits_d:
            payload["traits"] = dict(traits_d)
        try:
            gen = getattr(org, "generation", None)
            if gen is not None:
                payload["generation"] = int(gen)
        except Exception:
            pass
        return payload

    def save_all_states(self, dir_path) -> int:
        """Сохранить state всех зарегистрированных особей в `dir_path/{cid}.pt`.

        Возвращает число успешно сохранённых файлов. Каталог создаётся при
        отсутствии. Ошибки сериализации отдельных особей логируются и не
        прерывают остальные.
        """
        from pathlib import Path
        torch = self._torch
        dir_path = Path(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)
        n = 0
        for cid in list(self.organisms.keys()):
            payload = self.save_state(cid)
            if not payload:
                continue
            try:
                torch.save(payload, dir_path / f"{cid}.pt")
                n += 1
            except Exception as e:
                logger.warning("save_state %s torch.save failed: %s", cid, e)
        return n

    def snapshot_elite(self, elite_dir, min_alive: int = 4) -> int:
        """Elite preservation (02.06.2026, Фрай): снимок ЖИВЫХ обученных мозгов
        колонии в отдельный elite-слот, который ПЕРЕЖИВАЕТ вымирание.

        Основной persist (save_all_states) хранит ЖИВЫХ → при полном вымирании
        пуст (учить не с чего). Elite — снимок последней ЗДОРОВОЙ колонии
        (>= min_alive живых с energy>0). Снимаем только здоровых, чтобы elite
        не затёрся умирающей колонией. Recovery (ws_client._maybe_recover_from_
        elite) поднимает отсюда обученные мозги → эволюция продолжается, не
        с untrained-нуля. Возвращает число сохранённых.
        """
        from pathlib import Path
        torch = self._torch
        alive = [cid for cid in self.organisms
                 if cid not in self._dead_cids
                 and float(getattr(self.biochem.get(cid), "energy", 0.0)
                           or 0.0) > 0.0]
        if len(alive) < min_alive:
            return 0
        ed = Path(elite_dir)
        ed.mkdir(parents=True, exist_ok=True)
        for f in ed.glob("*.pt"):  # чистим устаревший снимок
            try:
                f.unlink()
            except Exception:
                pass
        n = 0
        for cid in alive:
            payload = self.save_state(cid)
            if not payload:
                continue
            try:
                torch.save(payload, ed / f"{cid}.pt")
                n += 1
            except Exception as e:
                logger.debug("snapshot_elite %s: %s", cid, e)
        if n:
            logger.info("snapshot_elite: %d обученных мозгов → elite", n)
        return n

    def _get_hw_capacity(self) -> Optional[int]:
        """Расчётная ёмкость колонии по железу (vision body_migration.md §40/
        §145: `benchmark.py:estimate_population`). Кэш — бенчмарк дорогой
        (~секунды). Возвращает int>0 или None (если бенчмарк недоступен/упал)."""
        if self._natural_selection_capacity is None:
            try:
                from .benchmark import estimate_population, run_full
                self._natural_selection_capacity = estimate_population(run_full())
                logger.info("hw capacity = %d (estimate_population, cached)",
                            self._natural_selection_capacity)
            except Exception as e:
                logger.debug("estimate_population failed: %s", e)
                self._natural_selection_capacity = -1  # sentinel: tried, failed
        return (self._natural_selection_capacity
                if self._natural_selection_capacity
                and self._natural_selection_capacity > 0 else None)

    @staticmethod
    def _cap_and_clean_pt(dir_path, cap: int) -> tuple:
        """Carrying-capacity cap для restore (31.05.2026, Шеф). Возвращает
        (keep_paths, culled_cids): N свежайших .pt (по mtime) + cid'ы удалённых.
        УДАЛЯЕТ остальные (persist bloat). cap = estimate_population.

        culled_cids возвращаются ДО удаления → caller шлёт owned_bye, чтобы
        P40 despawn'нул сброшенные проекции (иначе осиротеют на сервере,
        /stats показывает фантомное число — frozen-sticky до 24ч safety cap).
        Удаляет только сверх cap (старейшие); свежайшие cap сохраняет.
        """
        from pathlib import Path
        dir_path = Path(dir_path)
        all_pt = sorted(dir_path.glob("*.pt"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
        if cap < 0:
            cap = 0
        keep = all_pt[:cap]
        cull = all_pt[cap:]
        culled_cids = [p.stem for p in cull]  # собираем ДО unlink
        for old in cull:
            try:
                old.unlink()
            except Exception as e:
                logger.debug("_cap_and_clean_pt unlink %s: %s", old.name, e)
        if cull:
            logger.info(
                "_cap_and_clean_pt: cap=%d — оставлено %d свежих .pt, "
                "почищено %d устаревших (persist bloat) → owned_bye",
                cap, len(keep), len(cull))
        return keep, culled_cids

    def restore_colony_from_local(self, dir_path) -> list[str]:
        """Colony Ownership Migration §5.1: восстановить колонию из local.

        Сканирует `dir_path/*.pt`, для каждого:
          1. torch.load(weights_only=False) payload
          2. Build CompositeOrganism через seed_loader.organism_from_weights
             (skeleton от seed.norg + веса из payload)
          3. add_creature(cid, organism, lineage='zodchiy')
          4. restore_persisted_state(cid, payload) — биохимия + EMA + SFNN
          5. memory_store.load_memory_state(cid) — episodic recall

        Returns:
            List of restored cids. Empty list если dir не существует / пуст.

        Errors:
            Per-file ошибки логируются, не прерывают остальные.
        """
        from pathlib import Path
        torch = self._torch
        dir_path = Path(dir_path)
        if not dir_path.exists() or not dir_path.is_dir():
            logger.info("restore_colony_from_local: dir %s missing — empty start",
                        dir_path)
            return []
        # Resolve seed skeleton. КРИТИЧНО (fix 0.11.8): genesis-зодчий —
        # 20-tissue (brain stack + cerebellum/amygdala/episodic). Genesis
        # записал этот 20-tissue skeleton в wanderer.norg (seed_norg
        # lineage="wanderer"). elder/seed.norg = 10-tissue (nexus) — НЕ
        # подходит, resume дал бы mismatch (как nexus-vs-zodchiy #33).
        #
        # Выбираем skeleton по факту: пробуем wanderer (20-tissue zodchiy)
        # затем elder; берём тот что существует И даёт >=20 tissue, иначе
        # с максимумом. .pt tissues_by_role = 20 ролей, skeleton должен
        # совпадать для bit-exact restore_persisted_state.
        try:
            from .seed_loader import seed_cache_path, load_founders
        except Exception as e:
            logger.warning("restore_colony_from_local: seed_loader import: %s", e)
            return []
        seed_path = None
        best_n = -1
        for _lineage in ("wanderer", "elder"):
            cand = seed_cache_path(_lineage)
            if not cand.exists():
                continue
            try:
                probe = load_founders(cand, 1)[0]
                n_tis = len(getattr(probe, "tissues", {}) or {})
            except Exception as e:
                logger.debug("restore probe %s (%s) failed: %s", _lineage, cand, e)
                continue
            logger.info("restore skeleton probe: %s = %d tissues", _lineage, n_tis)
            if n_tis > best_n:
                best_n = n_tis
                seed_path = cand
        if seed_path is None:
            logger.warning(
                "restore_colony_from_local: no usable seed skeleton — cannot rebuild")
            return []
        logger.info("restore_colony_from_local: skeleton=%s (%d tissues)",
                    seed_path, best_n)

        # Carrying-capacity cap (31.05.2026, Шеф; vision §40/§145). Persist
        # накопил сотни .pt без чистки → restore всех → колония 20× сверх
        # ёмкости железа → cheef-PC не тикает handle_tick → ws-churn → Мир
        # замерзал → отбор не сходился (дедлок). Грузим только N свежайших,
        # где N = estimate_population (расчётная ёмкость). Лишние чистим.
        cap = self._get_hw_capacity() or 16  # fallback 16 если бенчмарк недоступен
        keep_pt, culled_cids = self._cap_and_clean_pt(dir_path, cap)
        # Сброшенные cid'ы → ws шлёт owned_bye (P40 despawn проекций), иначе
        # осиротеют на сервере (фантомное /stats). Читается в ws_client.
        self._cull_bye_cids = list(culled_cids)

        restored: list[str] = []
        for pt_file in keep_pt:
            cid = pt_file.stem
            if cid in self.organisms:
                logger.debug("restore_colony_from_local: %s already loaded, skip",
                              cid)
                continue
            try:
                # weights_only=False — payload содержит non-tensor (dicts, floats)
                payload = torch.load(pt_file, map_location="cpu",
                                      weights_only=False)
            except Exception as e:
                logger.warning("restore_colony_from_local: load %s failed: %s",
                                pt_file.name, e)
                continue
            if not isinstance(payload, dict):
                logger.warning("restore_colony_from_local: %s not a dict (skip)",
                                pt_file.name)
                continue
            # Build skeleton + apply tissue weights via organism_from_weights.
            # Это требует weights_bytes — у нас payload dict, не bytes.
            # Альтернатива: load_founders + restore_persisted_state.
            try:
                from .seed_loader import load_founders
                org = load_founders(seed_path, 1)[0]
                org.id = f"restored_{cid}"
            except Exception as e:
                logger.warning("restore_colony_from_local: load_founders %s: %s",
                                cid, e)
                continue
            try:
                self.add_creature(cid, org, lineage="zodchiy")
            except Exception as e:
                logger.warning("restore_colony_from_local: add_creature %s: %s",
                                cid, e)
                continue
            try:
                self.restore_persisted_state(cid, payload)
            except Exception as e:
                logger.warning(
                    "restore_colony_from_local: restore_persisted_state %s: %s",
                    cid, e)
            # Bootstrap (Фрай): пометить на омоложение — birth_tick выставится
            # в handle_tick (там известен текущий world_tick) → инстинкт
            # активен → особь учится есть, как newborn. На реконнекте/elite-
            # recovery каждый раз заново (ОК, пока колония не self-sustaining).
            self._bootstrap_pending.add(cid)
            restored.append(cid)
        logger.info("restore_colony_from_local: %d organisms restored from %s",
                    len(restored), dir_path)
        return restored

    # Серверный энергобаланс forage/hunt (Фрай 04.06, порт world.py):
    #   prey_kill_energy = φ⁷ ≈ 29.03 (yield убийства, v4.0 абсолютный)
    #   forage_yield     = φ⁴ ≈ 6.854 (yield добычи, ниже kill на φ³≈4.24×)
    #   predator_damage  ≈ φ⁷       (риск = зеркало yield жертвы)
    _KILL_YIELD: float = _PHI ** 7      # ≈ 29.03 = prey_kill_energy
    _FORAGE_YIELD: float = _PHI ** 4    # ≈ 6.854 (forage < hunt, дифференцировано)

    def _compute_immediate_reward(self, event: dict) -> float:
        """R3 immediate из событий тика.

        14.05.2026: усилен ate (1.0→5.0, равно killed) и δenergy (×10) —
        action-space collapse fix (r_imm тонул в метаболическом шуме).
        04.06.2026 (Фрай): плоские РАВНЫЕ +5·ate / +5·killed = два равных
        reward-аттрактора → бистабильность motor-головы (GATHER/+/suppress-ATTACK
        ↔ DIG/−/boost-ATTACK). Под `reward_balance` — порт серверного энергобаланса
        (environment/world.py): differentiated value через энергию + риск. Hunt
        выше-yield (φ⁷≈29) но риск (damage ≈ зеркало yield), forage ниже-yield
        (φ⁴≈6.85) без риска → НЕ равные аттракторы → контекстная policy (охоться
        когда выгодно/безопасно, иначе форажь), не бистабильный флип. Веса tunable.
        """
        delta_energy = float(event.get("delta_energy", 0.0))
        ate = bool(event.get("ate", False))
        killed = bool(event.get("killed", False))
        damage_taken = float(event.get("damage_taken", 0.0))
        if self._reward_balance_on > 0.0:
            # Энерго-дифференцированный: hunt φ⁷ vs forage φ⁴, risk = damage.
            # Дефолты весов 1.0: forage 6.85, safe-kill 29 (4.2×), full-damage
            # kill 29−29≈0 (избегать) → safe-hunt > forage > risky-hunt = контекст.
            return (delta_energy * 0.5
                    + (self._FORAGE_YIELD * self._reward_forage_w if ate else 0.0)
                    + (self._KILL_YIELD * self._reward_kill_w if killed else 0.0)
                    - damage_taken * self._reward_risk_w)
        # Дефолт (колония + safety): прежний плоский reward.
        return (delta_energy * 0.5
                + (5.0 if ate else 0.0)
                + (5.0 if killed else 0.0)
                - damage_taken * 0.1)

    # ── Internals ────────────────────────────────────────────────────────

    def _make_hebbian(self, organism, enabled: bool, lr: float, decay: float):
        if not enabled or not hasattr(organism, "tissues"):
            return None
        try:
            from core.hebbian import HebbianConfig, HebbianController

            cfg = HebbianConfig(
                lr_reward=float(lr),
                lr_oja=float(lr) * 2.0,
                eligibility_decay=float(decay),
            )
            return HebbianController(organism, cfg)
        except Exception as e:
            logger.debug("hebbian init failed: %s", e)
            return None

    # ── Phase 1 — Forward Model (predictor) ──────────────────────────────

    def _make_predictor_tissue(self, n_embd: int = 21):
        """Sidecar Tissue для Phase 1: obs_t → pred(obs_{t+1}).

        n_embd=21 (Fibonacci), n_head=3, n_layer=1. ~8.7K params. Identical
        с _make_predictor_tissue на P40.
        """
        try:
            from core.connection import CellGene
            from core.tissue import Tissue, TissuePort, TissueSpec
        except Exception as e:
            logger.warning("predictor: core imports failed: %s", e)
            return None
        try:
            cg = CellGene(innovation=1, n_embd=n_embd, n_head=3, n_layer=1)
            spec = TissueSpec(
                name="predictor",
                role="predictor",
                cell_genes=[cg],
                connection_genes=[],
                input_ports=[TissuePort("input", 1)],
                output_ports=[TissuePort("output", 1)],
                internal_lr_scale=1.0,
            )
            return Tissue(spec).to(self.device)
        except Exception as e:
            logger.warning("predictor: build failed: %s", e)
            return None

    def _apply_y50_to_predictor(self, predictor) -> None:
        """Y50: offspring = 0.5·parent + 0.5·noise(σ·std), σ=1/φ⁵≈0.0902."""
        torch = self._torch
        with torch.no_grad():
            for name, p in predictor.named_parameters():
                if p.dim() >= 2 and "weight" in name:
                    std = max(float(p.data.std().item()), 1e-6)
                    noise = torch.randn_like(p.data) * _PREDICTOR_Y50_SCALE * std
                    p.data.copy_(0.5 * p.data + 0.5 * noise)
                # bias / 1D — не трогаем (Y50 на P40 тоже).

    # ── Brain migration (10.05.2026) — S2.E/G/A/F sidecar tissues ────────

    def _make_higher_tissue(self, role: str, *, data_dim: int = 64,
                             n_embd: int = 21):
        """Universal sidecar Tissue 21/3/1 для высших тканей (S2.E/G/A/F).

        role: "dopamine" | "imagination" | "planner" | "insula".
        data_dim: 64 для большинства, 71 для insula (obs+intero_7).
        Возвращает Tissue или None при ошибке импортов.
        """
        try:
            from core.connection import CellGene
            from core.tissue import Tissue, TissuePort, TissueSpec
        except Exception as e:
            logger.warning("higher tissue %s: imports failed: %s", role, e)
            return None
        try:
            cg = CellGene(innovation=1, n_embd=n_embd, n_head=3, n_layer=1)
            kwargs = dict(
                name=role,
                role=role,
                cell_genes=[cg],
                connection_genes=[],
                input_ports=[TissuePort("input", 1)],
                output_ports=[TissuePort("output", 1)],
                internal_lr_scale=1.0,
            )
            if data_dim != 64:
                kwargs["data_dim"] = data_dim  # Phase B per-tissue data_dim
            spec = TissueSpec(**kwargs)
            return Tissue(spec).to(self.device)
        except Exception as e:
            logger.warning("higher tissue %s: build failed: %s", role, e)
            return None

    def _apply_y50_to_tissue(self, tissue) -> None:
        """Generic Y50 для любой ткани (мирор _apply_y50_to_predictor)."""
        torch = self._torch
        with torch.no_grad():
            for name, p in tissue.named_parameters():
                if p.dim() >= 2 and "weight" in name:
                    std = max(float(p.data.std().item()), 1e-6)
                    noise = torch.randn_like(p.data) * _HIGHER_TISSUE_Y50_SCALE * std
                    p.data.copy_(0.5 * p.data + 0.5 * noise)

    def _compute_higher_tissues(self, cid: str, obs_tensor,
                                  intero_tensor=None) -> None:
        """Forward всех 4 высших тканей: S2.E/G/A/F.

        Сохраняет:
          last_beta_local[cid]    — S2.E ∈ [0, 1/φ]
          last_imag_mult[cid]     — S2.G ∈ [1, 2]
          last_planner_delta[cid] — S2.A torch.Tensor [16]
          last_stress[cid]        — S2.F ∈ [0, 1] (если intero_tensor задан)

        Сервер прочитает их через get_phase_emas → actions_batch.phase_emas.
        Ablation_mask клиент не знает — гейт делает сервер на стороне fastpath.
        """
        torch = self._torch
        # S2.E — dopamine: β_local = sigmoid(out[0,0]) / φ ∈ [0, 1/φ].
        d_tissue = self.dopamine.get(cid)
        if d_tissue is not None:
            try:
                with torch.no_grad():
                    out = d_tissue({"input": obs_tensor.detach()})["output"]
                    raw = float(out[0, 0].item())
                    beta = 1.0 / (1.0 + math.exp(-raw)) / _PHI
                    self.last_beta_local[cid] = beta
                    # Phase 5d (NEOL): TD = β − EMA(β, α=0.01). Один раз за тик,
                    # rely on _compute_higher_tissues called once per cid per tick.
                    prev_ema = self.dopamine_ema.get(cid, 0.0)
                    self.dopamine_td[cid] = beta - prev_ema
                    self.dopamine_ema[cid] = (1.0 - _EMA_ALPHA) * prev_ema + _EMA_ALPHA * beta
            except Exception as e:
                logger.debug("dopamine forward %s: %s", cid, e)
        # S2.G — imagination: mult = 1 + sigmoid(out[0,0]) ∈ [1, 2].
        i_tissue = self.imagination.get(cid)
        if i_tissue is not None:
            try:
                with torch.no_grad():
                    out = i_tissue({"input": obs_tensor.detach()})["output"]
                    raw = float(out[0, 0].item())
                    mult = 1.0 + 1.0 / (1.0 + math.exp(-raw))
                    self.last_imag_mult[cid] = mult
            except Exception as e:
                logger.debug("imagination forward %s: %s", cid, e)
        # S2.A — planner: delta = scale · tanh(out[0, :16]) ∈ [-1, 1]^16.
        p_tissue = self.planner.get(cid)
        if p_tissue is not None:
            try:
                with torch.no_grad():
                    out = p_tissue({"input": obs_tensor.detach()})["output"]
                    delta = torch.tanh(out[0, :_PLANNER_N_ACTIONS]) * _PLANNER_SCALE
                    self.last_planner_delta[cid] = delta.detach().cpu()
            except Exception as e:
                logger.debug("planner forward %s: %s", cid, e)
        # S2.F — insula: stress = sigmoid(out[0,0]) ∈ [0, 1] над cat[obs, intero].
        ins_tissue = self.insula.get(cid)
        if ins_tissue is not None and intero_tensor is not None:
            try:
                with torch.no_grad():
                    full = torch.cat([obs_tensor.detach(),
                                       intero_tensor.detach()], dim=-1)
                    out = ins_tissue({"input": full})["output"]
                    raw = float(out[0, 0].item())
                    self.last_stress[cid] = 1.0 / (1.0 + math.exp(-raw))
            except Exception as e:
                logger.debug("insula forward %s: %s", cid, e)
        # S2.D — default_mode: floor = sigmoid(out[0,0]) · _DEFAULT_MODE_FLOOR_MAX.
        # Применяется на P40 как фоновый floor для Δsurprise → intrinsic
        # не схлопывается в ноль в «тихом» мире (delta≈0).
        dmn_tissue = self.default_mode.get(cid)
        if dmn_tissue is not None:
            try:
                with torch.no_grad():
                    out = dmn_tissue({"input": obs_tensor.detach()})["output"]
                    raw = float(out[0, 0].item())
                    dmn = 1.0 / (1.0 + math.exp(-raw))
                    self.last_dmn_floor[cid] = dmn * _DEFAULT_MODE_FLOOR_MAX
            except Exception as e:
                logger.debug("default_mode forward %s: %s", cid, e)
        # Z7.i.e (16.05.2026, Зодчий) — 3 уникальные ткани третьей линии.
        # Forward выполняется ТОЛЬКО для lineage="zodchiy" (для elder/wanderer
        # tissue is None — store пустой). Снимки → last_*_*; apply-step
        # переиспользует `_higher_tissue_sfnn_update_step` (см. handle_tick).
        # Cerebellum: motor error-loop, delta = tanh(out[0, :16]) ∈ [-1, 1]^16.
        # Аналог planner, но другая роль/τ (21 vs 233) — мутирует независимо.
        cer_tissue = self.cerebellum.get(cid)
        if cer_tissue is not None:
            try:
                with torch.no_grad():
                    out = cer_tissue({"input": obs_tensor.detach()})["output"]
                    delta = torch.tanh(out[0, :_PLANNER_N_ACTIONS])
                    self.last_cerebellum_delta[cid] = delta.detach().cpu()
            except Exception as e:
                logger.debug("cerebellum forward %s: %s", cid, e)
        # Amygdala: valence ∈ [-1, 1], tanh(out[0, 0]). Используется как
        # эмоциональный gating (negative → стресс, positive → reward).
        amy_tissue = self.amygdala.get(cid)
        if amy_tissue is not None:
            try:
                with torch.no_grad():
                    out = amy_tissue({"input": obs_tensor.detach()})["output"]
                    valence = float(torch.tanh(out[0, 0]).item())
                    self.last_amygdala_valence[cid] = valence
            except Exception as e:
                logger.debug("amygdala forward %s: %s", cid, e)
        # Episodic: long-term recall, vector [64] ∈ [-1, 1]^64. Полное окно
        # выходных каналов — позже служит для associative-retrieval (Z5+),
        # сейчас snapshot для apply-step + диагностики.
        epi_tissue = self.episodic.get(cid)
        if epi_tissue is not None:
            try:
                with torch.no_grad():
                    out = epi_tissue({"input": obs_tensor.detach()})["output"]
                    recall = torch.tanh(out[0, :64])
                    self.last_episodic_recall[cid] = recall.detach().cpu()
            except Exception as e:
                logger.debug("episodic forward %s: %s", cid, e)

    # ── S2.B — theory_of_mind supervised sidecar ─────────────────────────

    @staticmethod
    def _tom_target_action(prev_xy: tuple[int, int],
                            curr_xy: tuple[int, int],
                            world_size: int) -> int:
        """Маппинг Δposition → action class ∈ {NORTH, SOUTH, EAST, WEST, STAY}.

        Тор-метрика: |Δ| > size/2 → знак инвертируется через границу.
        Диагональное движение → ось с большим |Δ|. Если |dx| == |dy| != 0,
        предпочитаем горизонталь (EAST/WEST).
        """
        dx = int(curr_xy[0]) - int(prev_xy[0])
        dy = int(curr_xy[1]) - int(prev_xy[1])
        half = world_size // 2
        if dx > half:
            dx -= world_size
        elif dx < -half:
            dx += world_size
        if dy > half:
            dy -= world_size
        elif dy < -half:
            dy += world_size
        if dx == 0 and dy == 0:
            return _TOM_ACT_STAY
        if abs(dx) >= abs(dy):
            return _TOM_ACT_EAST if dx > 0 else _TOM_ACT_WEST
        return _TOM_ACT_SOUTH if dy > 0 else _TOM_ACT_NORTH

    def _build_tom_features(self, neighbors: list) -> "object":
        """neighbors: list[tuple(ncid, x, y, dx, dy, lineage, energy_norm, sig)].
        Возвращает torch.Tensor [1, _TOM_DATA_DIM=52]. Слоты соседей короче 4
        паддятся нулями.
        """
        torch = self._torch
        feat = torch.zeros(1, _TOM_DATA_DIM, dtype=torch.float32,
                            device=self.device)
        for i, n in enumerate(neighbors[:_TOM_N_NEIGHBORS]):
            base = i * _TOM_FEATURES_PER_NEIGHBOR
            _ncid, _x, _y, dx, dy, lineage, energy_norm, sig = n
            feat[0, base + 0] = float(dx) / _TOM_DELTA_NORM
            feat[0, base + 1] = float(dy) / _TOM_DELTA_NORM
            # lineage one-hot [elder, wanderer]
            if lineage == "elder":
                feat[0, base + 2] = 1.0
            elif lineage == "wanderer":
                feat[0, base + 3] = 1.0
            # signal one-hot [0..7]
            si = int(sig)
            if 0 <= si < _TOM_N_SIGNALS:
                feat[0, base + 4 + si] = 1.0
            # energy_norm в последний слот блока
            feat[0, base + 4 + _TOM_N_SIGNALS] = max(0.0, min(1.0,
                                                                float(energy_norm)))
        return feat

    def _compute_theory_of_mind(self, cid: str) -> None:
        """Один supervised step S2.B: предсказание следующего motor-action
        для focus-соседа (nearest на прошлом тике) и сравнение с реальным
        Δposition. Cross-entropy backward + Adam.

        Если world_cache не привязан — no-op. Если ткани/Adam нет — no-op.
        """
        torch = self._torch
        tissue = self.theory_of_mind.get(cid)
        opt = self.theory_of_mind_opt.get(cid)
        if tissue is None or opt is None:
            return
        wc = self.world_cache
        if wc is None or not getattr(wc, "is_bootstrapped", False):
            return
        try:
            neighbors = wc.tom_neighbors_view(cid, n=_TOM_N_NEIGHBORS)
        except Exception as e:
            logger.debug("tom_neighbors_view %s: %s", cid, e)
            return
        if not neighbors:
            self._tom_prev_focus[cid] = None
            return

        # SFNN S3.6 (14.05.2026): под флагом higher_tissue_sfnn_enabled
        # ткань обучается локальным правилом ΔW (Pedersen/Risi), не Adam'ом.
        # Forward всё равно делаем — иначе SFNN-hooks не словят активации.
        # Adam-шаг (cross_entropy/backward/opt.step) + accuracy EMA (которая
        # без обучения была бы misleading) пропускаются.
        org_obj = self.organisms.get(cid)
        sfnn_on = bool(getattr(getattr(org_obj, "genome", None),
                                "higher_tissue_sfnn_enabled", False))
        if sfnn_on:
            try:
                features = self._build_tom_features(neighbors)
                tissue.eval()
                with torch.no_grad():
                    tissue({"input": features})
            except Exception as e:
                logger.debug("tom sfnn forward %s: %s", cid, e)
            # focus всё равно обновляем — на случай ablation off, чтобы
            # Adam-путь стартовал с известным prev_focus.
            ncid_curr, x_curr, y_curr = neighbors[0][:3]
            self._tom_prev_focus[cid] = (str(ncid_curr), int(x_curr),
                                          int(y_curr))
            return

        size = int(wc.config.size) if wc.config is not None else 256
        prev_focus = self._tom_prev_focus.get(cid)
        # Текущая ближайшая особь (для следующего тика).
        ncid_curr, x_curr, y_curr = neighbors[0][:3]
        # Если на прошлом тике был focus и он жив сейчас — supervised step.
        if prev_focus is not None:
            prev_ncid, prev_x, prev_y = prev_focus
            # Ищем prev_ncid в текущих соседях, если есть — берём его (x, y).
            curr_pos = None
            for n in neighbors:
                if n[0] == prev_ncid:
                    curr_pos = (int(n[1]), int(n[2]))
                    break
            if curr_pos is not None:
                target = self._tom_target_action(
                    (int(prev_x), int(prev_y)), curr_pos, size)
                try:
                    import torch.nn.functional as F
                    features = self._build_tom_features(neighbors)
                    tissue.train()
                    with torch.enable_grad():
                        out = tissue({"input": features})["output"]
                        logits = out[0, :_TOM_N_ACTIONS].unsqueeze(0)
                        target_t = torch.tensor([int(target)],
                                                  dtype=torch.long,
                                                  device=self.device)
                        loss = F.cross_entropy(logits, target_t)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                    pred = int(logits.argmax(dim=-1).item())
                    hit = 1.0 if pred == int(target) else 0.0
                    prev_acc = float(self.last_tom_acc.get(cid, 0.0))
                    self.last_tom_acc[cid] = (
                        (1 - _TOM_ACC_EMA_ALPHA) * prev_acc
                        + _TOM_ACC_EMA_ALPHA * hit
                    )
                    self.tom_steps += 1
                except Exception as e:
                    logger.debug("tom train %s: %s", cid, e)
        # Обновить focus для следующего тика — текущий ближайший сосед.
        self._tom_prev_focus[cid] = (str(ncid_curr), int(x_curr), int(y_curr))

    # ── S2.C — language supervised sidecar ───────────────────────────────

    @staticmethod
    def _lang_event_class(event: Optional[dict]) -> int:
        """Маппинг event-dict → класс ∈ {ate, damage, killed, idle}.

        Приоритет (взаимоисключаем): killed > damage > ate > idle.
        Хищники чаще killed, травоядные — ate; damage — редкое значимое
        событие (укусили). idle = всё прочее (передвижение/спокойствие)
        — этот класс пропускаем при обучении (class imbalance).
        """
        if not event:
            return _LANG_EVT_IDLE
        try:
            killed = bool(event.get("killed", False))
            if killed:
                return _LANG_EVT_KILLED
            damage = float(event.get("damage_taken", 0.0) or 0.0)
            if damage > 0.0:
                return _LANG_EVT_DAMAGE
            ate = bool(event.get("ate", False))
            if ate:
                return _LANG_EVT_ATE
        except (TypeError, ValueError):
            return _LANG_EVT_IDLE
        return _LANG_EVT_IDLE

    def _build_lang_features(self, cid: str):
        """Собрать [1, 16] tensor контекста сигналов для cid.

        own_signal_one_hot[8] + neighbor_signal_counts/k[8]. Если world_cache
        не привязан или нет данных — возвращаем None (no-op).
        """
        torch = self._torch
        wc = self.world_cache
        if wc is None or not getattr(wc, "is_bootstrapped", False):
            return None
        feat = torch.zeros(1, _LANG_DATA_DIM, dtype=torch.float32,
                            device=self.device)
        # own signal (если known).
        own_tom = wc.creature_tom.get(cid)
        if own_tom is not None:
            try:
                own_sig = int(own_tom[3])
            except (TypeError, ValueError, IndexError):
                own_sig = -1
            if 0 <= own_sig < _LANG_N_SIGNALS:
                feat[0, own_sig] = 1.0
        # neighbor signal counts по K ближайшим.
        try:
            neighbors = wc.tom_neighbors_view(cid, n=_LANG_K_NEIGHBORS)
        except Exception:
            neighbors = []
        if neighbors:
            denom = float(len(neighbors))
            for n in neighbors:
                try:
                    sig = int(n[7])
                except (TypeError, ValueError, IndexError):
                    continue
                if 0 <= sig < _LANG_N_SIGNALS:
                    feat[0, _LANG_N_SIGNALS + sig] += 1.0 / denom
        return feat

    def _compute_language(self, cid: str, event: Optional[dict]) -> None:
        """Один supervised step S2.C: decode prev_context → текущее событие.

        Если на прошлом тике был сохранён контекст и текущий event != idle,
        делаем cross_entropy step на 4 классах (ate/damage/killed). Это
        учит ткань ассоциировать чужие сигналы с собственными исходами.

        Затем обновляем prev_context = свежим контекстом текущего тика —
        для использования на следующем supervised шаге.
        """
        torch = self._torch
        tissue = self.language.get(cid)
        opt = self.language_opt.get(cid)
        if tissue is None or opt is None:
            return

        # SFNN S3.7 (14.05.2026): под флагом higher_tissue_sfnn_enabled
        # ткань обучается локальным правилом ΔW. Forward выполняем по
        # свежему контексту (текущие сигналы) — иначе SFNN-hooks не словят
        # активации. Adam-шаг (cross_entropy/backward/opt.step) + accuracy
        # EMA пропускаются. prev_context всё равно обновляем — на случай
        # ablation off позже Adam-путь стартует с известного контекста.
        org_obj = self.organisms.get(cid)
        sfnn_on = bool(getattr(getattr(org_obj, "genome", None),
                                "higher_tissue_sfnn_enabled", False))
        if sfnn_on:
            try:
                curr_features = self._build_lang_features(cid)
                if curr_features is not None:
                    tissue.eval()
                    with torch.no_grad():
                        tissue({"input": curr_features})
                    self._lang_prev_context[cid] = curr_features.detach()
            except Exception as e:
                logger.debug("lang sfnn forward %s: %s", cid, e)
            return

        prev_context = self._lang_prev_context.get(cid)
        event_class = self._lang_event_class(event)
        # Supervised step: prev_context + non-idle event → cross_entropy на 3
        # классах (ate/damage/killed). idle → skip, чтобы не схлопнуть policy
        # на доминанту (idle ~95% тиков).
        if prev_context is not None and event_class != _LANG_EVT_IDLE:
            try:
                import torch.nn.functional as F
                tissue.train()
                with torch.enable_grad():
                    out = tissue({"input": prev_context})["output"]
                    logits = out[0, :_LANG_N_CLASSES].unsqueeze(0)
                    target_t = torch.tensor([int(event_class)],
                                              dtype=torch.long,
                                              device=self.device)
                    loss = F.cross_entropy(logits, target_t)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                pred = int(logits.argmax(dim=-1).item())
                hit = 1.0 if pred == int(event_class) else 0.0
                prev_acc = float(self.last_lang_acc.get(cid, 0.0))
                self.last_lang_acc[cid] = (
                    (1 - _LANG_ACC_EMA_ALPHA) * prev_acc
                    + _LANG_ACC_EMA_ALPHA * hit
                )
                self.lang_steps += 1
            except Exception as e:
                logger.debug("lang train %s: %s", cid, e)
        # Обновить prev_context: свежий слепок сигналов для следующего шага.
        try:
            curr_features = self._build_lang_features(cid)
            if curr_features is not None:
                self._lang_prev_context[cid] = curr_features.detach()
        except Exception as e:
            logger.debug("lang build features %s: %s", cid, e)

    # ── motor_policy forward (SFNN S4) ───────────────────────────────────

    _NEWBORN_INSTINCT_DECAY = 500.0  # тиков до затухания (server phase_a.py:749)
    _CARRIED_FOOD_CAP = 5            # F(5) (world.py:735, carried_food cap)

    def _apply_newborn_instinct(self, cid: str, logits, world_tick: int,
                                on_flora: bool,
                                carried_food: "Optional[int]" = None) -> None:
        """Newborn-инстинкт GATHER/EAT (порт server phase_a.py:748-755).

        age = world_tick - birth_tick; instinct = max(0, 1 - age/500) — линейно
        затухает за 500 тиков. Пока instinct>0:
          GATHER(13) += 2*instinct — если на флоре и carried_food<5;
          EAT(14)    += 2*instinct — если carried_food>0.
        on_flora + carried_food — P40 authoritative (9f8d99d), убирает desync;
        carried_food=None → fallback на client-зеркало (переходный период).
        Только client-рождённые (есть birth_tick); остальные → no-op."""
        # Single-organism pivot (ТЗ e3cc81b §1): newborn-scaffold — колониальная
        # механика (Адам не новорождённый, scaffold GATHER/EAT не нужен). Под
        # флагом no-op. Код сохранён для Зоопарка Эпохи 2.
        if self._single_organism:
            return
        bt = self._birth_tick.get(cid)
        if bt is None:
            return
        age = max(0, int(world_tick) - int(bt))
        instinct = 1.0 - age / self._NEWBORN_INSTINCT_DECAY
        if instinct <= 0.0:
            self._birth_tick.pop(cid, None)
            self._carried_food.pop(cid, None)
            return
        carried = (int(carried_food) if carried_food is not None
                   else int(self._carried_food.get(cid, 0)))
        if on_flora and carried < self._CARRIED_FOOD_CAP:
            logits[13] += 2.0 * instinct   # GATHER когда есть что собрать
        if carried > 0:
            logits[14] += 2.0 * instinct   # EAT когда есть что есть

    def _update_bias_curriculum(self, world_tick: int) -> None:
        """Curriculum bias_scale — кроссфейд shaping↔motor (порт server
        механизма loop.py:603 own_contribution; СИГНАЛ адаптирован под наш
        контекст). Каждые 1000 world-тиков.

        Сервер ведёт bias по населению (n_alive/target) — там население⟹здоровье
        (обученные мозги кормят). У НАС колония закреплена elite-restore на cap
        ПРИ net<0 → population-ratio=1.0 всегда → bias декеил до 0 → untrained
        motor на полную → голод (подтверждено монитором). Поэтому ведём bias по
        ПРЯМОМУ признаку self-sustaining — энергобалансу (income/cost):
          health < 1.0 (не кормит себя) → bias += 0.1 → shaping (флора-навигация)
                                          доминирует, motor подавлен;
          health >= 1.0 (self-sustaining) → bias -= 0.05 → отпускаем motor.
        Механизм (own_contribution=max(0,1-bias)×motor_delta) — серверный."""
        # Single-organism pivot (ТЗ e3cc81b §1): популяционный annealing
        # (по ratio населения/энергобалансу колонии) — колониальный. Под флагом
        # не ведём: bias_scale заморожен на 0 (автономный мотор) в
        # set_single_organism. Curriculum-усложнение переезжает на адаптивный
        # Storyteller (§6, зона Хьюберта).
        if self._single_organism:
            return
        if int(world_tick) - self._bias_last_update_tick < 1000:
            return
        self._bias_last_update_tick = int(world_tick)
        health = (float(self._last_window.get("ratio", 0.0))
                  if self._last_window else 0.0)
        old = self._bias_scale
        if health < 1.0:
            self._bias_scale = min(1.0, self._bias_scale + 0.1)
        else:
            self._bias_scale = max(0.0, self._bias_scale - 0.05)
        if abs(self._bias_scale - old) > 0.001:
            n_alive = sum(1 for c in self.organisms if c not in self._dead_cids)
            logger.info(
                "bias_scale curriculum %.2f → %.2f (health_ratio=%.2f alive=%d "
                "own_contrib=%.2f tick=%d)", old, self._bias_scale, health,
                n_alive, max(0.0, 1.0 - self._bias_scale), int(world_tick))

    def _apply_bootstrap_pending(self, world_tick: int) -> None:
        """Bootstrap (Фрай): омолодить restored-особей — birth_tick=now →
        инстинкт активен 500т → учатся есть тем же механизмом, что newborn.

        Делается в handle_tick (а не в restore), т.к. restore не знает текущий
        world_tick. Применяется только к ещё живым зарегистрированным; на
        каждом restore/elite-recovery заново (self-decay: пока колония не
        self-sustaining). Счётчик _n_bootstrap_rejuv — для verify ratio."""
        if not self._bootstrap_pending:
            return
        for cid in list(self._bootstrap_pending):
            if cid in self.organisms and cid not in self._dead_cids:
                self._birth_tick[cid] = int(world_tick)
                self._carried_food.setdefault(cid, 0)
                self._n_bootstrap_rejuv += 1
        self._bootstrap_pending.clear()

    def _update_carried_food_mirror(self, cid: str, action: int,
                                    on_flora: bool) -> None:
        """Клиентское зеркало carried_food (mirror server world.py:3048-3141).

        P40 carried_food клиенту не шлёт → реплицируем серверные правила резолва
        action: GATHER на флоре +1 (cap F(5)=5), EAT/SHARE_FOOD -1. Оценочно
        (редкий рассинхрон при гонке за флору), но bounded 0..5 и self-
        correcting — для затухающего scaffold достаточно. Нужно только инстинкту."""
        cur = int(self._carried_food.get(cid, 0))
        if action == 13 and on_flora and cur < self._CARRIED_FOOD_CAP:  # GATHER
            self._carried_food[cid] = cur + 1
        elif action in (14, 15) and cur > 0:                            # EAT/SHARE_FOOD
            self._carried_food[cid] = cur - 1

    def _shape_action_logits(self, logits, obs_arr, diet: float,
                             energy_ratio: float, nearest_flora=None,
                             recent_yield: bool = False,
                             on_flora: bool = False,
                             just_hit: bool = False,
                             flora_abandon_dir: "Optional[int]" = None) -> None:
        """Phase 4 Body Migration (01.06.2026): контекстный шейпинг логитов
        действия — порт server `_decide_action` (phase_a.py:668-765). Без него
        логиты org.forward однородны → ActionSelector коллапсирует в move (0-3),
        зодчие не едят/не охотятся/не бегут → голод + мёртвое поведение.

        Градиентные bias'ы (флора/prey/predator из obs-слотов) направляют
        движение; структурные φ-штрафы + контекстные boost'ы формируют action.
        motor_policy (SFNN) доучивается ПОВЕРХ. EAT/GATHER не трогаем —
        passive_eating кормит ambient на флоре.

        Конвенция Action (world.py): N=0,S=1,E=2,W=3,STAY=4,ATTACK=5,SIGNAL_FOOD
        =6,SIGNAL_DANGER=7,SHARE=8,REPRODUCE=9,FLEE=10,DIG=11,BUILD=12,GATHER=13,
        EAT=14,SHARE_FOOD=15. obs: [33/34]=флора-град NS/EW, [56-58]=prey NS/EW/
        prox, [59-61]=predator. logits: torch[16]-вью, мутируется in-place.
        """
        try:
            n = len(obs_arr)
            BS = float(self._bias_scale)  # curriculum (порт server _bias_scale)
            # DIRECTION-сила (Фрай 03.06): под single_organism развязана от
            # bias_scale (который =0 зануляло прекондишн) → инстинкт-приор всегда on,
            # умеренная сила. Колония — остаётся ×BS (curriculum crossfade сохранён).
            DS = (float(self._instinct_dir_strength)
                  if self._single_organism else BS)
            PHI = 1.6180339887
            # Нормализация направления (Фрай 03.06, scaffold-restore): под
            # single_organism obs-градиент МАЛ (~0.1, A/B-диагностик) → fw*g мизер,
            # пик не пробивается. Единичный вектор направления × fw → сильный
            # ОРИЕНТИР при УМЕРЕННОМ fw (не эскалация силы). Магнитуда «насколько
            # близко» теряется — но для ориентации важно «куда», не «как слабо видно».
            _norm = self._single_organism
            def _unit(a, b):
                m = (a * a + b * b) ** 0.5
                return (a / m, b / m) if (m > 1e-6 and _norm) else (a, b)
            # nearest_flora точный нав-ПОЛ (Хьюберт/Фрай 05.06): {dr,dc,dist,kind}|None,
            # raw nearest-flora delta (Elder argmin). Заменяет грубый smell-градиент
            # когда есть: dist==0 → GATHER/EAT буст (на тайле); dist>0 → точный шаг
            # N/S/E/W по dr/dc ×min(1/dist,1) (ближе→резче); None → smell-fallback.
            # Прайор = эвристика Старших как ПОЛ (точная нав + on-flora gather), мотор
            # учит высшее поверх. obs[62/63]=dist/kind для контекст-обучения мотора.
            # Конвенция: dr>0=флора южнее→SOUTH(1), dr<0→NORTH(0), dc>0→EAST(2), dc<0→WEST(3).
            _nf = nearest_flora
            if flora_abandon_dir is not None:
                # ARRIVAL ABANDON (Фрай 07.06 go): nearest_flora недостижима (стак
                # детектнут в caller) → НЕ коммить к ней (и НЕ smell-fallback к той
                # же цели) → обход препятствия перпендикулярно (flora_abandon_dir).
                # Сильный шаг (2φ·DS, как arrival-commit) + не стой. pos сменится →
                # nf пересчёт → стак-сброс → arrival-commit к достижимой флоре.
                logits[int(flora_abandon_dir)] += 2.0 * PHI * DS
                logits[4] -= PHI * DS          # не стой — двигайся в обход
                _nf = None                     # smell-fallback ниже ОТКЛ (см. гейт)
            elif _nf is not None:
                try:
                    _dr = float(_nf.get("dr", 0.0))
                    _dc = float(_nf.get("dc", 0.0))
                    _dist = float(_nf.get("dist", 0.0))
                    if recent_yield:
                        # «Доедай, не убегай» (Фрай 06.06): текущий тайл ещё
                        # отдаёт (ate/delta_energy в прошлый тик) → стой и собирай
                        # на месте, nav-толчок к ближайшей флоре НЕ эмитим. φ-маг:
                        # STAY+GATHER = PHI, EAT = 1/PHI. Отдача прекратится →
                        # yield-память истечёт → nav возобновится сам. FLEE от
                        # хищника (ниже, ×3) перебьёт STAY — бежим даже на еде.
                        logits[4] += PHI * DS            # STAY (no-move)
                        logits[13] += PHI * DS           # GATHER (доедай на месте)
                        logits[14] += (1.0 / PHI) * DS   # EAT
                    elif _dist <= 0.0:
                        logits[13] += 2.0 * DS    # GATHER (на флоре, пол Старших)
                        logits[14] += 1.0 * DS    # EAT
                    elif _dist <= 1.0:
                        # ARRIVAL COMMIT (Фрай 07.06, predator_defense-зеркало):
                        # флора ВПЛОТНУЮ (dist≈1, cardinal-смежная) → ДОМИНАНТНЫЙ
                        # финальный ШАГ НА тайл + анти-флип (гаси обратное напр.) +
                        # не стой. Корень thrash: на dist=1 обычный _w УЖЕ макс
                        # (min(1/dist,1)=1), но мотор (norm~1.6, flip 0.6) перебивал
                        # argmax ~60% → осцилляция в 1 тайле, onf_rate=0, income=0.
                        # Фикс — commit ×2φ (как ATTACK-контратака just_hit): сильный
                        # prior, мотор дотачивает; анти-флип гасит реверс. Флора-
                        # специфично (nearest_flora 62/63). С §4 predator (по obs[61],
                        # ниже) НЕ конфликтует — хищник вплотную перебьёт (×0.5 move +
                        # ATTACK/FLEE). Встанет НА тайл → след.тик dist=0 → GATHER/EAT.
                        _cw = 2.0 * PHI * DS          # commit-сила шага
                        _af = PHI * DS                # анти-флип обратного напр.
                        if _dr < 0:
                            logits[0] += _cw; logits[1] -= _af   # NORTH, гаси SOUTH
                        elif _dr > 0:
                            logits[1] += _cw; logits[0] -= _af   # SOUTH, гаси NORTH
                        if _dc > 0:
                            logits[2] += _cw; logits[3] -= _af   # EAST, гаси WEST
                        elif _dc < 0:
                            logits[3] += _cw; logits[2] -= _af   # WEST, гаси EAST
                        logits[4] -= _af              # не стой — делай финальный шаг
                    else:
                        _w = (4.0 - 2.0 * diet) * DS * min(1.0 / _dist, 1.0)
                        if _dr < 0:
                            logits[0] += _w       # NORTH
                        elif _dr > 0:
                            logits[1] += _w       # SOUTH
                        if _dc > 0:
                            logits[2] += _w       # EAST
                        elif _dc < 0:
                            logits[3] += _w       # WEST
                except Exception:
                    _nf = None                    # битое поле → smell-fallback
            if _nf is None and flora_abandon_dir is None:
                # Smell-градиент fallback (иди к еде, центр-масс). diet→0 сильнее.
                # ОТКЛ при abandon (_nf=None из-за абандона) — иначе smell к ТОЙ ЖЕ
                # недостижимой флоре-кластеру вернёт стак; обход уже задан выше.
                g_ns = float(obs_arr[33]) if n > 34 else 0.0
                g_ew = float(obs_arr[34]) if n > 34 else 0.0
                g_ns, g_ew = _unit(g_ns, g_ew)
                fw = (4.0 - 2.0 * diet) * DS
                logits[0] -= fw * g_ns; logits[1] += fw * g_ns
                logits[2] += fw * g_ew; logits[3] -= fw * g_ew
            # Prey-градиент (охота). Карнивор (diet→1) сильнее.
            p_ns = float(obs_arr[56]) if n > 58 else 0.0
            p_ew = float(obs_arr[57]) if n > 58 else 0.0
            p_prox = float(obs_arr[58]) if n > 58 else 0.0
            p_ns, p_ew = _unit(p_ns, p_ew)
            pw = (2.0 + 4.0 * diet) * DS * min(p_prox + 0.1, 1.0)
            logits[0] -= pw * p_ns; logits[1] += pw * p_ns
            logits[2] += pw * p_ew; logits[3] -= pw * p_ew
            # Predator-градиент (беги ПРОТИВ градиента).
            d_ns = float(obs_arr[59]) if n > 61 else 0.0
            d_ew = float(obs_arr[60]) if n > 61 else 0.0
            d_prox = float(obs_arr[61]) if n > 61 else 0.0
            d_ns, d_ew = _unit(d_ns, d_ew)
            # Predator flee-MOVE (направление ПРОТИВ градиента) — ТОЛЬКО при
            # ПРИБЛИЖЕНИИ (на контакте = стоять и бить, §4 predator_defense.md).
            # Ослаблен до PHI*DS (был 4.0) — даёт направление, но FLEE-action
            # (ниже, рывок 3 тайла, растит move_speed) должен доминировать.
            if 0.05 < d_prox < 0.8:
                pf = PHI * DS * min(d_prox, 1.0)
                logits[0] += pf * d_ns; logits[1] -= pf * d_ns
                logits[2] -= pf * d_ew; logits[3] += pf * d_ew
            # §4 PREDATOR DEFENSE (predator_defense.md §11, Фрай 07.06): рефлекс-bias
            # ACTION по obs[61], DS-scaled (активен под single_organism, где BS=0
            # зануляет старые BS-бусты → мотор машет невпопад). Цель СТРОГО хищник.
            # ТРИГГЕР БОЯ = АТАКА (damage_taken>0), НЕ КОНТАКТ (Фрай §11 durable):
            # старый mere-contact ATTACK (d_prox≥0.85) бил по ПРИСУТСТВИЮ → spam на
            # predator-транзиенты (dmg=0, pred_ticks=0, attack=144-185) рвал forage
            # И возвращал berserk-on-presence при climb pressure. Теперь: реально
            # ударили → контратака; контакт БЕЗ урона (пассив/транзиент) → НЕ бей,
            # пасись/настороже; приближается (не контакт) → уйди до удара.
            if just_hit:
                # damage_taken>0 = хищник ТОЧНО ударил ЭТОТ тик → бей в ответ СИЛЬНО,
                # гаси бегство + move-прочь (стой в radius=1, добей).
                logits[5] += 2.0 * PHI * DS      # сильный ATTACK
                logits[10] -= PHI * DS           # НЕ беги — контратакуй
                logits[0] *= 0.5; logits[1] *= 0.5
                logits[2] *= 0.5; logits[3] *= 0.5
            elif 0.15 < d_prox < 0.85:   # ПРИБЛИЖАЕТСЯ (не контакт) → создай дистанцию
                logits[10] += PHI * PHI * DS * min(d_prox, 1.0)  # FLEE рывок (уйди до удара)
            # Контакт d_prox≥0.85 БЕЗ just_hit → НИ одна ветка: пасись/настороже.
            # Подавляем ATTACK ВСЕГДА когда нас НЕ ударили (мотор не должен спамить
            # бой на присутствие/в воздух) — единственный буст ATTACK = just_hit выше.
            if not just_hit:
                logits[5] -= PHI * DS
            # Структурные φ-штрафы (постоянные).
            logits[4] -= 1.0                 # STAY
            logits[6] -= 1.0 / (PHI * PHI)   # SIGNAL_FOOD
            logits[7] -= 1.0 / (PHI * PHI)   # SIGNAL_DANGER
            logits[8] -= 1.0 / PHI           # SHARE
            logits[9] -= 1.0                 # REPRODUCE (через mate_pairs, не action)
            logits[11] -= 1.0                # DIG
            logits[12] -= 1.0 / (PHI * PHI)  # BUILD
            # Контекстные. ATTACK: boost ТОЛЬКО когда добыча ДОСТИЖИМА (prox>0.3),
            # не просто видна (prox>0.1) — иначе фуражёры впустую лезут в драку
            # (Хьюберт: ATTACK доминировал 36%). prox 0.1..0.3: prey виден но
            # далеко → нейтрально, prey-градиент сам подведёт. Магнитуда 2.0→1.5.
            # Prey-ATTACK (охота) — ТОЛЬКО хищник/всеядный (diet>0.5). Травоядному
            # Адаму ВЫКЛ (§4 predator_defense.md): удар-по-добыче = удары в воздух
            # (ATTACK_REACH подтвердил: atk_contact~0). ATTACK теперь только на хищника.
            if diet > 0.5:
                if p_prox <= 0.1:
                    logits[5] -= 1.0 * BS        # добычи нет — ATTACK бессмысленна
                elif p_prox > 0.3:
                    logits[5] += 1.5 * BS        # добыча достижима
                    if diet > 0.7:
                        logits[5] += 3.0 * diet * min(p_prox, 1.0)  # карнивор-охота
            logits[10] -= 0.3 * BS           # FLEE базовый штраф
            if d_prox > 0.15:
                logits[10] += 3.0 * BS * min(d_prox, 1.0)  # FLEE у хищника
            if energy_ratio < 0.3:
                logits[12] += 1.0 * BS       # BUILD при низкой энергии (оборона)
            # ИЗОЛИРУЮЩИЙ ТЕСТ override-мотора (Фрай 06.06): на on-flora тиках
            # STAY выигрывает БЕЗУСЛОВНО — паркуем (перебивает структурный −1.0
            # и все градиенты, включая хищника: тест короткий, §3/halo держат).
            # Обычно ставят с motor_voice=0. Если паркуется + net flip → виноват
            # override мотора; не паркуется → прайор сам не держит hold на dist=0.
            if self._motor_park_test > 0.0 and on_flora:
                try:
                    _mx = float(logits[:16].max().item())
                except Exception:
                    _mx = 5.0
                logits[4] = _mx + 5.0        # STAY доминирует безусловно
            # STAY-исполнение контроль (Фрай 06.06): БЕЗУСЛОВНЫЙ STAY каждый тик
            # (не только on-flora) — чистый тест honored-ли STAY на P40.
            if self._motor_stay_force > 0.0:
                try:
                    _mx2 = float(logits[:16].max().item())
                except Exception:
                    _mx2 = 5.0
                logits[4] = _mx2 + 5.0
        except Exception as e:
            logger.debug("shape_action_logits failed: %s", e)

    def _motor_forward(self, cid: str, obs_tensor):
        """Forward motor_policy → tanh-delta [-1, 1]^16. Hooks захватывают
        pre/post активации для SFNN update_step на следующем тике. Возвращает
        torch.Tensor [16] или None если ткани нет.

        S5 (15.05.2026): pre-tanh делится на T из SFNNRule (эволюционирует).
        T>1 расширяет линейную зону tanh при насыщенных весах
        (motor_delta_norm≈3.97). T=1.0 (дефолт) — no-op.
        """
        torch = self._torch
        tissue = self.motor_policy.get(cid)
        if tissue is None:
            return None
        try:
            tissue.eval()
            with torch.no_grad():
                out = tissue({"input": obs_tensor.detach()})["output"]
                rule = self.motor_sfnn_rule.get(cid)
                # De-saturation (Фрай 04.06): motor_temp>0 override'ит T ткани —
                # делит pre-tanh, возвращая залипшую голову в отзывчивый диапазон.
                if self._motor_temp > 0.0:
                    T = self._motor_temp
                else:
                    T = rule.temperature if rule is not None else 1.0
                delta = (torch.tanh(out[0, :_MOTOR_POLICY_N_ACTIONS] / T)
                          * _MOTOR_POLICY_SCALE)
            self.last_motor_delta[cid] = delta.detach().cpu()
            return delta
        except Exception as e:
            logger.debug("motor_policy forward %s: %s", cid, e)
            return None

    # Маппинг суффикса имени модуля → имя типа синапса в SFNNRule.
    # Привязан к структуре Tissue 21/3/1 (см. core/tissue.py).
    _MOTOR_SFNN_SYNAPSE_MAP = (
        ("input_proj", "input_proj"),
        ("attn.qkv", "attn_qkv"),
        ("attn.proj", "attn_proj"),
        ("mlp.fc1", "mlp_fc1"),
        ("mlp.fc2", "mlp_fc2"),
        ("output_proj", "output_proj"),
    )
    # Защитный clip ΔW по infinity-норме (см. ТЗ S1).
    _MOTOR_SFNN_DW_CLIP = 0.01
    # SFNN S1.2c: floor для row-norm деления (защита от 0).
    _MOTOR_SFNN_RENORM_EPS = 1e-8

    def _snapshot_row_norms(self, tissue) -> dict:
        """SFNN S1.2c (14.05.2026): snapshot baseline L2-норм строк 6 weight
        matrices Tissue 21/3/1. Используется в update_step как target для
        row-wise renormalization после применения ΔW — защита от
        tanh-saturation и долгосрочного дрейфа весов.

        Возвращает {synapse_type → Tensor[rows]} с detached clone (W меняется,
        baseline остаётся фиксирован — это и есть точка отсчёта).
        """
        torch = self._torch
        out: dict = {}
        if tissue is None:
            return out
        params = dict(tissue.named_parameters())
        for module_name, synapse_type in self._MOTOR_SFNN_SYNAPSE_MAP:
            for n in params:
                if n.endswith(module_name + ".weight"):
                    W = params[n]
                    if W.dim() == 2:
                        out[synapse_type] = W.data.norm(dim=1).detach().clone()
                    break
        return out

    def _register_motor_sfnn_hooks(self, cid: str, tissue) -> None:
        """SFNN S1.2b: forward-hooks на 6 Linear motor_policy Tissue для
        захвата pre (input) и post (output) активаций.

        Hooks записывают в `self.motor_sfnn_acts[cid][synapse_type]` пару
        (pre, post), усреднённую по всем batch/sequence-измерениям. На
        следующем тике `_motor_sfnn_update_step` использует эти активации
        для применения локального правила пластичности.

        Регистрируется при add_creature; снимается при remove_creature.
        Hooks дёшевы (≪1us на каждый) и всегда активны — даже при
        sfnn_enabled=False активации просто перезаписываются и не читаются.
        """
        self.motor_sfnn_acts[cid] = {}
        handles: list = []
        for module_name, synapse_type in self._MOTOR_SFNN_SYNAPSE_MAP:
            mod = None
            for n, m in tissue.named_modules():
                if n.endswith(module_name):
                    mod = m
                    break
            if mod is None:
                logger.debug("motor_sfnn hook: module %s not found for %s",
                              module_name, cid)
                continue

            def _make_hook(_cid: str, _syn: str):
                def hook(_module, _input, _output):
                    try:
                        pre_t = _input[0] if isinstance(_input, tuple) else _input
                        # Усредняем по всем измерениям кроме последнего.
                        pre = pre_t.detach().reshape(-1, pre_t.shape[-1]).mean(dim=0)
                        post = _output.detach().reshape(-1, _output.shape[-1]).mean(dim=0)
                        acts = self.motor_sfnn_acts.get(_cid)
                        if acts is not None:
                            acts[_syn] = (pre, post)
                    except Exception:
                        pass
                return hook

            try:
                h = mod.register_forward_hook(_make_hook(cid, synapse_type))
                handles.append(h)
            except Exception as e:
                logger.debug("motor_sfnn hook %s register: %s", synapse_type, e)
        self.motor_sfnn_hook_handles[cid] = handles

    def _motor_sfnn_update_step(self, cid: str, events_per_cid,
                                  intrinsic_now: float) -> None:
        """SFNN S1.2 (14.05.2026): локальное правило пластичности для
        motor_policy под флагом `genome.sfnn_enabled=True`.

        SFNN S4 (14.05.2026) — единственный путь обучения motor. Для каждого
        из 6 типов синапсов Tissue 21/3/1 motor_policy применяется:

            hebb_A = outer(post, pre) − post²·W       # Oja-стабилизация
            ΔW = η · (1 + r_imm_total) · (A·hebb_A
                                            + B·post + C·preᵀ + D)
            ΔW ← clip(ΔW, ±0.01)            # защита от взрыва
            W  ← W + ΔW
            W  ← W·(target_row_norm / cur_row_norm)  # safety renorm

        SFNN S1.2c (14.05.2026): Oja подсложил вычитающий член `−post²·W`
        в A-Hebbian-слагаемое. Без него чистое `outer(post, pre)`
        сонаправляло строки W с avg(pre) — output[i] = ‖W[i]‖·‖pre‖·cos
        → tanh saturated. Oja-rule (Oja 1982) фиксирует и направление, и
        магнитуду: ‖W_i‖ → ‖input‖. L2-row-renorm к baseline оставлен
        страховкой. Наблюдалось 14.05 на cheef: motor_delta_norm рос
        3.72 → 3.97 за 2100 шагов (sqrt(16)=4.0 max) до Oja.

        pre/post захватываются forward-hooks (`_register_motor_sfnn_hooks`)
        при предыдущем `_motor_forward`. Если активаций нет (первый тик
        или ткани нет) — обновление пропускается.

        Reward-модуляция: r_imm (immediate reward от событий) +
        intrinsic_now (surprise predictor).
        """
        if events_per_cid is None:
            return
        event = events_per_cid.get(cid)
        if event is None:
            return
        rule = self.motor_sfnn_rule.get(cid)
        if rule is None:
            return
        tissue = self.motor_policy.get(cid)
        if tissue is None:
            return
        acts = self.motor_sfnn_acts.get(cid)
        if not acts:
            return
        torch = self._torch
        row_norms = self.motor_sfnn_row_norms.get(cid)
        try:
            r_imm = self._compute_immediate_reward(event)
            r_imm_total = float(r_imm + intrinsic_now)
            # REINFORCE baseline (Phase 4 #1, 01.06.2026): advantage =
            # reward − бегущее среднее. Выше среднего (eat/kill) → gain>1
            # reinforce; ниже (пустой move/damage) → gain<1 ослабить →
            # политика УЧИТСЯ отличать yield-действия. Без baseline (было)
            # gain=1+r всегда>0 → только reinforce, нет дискриминации.
            b = self._motor_reward_baseline.get(cid, 0.0)
            b = 0.99 * b + 0.01 * r_imm_total      # EMA (τ≈100 тиков)
            self._motor_reward_baseline[cid] = b
            advantage = r_imm_total - b
            # clamp [-2,2] → gain ∈ [-1,3] (ΔW-clip ±0.01 + row-renorm
            # ниже бортуют величину; знак-флип = анти-Hebbian «отучивание»).
            advantage = max(-2.0, min(2.0, advantage))
            reward_gain = 1.0 + advantage
            # MOTOR_LEARN: EMA |advantage| — дискриминирует ли reward (≠0 → доехал
            # и варьирует; ≈0 → плоский reward = (i) или нет discrimination).
            self._motor_adv_ema[cid] = (
                0.98 * self._motor_adv_ema.get(cid, 0.0) + 0.02 * abs(advantage))
            clip_val = self._MOTOR_SFNN_DW_CLIP
            eps = self._MOTOR_SFNN_RENORM_EPS
            _dw_norm_sum = 0.0   # ‖ΔW‖ применённого (до renorm) — policy движется?
            with torch.no_grad():
                # Build name → parameter map для прямого in-place обновления.
                params = dict(tissue.named_parameters())
                for module_name, synapse_type in self._MOTOR_SFNN_SYNAPSE_MAP:
                    if synapse_type not in acts:
                        continue
                    # Найти соответствующий weight tensor.
                    p_name = None
                    for n in params:
                        if n.endswith(module_name + ".weight"):
                            p_name = n
                            break
                    if p_name is None:
                        continue
                    W = params[p_name]
                    pre, post = acts[synapse_type]
                    # Размерности: W (out, in), pre (in,), post (out,).
                    if W.dim() != 2:
                        continue
                    if pre.shape[0] != W.shape[1] or post.shape[0] != W.shape[0]:
                        continue
                    coef = rule.coeffs.get(synapse_type)
                    if coef is None:
                        continue
                    eta = float(coef.eta) * self._motor_lr_scale  # anti-saddle-flip (Фрай 04.06)
                    A = float(coef.A)
                    B = float(coef.B)
                    C = float(coef.C)
                    D = float(coef.D)
                    # Policy-gradient на output_proj (Фрай 04.06, rule-upgrade):
                    # тангенциальный REINFORCE credit к СЭМПЛИРОВАННОМУ действию
                    # вместо correlational Hebbian (radial=1 → залочен на колониальной
                    # политике). ΔW = pg_lr·advantage·outer(g, pre),
                    #   g[j] = own·SCALE·(1−tanh²(post[j]/T))/T·(δ_aj − π[j]).
                    # δ_a−π тангенциален (поднимает выбранное действие, опускает
                    # прочие) → может разучить колонию + выучить forage. Прочие
                    # синапсы и колониальный режим — Hebbian-Oja.
                    # ВАЖНО: output_proj выдаёт 64-dim (DATA_DIM), а действий 16 —
                    # motor_delta берёт post[:16]. PG градиент только на эти 16 строк
                    # output_proj (rows 0:16 = action-логиты, rows 16:64 = 0).
                    _NA = _MOTOR_POLICY_N_ACTIONS
                    _pg_ctx = (self._motor_pg_ctx.get(cid)
                               if self._motor_pg_on > 0.0 else None)
                    _use_pg = (synapse_type == "output_proj" and _pg_ctx is not None
                               and _pg_ctx[1].shape[0] == _NA
                               and post.shape[0] >= _NA)
                    if _use_pg:
                        _a_idx, _pi = _pg_ctx
                        _T = (self._motor_temp if self._motor_temp > 0.0
                              else (rule.temperature if rule is not None else 1.0))
                        _own_pg = (float(self._motor_voice) if self._single_organism
                                   else max(0.0, 1.0 - float(self._bias_scale)))
                        _dtanh = 1.0 - torch.tanh(post[:_NA] / _T) ** 2  # [16]
                        _indic = -_pi.clone()                            # [16]
                        if 0 <= _a_idx < _NA:
                            _indic[_a_idx] += 1.0                  # δ_a − π
                        _g16 = ((_own_pg * _MOTOR_POLICY_SCALE)
                                * (_dtanh / max(_T, 1e-6)) * _indic)  # [16]
                        _g = torch.zeros_like(post)                   # [64]
                        _g[:_NA] = _g16
                        dW = (self._motor_pg_lr * advantage) * torch.outer(_g, pre)
                        self._motor_pg_steps += 1                     # верификация: PG бежит
                    else:
                        # SFNN S1.2c (14.05.2026): Hebbian-A с тремя стабилизаторами
                        # против tanh-saturation: (1) mean-центрирование post/pre;
                        # (2) Oja-вычитающий член (−post_c²·W, Oja 1982); (3) row-L2-
                        # renorm к baseline (ниже). output_proj-specific Oja-scale.
                        post_c = post - post.mean()
                        pre_c = pre - pre.mean()
                        _oja_eff = (self._motor_oja_scale_out
                                    if synapse_type == "output_proj"
                                    else self._motor_oja_scale)
                        hebb_A = (torch.outer(post_c, pre_c)
                                  - (_oja_eff
                                     * post_c.square().unsqueeze(1) * W.data))
                        dW = A * hebb_A
                        if B != 0.0:
                            dW = dW + B * post.unsqueeze(1).expand_as(W)
                        if C != 0.0:
                            dW = dW + C * pre.unsqueeze(0).expand_as(W)
                        if D != 0.0:
                            dW = dW + D
                        dW = dW * (eta * reward_gain)
                    dW.clamp_(-clip_val, clip_val)
                    _dw_norm_sum += float(dW.norm().item())  # MOTOR_LEARN
                    # ΔW-инструментирование output_proj (Фрай 04.06): (a) vs (b).
                    # ДО add_ — W.data ещё пред-апдейтный весовой вектор.
                    if synapse_type == "output_proj":
                        _wn = W.data.norm(dim=1).clamp(min=eps)   # ‖W[i]‖ per row
                        _dwn = dW.norm(dim=1).clamp(min=eps)      # ‖ΔW[i]‖ per row
                        # |cos(ΔW[i], W[i])| = доля ΔW вдоль W (renorm режет именно её)
                        _radial = ((dW * W.data).sum(dim=1).abs() / (_wn * _dwn))
                        _rf = float(_radial.mean().item())
                        self._motor_dw_radial_ema[cid] = (
                            0.95 * self._motor_dw_radial_ema.get(cid, _rf) + 0.05 * _rf)
                        _dwf = dW.reshape(-1)
                        _last = self._motor_dw_last.get(cid)
                        if _last is not None and _last.shape == _dwf.shape:
                            _cos = float(torch.nn.functional.cosine_similarity(
                                _dwf, _last, dim=0).item())
                            self._motor_dw_cos_ema[cid] = (
                                0.95 * self._motor_dw_cos_ema.get(cid, _cos)
                                + 0.05 * _cos)
                        self._motor_dw_last[cid] = _dwf.clone()
                    W.data.add_(dW)
                    # SFNN S1.2c safety net: row-wise L2-renorm к baseline.
                    # Growth-cap (Фрай 03.06): cap=1.0 → пин к target (текущее);
                    # cap>1 → строка растёт до target×cap (заострение возможно),
                    # renorm DOWN только при превышении → анти-взрыв сохранён.
                    if row_norms is not None:
                        target = row_norms.get(synapse_type)
                        if (target is not None
                                and target.shape[0] == W.shape[0]):
                            cur = W.data.norm(dim=1).clamp(min=eps)
                            # output_proj-specific renorm-cap (Фрай: вторично — дать
                            # тангенциальной магнитуде survive на policy-выходе).
                            cap = float(self._motor_renorm_cap_out
                                        if synapse_type == "output_proj"
                                        else self._motor_renorm_growth_cap)
                            if cap <= 1.0:
                                W.data.mul_((target / cur).unsqueeze(1))
                            else:
                                limit = target * cap
                                scale = torch.where(
                                    cur > limit, limit / cur,
                                    torch.ones_like(cur))
                                W.data.mul_(scale.unsqueeze(1))
            self._motor_dw_ema[cid] = (   # MOTOR_LEARN: EMA ‖ΔW‖ применённого
                0.98 * self._motor_dw_ema.get(cid, 0.0) + 0.02 * _dw_norm_sum)
            self.motor_sfnn_steps += 1
        except Exception as e:
            logger.debug("motor_sfnn_update %s: %s", cid, e)

    def _register_higher_tissue_sfnn_hooks(self, tissue_name: str,
                                              cid: str, tissue) -> None:
        """SFNN S3.1: forward-hooks pre/post активаций 6 Linear высшей ткани.

        Структура Tissue 21/3/1 одинакова у motor_policy и всех 7 высших
        тканей — поэтому переиспользуем `_MOTOR_SFNN_SYNAPSE_MAP`.

        Hooks записывают (pre, post) в
        `self.higher_tissue_sfnn_acts[tissue_name][cid][synapse_type]`.
        Цикл апдейта на следующем тике использует эти активации.

        Регистрируется на лету в add_creature только для активных-в-S3.x
        тканей; снимается при remove_creature. Hooks дёшевы и всегда
        включены — даже при higher_tissue_sfnn_enabled=False активации
        перезаписываются, но не читаются.
        """
        if tissue is None:
            return
        self.higher_tissue_sfnn_acts[tissue_name][cid] = {}
        # SFNN S1.2c (14.05.2026): baseline row-norms для renorm после ΔW.
        self.higher_tissue_sfnn_row_norms[tissue_name][cid] = (
            self._snapshot_row_norms(tissue))
        handles: list = []
        for module_name, synapse_type in self._MOTOR_SFNN_SYNAPSE_MAP:
            mod = None
            for n, m in tissue.named_modules():
                if n.endswith(module_name):
                    mod = m
                    break
            if mod is None:
                logger.debug("higher_sfnn hook: module %s not found %s/%s",
                              module_name, tissue_name, cid)
                continue

            def _make_hook(_tissue: str, _cid: str, _syn: str):
                def hook(_module, _input, _output):
                    try:
                        pre_t = (_input[0] if isinstance(_input, tuple)
                                 else _input)
                        pre = pre_t.detach().reshape(
                            -1, pre_t.shape[-1]).mean(dim=0)
                        post = _output.detach().reshape(
                            -1, _output.shape[-1]).mean(dim=0)
                        acts = self.higher_tissue_sfnn_acts[_tissue].get(_cid)
                        if acts is not None:
                            acts[_syn] = (pre, post)
                    except Exception:
                        pass
                return hook

            try:
                h = mod.register_forward_hook(
                    _make_hook(tissue_name, cid, synapse_type))
                handles.append(h)
            except Exception as e:
                logger.debug("higher_sfnn hook %s/%s register: %s",
                              tissue_name, synapse_type, e)
        self.higher_tissue_sfnn_hook_handles[tissue_name][cid] = handles

    def _higher_tissue_sfnn_update_step(self, tissue_name: str,
                                          cid: str,
                                          *,
                                          dopa_td_mult: float = 1.0,
                                          r_imm_eff: float = 0.0,
                                          r_med_eff: float = 0.0,
                                          r_long_eff: float = 0.0) -> None:
        """SFNN S6.5/Z1 — унифицированный apply-step для 7 высших тканей
        (dopamine, imagination, planner, insula, default_mode, theory_of_mind,
        language). Зеркало `_basic_sfnn_update_step` для 10 базовых.

        Формула:
            hebb_A = outer(post_c, pre_c) − post_c²·W       # Oja S1.2c
            e_t   = exp(-1/τ)·e_{t-1} + hebb_A              # eligibility trace
            r_eff = w_imm·r_imm_eff + w_med·r_med_eff + w_long·r_long_eff
            η_eff = η · (1 + td_coupling · (dopa_td_mult − 1))
            ΔW    = clip(η_eff · (1 + r_eff) · (A·e + B·post + C·pre + D), ±0.01)
            W    ← W + ΔW

        Поля τ/R3/TD — части SFNNRule (core/sfnn_rule.py) и эволюционируют
        через σ-мутации. ROLE_DEFAULTS для 7 высших стартуют с R3=0, TD=0
        → r_eff=0, η_eff=η; τ=233 даёт slow-accumulating trace
        (decay ≈ exp(-1/233) ≈ 0.9957). По мере эволюции R3/TD расходятся
        от нуля и канал R3 включается per-tissue.

        r_*_eff подаются вызывающим как baseline-subtracted (heb.update
        внутри обновил baseline EMA). pre/post захватываются forward-hooks
        при предыдущем `_compute_higher_tissues` / `_compute_theory_of_mind` /
        `_compute_language`. Если активаций нет (первый тик / ткани нет /
        hook не зарегистрирован) — обновление пропускается.
        """
        rule_store = self.higher_tissue_sfnn_rule.get(tissue_name)
        if rule_store is None:
            return
        rule = rule_store.get(cid)
        if rule is None:
            return
        tissue_store = getattr(self, tissue_name, None)
        if tissue_store is None:
            return
        tissue = tissue_store.get(cid)
        if tissue is None:
            return
        acts = self.higher_tissue_sfnn_acts.get(tissue_name, {}).get(cid)
        if not acts:
            return
        torch = self._torch
        row_norms = (self.higher_tissue_sfnn_row_norms.get(tissue_name, {})
                       .get(cid))
        try:
            clip_val = self._MOTOR_SFNN_DW_CLIP
            eps = self._MOTOR_SFNN_RENORM_EPS
            tau = max(1.0, float(rule.tau))
            decay = math.exp(-1.0 / tau)
            r_eff = (rule.r_imm_weight * float(r_imm_eff)
                      + rule.r_med_weight * float(r_med_eff)
                      + rule.r_long_weight * float(r_long_eff))
            td_gain = (1.0 + rule.td_coupling
                        * (float(dopa_td_mult) - 1.0))
            reward_gain = 1.0 + r_eff
            trace_per_cid = self.higher_tissue_sfnn_trace[
                tissue_name].setdefault(cid, {})
            with torch.no_grad():
                params = dict(tissue.named_parameters())
                for module_name, synapse_type in self._MOTOR_SFNN_SYNAPSE_MAP:
                    if synapse_type not in acts:
                        continue
                    p_name = None
                    for n in params:
                        if n.endswith(module_name + ".weight"):
                            p_name = n
                            break
                    if p_name is None:
                        continue
                    W = params[p_name]
                    pre, post = acts[synapse_type]
                    if W.dim() != 2:
                        continue
                    if (pre.shape[0] != W.shape[1]
                            or post.shape[0] != W.shape[0]):
                        continue
                    coef = rule.coeffs.get(synapse_type)
                    if coef is None:
                        continue
                    eta = float(coef.eta)
                    A = float(coef.A)
                    B = float(coef.B)
                    C = float(coef.C)
                    D = float(coef.D)
                    # SFNN S1.2c (14.05.2026): три стабилизатора Hebbian-A
                    # (см. подробности в _motor_sfnn_update_step).
                    post_c = post - post.mean()
                    pre_c = pre - pre.mean()
                    hebb_A = (torch.outer(post_c, pre_c)
                              - post_c.square().unsqueeze(1) * W.data)
                    # Eligibility trace per (tissue, cid, synapse).
                    trace = trace_per_cid.get(synapse_type)
                    if trace is None or trace.shape != hebb_A.shape:
                        trace = torch.zeros_like(hebb_A)
                    else:
                        trace.mul_(decay)
                    trace.add_(hebb_A)
                    trace_per_cid[synapse_type] = trace
                    dW = A * trace
                    if B != 0.0:
                        dW = dW + B * post.unsqueeze(1).expand_as(W)
                    if C != 0.0:
                        dW = dW + C * pre.unsqueeze(0).expand_as(W)
                    if D != 0.0:
                        dW = dW + D
                    dW = dW * (eta * td_gain * reward_gain)
                    dW.clamp_(-clip_val, clip_val)
                    W.data.add_(dW)
                    # SFNN S1.2c safety net: row-wise L2-renorm к baseline.
                    if row_norms is not None:
                        target = row_norms.get(synapse_type)
                        if (target is not None
                                and target.shape[0] == W.shape[0]):
                            cur = W.data.norm(dim=1).clamp(min=eps)
                            W.data.mul_((target / cur).unsqueeze(1))
            self.higher_tissue_sfnn_steps[tissue_name] = (
                self.higher_tissue_sfnn_steps.get(tissue_name, 0) + 1)
        except Exception as e:
            logger.debug("higher_sfnn_update %s/%s: %s",
                          tissue_name, cid, e)

    def _basic_sfnn_update_step(self, cid: str, hebbian, *,
                                  dopa_td_mult: float = 1.0,
                                  r_imm_eff: float = 0.0,
                                  r_med_eff: float = 0.0,
                                  r_long_eff: float = 0.0) -> None:
        """SFNN S6.5 (16.05.2026): унифицированный apply-step для 10 базовых
        тканей organism graph (sensory, attention, brain, memory, consciousness,
        communication, motor, manipulator, digestive, immune).

        Формула:
            e_t   = exp(-1/τ)·e_{t-1} + (outer(post_c, pre_c) − post_c²·W_sub)
            r_eff = w_imm·r_imm_eff + w_med·r_med_eff + w_long·r_long_eff
            η_eff = η · (1 + td_coupling · (dopa_td_mult − 1))
            ΔW    = clip(η_eff · (1 + r_eff) · (A·e + B·post + C·pre + D), ±0.01)
            W    ← W + ΔW

        Источник pre/post:
          oja_input    → pre = heb._last_input,  W = cell.input_proj.weight
          reward_output → pre = heb._last_output, W = cell.output_proj.weight[:16]

        Эти роли должны быть переданы как `skip_roles` в heb.update() (S6.6),
        иначе классика обновит W параллельно и эволюционируемое правило тонет
        в Phase 5d-шуме. r_*_eff подаются вызывающим как baseline-subtracted
        (heb внутри уже считает baseline EMA → r_imm_eff = r_imm − baseline_imm).
        """
        if hebbian is None:
            return
        if hebbian._last_input is None or hebbian._last_output is None:
            return
        torch = self._torch
        clip_val = self._MOTOR_SFNN_DW_CLIP
        for info in hebbian._tissue_info:
            role = info['role']
            if role not in _BASIC_SFNN_TISSUES:
                continue
            rule_store = self.basic_tissue_sfnn_rule.get(role)
            if rule_store is None:
                continue
            rule = rule_store.get(cid)
            if rule is None:
                continue
            cell = info['cell']
            algorithm = info['algorithm']
            try:
                if algorithm == 'oja_input':
                    if not hasattr(cell, 'input_proj'):
                        continue
                    synapse_type = 'input_proj'
                    W_full = cell.input_proj.weight.data
                    x = hebbian._last_input  # [1, 64]
                    data_cols = min(x.shape[1], W_full.shape[1])
                    W_sub = W_full[:, :data_cols]
                    pre = x[0, :data_cols]
                    post = W_sub @ pre
                elif algorithm == 'reward_output':
                    if not hasattr(cell, 'output_proj'):
                        continue
                    synapse_type = 'output_proj'
                    W_full = cell.output_proj.weight.data
                    x = hebbian._last_output  # [1, 64]
                    n_embd = W_full.shape[1]
                    n_actions = min(16, W_full.shape[0])
                    W_sub = W_full[:n_actions, :]
                    pre = x[0, :n_embd]
                    post = W_sub @ pre
                else:
                    continue

                coef = rule.coeffs.get(synapse_type)
                if coef is None:
                    continue

                eta = float(coef.eta)
                A = float(coef.A)
                B = float(coef.B)
                C = float(coef.C)
                D = float(coef.D)

                # Oja-style Hebb-A с mean-centering (S1.2c).
                post_c = post - post.mean()
                pre_c = pre - pre.mean()
                hebb_A = (torch.outer(post_c, pre_c)
                            - post_c.square().unsqueeze(1) * W_sub)

                # Eligibility trace, decay = exp(-1/τ).
                tau = max(1.0, float(rule.tau))
                decay = math.exp(-1.0 / tau)
                trace_store = self.basic_tissue_sfnn_trace[role]
                trace = trace_store.get(cid)
                if trace is None or trace.shape != hebb_A.shape:
                    trace = torch.zeros_like(hebb_A)
                else:
                    trace.mul_(decay)
                trace.add_(hebb_A)
                trace_store[cid] = trace

                # r_eff от правила (per-role веса R3 горизонтов).
                r_eff = (rule.r_imm_weight * float(r_imm_eff)
                          + rule.r_med_weight * float(r_med_eff)
                          + rule.r_long_weight * float(r_long_eff))
                # η_eff = η · (1 + td_coupling · (dopa_td_mult − 1)).
                eta_eff = eta * (1.0 + rule.td_coupling
                                  * (float(dopa_td_mult) - 1.0))
                reward_gain = 1.0 + r_eff

                dW = A * trace
                if B != 0.0:
                    dW = dW + B * post.unsqueeze(1).expand_as(W_sub)
                if C != 0.0:
                    dW = dW + C * pre.unsqueeze(0).expand_as(W_sub)
                if D != 0.0:
                    dW = dW + D
                dW = dW * (eta_eff * reward_gain)
                dW.clamp_(-clip_val, clip_val)
                W_sub.add_(dW)
                self.basic_tissue_sfnn_steps[role] = (
                    self.basic_tissue_sfnn_steps.get(role, 0) + 1)
            except Exception as e:
                logger.debug("basic_sfnn_update %s/%s: %s", role, cid, e)

    def _cerebellum_tissue_id(self, cid: str, org) -> "str | None":
        """tissue_id ткани role=='cerebellum' (cache per cid). Вариант A:
        cerebellum — каноничный in-graph узел (бит 18), через который связи
        кормят прогноз. None если ткани нет → predictor остаётся на raw obs."""
        if cid in self._cerebellum_tid:
            return self._cerebellum_tid[cid]
        tid = None
        try:
            for _tid, tissue in (getattr(org, "tissues", {}) or {}).items():
                role = getattr(tissue, "role", None)
                if role is None:
                    spec = getattr(tissue, "spec", None)
                    role = getattr(spec, "role", None) if spec is not None else None
                if role == "cerebellum":
                    tid = _tid
                    break
        except Exception as e:
            logger.debug("cerebellum resolve %s: %s", cid, e)
        self._cerebellum_tid[cid] = tid
        return tid

    def _ensure_cerebellum_hook(self, cid: str, org) -> None:
        """Регистрирует forward-hook на cerebellum-ткани (один раз/cid): ловит
        её выход в self._cerebellum_out[cid] (detach). Tissue — nn.Module,
        forward → {port: tensor}. No-op (но помечает cid), если ткани нет."""
        if cid in self._cerebellum_hooked:
            return
        tid = self._cerebellum_tissue_id(cid, org)
        tissues = getattr(org, "tissues", {}) or {}
        tissue = tissues.get(tid) if tid is not None else None
        if tissue is None or not hasattr(tissue, "register_forward_hook"):
            self._cerebellum_hooked.add(cid)  # нет ткани — больше не пытаемся
            return

        def _hook(_module, _inp, output, _cid=cid):
            try:
                if isinstance(output, dict) and output:
                    t = next(iter(output.values()))
                    self._cerebellum_out[_cid] = t.detach()
            except Exception:
                pass

        try:
            tissue.register_forward_hook(_hook)
            logger.info("cerebellum hook cid=%s tid=%s (variant A: predictor "
                        "input ← cerebellum)", cid, tid)
        except Exception as e:
            logger.warning("cerebellum hook %s: %s", cid, e)
        self._cerebellum_hooked.add(cid)

    # ── Шаг 2 — петля роста связей (in-life topology growth) ──────────────

    def _brain_growth_step(self, cid: str) -> None:
        """Один тик петли роста. idle→(плато intrinsic)→propose+dwell→измерить
        Δloss_ema→keep/backoff. keep на ЗНАЧИМОМ сошедшемся улучшении; backoff
        revert'ит ребро и защищает net/§3. Одна связь в полёте. Gated снаружи
        (_growth_enabled + single_organism)."""
        org = self.organisms.get(cid)
        pred = self.predictor.get(cid)
        if org is None or pred is None:
            return
        if self._cerebellum_tissue_id(cid, org) is None:
            return  # нет cerebellum → расти некуда (драйвер не отзовётся)
        st = self._growth_state.get(cid)
        if st is None:
            if self._growth_saturated:
                return  # связи насыщены — петля на паузе (до фазы тканей §3.2)
            # idle: ОТНОСИТЕЛЬНОЕ плато — intrinsic_ema у своего трейлинг-floor
            # (min за окно) N тиков подряд = learning-progress застрял (весами
            # учиться нечему → менять структуру). intrinsic выше floor (прогресс
            # вернулся) → сброс. Self-referencing, drift-robust (Фрай 08.06).
            intr = float(self.intrinsic_ema.get(cid, 0.0))
            hist = self._growth_intr_hist.get(cid)
            if hist is None:
                hist = deque(maxlen=self._growth_intr_window)
                self._growth_intr_hist[cid] = hist
            hist.append(intr)
            floor = min(hist)
            near_floor = intr <= floor * (1.0 + self._growth_near_floor_margin)
            if near_floor:
                n = self._growth_stagnation_n.get(cid, 0) + 1
                self._growth_stagnation_n[cid] = n
                if n >= self._growth_plateau_ticks:
                    self._propose_growth_edge(cid, org)
            else:
                self._growth_stagnation_n[cid] = 0  # intrinsic поднялся над floor → прогресс
            return
        # dwell: ждём пере-сходимости predictor'а с новым входом (Фрай: достаточно,
        # чтобы не принять шум за сигнал).
        st["ticks"] += 1
        if st["ticks"] < self._growth_dwell_ticks:
            return
        # measure + keep/backoff
        loss_before = float(st["loss_before"])
        loss_after = float(self.loss_ema.get(cid, loss_before))
        delta = loss_before - loss_after  # >0 = прогноз улучшился
        signif = delta >= loss_before * self._growth_min_delta_frac
        # net/§3-защита: §3 не ухудшился (par не вырос) И энергия не обвалилась.
        net_ok = int(self._paralysis_window_n) <= int(st["par_before"])
        bc = self.biochem.get(cid)
        if bc is not None and st.get("energy_before") is not None:
            net_ok = net_ok and (float(getattr(bc, "energy", 0.0))
                                 >= float(st["energy_before"]) * 0.618)
        keep = bool(signif and net_ok)
        gene = st["gene"]
        if keep:
            self._growth_kept += 1
            # KEEP = поиск ещё продуктивен → сбрасываем насыщение.
            self._growth_fallback_streak = 0
            self._growth_saturated = False
            logger.info("brain-growth KEEP cid=%s edge=%s→cerebellum Δloss=%.5f "
                        "(%.1f%%) kept=%d", cid, gene.source_role, delta,
                        100.0 * delta / max(1e-9, loss_before), self._growth_kept)
        else:
            try:
                gene.enabled = False
                from core.tissue_topology import apply_topology_overlay_to_org
                apply_topology_overlay_to_org(org)
            except Exception as e:
                logger.warning("brain-growth backoff %s: %s", cid, e)
            # cooldown: отметить ребро отвергнутым (не ретраить ~cooldown проб).
            self._growth_rejected.setdefault(cid, {})[gene.source_role] = (
                self._growth_propose_count)
            self._growth_reverted += 1
            # SATURATION: backoff из fallback (свежих кандидатов не было) → копим
            # streak. Порог → пауза петли + сигнал готовности к фазе тканей.
            if st.get("from_fallback"):
                self._growth_fallback_streak += 1
                if (not self._growth_saturated and self._growth_fallback_streak
                        >= self._growth_saturation_threshold):
                    self._growth_saturated = True
                    n_active = sum(1 for g in (
                        getattr(org, "tissue_topology_genes", []) or [])
                        if getattr(g, "enabled", False))
                    logger.info("brain-growth SATURATED cid=%s: полезные связи "
                                "исчерпаны (active=%d), петля связей НА ПАУЗЕ — "
                                "готов к фазе тканей §3.2", cid, n_active)
            else:
                self._growth_fallback_streak = 0  # был свежий кандидат → не насыщение
            logger.info("brain-growth BACKOFF cid=%s edge=%s→cerebellum Δloss=%.5f "
                        "signif=%s net_ok=%s reverted=%d", cid, gene.source_role,
                        delta, signif, net_ok, self._growth_reverted)
        self._growth_state.pop(cid, None)
        # После keep/backoff — заново копим стагнацию у floor до следующего propose.
        self._growth_stagnation_n[cid] = 0

    def _propose_growth_edge(self, cid: str, org) -> bool:
        """Предложить ОДНУ связь {роль}→cerebellum (gene-формат + innovation
        tracker как у Хьюберта, target=cerebellum чтобы двигать вход прогноза).
        Записывает state для dwell. True если ребро добавлено."""
        try:
            from core.tissue_topology import (
                TissueConnectionGene, apply_topology_overlay_to_org,
                TissueInnovationTracker)
            from core.connection import ConnectionType
        except Exception as e:
            logger.warning("brain-growth import %s: %s", cid, e)
            return False
        if self._growth_tracker is None:
            self._growth_tracker = TissueInnovationTracker()
        role_to_id: dict = {}
        for _tid, tissue in (getattr(org, "tissues", {}) or {}).items():
            role = getattr(tissue, "role", None)
            if role is None:
                spec = getattr(tissue, "spec", None)
                role = getattr(spec, "role", None) if spec is not None else None
            if role:
                role_to_id[role] = _tid
        if "cerebellum" not in role_to_id:
            return False
        genes = getattr(org, "tissue_topology_genes", None)
        if genes is None:
            genes = []
            org.tissue_topology_genes = genes
        taken = {g.source_role for g in genes
                 if g.enabled and g.target_role == "cerebellum"}
        all_cand = sorted(r for r in role_to_id
                          if r != "cerebellum" and r not in taken)
        if not all_cand:
            return False
        # COOLDOWN: предпочитаем рёбра, не отвергнутые за последние
        # _growth_retry_cooldown проб. Если свежих нет (всё в cooldown) —
        # fallback на all_cand (cooldown = предпочтение, не сталл).
        rej = self._growth_rejected.get(cid, {})
        cd = int(self._growth_retry_cooldown)
        fresh = [r for r in all_cand
                 if r not in rej or (self._growth_propose_count - rej[r]) >= cd]
        used_fallback = not fresh   # свежих кандидатов нет → гоняем оставшиеся
        candidates = fresh if fresh else all_cand
        self._growth_propose_count += 1
        src = self._growth_rng.choice(candidates)
        gene = TissueConnectionGene(
            innovation=self._growth_tracker.reserve(src, "cerebellum"),
            source_role=src, target_role="cerebellum",
            conn_type=ConnectionType.DIRECT, weight=1.0, enabled=True)
        genes.append(gene)
        try:
            apply_topology_overlay_to_org(org)
        except Exception as e:
            logger.warning("brain-growth apply %s: %s", cid, e)
            genes.remove(gene)
            return False
        bc = self.biochem.get(cid)
        self._growth_state[cid] = {
            "gene": gene,
            "loss_before": float(self.loss_ema.get(cid, 0.0)),
            "ticks": 0,
            "par_before": int(self._paralysis_window_n),
            "energy_before": (float(getattr(bc, "energy", 0.0))
                              if bc is not None else None),
            "from_fallback": used_fallback,
        }
        logger.info("brain-growth PROPOSE cid=%s edge=%s→cerebellum loss_before=%.5f",
                    cid, src, self._growth_state[cid]["loss_before"])
        return True

    def reset_growth_saturation(self) -> None:
        """Снять насыщение роста связей. Вызывать при добавлении ТКАНЕЙ (§3.2):
        новые роли = новые {роль}→cerebellum кандидаты → петле снова есть что
        пробовать. Идемпотентно."""
        if self._growth_saturated or self._growth_fallback_streak:
            logger.info("brain-growth saturation reset (новые ткани/роли — "
                        "петля связей возобновлена)")
        self._growth_saturated = False
        self._growth_fallback_streak = 0

    def set_tissue_growth(self, on: bool) -> bool:
        """Канал client_flags: вкл/выкл §10.8 рост ТКАНЕЙ (узлами) МГНОВЕННО.
        on=False (дефолт, kill-switch): петля тканей no-op (узлы не растут).
        on=True: после насыщения связей петля растит ткань-кандидата. Растущие
        ткани уже KEEP'нутые НЕ удаляются при off (живут; off лишь стопит новые
        propose). Реверс мгновенный."""
        self._tissue_growth_enabled = bool(on)
        logger.info("set_tissue_growth: %s", on)
        return self._tissue_growth_enabled

    def _propose_growth_tissue(self, cid: str, org) -> bool:
        """§10.8: предложить ОДНУ новую ТКАНЬ-кандидата (узел). Минт sidecar
        21/3/1 (читает obs как сенсор — без входящего ребра) + вставка в
        org.tissues + проводка {роль}→cerebellum (двигает вход прогноза) +
        re-apply overlay. Записывает state для dwell. True если ткань добавлена."""
        try:
            from core.tissue_topology import (
                TissueConnectionGene, apply_topology_overlay_to_org)
            from core.connection import ConnectionType
        except Exception as e:
            logger.warning("tissue-growth import %s: %s", cid, e)
            return False
        tissues = getattr(org, "tissues", None)
        if tissues is None:
            return False
        # cerebellum должен существовать (цель проводки)
        if self._cerebellum_tissue_id(cid, org) is None:
            return False
        self._tissue_propose_count += 1
        n = self._tissue_propose_count
        role = f"grown{n}"
        tid = f"gt{n}"
        data_dim = int(self._TISSUE_GROWTH_DATA_DIM)
        n_embd = int(self._TISSUE_GROWTH_N_EMBD)
        tissue = self._make_higher_tissue(role, data_dim=data_dim, n_embd=n_embd)
        if tissue is None:
            return False
        # Вставка узла в живой граф. Нет входящего ребра → forward трактует как
        # сенсор → ткань читает obs (workbench.forward). Выход → cerebellum.
        tissues[tid] = tissue
        genes = getattr(org, "tissue_topology_genes", None)
        if genes is None:
            genes = []
            org.tissue_topology_genes = genes
        if self._growth_tracker is None:
            try:
                from core.tissue_topology import TissueInnovationTracker
                self._growth_tracker = TissueInnovationTracker()
            except Exception:
                pass
        innov = (self._growth_tracker.reserve(role, "cerebellum")
                 if self._growth_tracker is not None else n)
        gene = TissueConnectionGene(
            innovation=innov, source_role=role, target_role="cerebellum",
            conn_type=ConnectionType.DIRECT, weight=1.0, enabled=True)
        genes.append(gene)
        try:
            apply_topology_overlay_to_org(org)
        except Exception as e:
            logger.warning("tissue-growth apply %s: %s", cid, e)
            tissues.pop(tid, None)
            genes.remove(gene)
            return False
        bc = self.biochem.get(cid)
        self._tissue_growth_state[cid] = {
            "role": role, "tid": tid, "gene": gene,
            "spec": {"role": role, "data_dim": data_dim, "n_embd": n_embd},
            "loss_before": float(self.loss_ema.get(cid, 0.0)),
            "ticks": 0,
            "par_before": int(self._paralysis_window_n),
            "energy_before": (float(getattr(bc, "energy", 0.0))
                              if bc is not None else None),
        }
        logger.info("brain-growth TISSUE-PROPOSE cid=%s role=%s loss_before=%.5f",
                    cid, role, self._tissue_growth_state[cid]["loss_before"])
        return True

    def _remove_grown_tissue(self, cid: str, org, tid: str, role: str,
                              gene=None) -> None:
        """§10.8 backoff: убрать выросшую ткань-узел + её ребро из живого графа
        (отключаем ген, удаляем ткань из org.tissues, re-apply overlay → forward
        её больше не выполняет)."""
        try:
            genes = getattr(org, "tissue_topology_genes", None) or []
            for g in list(genes):
                if (getattr(g, "source_role", None) == role
                        and getattr(g, "target_role", None) == "cerebellum"):
                    try:
                        genes.remove(g)
                    except ValueError:
                        pass
            tissues = getattr(org, "tissues", None)
            if tissues is not None:
                tissues.pop(tid, None)
            from core.tissue_topology import apply_topology_overlay_to_org
            apply_topology_overlay_to_org(org)
        except Exception as e:
            logger.warning("tissue-growth remove %s role=%s: %s", cid, role, e)

    def _tissue_growth_step(self, cid: str) -> None:
        """§10.8: propose→dwell→Δloss_ema→keep/backoff для ТКАНЕЙ-узлов. Растит
        узел ТОЛЬКО когда связи насыщены (19/19) И intrinsic у floor (тот же
        плато-критерий). keep на ЗНАЧИМОМ Δloss И net/§3 ok; иначе backoff (узел
        удаляется). Драйвер prediction спит, пока мир не вернёт давление — тогда
        keep оживёт; механизм/персист готовы заранее."""
        org = self.organisms.get(cid)
        pred = self.predictor.get(cid)
        if org is None or pred is None:
            return
        if self._cerebellum_tissue_id(cid, org) is None:
            return
        st = self._tissue_growth_state.get(cid)
        if st is None:
            # Узлы — ПОСЛЕ рёбер: пока связь-петля не насытилась, ткани ждут.
            if not self._growth_saturated:
                return
            # Плато intrinsic у трейлинг-floor (тот же критерий, что у рёбер).
            intr = float(self.intrinsic_ema.get(cid, 0.0))
            hist = self._growth_intr_hist.get(cid)
            if not hist:
                return
            floor = min(hist)
            if intr <= floor * (1.0 + self._growth_near_floor_margin):
                self._propose_growth_tissue(cid, org)
            return
        # dwell — ждём пере-сходимости predictor'а с новым узлом.
        st["ticks"] += 1
        if st["ticks"] < self._growth_dwell_ticks:
            return
        loss_before = float(st["loss_before"])
        loss_after = float(self.loss_ema.get(cid, loss_before))
        delta = loss_before - loss_after
        signif = delta >= loss_before * self._growth_min_delta_frac
        net_ok = int(self._paralysis_window_n) <= int(st["par_before"])
        bc = self.biochem.get(cid)
        if bc is not None and st.get("energy_before") is not None:
            net_ok = net_ok and (float(getattr(bc, "energy", 0.0))
                                 >= float(st["energy_before"]) * 0.618)
        keep = bool(signif and net_ok)
        role = st["role"]
        tid = st["tid"]
        if keep:
            self._tissue_kept += 1
            # персист спек узла для recreate-on-restore (durable-инвариант).
            self._tissue_grown_specs.setdefault(cid, []).append(st["spec"])
            logger.info("brain-growth TISSUE-KEEP cid=%s role=%s Δloss=%.5f "
                        "(%.1f%%) kept=%d", cid, role, delta,
                        100.0 * delta / max(1e-9, loss_before), self._tissue_kept)
            # новая ткань = новая роль → связь-петле снова есть {роль}→cerebellum.
            self.reset_growth_saturation()
        else:
            self._remove_grown_tissue(cid, org, tid, role, st.get("gene"))
            self._tissue_reverted += 1
            logger.info("brain-growth TISSUE-BACKOFF cid=%s role=%s Δloss=%.5f "
                        "signif=%s net_ok=%s reverted=%d", cid, role, delta,
                        signif, net_ok, self._tissue_reverted)
        self._tissue_growth_state.pop(cid, None)
        self._growth_stagnation_n[cid] = 0  # заново копим стагнацию до след. propose

    def _predictor_train_step(self, cid: str, obs_tensor,
                              input_tensor=None) -> float:
        """Phase 1+2: один MSE-шаг predictor + intrinsic reward.

        Идентично _predictor_train_step на P40 (routes_world.py:1149).

        target = obs_tensor (obs64, env_{t+1} — что предсказываем). input_tensor
        (Track 2) = то, что СОХРАНЯЕМ как prev_obs (вход следующего forward): obs68
        (env+self) при расширенном восприятии, иначе obs64. Output ткани всегда
        DATA_DIM=64 → MSE с obs64-target. None → input=target (backward-compat 64).

        Возвращает intrinsic_last (β·max(0, loss_ema_prev - loss_curr)).
        Если predictor нет или prev_obs пустой — 0.0 (но prev_obs обновится).
        """
        torch = self._torch
        pred = self.predictor.get(cid)
        opt = self.predictor_opt.get(cid)
        prev = self.prev_obs.get(cid)
        if input_tensor is None:
            input_tensor = obs_tensor
        intrinsic = 0.0
        self.intrinsic_last[cid] = 0.0
        if pred is not None and opt is not None and prev is not None:
            try:
                import torch.nn.functional as F
                pred.train()
                with torch.enable_grad():
                    out = pred({"input": prev})["output"]
                    loss = F.mse_loss(out, obs_tensor.detach())
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                loss_f = float(loss.item())
                surprise_prev = self.loss_ema.get(cid, 0.0)
                self.loss_ema[cid] = (1 - _EMA_ALPHA) * surprise_prev + _EMA_ALPHA * loss_f
                self.pred_loss_history[cid].append(loss_f)
                # §10.9 v0.2-валидация: per-region predictor-loss — ГДЕ давление.
                # [35]=temperature (прямой сигнал), [44:55]=density bins (temp-
                # зависимы в v0.2 через respawn-лаг → сюда сядет Path-B давление),
                # rest=базлайн. v0.1: dens≈базлайн. v0.2: dens↑ И УСТОЙЧИВО (лаг/
                # нелинейность не выучиваются persistence'ом, в отличие от гладкого
                # temp@35) = реальное давление. Throttled, read-only (не трогает train).
                self._pred_region_diag_n = getattr(self, "_pred_region_diag_n", 0) + 1
                if self._pred_region_diag_n % 600 == 1:
                    try:
                        with torch.no_grad():
                            _t = obs_tensor.detach()
                            _lt = float(F.mse_loss(out[:, 35:36], _t[:, 35:36]).item())
                            _ld = float(F.mse_loss(out[:, 44:56], _t[:, 44:56]).item())
                            _rest_o = torch.cat([out[:, :35], out[:, 36:44], out[:, 56:]], dim=1)
                            _rest_t = torch.cat([_t[:, :35], _t[:, 36:44], _t[:, 56:]], dim=1)
                            _lr = float(F.mse_loss(_rest_o, _rest_t).item())
                        logger.info("PRED_REGION_DIAG cid=%s loss=%.5f temp35=%.5f "
                                    "dens44_55=%.5f rest=%.5f", cid, loss_f, _lt, _ld, _lr)
                    except Exception:
                        pass
                delta = max(0.0, surprise_prev - loss_f)
                intrinsic = _BETA_INTRINSIC * delta
                self.intrinsic_last[cid] = intrinsic
                self.intrinsic_ema[cid] = (
                    (1 - _EMA_ALPHA) * self.intrinsic_ema.get(cid, 0.0)
                    + _EMA_ALPHA * intrinsic
                )
                if self.predictor_steps == 0:
                    logger.info(
                        "predictor train OK first time cid=%s loss=%.4f",
                        cid, loss_f)
                self.predictor_steps += 1
            except Exception as e:
                logger.warning("predictor train %s: %s", cid, e)
        if pred is not None:
            # Track 2: храним ВХОД (obs68 при расширенном восприятии) — он станет
            # prev для следующего forward. target (obs64) только для loss.
            self.prev_obs[cid] = input_tensor.detach()
        return intrinsic

    # ── Track 2 (этап 4): self-observable obs — расширение восприятия ─────

    def _build_self_observable(self, cid: str) -> "np.ndarray":
        """4 self-observable сигнала для STATE_DIM obs[64:68] (Track 2).

        Интероцепция ВЫСШЕГО порядка — мозг ощущает не тело, а свой УМ. Контракт
        зафиксирован (cb8180b, порядок постоянный): entropy_ema («не уверен») /
        trace_norm_ema («учусь») / reward_var_ema («среда изменилась») / paralyzed
        («осознаёт паралич» — закрывает §3 обучающую половину). См. _SELF_OBS_*.
        """
        par, _ = self._paralysis_state(cid)
        return np.array([
            float(self.entropy_ema.get(cid, 0.0)),
            float(self.trace_norm_ema.get(cid, 0.0)),
            float(self.reward_var_ema.get(cid, 0.0)),
            1.0 if par else 0.0,
        ], dtype=np.float32)

    def _build_client_intero(self, cid: str):
        """§3.2: client-authoritative интероцепция [7] из self.biochem.

        Зеркало server `_gather_interoception` (sidecars.py:48). Slots 0,1,3,4,5
        (energy/hydration/cortisol/serotonin/infection) — ТОЧНЫЕ из биохимии
        Адама. Slots 2 (age) и 6 (valence=comfort−discomfort) требуют member-
        стейта P40 (нет birth_tick / comfort-EMA client-side) → carry последнего
        P40-intero (или 0). Снимает P40-blind risk — insula не слепнет, если P40
        перестанет слать `intero`. Возвращает np.ndarray[7] или None (нет биохимии
        → insula пропустится штатно).

        Нормировки совпадают с server ТОЧЬ-В-ТОЧЬ:
          slot[1] = hydration / (100·camel)  — raw camel, как в _gather (НЕ
          kleiber); даёт ту же squashed-шкалу, что insula видела от P40. Для
          felt-DRIVE используется нативная [0,100] шкала (см. ws_client), НЕ
          этот squashed slot — у них разные потребители.
        """
        bc = self.biochem.get(cid)
        if bc is None:
            return None
        energy = float(getattr(bc, "energy", 0.0) or 0.0)
        hydration = float(getattr(bc, "hydration", 0.0) or 0.0)
        camel = float((self.traits.get(cid, {}) or {}).get(
            "camel", _CLIENT_DEFAULT_CAMEL) or _CLIENT_DEFAULT_CAMEL)
        max_h = 100.0 * camel
        cortisol = float(getattr(bc, "cortisol", 0.0) or 0.0)
        serotonin = float(getattr(bc, "serotonin", 0.0) or 0.0)
        infection = float(getattr(bc, "infection_severity", 0.0) or 0.0)
        # slots 2/6 — carry последнего P40-intero (member-стейт), пока нет
        # client-side birth_tick/comfort-EMA. Полный паритет — отд. итерацией.
        carry = self._last_p40_intero.get(cid)
        age_norm = (float(carry[2]) if carry is not None
                    and len(carry) > 2 else 0.0)
        valence = (float(carry[6]) if carry is not None
                   and len(carry) > 6 else 0.0)
        return np.array([
            energy / _CLIENT_MAX_ENERGY if _CLIENT_MAX_ENERGY > 0 else 0.0,
            hydration / max_h if max_h > 0 else 0.0,
            age_norm,
            cortisol / 100.0,
            serotonin / 100.0,
            infection,
            valence,
        ], dtype=np.float32)

    def _upgrade_tissue_input_dim(self, tissue, new_data_dim: int) -> bool:
        """Расширить read-окно ткани DATA_DIM(64)→new_data_dim (Track 2).

        Добавляет input_proj Linear(new_data_dim→64) с PASSTHROUGH-init [I_64 | 0]:
        первые 64 проходят как есть, новые dims → 0-влияние. Обученное ядро ткани
        НЕ трогается → founding-мозг c103927 не дисраптится (math-equivalence на
        первых 64); новые self-observable dims Hebbian/predictor доучивают сами.
        Использует штатный механизм Tissue (data_dim>64 → input_proj, см.
        core/tissue.py:163). Идемпотентно. Returns True если применено.
        """
        torch = self._torch
        nn = torch.nn
        try:
            cur = int(getattr(tissue, "data_dim", _SELF_OBS_OFFSET))
            if cur >= new_data_dim:
                return False
            proj = nn.Linear(new_data_dim, _SELF_OBS_OFFSET).to(self.device)
            with torch.no_grad():
                w = torch.zeros(_SELF_OBS_OFFSET, new_data_dim,
                                dtype=proj.weight.dtype, device=self.device)
                w[:, :_SELF_OBS_OFFSET] = torch.eye(
                    _SELF_OBS_OFFSET, dtype=proj.weight.dtype, device=self.device)
                proj.weight.copy_(w)   # [I_64 | 0] — passthrough первые 64
                proj.bias.zero_()
            tissue.input_proj = proj
            tissue.data_dim = new_data_dim
            return True
        except Exception as e:
            logger.warning("upgrade_tissue_input_dim failed: %s", e)
            return False

    def _enable_self_observable(self, cid: str) -> bool:
        """Track 2: расширить восприятие predictor cid'а до obs68 (env+self).

        Upgrade input_proj [I|0] + ПЕРЕСОЗДАТЬ optimizer (новый input_proj должен
        обучаться, иначе self-observable dims навсегда нулевые). Идемпотентно
        (если уже расширен — opt не трогаем). Returns True если применено сейчас.
        """
        pred = self.predictor.get(cid)
        if pred is None:
            return False
        if self._upgrade_tissue_input_dim(pred, _BRAIN_INPUT_DIM):
            try:
                self.predictor_opt[cid] = self._torch.optim.Adam(
                    pred.parameters(), lr=1e-3)
            except Exception as e:
                logger.warning("self-observable opt recreate %s: %s", cid, e)
            # Stale prev_obs (obs64, до upgrade) не совпадёт с input_proj(68) →
            # чистим: следующий forward стартует с prev=None (skip) → сохранит
            # obs68 → дальше согласовано.
            self.prev_obs.pop(cid, None)
            logger.info("self-observable enabled cid=%s (predictor 64→68, [I|0])",
                        cid)
            return True
        return False

    def _load_predictor_sd(self, cid: str, pred_sd: dict) -> None:
        """Robust load predictor state_dict (Track 2). Если сохранён РАСШИРЕННЫЙ
        predictor (data_dim=68 → есть input_proj.* в sd), сначала upgrade rebuilt
        predictor [I|0], потом load — ключи совпадут, обученный self-observable
        input_proj переживёт restart. Иначе обычный load (64)."""
        pred = self.predictor.get(cid)
        if pred is None:
            return
        if any(str(k).startswith("input_proj") for k in pred_sd.keys()):
            self._upgrade_tissue_input_dim(pred, _BRAIN_INPUT_DIM)
        pred.load_state_dict(pred_sd)

    def _enable_self_obs_action_head(self, cid: str) -> bool:
        """Track 2: создать self-obs→action REINFORCE-голову для cid.

        Linear(_SELF_OBS_DIM→N_ACTIONS), ZERO-init (вес+bias=0) → на старте bias=0
        → действие Адама НЕ меняется (non-destructive). Учится REINFORCE: мапит
        self-observable в bias логитов под награду. Идемпотентно.
        """
        if cid in self.self_obs_head:
            return False
        torch = self._torch
        nn = torch.nn
        try:
            head = nn.Linear(_SELF_OBS_DIM, N_ACTIONS).to(self.device)
            with torch.no_grad():
                head.weight.zero_()
                head.bias.zero_()
            self.self_obs_head[cid] = head
            self.self_obs_head_opt[cid] = torch.optim.Adam(
                head.parameters(), lr=1e-3)
            logger.info("self-obs action head enabled cid=%s (%d→%d, zero-init)",
                        cid, _SELF_OBS_DIM, N_ACTIONS)
            return True
        except Exception as e:
            logger.warning("enable_self_obs_action_head %s: %s", cid, e)
            return False

    def _self_obs_head_reinforce(self, cid: str, ctx, advantage: float) -> None:
        """REINFORCE-шаг головы: ctx=(so4, action, base_logits). loss =
        -log_softmax(base + head(so4))[action] · advantage. base — detached
        (вклад organism+motor); градиент только через голову. Adam-step."""
        head = self.self_obs_head.get(cid)
        opt = self.self_obs_head_opt.get(cid)
        if head is None or opt is None or ctx is None:
            return
        torch = self._torch
        so4, action, base = ctx
        try:
            import torch.nn.functional as F
            bias = head(so4)                       # grad через голову
            final = base + bias                    # base detached
            logp = F.log_softmax(final, dim=-1)[int(action)]
            loss = -logp * float(advantage)
            opt.zero_grad()
            loss.backward()
            opt.step()
        except Exception as e:
            logger.debug("self_obs_head reinforce %s: %s", cid, e)

    # ── Track 2 / направление (б): insula-стресс → temperature-модуляция ──

    def _enable_insula_temp(self, cid: str) -> bool:
        """Создать insula-temp голову для cid (направление (б), Фрай).

        Linear(_SELF_OBS_DIM→1), ZERO-init (вес+bias=0) → mu=0 → log_tmod≈0 →
        T_mod≈1.0 (near-identity старт, действие НЕ меняется). Учится 1-dim
        Gaussian-policy REINFORCE мапить интероцепцию (стресс/неуверенность) в
        temperature (sharpen при уверенности, flatten при стрессе). Temperature
        НЕ меняет направление действия → структурно безопасна. Идемпотентно.
        """
        if cid in self.insula_temp_head:
            return False
        torch = self._torch
        nn = torch.nn
        try:
            head = nn.Linear(_SELF_OBS_DIM, 1).to(self.device)
            with torch.no_grad():
                head.weight.zero_()
                head.bias.zero_()
            self.insula_temp_head[cid] = head
            self.insula_temp_head_opt[cid] = torch.optim.Adam(
                head.parameters(), lr=_INSULA_TEMP_LR)
            self._it_baseline.setdefault(cid, 0.0)
            logger.info("insula-temp head enabled cid=%s (%d→1, zero-init, "
                        "near-identity T≈1)", cid, _SELF_OBS_DIM)
            return True
        except Exception as e:
            logger.warning("enable_insula_temp %s: %s", cid, e)
            return False

    def _insula_temp_factor(self, cid: str, so4, deterministic: bool = False):
        """Вернуть (T_mod, log_tmod, ctx) для cid из self-observable so4.

        T_mod = exp(clamp(mu + σ·ε, ±ln2)), mu = head(so4) (detached для forward).
        deterministic=True → ε=0 (для тестов/проверки near-identity: zero-init
        head → T_mod=1.0 ровно). ctx=(so4, log_tmod) для последующего REINFORCE.
        Возвращает (None, None, None) если головы нет/ошибка.
        """
        head = self.insula_temp_head.get(cid)
        if head is None:
            return None, None, None
        torch = self._torch
        try:
            with torch.no_grad():
                mu = head(so4).reshape(())          # скаляр
                if deterministic:
                    eps = torch.zeros((), device=mu.device, dtype=mu.dtype)
                else:
                    eps = torch.randn((), device=mu.device, dtype=mu.dtype)
                log_tmod = (mu + _INSULA_TEMP_SIGMA * eps).clamp(
                    -_INSULA_TEMP_LOG_CLAMP, _INSULA_TEMP_LOG_CLAMP)
                t_mod = float(torch.exp(log_tmod).item())
            return t_mod, float(log_tmod.item()), (so4, float(log_tmod.item()))
        except Exception as e:
            logger.debug("insula_temp_factor %s: %s", cid, e)
            return None, None, None

    def _insula_temp_reinforce(self, cid: str, ctx, advantage: float) -> None:
        """1-dim Gaussian-policy REINFORCE-шаг temperature-головы.

        ctx=(so4, log_tmod_sampled). mu=head(so4) (grad). logπ(log_tmod|mu) =
        -0.5·((log_tmod−mu)/σ)² (+const, отбрасываем). loss = −(adv−baseline)·logπ.
        baseline = бегущая средняя advantage (variance-reduction, как Фрай просил).
        Малый lr → медленный отход от near-identity. Adam-step.
        """
        head = self.insula_temp_head.get(cid)
        opt = self.insula_temp_head_opt.get(cid)
        if head is None or opt is None or ctx is None:
            return
        torch = self._torch
        so4, log_tmod = ctx
        try:
            base = self._it_baseline.get(cid, 0.0)
            adv = float(advantage) - float(base)
            # обновляем baseline (EMA) ПОСЛЕ вычисления adv
            self._it_baseline[cid] = (
                (1.0 - _EMA_ALPHA) * float(base) + _EMA_ALPHA * float(advantage))
            mu = head(so4).reshape(())             # grad через голову
            logp = -0.5 * ((float(log_tmod) - mu) / _INSULA_TEMP_SIGMA) ** 2
            loss = -logp * adv
            opt.zero_grad()
            loss.backward()
            opt.step()
        except Exception as e:
            logger.debug("insula_temp reinforce %s: %s", cid, e)

    def _apply_insula_temp(self, cid: str, slice_t):
        """Гейт + применение (б): slice / T_mod до select. Returns (slice, ctx).

        Флаг off ИЛИ головы нет → возвращает (slice, None) (полный no-op). Иначе
        строит so4, берёт T_mod (~1 near-identity), делит логиты, отдаёт ctx для
        next-тик REINFORCE. T_mod>1 → flatter (explore при стрессе), <1 → sharper.
        Никогда не меняет НАПРАВЛЕНИЕ — только остроту распределения.
        """
        if not self._insula_temp_enabled or cid not in self.insula_temp_head:
            return slice_t, None
        torch = self._torch
        try:
            so4 = torch.from_numpy(
                self._build_self_observable(cid)).to(self.device)
            t_mod, _log_tmod, ctx = self._insula_temp_factor(cid, so4)
            if t_mod is None or not (0.0 < t_mod < 1e6):
                return slice_t, None
            self._it_last_tmod[cid] = t_mod   # телеметрия обучения моста
            return slice_t / t_mod, ctx
        except Exception as e:
            logger.debug("apply_insula_temp %s: %s", cid, e)
            return slice_t, None

    # ── Phase 6 — Self-observable states ─────────────────────────────────

    def _update_entropy_ema(self, cid: str, logits) -> None:
        """Нормированная энтропия action-distribution ∈ [0, 1]."""
        if cid not in self.entropy_ema:
            return
        torch = self._torch
        try:
            n = int(logits.shape[-1])
            if n <= 1:
                return
            probs = torch.softmax(logits, dim=-1)
            ent = float(-(probs * probs.clamp_min(1e-9).log()).sum().item())
            ent_norm = max(0.0, min(1.0, ent / math.log(n)))
            self.entropy_ema[cid] = (
                (1 - _EMA_ALPHA) * self.entropy_ema[cid] + _EMA_ALPHA * ent_norm
            )
        except Exception:
            pass

    def _update_trace_norm_ema(self, cid: str, hebbian) -> None:
        """RMS-норма Hebbian eligibility-traces ∈ [0, 1)."""
        if cid not in self.trace_norm_ema or hebbian is None:
            return
        try:
            sq = 0.0
            cnt = 0
            for info in getattr(hebbian, "_tissue_info", []):
                t = info.get("trace")
                if t is None:
                    continue
                sq += float((t * t).sum().item())
                cnt += int(t.numel())
            if cnt == 0:
                return
            rms = (sq / cnt) ** 0.5
            tn = 1.0 - math.exp(-rms)
            self.trace_norm_ema[cid] = (
                (1 - _EMA_ALPHA) * self.trace_norm_ema[cid] + _EMA_ALPHA * tn
            )
        except Exception:
            pass

    def _update_reward_var_ema(self, cid: str, r_imm: float) -> None:
        """Variance последних 10 r_imm, rescaled ∈ [0, 1]."""
        hist = self.reward_history.get(cid)
        if hist is None:
            return
        hist.append(float(r_imm))
        if len(hist) < 3:
            return
        mean = sum(hist) / len(hist)
        var = sum((r - mean) ** 2 for r in hist) / len(hist)
        rv = min(1.0, var * 10.0)
        self.reward_var_ema[cid] = (
            (1 - _EMA_ALPHA) * self.reward_var_ema[cid] + _EMA_ALPHA * rv
        )

    # ── Body Migration Phase 2 (27.05.2026, Бендер): biochem events + decay ─

    def _build_biochem_snapshot(self) -> dict:
        """Snapshot для diagnostics push: агрегаты + mental_break heatmap.

        Цель — Хьюберт + frontend StatsPage увидят:
          - сколько owned Зодчих с активной биохимией (`n_active`)
          - распределение mental_break по states (`mental_break_counts`)
          - average levels всех 8 веществ (для радара/тренда UI)

        Per-cid raw данные **не пушим** в diagnostics — это шум для
        observability. Если потребуется детальный per-cid breakdown
        (для debug), запрашивается через отдельный admin endpoint.
        """
        biochem = self.biochem
        n = len(biochem)
        if n == 0:
            return {
                "n_active": 0,
                "mental_break_counts": {},
                "cortisol_avg": 0.0,
                "serotonin_avg": 0.0,
                "dopamine_avg": 0.0,
                "oxytocin_avg": 0.0,
                "adrenaline_avg": 0.0,
                "glucose_avg": 0.0,
                "fatigue_avg": 0.0,
                "histamine_avg": 0.0,
            }
        mb_counts: dict[str, int] = {}
        sums = {
            "cortisol": 0.0, "serotonin": 0.0, "dopamine": 0.0,
            "oxytocin": 0.0, "adrenaline": 0.0,
            "glucose": 0.0, "fatigue": 0.0, "histamine": 0.0,
        }
        for bc in biochem.values():
            mb = str(getattr(bc, "mental_break", "") or "normal")
            mb_counts[mb] = mb_counts.get(mb, 0) + 1
            for chem in sums:
                try:
                    sums[chem] += float(getattr(bc, chem, 0.0))
                except (TypeError, ValueError):
                    pass
        # DEBUG: per-cid biochem log (temp diagnostic). +cortisol/serotonin
        # для verify cortisol-гомеостаза 0.11.7 (catatonic = cort>80 + ser<MAX).
        try:
            per_cid_e = sorted([
                (cid, round(float(getattr(bc, "energy", 0.0)), 1),
                 round(float(getattr(bc, "cortisol", 0.0)), 1),
                 round(float(getattr(bc, "serotonin", 0.0)), 1),
                 round(float(getattr(bc, "glucose", 0.0)), 1),
                 str(getattr(bc, "mental_break", "") or ""))
                for cid, bc in biochem.items()
            ])
            logger.info(
                "BIOCHEM_DEBUG cids_e_cort_ser_g_mb: %s",
                "; ".join(f"{cid}:e={e},cort={c},ser={s},g={g},mb={mb}"
                          for cid, e, c, s, g, mb in per_cid_e))
        except Exception as _e:
            logger.debug("biochem debug log failed: %s", _e)
        # INSULA_TEMP_DEBUG (Track 2 (б)): сигнал ОБУЧЕНИЯ моста — отличить
        # «мало времени» от «не работает» (Фрай). t_mod (расход от 1.0),
        # wnorm (L2 веса головы; zero-init=0 → расход = обучение), baseline
        # (REINFORCE variance-reduction; растёт = мост видит advantage).
        try:
            if self.insula_temp_head:
                _it_dbg = []
                for cid, head in self.insula_temp_head.items():
                    try:
                        wnorm = float(head.weight.detach().norm().item())
                    except Exception:
                        wnorm = 0.0
                    _it_dbg.append(
                        f"{cid}:t_mod={round(float(self._it_last_tmod.get(cid, 1.0)), 4)},"
                        f"wnorm={round(wnorm, 5)},"
                        f"baseline={round(float(self._it_baseline.get(cid, 0.0)), 3)}")
                # enabled= — явный маркер состояния флага для A/B-окон (Ступень 0
                # full-world): головы персистят при off → без маркера t_mod stale
                # и on/off-окна не разделить. enabled=1 → мост влияет, =0 → no-op.
                logger.info(
                    "INSULA_TEMP_DEBUG enabled=%d cids_tmod_wnorm_baseline: %s",
                    1 if self._insula_temp_enabled else 0, "; ".join(_it_dbg))
        except Exception as _e:
            logger.debug("insula_temp debug log failed: %s", _e)
        # MOTOR_LEARN_DEBUG (03.06, Ступень 2): доставка reward→motor-policy.
        # steps растёт → апдейт бежит (не (i)-skip); baseline≠0 → reward доезжает;
        # adv_ema≠0 → reward дискриминирует; dw_ema≠0 → policy реально движется.
        # Все ≠0 + flip держит 0.99 → (ii) credit-fail; dw_ema≈0/steps плоский →
        # (i) delivery-bug. Логируем для cid с motor-baseline (обновлялись).
        try:
            if self._motor_reward_baseline:
                _ml = "; ".join(
                    f"{cid}:baseline={round(float(self._motor_reward_baseline.get(cid, 0.0)), 3)},"
                    f"adv_ema={round(float(self._motor_adv_ema.get(cid, 0.0)), 4)},"
                    f"dw_ema={round(float(self._motor_dw_ema.get(cid, 0.0)), 5)},"
                    # ΔW-направление (Фрай): radial→1 = renorm режет=(a); cos→0/<0 = incoherent=(b)
                    f"dw_radial={round(float(self._motor_dw_radial_ema.get(cid, -1.0)), 3)},"
                    f"dw_cos={round(float(self._motor_dw_cos_ema.get(cid, -9.0)), 3)}"
                    for cid in self._motor_reward_baseline)
                logger.info(
                    "MOTOR_LEARN_DEBUG steps=%d renorm_cap=%.2f oja_scale=%.2f cids: %s",
                    int(self.motor_sfnn_steps),
                    float(self._motor_renorm_growth_cap),
                    float(self._motor_oja_scale), _ml)
        except Exception as _e:
            logger.debug("motor_learn debug log failed: %s", _e)
        # SELECTOR_DEBUG (03.06, Фрай Шаг 1): downstream-кап ActionSelector —
        # биндит ли temperature(→5.0)/ε решительный логит обратно к высокой
        # entropy (anti-collapse, колониальное наследие, инверсия пивота).
        # temp pinned high + entropy высокая + random_pct высокий → селектор
        # флэттит policy (НЕ SFNN weight-update). argmax_share = lock-in watch.
        try:
            for cid, sel in self.action_selectors.items():
                st = sel.get_stats()
                ad = st.get("action_distribution") or []
                _amax = max(range(len(ad)), key=lambda i: ad[i]) if ad else -1
                _ashare = ad[_amax] if ad else 0.0
                logger.info(
                    "SELECTOR_DEBUG %s: temp=%.3f eps=%.4f avg_entropy=%.4f "
                    "random_pct=%.1f argmax_action=%d argmax_share=%.3f",
                    cid, st.get("temperature", 0.0), st.get("epsilon", 0.0),
                    st.get("avg_entropy", 0.0), st.get("random_pct", 0.0),
                    _amax, _ashare)
        except Exception as _e:
            logger.debug("selector debug log failed: %s", _e)
        # LOGIT_DEBUG (Фрай): локализация uniformity. base (organism+shaping, до
        # motor) vs final (после motor). base_ent/final_ent = softmax-энтропия
        # (макс ln16=2.77). md_unif = std/|mean| motor_delta (мал → uniform blob →
        # shift-инвариант → не дифференцирует). base_peak/final_peak = argmax-доля.
        # Читать: base пикует, final flat → motor смазывает; обе flat → shaping/
        # base не пикует (reward не выучил preference).
        try:
            torch = self._torch
            for cid, _ld in self._logit_dbg.items():
                raw_t, base_t, final_t, _own_v, g_ns, g_ew = _ld[:6]
                md_t = _ld[6] if len(_ld) > 6 else None
                rp = torch.softmax(raw_t, dim=-1)
                bp = torch.softmax(base_t, dim=-1)
                fp = torch.softmax(final_t, dim=-1)
                raw_ent = float(-(rp * rp.clamp_min(1e-9).log()).sum().item())
                base_ent = float(-(bp * bp.clamp_min(1e-9).log()).sum().item())
                final_ent = float(-(fp * fp.clamp_min(1e-9).log()).sum().item())
                raw_spread = float(raw_t.std().item())   # (A): спред org.forward
                _b_argmax = int(bp.argmax().item())
                # Phase-1 alignment-reader (Фрай 04.06): куда толкает САМ обучаемый
                # motor_delta (SFNN-REINFORCE). m_argmax — топ-действие мотора;
                # m_attack=motor_delta[5] — ATTACK-override (спадает→0 = переучен);
                # m_atbase=motor_delta[base_argmax] — реинфорсит ли мотор пик прайора
                # (food-dir): >0 = выравнивается, <0 = воюет с прайором; align=1 если
                # топ мотора == пик прайора. ATTACK=5, food-movement=0..3, GATHER=13.
                if md_t is not None:
                    _m_argmax = int(md_t.argmax().item())
                    _m_attack = float(md_t[5].item())
                    _m_atbase = float(md_t[_b_argmax].item())
                    _align = 1 if _m_argmax == _b_argmax else 0
                    _motor_str = (" | motor: m_argmax=%d m_attack=%+.4f "
                                  "m_atbase=%+.4f align=%d" % (
                                      _m_argmax, _m_attack, _m_atbase, _align))
                else:
                    _motor_str = " | motor: n/a (own=0)"
                # Threat-телеметрия (Шеф 04.06): prey-близость(58)/predator-близость(61)
                # — есть ли персистентная цель/угроза у Адама (1/dist, высокое=близко).
                _prey_prox = float(_ld[7]) if len(_ld) > 8 else -1.0
                _pred_prox = float(_ld[8]) if len(_ld) > 8 else -1.0
                _threat_str = " | threat: prey_prox=%.3f pred_prox=%.3f" % (
                    _prey_prox, _pred_prox)
                logger.info(
                    "LOGIT_DEBUG %s: own=%.2f raw_ent=%.3f raw_spread=%.3f "
                    "base_ent=%.3f final_ent=%.3f (max2.77) base_peak=%.3f "
                    "obs_grad=|%.3f,%.3f| raw_argmax=%d base_argmax=%d final_argmax=%d%s%s",
                    cid, _own_v, raw_ent, raw_spread, base_ent, final_ent,
                    float(bp.max().item()), g_ns, g_ew,
                    int(rp.argmax().item()), _b_argmax,
                    int(fp.argmax().item()), _motor_str, _threat_str)
        except Exception as _e:
            logger.debug("logit debug log failed: %s", _e)
        return {
            "n_active": n,
            "mental_break_counts": mb_counts,
            "cortisol_avg": round(sums["cortisol"] / n, 2),
            "serotonin_avg": round(sums["serotonin"] / n, 2),
            "dopamine_avg": round(sums["dopamine"] / n, 2),
            "oxytocin_avg": round(sums["oxytocin"] / n, 2),
            "adrenaline_avg": round(sums["adrenaline"] / n, 2),
            "glucose_avg": round(sums["glucose"] / n, 2),
            "fatigue_avg": round(sums["fatigue"] / n, 2),
            "histamine_avg": round(sums["histamine"] / n, 2),
        }

    def _build_natural_selection_snapshot(self) -> dict:
        """Phase 4 этап G (28.05.2026, Бендер): естественный отбор tracking.

        Pure observability — client НЕ вмешивается в жизнь организмов
        (vision §3.1 «вариант D»). Server-side рычаги (flora_density,
        TelomerePhase decay, safe_grove_mult) делают давление сами.
        Здесь только snapshot для diag и frontend.

        Эмитится в diag['natural_selection']:
            {n_organisms, capacity, weakest_cids, scores, mean_score, max_score}

        Если biochem empty (нет zodchiy) — n_organisms=0, capacity сохра-
        няется (для frontend context).
        """
        try:
            from .natural_selection import natural_selection_snapshot
        except Exception as e:
            logger.debug("natural_selection module import failed: %s", e)
            return {"n_organisms": 0, "capacity": None,
                    "weakest_cids": [], "scores": {},
                    "mean_score": 0.0, "max_score": 0.0}
        # capacity = estimate_population() — потолок по железу (vision
        # body_migration.md §40/§145: размер колонии определяется
        # benchmark.py:estimate_population).
        capacity = self._get_hw_capacity()
        return natural_selection_snapshot(
            self.biochem, capacity=capacity, top_n_to_emit=3)

    # ── §3 paralysis state (single-organism) ────────────────────────────

    def _paralysis_state(self, cid: str) -> tuple[bool, int]:
        """(paralyzed, ticks_remaining) для cid. ticks ≈ P40-тики @30TPS из
        wall-clock остатка. Снимает истёкший дедлайн как побочный эффект чтения
        в projection (recovery-грант энергии делает _apply_metabolism)."""
        until = self._paralysis_until.get(cid)
        if until is None:
            return False, 0
        remaining = until - time.monotonic()
        if remaining <= 0.0:
            return False, 0
        return True, max(0, round(remaining * 30.0))

    def _enter_paralysis(self, cid: str, reason: str) -> None:
        """Войти в паралич (единая точка для обоих триггеров: energy≤0 и
        death_suppressed от P40). Idempotent: повторный вызов в активном
        параличе НЕ продлевает (избегаем «вечного» паралича от потока событий).
        record paralysis_enter — P40 через projection False→True."""
        # Idempotent: если cid уже в паралич-цикле (ДАЖЕ с истёкшим дедлайном) —
        # НЕ ре-армим. Иначе поток триггеров (P40 шлёт death_suppressed каждый
        # тик при energy=0) ре-армил бы дедлайн на каждом expiry → recovery в
        # _apply_metabolism никогда не срабатывал → вечный паралич, energy
        # застряла на 0 (баг найден live 01.06: 419 start / 1 recovery). Снятие
        # дедлайна — ТОЛЬКО recovery (по expiry) или remove_creature.
        if cid in self._paralysis_until:
            return
        self._paralysis_until[cid] = time.monotonic() + _PARALYSIS_SEC
        logger.info("paralysis start cid=%s reason=%s -> STAY %.1fs (NOT death)",
                    cid, reason, _PARALYSIS_SEC)

    # ── Colony Ownership Migration §5.2: projection_batch ────────────────

    def build_projection_batch(self) -> list[dict]:
        """Собрать lightweight projection всех owned Zodchiy для P40.

        Schema (projection_batch_draft.md §3 ФИНАЛ 28.05):
            {cid, species_id, alive, frozen, action, chem{7}, mental_break}

        P40 размещает проекции в _world.creatures, использует только для
        физики Мира (AOI, флора, видимость, арбитраж). НЕ держит
        authoritative веса — те только у клиента.

        Throttling 5 Hz делается caller'ом (main loop).
        """
        projections: list[dict] = []
        for cid, org in list(self.organisms.items()):
            bc = self.biochem.get(cid)
            last_action = self.last_action.get(cid, -1) if hasattr(self, "last_action") else -1
            proj: dict = {
                "cid": str(cid),
                "species_id": self.species_id.get(cid) if hasattr(self, "species_id") else None,
                # Body Migration метаболизм (31.05.2026): client-authoritative
                # death. _dead_cids заполняется в _apply_metabolism (energy<=0 /
                # telomere AGONY). P40 на alive=False убирает + decrement.
                "alive": cid not in self._dead_cids,
                "frozen": False,
                "action": int(last_action) if last_action >= 0 else -1,
            }
            # Body Migration метаболизм: client-authoritative energy/hydration/
            # telomere → P40 accept'ит как chem на CreatureState (для death-
            # триггеров + физики). Контракт Хьюберт 31.05.
            try:
                if bc is not None:
                    proj["energy"] = float(getattr(bc, "energy", 0.0))
                    proj["hydration"] = float(getattr(bc, "hydration", 0.0))
                    # Infection (01.06.2026, Фрай): client-authoritative →
                    # P40 зеркалит (как energy/hydration). Тик/death — клиент.
                    proj["infected"] = bool(getattr(bc, "infected", False))
                    proj["infection_severity"] = float(
                        getattr(bc, "infection_severity", 0.0) or 0.0)
                proj["telomere_scale"] = float(getattr(org, "telomere", 1.0))
            except Exception:
                pass
            # §3 paralysis mirror (контракт Хьюберта 2b0f3a2): client →  P40.
            # P40 читает для force-STAY override + record_adam_event на
            # transition False→True. client = единственный хозяин paralysis.
            _par, _ticks = self._paralysis_state(cid)
            proj["paralyzed"] = bool(_par)
            proj["paralysis_ticks_remaining"] = int(_ticks)
            if bc is not None:
                # 7 ephemeral (без histamine = mirror infection_severity)
                proj["chem"] = {
                    "cortisol": float(getattr(bc, "cortisol", 0.0)),
                    "dopamine": float(getattr(bc, "dopamine", 0.0)),
                    "serotonin": float(getattr(bc, "serotonin", 0.0)),
                    "oxytocin": float(getattr(bc, "oxytocin", 0.0)),
                    "adrenaline": float(getattr(bc, "adrenaline", 0.0)),
                    "glucose": float(getattr(bc, "glucose", 0.0)),
                    "fatigue": float(getattr(bc, "fatigue", 0.0)),
                }
                mb = str(getattr(bc, "mental_break", "") or "")
                proj["mental_break"] = mb if mb else None
            else:
                proj["chem"] = None
                proj["mental_break"] = None
            projections.append(proj)
        return projections

    def build_creature_stats(self) -> dict:
        """Компактный per-creature CLIENT-only для UI /stats — keyed by cid.

        Только поля, которых НЕТ в P40 owner_creatures (world_save.py:518):
          species_id — client-local speciation (по графу Z2.b);
          topo       — число межтканевых рёбер-генов (сложность графа мозга);
          inst       — newborn_instinct 0..1 (None для не-newborn).
        Идёт heartbeat-каналом (push_stats → public_meta.extra.creature_stats),
        тем же, что colony_summary — проверенно доходит (diag-push ловит 502).
        Фронт мёржит по cid на свой ряд особи."""
        out: dict = {}
        for cid, org in list(self.organisms.items()):
            if cid in self._dead_cids:
                continue
            bt = self._birth_tick.get(cid)
            inst = (round(max(0.0, 1.0 - max(
                0, int(self._last_world_tick) - int(bt)) / 500.0), 3)
                if bt is not None else None)
            out[str(cid)] = {
                "species_id": self.species_id.get(cid),
                "topo": len(getattr(org, "tissue_topology_genes", []) or []),
                "inst": inst,
            }
        return out

    def build_colony_summary(self) -> dict:
        """Агрегат для UI /stats (extra.colony_summary): выживание (energy/
        deaths), эволюция (species), обучение (newborn) + downsampled history.

        Энерго-поля — из последнего 300-тик окна (_last_window, обновляется в
        ENERGY_CALIB-блоке); n_alive/species/eff_mean/instinct_active — live;
        deaths/natural/bootstrap — кумулятивно. history — ring (~окно 2ч)."""
        alive = [c for c in self.organisms if c not in self._dead_cids]
        n_alive = len(alive)
        n_species = len({self.species_id.get(c) for c in alive
                         if self.species_id.get(c) is not None})
        effs = [int((self.traits.get(c) or {}).get("efficiency", 10))
                for c in alive]
        eff_mean = round(sum(effs) / len(effs), 2) if effs else 0.0
        instinct_active = sum(1 for c in self._birth_tick if c in self.organisms)
        w = self._last_window or {}
        return {
            "n_alive": n_alive,
            "n_species_alive": n_species,
            "eff_mean": eff_mean,
            "energy": {
                "income": float(w.get("inc", 0.0)),
                "cost": float(w.get("cost", 0.0)),
                "net": float(w.get("net", 0.0)),
                "ratio": float(w.get("ratio", 0.0)),
            },
            "newborn": {
                "natural": int(self._n_natural_newborn),
                "bootstrap": int(self._n_bootstrap_rejuv),
                "instinct_active": int(instinct_active),
            },
            "deaths": dict(self._deaths_by_cause),
            "history": list(self._summary_history),
            "ts": int(self._last_world_tick),
        }

    # ── Phase 4 этап E+F (28.05.2026): local-only reproduction flow ────

    # 9 traits child inheritance — projection model (ТЗ 8b8f184 §4.1).
    # Диапазоны (ТЗ §6 R1): client clamps до отправки P40.
    _TRAIT_RANGES: dict = {
        "vision_radius": (3, 12),
        "smell_radius": (5, 40),
        "attack_radius": (1, 5),
        "move_speed": (1, 10),
        "attack_power": (1, 10),
        "armor": (0, 10),
        "efficiency": (1, 10),
        "camel": (5, 30),
        "diet_gene": (0.0, 1.0),
    }

    def _parent_trait(self, organism, cid, name):
        """Resolve один trait родителя: authoritative стор по cid → fallback
        loose-атрибут организма → None.

        Evolved-traits recovery (30.05.2026): стор — источник истины (он
        переживает рестарт и наполняется owned-handoff'ом). getattr оставлен
        как back-compat для путей, где traits ещё не попали в стор (старые
        client-born до миграции, тесты с атрибутами на организме)."""
        if cid is not None:
            stored = self.traits.get(cid)
            if isinstance(stored, dict) and name in stored \
                    and stored[name] is not None:
                return stored[name]
        return getattr(organism, name, None)

    def _inherit_traits_for_newborn(self, mother, father,
                                    mother_cid=None, father_cid=None) -> dict:
        """Compute child 9 traits via parent crossover + clamp.

        Per-trait: 50/50 от родителя + малый Gaussian noise (σ=φ⁻⁵≈0.09 от range).
        clamp в _TRAIT_RANGES (R1 R-mitigation: invalid not sent P40).

        Evolved-traits recovery: значения родителя берутся из authoritative
        стора `self.traits` по cid (с fallback на loose-атрибут) — иначе
        crossover терял эволюцию, читая baseline-атрибуты пересозданного
        self-heal'ом организма.
        """
        import random as _random
        out: dict = {}
        sigma_frac = 0.0902  # 1/φ⁵
        for name, (lo, hi) in self._TRAIT_RANGES.items():
            m_val = self._parent_trait(mother, mother_cid, name)
            f_val = self._parent_trait(father, father_cid, name)
            if m_val is None and f_val is None:
                # Default median
                base = (lo + hi) / 2.0
            elif m_val is None:
                base = float(f_val)
            elif f_val is None:
                base = float(m_val)
            else:
                base = float(m_val) if _random.random() < 0.5 else float(f_val)
            # Small Gaussian noise — scale to range
            span = float(hi - lo)
            noise = _random.gauss(0.0, sigma_frac * span)
            val = base + noise
            # Clamp
            val = max(float(lo), min(float(hi), val))
            # Integer fields → int
            if isinstance(lo, int) and isinstance(hi, int):
                val = int(round(val))
            else:
                val = float(val)
            out[name] = val
        return out

    def detect_and_emit_mate_pairs(
        self,
        world_tick: int,
        embodied_client=None,
    ) -> list[str]:
        """Phase 4 reproduction под projection-модель (ТЗ 8b8f184).

        Flow per detected pair:
          1. Pick (mother_cid, father_cid) — only among own organisms
             (cross-owner ЗАПРЕЩЁН, vision §3.2)
          2. Clone mother organism + crossover with father → child organism
          3. Allocate UUID `child_cid`
          4. `add_creature(child_cid, child_organism, lineage="zodchiy")`
          5. Compute child traits via parent crossover (9 полей, clamped)
          6. Compute generation = max(parents.generation) + 1
          7. Build newborn_announce envelope (projection schema):
             {type, child_cid, parent_cid, parent2_cid, traits, species_id, generation}
          8. Emit + register pending → ждём ack

        Args:
            world_tick: текущий мировой тик (для cooldown + envelope ts)
            embodied_client: EmbodiedWSClient — для send_state. Если None
                (тестирование/disconnected), build child локально но НЕ
                эмитим (pending не записывается → caller узнает по
                returns list пустой).

        Returns:
            List of child cids которые были built + emitted (или пустой
            если embodied_client недоступен).
        """
        # Single-organism pivot (01.06.2026, ТЗ e3cc81b §1): репродукция —
        # колониальная механика, под флагом отключена (Адам не размножается,
        # развитие — не через поколения). Defense-in-depth: main.py уже гейтит
        # вызов через cfg["local_repro_enabled"], но гейтим и на входе, чтобы
        # любой путь (тест/будущий caller) уважал режим. Код ниже сохранён.
        if self._single_organism:
            return []

        import time as _time
        import uuid as _uuid

        try:
            from .mate_detection import detect_mate_pairs, mark_mate_event
            from .crossover import apply_crossover_inheritance
        except Exception as e:
            logger.warning("detect_and_emit_mate_pairs imports failed: %s", e)
            return []

        # Популяционный КЭП (0.11.38, Шеф): размножение до ёмкости железа, не
        # выше (vision §40/§145). Без кэпа eat-income разгонял колонию до 50 →
        # perf-шторм (message-too-big/keepalive). Кэп = bounded self-sustaining
        # цикл: рождения заменяют естественные смерти ДО потолка, без death-
        # налога (тот опрокидывал energy<порог-репро → вымирание). cap=None
        # (бенчмарк упал) → fallback 20 (не runaway).
        _cap = self._get_hw_capacity() or 20
        _alive = sum(1 for cid in self.organisms if cid not in self._dead_cids)
        if _alive >= _cap:
            return []

        # 1. Scan биохимии — только own organisms (compute.biochem все own)
        pairs = detect_mate_pairs(
            self.biochem, self._last_mate_tick, world_tick)
        if not pairs:
            return []

        born_cids: list[str] = []
        for mother_cid, father_cid in pairs:
            mother = self.organisms.get(mother_cid)
            father = self.organisms.get(father_cid)
            if mother is None or father is None:
                logger.debug("mate pair %s+%s: organism missing", mother_cid, father_cid)
                continue

            # 2. Clone mother + apply crossover with father weights
            # Deep clone организма матери — нужно потому что crossover
            # пишет в child (нам нужна копия не задействующая мать).
            try:
                import copy as _copy
                child_org = _copy.deepcopy(mother)
            except Exception as e:
                logger.warning("deepcopy mother %s failed: %s", mother_cid, e)
                continue

            try:
                apply_crossover_inheritance(
                    child_org=child_org,
                    mother_org=mother,
                    father_org=father,
                )
            except Exception as e:
                logger.warning("crossover %s+%s failed: %s",
                               mother_cid, father_cid, e)
                continue

            # 2b. Z2.b (01.06.2026, Фрай): межтканевой NEAT-overlay — суть
            # Зодчего: перестраивает граф тканей, не только веса. crossover
            # рёбер родителей (по innovation_id) + σ-мутация add/remove/
            # change_type/weight → apply_topology_overlay_to_org переписывает
            # child.connections. motor_policy исключён (sidecar-policy, учится
            # весами REINFORCE, не топологией). Гены наполняются органично
            # (p_add=0.02/поколение) → speciation расходится постепенно, не
            # скачком. Делаем ДО add_creature, чтобы _assign_species увидел
            # финальные genes.
            # p_add=0.05/p_remove=0.01 — config-замысел WorldConfig (world.py:378,
            # zodchiy_topology_p_add/p_remove), НЕ функц.дефолт 0.02 (это getattr-
            # fallback в server loop/mate). Z6.c на 0.05+threshold0.9: 5 speciation
            # events за 2ч39м. p_change_type=0.005/p_weight=0.05 совпадают с
            # серверными дефолтами mate_tissue_topology.
            try:
                from core.tissue_topology import (
                    crossover_org_topology_for_zodchiy)
                from .reproduce import _default_zodchiy_available_roles
                crossover_org_topology_for_zodchiy(
                    child_org, mother, father, lineage="zodchiy",
                    available_roles=_default_zodchiy_available_roles(),
                    p_add=0.05, p_remove=0.01,
                )
            except Exception as e:
                logger.debug("Z2.b topology crossover %s+%s: %s",
                             mother_cid, father_cid, e)

            # 3. Allocate UUID
            child_cid = _uuid.uuid4().hex

            # 4. Register child локально — биохимия и episodic подтянутся
            # через add_creature (auto-load episodic + biochem init).
            # Биохимия Z7 inheritance — через make_from_inheritance в
            # отдельном flow (Phase 2 уже работает для asexual).
            try:
                self.add_creature(child_cid, child_org, lineage="zodchiy")
            except Exception as e:
                logger.warning("add_creature %s failed: %s", child_cid, e)
                continue

            # Newborn-инстинкт (Фрай): отметить рождение → тяга к GATHER/EAT
            # первые 500 тиков (см. _apply_newborn_instinct). Только mate-
            # рождённые трекаются — restored/seed (age>500) инстинкта не имеют.
            self._birth_tick[child_cid] = int(world_tick)
            self._carried_food[child_cid] = 0
            self._n_natural_newborn += 1  # verify: natural vs bootstrap ratio

            # 5. Compute child traits via parent crossover (ТЗ §4.1)
            # 9 полей: vision_radius, smell_radius, attack_radius, move_speed,
            #          attack_power, armor, efficiency, camel, diet_gene
            # Clamping (R1): защита диапазонов до отправки P40.
            child_traits = self._inherit_traits_for_newborn(
                mother, father, mother_cid=mother_cid, father_cid=father_cid)

            # 6. Generation = max(parents) + 1
            mother_gen = int(getattr(mother, "generation", 0))
            father_gen = int(getattr(father, "generation", 0))
            child_generation = max(mother_gen, father_gen) + 1
            try:
                child_org.generation = child_generation
            except Exception:
                pass

            # 7. Build envelope (projection-model schema, ТЗ §2)
            # species_id: Z2.b — add_creature уже назначил ребёнку species по
            # его genes (мог разойтись с родителями через topology overlay).
            # Шлём собственный вид ребёнка, fallback на родительский.
            child_species_id = (
                self.species_id.get(child_cid)
                or self.species_id.get(mother_cid)
                or self.species_id.get(father_cid)
                or 0
            )
            envelope = {
                "type": "newborn_announce",
                "child_cid": child_cid,
                "parent_cid": mother_cid,
                "parent2_cid": father_cid,
                "traits": child_traits,
                "species_id": int(child_species_id),
                "generation": child_generation,
                "ts_client": _time.time(),
            }

            # 6. Emit
            sent = False
            if embodied_client is not None:
                try:
                    sent = bool(embodied_client.send_state(envelope))
                except Exception as e:
                    logger.warning("embodied send newborn_announce failed: %s", e)
                    sent = False

            if sent:
                # 8. Register pending — ждём ack
                self._pending_newborn_envelopes[child_cid] = {
                    "parent_cid": mother_cid,
                    "parent2_cid": father_cid,
                    "generation": child_generation,
                    "ts_emit": _time.time(),
                }
                mark_mate_event(
                    self._last_mate_tick,
                    [mother_cid, father_cid],
                    world_tick,
                )
                born_cids.append(child_cid)
                logger.info(
                    "newborn_announce emitted cid=%s mother=%s father=%s",
                    child_cid, mother_cid, father_cid,
                )
            else:
                # Emit failed — rollback child creation
                logger.debug(
                    "newborn emit failed, removing local child %s", child_cid)
                try:
                    self.remove_creature(child_cid)
                except Exception:
                    pass

        return born_cids

    def handle_newborn_announce_ack(self, ack_payload: dict) -> bool:
        """Phase 4 F: process incoming newborn_announce_ack from P40.

        ack_payload schema (Хьюберт `server/embodied_newborn.py`):
            {
                "type": "newborn_announce_ack",
                "cid": str,
                "accepted": bool,
                "reason": str | None,
                "traits": {vision_radius, smell_radius, attack_radius,
                          move_speed, attack_power, armor, efficiency,
                          camel, diet_gene} | None,
                "ts_server": float,
            }

        Если `accepted=True` — apply traits, удалить из pending.
        Если `accepted=False` — `remove_creature(cid)` cleanup.

        Returns:
            True если ack обработан корректно, False иначе.
        """
        if not isinstance(ack_payload, dict):
            return False
        # Schema: новая projection-модель использует child_cid; legacy embodied
        # путь использовал cid. Принимаем оба.
        cid = str(ack_payload.get("child_cid") or ack_payload.get("cid") or "")
        if not cid:
            logger.debug("ack missing child_cid/cid: %s", ack_payload)
            return False

        pending = self._pending_newborn_envelopes.pop(cid, None)
        if pending is None:
            logger.debug("ack for unknown pending cid=%s", cid)
            # Не критично — может прийти повторно
            return False

        accepted = bool(ack_payload.get("accepted", False))
        if not accepted:
            reason = ack_payload.get("reason", "unknown")
            logger.warning(
                "newborn_announce rejected cid=%s reason=%s — cleanup",
                cid, reason)
            try:
                self.remove_creature(cid)
            except Exception as e:
                logger.warning("cleanup remove_creature %s failed: %s", cid, e)
            return True

        # accepted=True — apply traits if provided.
        # Evolved-traits recovery (30.05.2026): P40 ack несёт authoritative
        # (clamped) traits → пишем в стор `self.traits` + дублируем на
        # атрибуты организма (ingest_owned_traits делает оба). Стор теперь
        # переживает рестарт (save_state) и используется в crossover.
        traits = ack_payload.get("traits")
        org = self.organisms.get(cid)
        if traits and org is not None:
            try:
                applied = self.ingest_owned_traits(cid, traits)
                logger.info(
                    "newborn accepted cid=%s traits_applied=%d", cid, applied)
            except Exception as e:
                logger.warning("apply traits %s failed: %s", cid, e)
        else:
            logger.info("newborn accepted cid=%s (no traits)", cid)
        return True

    # ── Evolved-traits recovery (30.05.2026, Бендер) ─────────────────────
    # Closure Body Migration: тело (9 traits) переживает рестарт как мозг.
    # Согласовано Фрай (client authoritative по traits) + Хьюберт (owned-
    # handoff в seed/hello + accept-поверх-baseline на P40 self-heal).

    def _sanitize_traits(self, traits: dict) -> dict:
        """Привести входящие traits к 9 валидным полям в диапазонах _TRAIT_RANGES.

        Игнорирует лишние ключи, клампит значения, int-поля → int. Возвращает
        только валидные поля (частичный dict допустим — owned-handoff может
        нести подмножество)."""
        out: dict = {}
        if not isinstance(traits, dict):
            return out
        for name, (lo, hi) in self._TRAIT_RANGES.items():
            if name not in traits or traits[name] is None:
                continue
            try:
                val = float(traits[name])
            except (TypeError, ValueError):
                continue
            val = max(float(lo), min(float(hi), val))
            if isinstance(lo, int) and isinstance(hi, int):
                val = int(round(val))
            else:
                val = float(val)
            out[name] = val
        return out

    def ingest_owned_traits(self, cid: str, traits: dict,
                            overwrite: bool = True) -> int:
        """Записать traits в стор + дублировать на организм. Возвращает число
        фактически применённых полей.

        Источники и режим:
          - newborn-ack (handle_newborn_announce_ack) — overwrite=True
            (P40 authoritative для только что рождённого);
          - restore_persisted_state (.pt) — overwrite=True (свой evolved,
            стор пуст → режим без разницы);
          - owned-handoff P40 (seed_pack.meta / welcome.owned_traits_snapshot)
            — overwrite=False (FILL-ONLY): client authoritative по traits
            (принцип миграции). После P40 self-heal snapshot = baseline — НЕЛЬЗЯ
            затирать клиентский evolved. Заполняем только НЕизвестные поля
            (founders/never-persisted), известные — оставляем своими.

        Дублирование на loose-атрибуты сохранено для back-compat (тесты/
        проекция читают getattr).
        """
        clean = self._sanitize_traits(traits)
        if not clean:
            return 0
        cur = self.traits.get(cid)
        if not isinstance(cur, dict):
            cur = {}
            self.traits[cid] = cur
        applied: dict = {}
        for name, val in clean.items():
            if not overwrite and name in cur:
                continue  # client-authoritative: своё значение не трогаем
            cur[name] = val
            applied[name] = val
        org = self.organisms.get(cid)
        if org is not None:
            for name, val in applied.items():
                try:
                    setattr(org, name, val)
                except Exception:
                    pass
        return len(applied)

    def get_traits(self, cid: str) -> Optional[dict]:
        """Снимок traits особи из стора (copy), либо None если неизвестна."""
        t = self.traits.get(cid)
        return dict(t) if isinstance(t, dict) else None

    def build_traits_announce(self, cids=None) -> list[dict]:
        """Собрать `creatures`-список для batch traits_announce (existing owned).

        P40 после self-heal держит baseline → клиент возвращает evolved.
        Включаем только особей с известными traits в сторе (founders без
        handoff'а — пропуск, у P40 для них baseline корректный bootstrap).

        Schema items (контракт Хьюберта f673741): `{cid, traits{9}}` — БЕЗ
        generation/species_id (P40 держит их в CreatureState authoritative).
        Конверт собирает `build_traits_announce_envelope`.
        """
        if cids is None:
            cids = list(self.organisms.keys())
        out: list[dict] = []
        for cid in cids:
            traits = self.traits.get(cid)
            if not isinstance(traits, dict) or not traits:
                continue
            out.append({"cid": cid, "traits": dict(traits)})
        return out

    def build_traits_announce_envelope(self, cids=None) -> Optional[dict]:
        """Batch-конверт traits_announce для отправки по main ws (client→P40).

        Контракт Хьюберта f673741:
            {type: 'traits_announce', creatures: [{cid, traits{9}}, ...]}

        Возвращает None если нет ни одной особи с известными traits (нечего
        слать). Pending-учёт — отдельно через `mark_traits_announce_sent`
        (после фактической отправки), т.к. ws-send асинхронный."""
        creatures = self.build_traits_announce(cids=cids)
        if not creatures:
            return None
        return {"type": "traits_announce", "creatures": creatures}

    def mark_traits_announce_sent(self, cids) -> None:
        """Зарегистрировать pending для отправленных announce (ждём traits_ack).
        Эфемерное — снимается в `handle_traits_ack` по applied_cids."""
        import time as _time
        ts = _time.time()
        for cid in cids:
            self._pending_traits_announce[cid] = ts

    def handle_traits_ack(self, ack_payload: dict) -> int:
        """Process traits_ack от P40 (batch, контракт Хьюберта f673741).

        ack schema:
            {type: 'traits_ack', applied_cids: [cid,...],
             invalid: int, unknown: int}

        P40 применил evolved поверх baseline для `applied_cids` (authoritative
        теперь совпадает с тем, что клиент отправил — стор уже верный, доп.
        мёрж не нужен). Снимаем pending по applied_cids; invalid/unknown —
        warn для observability (несовпадение стора и server-side owner/range).

        Returns:
            Число cid'ов, снятых с pending.
        """
        if not isinstance(ack_payload, dict):
            return False  # type: ignore[return-value]
        applied = ack_payload.get("applied_cids") or []
        cleared = 0
        for cid in applied:
            if self._pending_traits_announce.pop(str(cid), None) is not None:
                cleared += 1
        invalid = int(ack_payload.get("invalid", 0) or 0)
        unknown = int(ack_payload.get("unknown", 0) or 0)
        if invalid or unknown:
            logger.warning(
                "traits_ack: applied=%d invalid=%d unknown=%d "
                "(invalid=range-reject, unknown=not-owned/absent)",
                cleared, invalid, unknown)
        else:
            logger.info("traits_ack: applied=%d", cleared)
        return cleared

    def _apply_biochem_mental_break(
        self, cid: str, world_tick: int = 0,
    ) -> None:
        """Hysteresis-aware update mental_break state.

        Pattern (тот же что server-side в `_update_creature_mechanics`):
          1. Декрементировать `mental_break_ticks` — пока > 0 текущий
             state удерживается (hold через `MENTAL_BREAK_DURATIONS`).
          2. Когда ticks <= 0 — recompute через `compute_mental_break`.
             Если возвращает новый state — set + reset ticks на duration.

        Это даёт hysteresis: cortisol-spike не «прыгает» в catatonic
        каждый тик; новый mental_break удерживается N тиков перед
        возможным переходом.
        """
        bc = self.biochem.get(cid)
        if bc is None:
            return
        try:
            from environment.biochemistry import (  # type: ignore
                MENTAL_BREAK_DURATIONS,
                compute_mental_break,
            )
        except Exception as e:
            logger.debug("biochem mental_break import failed cid=%s: %s",
                         cid, e)
            return
        try:
            # §1 RE-GATE (Фрай 04.06): держащийся колониальный mb (loner/pair-bond,
            # выставленный в restart-окне где single_organism ещё False) снять
            # НЕМЕДЛЕННО под single_organism — НЕ ждать истечения hysteresis-ticks.
            # Иначе loner держится N тиков и раздувает cortisol (3→39.6) →
            # конфаундит Phase-1 alignment-reader. loner у by-design одиночки —
            # колониальный артефакт, не реальный сигнал.
            if (self._single_organism
                    and bc.mental_break in _COLONIAL_MENTAL_BREAKS):
                bc.mental_break = ""
                bc.mental_break_ticks = 0
            # Hysteresis-hold (Фрай 04.06): под single_organism УБИРАЕМ — recompute
            # КАЖДЫЙ тик, чтобы держащийся stress-mb (catatonic от утихшего
            # blight/cold_snap) снимался НЕМЕДЛЕННО когда условие ушло (cort упал
            # <80), а не держался стейл N тиков. Флаг → проекция шлёт mb на P40 →
            # P40 mirror-force-STAY → Адам не форажит даже при client-гейте; stale-
            # hold блокирует. НЕ маскировка — условие реально изменилось (recompute
            # вернёт текущее). Колония — hysteresis как было (anti-flicker).
            if (not self._single_organism) and bc.mental_break_ticks > 0:
                bc.mental_break_ticks -= 1
                return
            new_state = compute_mental_break(bc, world_tick)
            # §1 (Фрай): под single_organism гейтим колониально-специфичные
            # mental_break (loner/pair-bond) — для одиночки always-true/
            # бессодержательны, сбивают solo-поведение. Не маскируем стресс.
            if self._single_organism and new_state in _COLONIAL_MENTAL_BREAKS:
                new_state = ""
            if new_state != bc.mental_break:
                bc.mental_break = new_state
                # duration = 0 для "" (normal) → выйдет из hold сразу,
                # перерасчёт каждый тик пока not normal.
                bc.mental_break_ticks = int(
                    MENTAL_BREAK_DURATIONS.get(new_state, 0))
        except Exception as e:
            logger.debug("biochem mental_break update cid=%s: %s", cid, e)

    def _maybe_force_stay(self, cid: str, out: dict) -> None:
        """Override action на STAY если биохимия требует.

        Server-side equivalent: `client_actions_batch` обрабатывает
        `should_force_stay()` и подменяет action на STAY=4. На клиенте
        делаем то же самое перед формированием actions_batch к P40.

        Triggers (см. `environment.biochemistry.should_force_stay`):
          - `glucose < 5.0` (обморок от истощения)
          - `mental_break == "catatonic"` (выгорание)
          - `mental_break == "exhaustion"` (истощение)

        Side-effect: out[cid] mutируется в-place. Безопасно — out это
        локальный dict handle_tick'а.
        """
        bc = self.biochem.get(cid)
        if bc is None or cid not in out:
            return
        # §3 paralysis (single-organism): пока паралич не снят — не движется.
        # «Осознаёт, но не движется» — обучающий сигнал. Проверяем ПЕРВЫМ:
        # paralysis независим от биохимического force_stay.
        if cid in self._paralysis_until:
            if time.monotonic() < self._paralysis_until[cid]:
                out[cid] = {"action": STAY, "target_id": None}
                return
        try:
            from environment.biochemistry import should_force_stay  # type: ignore
        except Exception:
            return
        try:
            if should_force_stay(bc):
                # §3 (Фрай 04.06, инвариант recoverable-не-absorbing): под
                # single_organism ВЕСЬ biochem force-STAY (catatonic/exhaustion/
                # glucose<5-faint) АБСОРБИРУЕТ — force-STAY → не форажит → energy/
                # glucose 0 → стресс → force-STAY держит (deep trough re-absorbing,
                # даже после recovery). Гейчу ВСЕ (СИГНАЛ mb/cortisol/glucose
                # ОСТАЁТСЯ — не маскируем). Адам форажит когда истощён/стрессован
                # → relief (eat→glucose/energy↑) → размыкание. §3-паралич (выше,
                # energy≤0, 3с) — единственный легитимный force-STAY (с recovery).
                # Для §3-immortal одиночки biochem-lock net-вреден (нет death-исхода).
                if not self._single_organism:
                    out[cid] = {"action": STAY, "target_id": None}
        except Exception as e:
            logger.debug("biochem force_stay cid=%s: %s", cid, e)

    def _apply_biochem_events(self, cid: str, event: "Optional[dict]") -> None:
        """Apply event-driven biochem deltas из per-cid event_dict.

        Минимальный set который покрывают server-side fields из
        `_compute_immediate_reward`:
          - `ate=True` → `apply_feed(creature)` (dopamine +5, glucose +10)
          - `killed=True` → `apply_kill_prey(creature)` (dopamine +8).
            Note: client не знает lineage жертвы → assume prey (не PvP).
            Если жертва была zodchiy, P40 не пришлёт `killed=True` без
            доп. поля; для безопасности этот path — только prey.
          - `damage_taken>0` → `apply_pvp_hit(creature, kind="cross_clan_target")`.
            Conservative default — cortisol +2. Реальное разделение
            fratricide vs cross_clan требует attacker lineage в event,
            которого сейчас нет; добавится в Phase 2 этап 4.5 при
            расширении event schema.

        Не покрыты в этом этапе (требуют cross-cid data / новых fields):
          - apply_mate_pair (нужен partner cid)
          - apply_share_food (нужен donor/recipient pair)
          - apply_signal_received (нужно channel поле)
          - apply_predator_visible / apply_panic_low_energy (нужно flag
            что предатор в visible зоне)
          - apply_drink (нужно DRINK event поле)

        Safety: try/except — никогда не ломает handle_tick.
        """
        if event is None:
            return
        bc = self.biochem.get(cid)
        if bc is None:
            return
        try:
            from environment.biochemistry import (  # type: ignore
                apply_feed, apply_kill_prey, apply_pvp_hit,
            )
        except Exception as e:
            logger.debug("biochem apply import failed cid=%s: %s", cid, e)
            return
        try:
            delta_e = float(event.get("delta_energy", 0.0) or 0.0)
            if event.get("ate"):
                apply_feed(bc)
            elif delta_e > 0.5:
                # Phase 4 fix 0.11.5 (29.05): P40 шлёт delta_energy>0 (organism
                # ест физически) но без ate=True flag → apply_feed не вызывался
                # → glucose decay → 0 → catatonic → reproduction блокирован.
                # Питание = и energy И glucose: при положительном delta_energy
                # без явного ate всё равно восполняем glucose (apply_feed).
                apply_feed(bc)
            if event.get("killed"):
                apply_kill_prey(bc)
                self._skill_kill[cid] = self._skill_kill.get(cid, 0) + 1  # F5
            # F5 skill-growth (Фрай): счётчик еды (ate или income energy>0) —
            # опыт фуражёра → efficiency растёт каждые 200 тиков.
            if event.get("ate") or delta_e > 0.0:
                self._skill_eat[cid] = self._skill_eat.get(cid, 0) + 1
            damage = float(event.get("damage_taken", 0.0) or 0.0)
            if damage > 0:
                apply_pvp_hit(bc, kind="cross_clan_target")
            # Phase 4 fix 0.11.2 (28.05.2026): apply delta_energy от P40 eat/move
            # events. Раньше delta_energy шёл только в reward signal, biochem.energy
            # был изолирован → особи не размножались + умирали голодом.
            # Server-side max_energy = 1000 (WorldConfig default), clamp [0, 1000].
            if delta_e != 0.0:
                bc.energy = max(0.0, min(1000.0, float(bc.energy) + delta_e))
                if delta_e > 0.0:
                    self._e_income_sum += delta_e  # ENERGY_CALIB income
                else:
                    # §3.5-ПОЛНОТА (Фрай 07.06): ОТРИЦАТЕЛЬНЫЙ delta_energy =
                    # серверная per-event цена (атаки/combat/действия) — списывал
                    # energy, но в ledger НЕ попадал (только delta>0 шёл в income).
                    # Это и есть «непосчитанная метаб-цена berserk/атак»: LEDGER_FULL
                    # вскрыл residual≈−250/окно при attack=140-202, dmg=0. Теперь в
                    # cost → net_comp = истинный метаб-баланс (net+ перестанет врать).
                    self._e_cost_sum += -delta_e
                    self._e_srv_cost_sum += -delta_e  # отдельный диаг-аккумулятор
            # Hydration income (31.05.2026): присутствие ключа delta_hydration
            # в event = P40 шлёт питьё → активируем hydration-ось отбора для
            # этого cid (thirst-декей + жажда-смерть включатся в
            # _apply_metabolism). delta_hydration>0 = попил (income).
            if "delta_hydration" in event:
                self._hydration_active.add(cid)
                delta_h = float(event.get("delta_hydration", 0.0) or 0.0)
                if delta_h > 0.0:
                    self._hyd_drink_sum += delta_h  # calib: income питья
                if delta_h != 0.0:
                    max_h = float(getattr(bc, "max_hydration", 100.0))
                    bc.hydration = max(
                        0.0, min(max_h, float(getattr(bc, "hydration", max_h)) + delta_h))
                # Вода лечит инфекцию (01.06.2026, Фрай; mirror world.py:3552):
                # питьё + severity<0.5 → 30% шанс полного выздоровления.
                if delta_h > 0.0:
                    _sev = float(getattr(bc, "infection_severity", 0.0) or 0.0)
                    if 0.0 < _sev < 0.5 and random.random() < 0.3:
                        bc.infection_severity = 0.0
                        bc.infected = False
            # Infection contact (01.06.2026, Фрай): owned-инфекцию ВЕДЁТ КЛИЕНТ
            # (P40 phase-out). Заражение от соседа — P40 шлёт infected/
            # infection_contact → начальная severity 0.05 (mirror world.py).
            # Прогресс severity + death тикает _apply_metabolism.
            _ic = event.get("infection_contact")
            if _ic or event.get("infected"):
                # _ic: список [{from_cid, severity_hint}] (Хьюберт). Берём max
                # severity_hint (несколько контактов за тик), дефолт 0.05.
                sev_hint = 0.05
                if isinstance(_ic, list) and _ic:
                    try:
                        sev_hint = max(
                            float(d.get("severity_hint", 0.05) or 0.05)
                            for d in _ic if isinstance(d, dict))
                    except Exception:
                        sev_hint = 0.05
                _s = float(getattr(bc, "infection_severity", 0.0) or 0.0)
                bc.infection_severity = max(_s, sev_hint)
                bc.infected = True
        except Exception as e:
            logger.debug("biochem apply events cid=%s: %s", cid, e)

    def _skill_growth_step(self, cid: str) -> None:
        """Lamarckian skill-growth (F5, 01.06.2026, Фрай; mirror world.py:4595).
        Каждые 200 тиков из опыта (eat/kill/move counts): efficiency/attack_power/
        move_speed растут/падают. P40 phase-out owned skill-growth → клиент ведёт.
        Внутрижизненная эволюция тела → наследуется crossover'ом. Изменённые
        traits → _skill_changed_cids (re-announce P40)."""
        tr = self.traits.get(cid)
        eat = self._skill_eat.get(cid, 0)
        kill = self._skill_kill.get(cid, 0)
        move = self._skill_move.get(cid, 0)
        atk_use = self._skill_atk.get(cid, 0)  # §6: melee-ATTACK в окне
        if tr is not None:
            try:
                eff0 = int(tr.get("efficiency", 10))  # server default 10 (не 5)
                atk0 = int(tr.get("attack_power", 3))
                spd0 = int(tr.get("move_speed", 3))
                eff, atk, spd = eff0, atk0, spd0
                # Growth
                if eat > 10:
                    eff = min(15, eff + 1)
                # §6 attack_power от боевого использования (melee-ATTACK) ИЛИ
                # киллов. atk_use клиент-надёжен (kill дропается на §3.5 tick-skip).
                # Порог φ-выровнен: ~13 melee-атак/окно или kill≥φ-порог.
                if (kill >= max(2, round(atk / 1.6180339887))
                        or atk_use >= 13):
                    atk = min(10, atk + 1)
                if move > 100:
                    spd = min(10, spd + 1)
                # Decay (attack_power НЕ деградирует — cold-start trap)
                if eat <= 2:
                    eff = max(5, eff - 1)
                if move < 30:
                    spd = max(2, spd - 1)
                if (eff, atk, spd) != (eff0, atk0, spd0):
                    tr["efficiency"], tr["attack_power"], tr["move_speed"] = (
                        eff, atk, spd)
                    self._skill_changed_cids.add(cid)
                # SKILL_DIAG (§6 predator_defense, Фрай 07.06): видимость роста
                # трейтов от использования. atk растёт от киллов (φ-порог), spd
                # от move+FLEE. move_speed>3 → бегство обгоняет хищника (speed=3).
                if self._single_organism:
                    logger.info(
                        "SKILL_DIAG cid=%s atk=%d->%d move_speed=%d->%d eff=%d "
                        "| kill=%d atk_use=%d eat=%d move=%d", cid, atk0, atk,
                        spd0, spd, eff, kill, atk_use, eat, move)
            except Exception as e:
                logger.debug("skill_growth %s: %s", cid, e)
        self._skill_eat[cid] = 0
        self._skill_kill[cid] = 0
        self._skill_atk[cid] = 0
        self._skill_move[cid] = 0

    def _apply_metabolism(self, cid: str, rates: "Optional[dict]") -> None:
        """Body Migration метаболизм (31.05.2026, Бендер; контракт Хьюберт).

        Client-authoritative энергозатрата/гидрация/теломеры. P40 phase-out
        перестал тикать owned-zodchiy (energy=500/age=0 forever → death-триггеры
        молчали, колония росла 9× сверх ёмкости). Теперь тело целиком на client:
        P40 шлёт effective per-tick rates (формулы single-source, зависят от
        night/season/biome — client их не знает), client интегрирует state.

        rates = {step_cost_now, telomere_decay_now, thirst_now}. income
        (delta_energy от еды) уже добавлен в _apply_biochem_events. Здесь cost +
        death-check: energy<=0 (голод) / telomere в фазе AGONY (старость) →
        cid в _dead_cids → build_projection_batch шлёт alive=False → P40 убирает
        (естественный отбор; client сообщает объективную смерть, не «убивает»).
        """
        if not rates:
            return
        # Contract per-sec (Хьюберт): rate в energy/СЕК; интегрируем по wall-clock
        # dt между applies → energy -= rate × dt. Tick-mismatch уходит независимо
        # от client TPS (handle_tick ~раз в 2.4с при dworld~36). dt клемпуется,
        # чтобы reconnect-разрыв не убил разом; первый apply — только засечь время.
        # LEGACY (P40 ещё шлёт *_now): dt=1 → per-apply, старое поведение без
        # over-drain. Авто-апгрейд на dt-интеграцию когда P40 пришлёт *_per_sec.
        _per_sec = bool(rates.get("_per_sec"))
        _now = time.time()
        _last_wall = self._last_metab_wall.get(cid)
        self._last_metab_wall[cid] = _now
        # §3.5 АСИНХРОННЫЕ ТЕМПЫ (Фрай 06.06, body_migration.md): для owned-
        # организма (Адам) тик клиента = ЕГО метаболизм, НЕ синхрон с миром
        # (черепаха/гепард). Income копится per-client-tick (свой темп). Cost
        # ДОЛЖЕН быть на той же тиковой шкале — иначе step_cost жжётся по wall-dt
        # («темп гепарда»), а ест в своём → дренаж к §3 (баг, не «доход мал»).
        # Фикс: step_cost_per_tick (Хьюберт: = step_cost_per_sec / world_TPS,
        # сервер знает TPS) применяем per-apply БЕЗ wall-dt. Пока сервер не шлёт
        # per-tick — fallback на wall-dt (колонии/legacy, поведение не меняется).
        _tick_metab = (self._single_organism
                       and rates.get("step_cost_per_tick") is not None)
        if _tick_metab:
            dt = 1.0  # per-client-tick: cost на шкале income (свой метаболизм)
        elif _per_sec:
            if _last_wall is None:
                return
            dt = min(_MAX_METAB_DT, max(0.0, _now - _last_wall))
            if dt <= 0.0:
                return
        else:
            dt = 1.0  # legacy *_now: per-apply (rate as-is), без dt-масштаба
        self._metab_applies += 1
        self._metab_dt_sum += dt
        bc = self.biochem.get(cid)
        org = self.organisms.get(cid)
        try:
            if _tick_metab:
                # §3.5: per-client-tick rate, применяется один раз за apply (БЕЗ
                # wall-dt). Та же шкала, что income → per-tick net = серверный.
                step_cost = float(rates.get("step_cost_per_tick", 0.0) or 0.0)
                thirst = float(rates.get("thirst_per_tick", 0.0) or 0.0)
                tel_decay = float(rates.get("telomere_decay_per_tick", 0.0) or 0.0)
                self._metab_sc_sum += step_cost     # METAB_DIAG (теперь per-tick)
                self._metab_tick_n = getattr(self, "_metab_tick_n", 0) + 1
                # DAMAGE-канал (Фрай 07.06): predator damage_per_tick → energy
                # per-client-tick (§3.5-симметрично, без drop). ×_damage_factor
                # (калибровка мал→расти). §3 = защита от смерти.
                if "damage_per_tick" in rates:
                    self._dmg_key_n = getattr(self, "_dmg_key_n", 0) + 1
                _dmg_rate = float(rates.get("damage_per_tick", 0.0) or 0.0)
                _dmg = _dmg_rate * float(self._damage_factor)
                if _dmg_rate > 0.0:
                    self._dmg_rate_sum += _dmg_rate
                    self._dmg_apply_n += 1
            else:
                # *_per_sec (контракт Хьюберта); ws_client нормализует *_now→
                # *_per_sec на переходный период (fallback), чтобы не было rate=0.
                _sc_rate = float(rates.get("step_cost_per_sec", 0.0) or 0.0)
                self._metab_sc_sum += _sc_rate      # per-sec rate (METAB_DIAG)
                step_cost = _sc_rate * dt           # energy/сек × сек (wall-dt)
                thirst = float(rates.get("thirst_per_sec", 0.0) or 0.0) * dt
                tel_decay = float(rates.get("telomere_decay_per_sec", 0.0) or 0.0) * dt
                _dmg = 0.0                           # damage только per-tick (Адам)
        except (TypeError, ValueError):
            return
        if bc is not None:
            if step_cost > 0.0:
                bc.energy = max(0.0, float(getattr(bc, "energy", 0.0)) - step_cost)
                self._e_cost_sum += step_cost  # ENERGY_CALIB
            # DAMAGE-канал: применяем predator-урон к авторитетной energy
            # (per-client-tick, §3.5). Идёт в _e_cost_sum → ENERGY_CALIB net
            # учитывает урон. §3 ловит energy≤0 → паралич, не смерть.
            if _dmg > 0.0:
                bc.energy = max(0.0, float(getattr(bc, "energy", 0.0)) - _dmg)
                self._e_cost_sum += _dmg
                self._dmg_sum += _dmg
            # Glucose→energy конверсия (Фрай 04.06): излишек glucose (>baseline 50,
            # от плотной еды) → energy. Делает экономику выигрываемой при хорошей
            # добыче. glucose потребляется, но не ниже baseline. rate=0 → no-op.
            _ger = float(self._glucose_energy_rate)
            if _ger > 0.0:
                _g = float(getattr(bc, "glucose", 0.0))
                _surplus = _g - 50.0  # baseline_glucose
                if _surplus > 0.0:
                    _conv = min(_ger * _surplus * dt, _surplus)
                    bc.energy = min(1000.0, float(getattr(bc, "energy", 0.0)) + _conv)
                    bc.glucose = _g - _conv
                    self._e_income_sum += _conv  # ENERGY_CALIB (доход)
            # Водный контур client-authoritative (01.06.2026, Шеф). Расход
            # жажды теперь БЕЗУСЛОВНЫЙ для всех owned-zodchiy (как на сервере):
            # доход (delta_hydration) теперь client-side из террейна
            # (ws_client._near_water), не зависит от P40 → gate _hydration_active
            # больше не нужен. Аккумулируем для calib-лога баланса.
            if thirst > 0.0:
                max_h = float(getattr(bc, "max_hydration", 100.0))
                bc.hydration = max(
                    0.0, min(max_h, float(getattr(bc, "hydration", max_h)) - thirst))
                self._hyd_thirst_sum += thirst
            # Обезвоживание → стадийный урон ЭНЕРГИИ (mirror world.py:3569):
            # dh_stage>=2 → energy -= energy_drain (φ²/φ³). Смерть наступает
            # органически через energy<=0 (starvation, см. death-check ниже),
            # не отдельной hydration-смертью. Gate: включаем урон только после
            # подтверждения дохода (drink_sum>0) — урок 0.11.24.
            if self._dehydration_damage_enabled:
                max_h = float(getattr(bc, "max_hydration", 100.0))
                stage = _dehydration_stage(
                    float(getattr(bc, "hydration", max_h)), max_h)
                drain = _DEHYDRATION_DRAIN.get(stage, 0.0) * dt  # per-sec × dt
                if drain > 0.0:
                    bc.energy = max(
                        0.0, float(getattr(bc, "energy", 0.0)) - drain)
        if org is not None and tel_decay > 0.0:
            try:
                org.telomere = max(0.0, min(
                    1.0, float(getattr(org, "telomere", 1.0)) - tel_decay))
            except Exception:
                pass
        # Infection (01.06.2026, Фрай): owned-инфекцию ТИКАЕТ КЛИЕНТ (P40
        # phase-out перестаёт убивать owned от infection). Mirror world.py:
        # 4574-4588 (per-sec × dt, как остальные оси — Хьюберт): severity
        # +0.15/сек (cap 1.0), energy -= 60/сек × severity. Затрагивает ТОЛЬКО
        # заражённых; вода лечит; contact от соседа — P40 шлёт infected event.
        if bc is not None:
            _sev = float(getattr(bc, "infection_severity", 0.0) or 0.0)
            if _sev > 0.0:
                _sev = min(1.0, _sev + _INFECTION_SEVERITY_PER_SEC * dt)
                bc.infection_severity = _sev
                _idrain = _sev * _INFECTION_DRAIN_PER_SEC * dt
                bc.energy = max(
                    0.0, float(getattr(bc, "energy", 0.0)) - _idrain)
                self._e_infdrain_sum += _idrain  # ENERGY_CALIB
        # §3 paralysis (single-organism, контракт Фрая): owned energy+смерть
        # client-authoritative. Под флагом energy≤0 — НЕ смерть, а паралич ~3с
        # (осознаёт, motor=STAY в _maybe_force_stay), через N клиент сам даёт
        # recovery (+45 energy). Agony/infection Адама НЕ убивают (персистентный;
        # P40 immortality бэкстопит свои death-вектора). Цикл паралич→recovery→
        # (нет еды)→паралич держит мозг живым (stuck, не dead); если зацикливает
        # — поднимаю self._recovery_energy.
        if self._single_organism:
            if bc is not None:
                now_m = time.monotonic()
                until = self._paralysis_until.get(cid)
                if until is not None and now_m >= until:
                    # Конец паралича → recovery-грант (ЭНЕРГИЯ, не время).
                    self._paralysis_until.pop(cid, None)
                    bc.energy = float(self._recovery_energy)
                    self._paralysis_window_n += 1  # §3.5-ledger: грант искажает Δ
                    # Фрай-инвариант (genuine response → recoverable, НЕ absorbing):
                    # recovery = «передышка» → снимает не только паралич, но и
                    # стресс-залипание catatonic (cortisol застревает на 99.5 от
                    # хронического голода, ×0.995-декей не пробивает ре-спайк →
                    # absorbing-STAY). На recovery облегчаем cortisol/serotonin →
                    # catatonic self-limiting: Адам выходит в активное окно и
                    # снова фуражит. НЕ маскировка (recovery — реальное событие
                    # relief), а размыкание absorbing-петли.
                    try:
                        bc.cortisol = float(getattr(bc, "baseline_cortisol", 10.0))
                        bc.serotonin = max(
                            float(getattr(bc, "serotonin", 0.0)),
                            float(getattr(bc, "baseline_serotonin", 50.0)))
                        # Glucose-relief (Фрай 04.06): recovery восстанавливает и
                        # glucose до baseline — иначе deep-trough (glucose→0) после
                        # recovery держит glucose<5 faint → re-absorbing. Recovery =
                        # полная «передышка» (energy+cort+ser+mb+glucose).
                        bc.glucose = max(
                            float(getattr(bc, "glucose", 0.0)),
                            float(getattr(bc, "baseline_glucose", 50.0)))
                        # Infection-relief (Фрай 08.06, вариант b): recovery чистит
                        # и infection_severity — иначе absorbing-петля (recovery
                        # снимает energy+стресс+mb, но инфекция ре-дренит сразу →
                        # mb=inflammation возвращается → §3-цикл вечно; water лечит
                        # ТОЛЬКО при severity<0.5 + water-seek гатнут → у Адама
                        # severity~1.0 вода бессильна, recovery-клир единственный
                        # путь). Инфекция остаётся ЖИВОЙ механикой (заражение тикает),
                        # recovery её снимает как стресс — §3=«полная передышка».
                        bc.infection_severity = 0.0
                        bc.infected = False
                        bc.mental_break = ""
                        bc.mental_break_ticks = 0
                    except Exception:
                        pass
                    logger.info("paralysis recovery cid=%s -> energy=%.1f "
                                "(+стресс-relief)", cid, self._recovery_energy)
                elif until is None and float(getattr(bc, "energy", 1.0)) <= 0.0:
                    # Триггер 1: голод (energy≤0). Триггер 2 (death_suppressed
                    # от P40, energy-независим) — в handle_tick.
                    self._enter_paralysis(cid, "starved")
        else:
            # Death-check (колониальный режим): голод (energy<=0) + старость
            # (telomere AGONY) + инфекция (severity>=1.0). Жажда/инфекция грызут
            # energy → starvation. Единый death-envelope (Фрай: градиентно).
            dead = bool(bc is not None and float(getattr(bc, "energy", 1.0)) <= 0.0)
            cause = "starvation" if dead else ""
            if not dead and bc is not None and float(
                    getattr(bc, "infection_severity", 0.0) or 0.0) >= 1.0:
                dead = True
                cause = "infection"
            if not dead and org is not None:
                try:
                    from core.telomere_phase import get_phase, TelomerePhase
                    if get_phase(float(getattr(org, "telomere", 1.0))) == TelomerePhase.AGONY:
                        dead = True
                        cause = "agony"
                except Exception:
                    pass
            if dead and cid not in self._dead_cids:
                self._dead_cids.add(cid)
                # deaths-by-cause для colony_summary (agony=telomere-фаза).
                _bucket = "telomere" if cause == "agony" else cause
                if _bucket in self._deaths_by_cause:
                    self._deaths_by_cause[_bucket] += 1
                logger.info(
                    "metabolic death cid=%s cause=%s energy=%.1f hydration=%.1f "
                    "telomere=%.3f infection=%.2f",
                    cid, cause, float(getattr(bc, "energy", 0.0)) if bc else -1.0,
                    float(getattr(bc, "hydration", -1.0)) if bc else -1.0,
                    float(getattr(org, "telomere", -1.0)) if org else -1.0,
                    float(getattr(bc, "infection_severity", 0.0)) if bc else -1.0)
        # Water calibration лог (31.05.2026): раз в ~300 тиков сводка баланса
        # income(drink)/cost(thirst) + распределение hydration. Из тренда
        # (drink≈thirst → баланс; drink<<thirst → income мал) калибруем.
        self._hyd_calib_ticks += 1
        if self._hyd_calib_ticks >= 300:
            try:
                hyds = [float(getattr(b, "hydration", 0.0))
                        for b in self.biochem.values()]
                n = len(hyds)
                hmin = min(hyds) if hyds else 0.0
                hmax = max(hyds) if hyds else 0.0
                hmean = (sum(hyds) / n) if n else 0.0
                nlow = sum(1 for h in hyds if h < 10.0)
                logger.info(
                    "WATER_CALIB ticks=300 active=%d thirst_sum=%.1f "
                    "drink_sum=%.1f ratio=%.2f | hydration n=%d min=%.1f "
                    "mean=%.1f max=%.1f low(<10)=%d",
                    len(self._hydration_active), self._hyd_thirst_sum,
                    self._hyd_drink_sum,
                    (self._hyd_drink_sum / self._hyd_thirst_sum
                     if self._hyd_thirst_sum > 0 else -1.0),
                    n, hmin, hmean, hmax, nlow)
                # ENERGY_CALIB (Фрай): income(eat) vs cost(step+infection) +
                # efficiency. net>0 → колония растёт; net<0 → вымирание.
                effs = [int((self.traits.get(c) or {}).get("efficiency", 10))
                        for c in self.organisms]
                eff_mean = (sum(effs) / len(effs)) if effs else 0.0
                _net = self._e_income_sum - self._e_cost_sum - self._e_infdrain_sum
                # §3.5-ПОЛНОТА (Фрай 07.06): net_true = реальная Δenergy за окно
                # (authoritative bc.energy, не врёт); residual = net_true − net_comp
                # = непосчитанные расходы/гранты (berserk/атаки/дегидрация/§3-грант).
                # Автотюн ДОЛЖЕН читать net_true, не компонентный net.
                _e_now = -1.0
                try:
                    for _bcv in self.biochem.values():
                        _e_now = float(getattr(_bcv, "energy", -1.0))
                        break  # single-organism: один Адам
                except Exception:
                    pass
                _net_true = (_e_now - self._e_window_e0
                             if (self._e_window_e0 >= 0.0 and _e_now >= 0.0)
                             else _net)
                _residual = _net_true - _net
                logger.info(
                    "LEDGER_FULL net_comp=%.1f net_true=%.1f residual=%.1f "
                    "srv_cost=%.1f paralysis=%d e0=%.1f e1=%.1f (srv_cost=серверная "
                    "per-event цена delta_e<0, теперь в ledger; residual→0+гранты "
                    "после фикса; paralysis>0 = §3-несустейнабельно)",
                    _net, _net_true, _residual, self._e_srv_cost_sum,
                    self._paralysis_window_n, self._e_window_e0, _e_now)
                # INSTINCT_DIAG (Фрай): natural-роды vs bootstrap-омоложение +
                # сколько особей сейчас в активном инстинкте. Критерий успеха:
                # natural>0 (пошли роды) + eff_mean ползёт 5→10. Если держится
                # ТОЛЬКО на bootstrap (natural=0) — scaffold лечит симптом.
                _instinct_active = sum(1 for c in self._birth_tick
                                       if c in self.organisms)
                logger.info(
                    "ENERGY_CALIB ticks=300 income=%.1f cost=%.1f infdrain=%.1f "
                    "net=%.1f ratio=%.2f eff_mean=%.1f natural=%d bootstrap=%d "
                    "instinct_active=%d",
                    self._e_income_sum, self._e_cost_sum, self._e_infdrain_sum,
                    _net, (self._e_income_sum /
                           max(self._e_cost_sum + self._e_infdrain_sum, 1e-6)),
                    eff_mean, self._n_natural_newborn, self._n_bootstrap_rejuv,
                    _instinct_active)
                # colony_summary history-точка (downsampled, шаг 300 тиков).
                _ratio = (self._e_income_sum /
                          max(self._e_cost_sum + self._e_infdrain_sum, 1e-6))
                _n_alive = sum(1 for c in self.organisms
                               if c not in self._dead_cids)
                _n_sp = len({self.species_id.get(c) for c in self.organisms
                             if c not in self._dead_cids
                             and self.species_id.get(c) is not None})
                _pt = {"t": int(self._last_world_tick), "alive": _n_alive,
                       "inc": round(self._e_income_sum, 1),
                       "cost": round(self._e_cost_sum, 1),
                       "net": round(_net, 1), "ratio": round(_ratio, 3),
                       "eff": round(eff_mean, 2), "sp": _n_sp}
                self._summary_history.append(_pt)
                self._last_window = _pt
                # NAV_DIAG (Фрай): навигация к еде. onf_rate низкий → не доходят
                # до флоры; flip_rate высокий → motor_policy перебивает shaping
                # (→ нужен bias_scale curriculum); gather_onf≈0 → нечего собрать.
                _nt = max(1, self._nav["ticks"])
                logger.info(
                    "NAV_DIAG ticks=%d onf_rate=%.3f gather=%d gather_onf=%d "
                    "eat=%d p40_ate=%d flip_rate=%.3f motor_norm=%.3f",
                    self._nav["ticks"], self._nav["onf"] / _nt,
                    self._nav["gather"], self._nav["gather_onf"],
                    self._nav["eat"], self._nav["p40_ate"],
                    self._nav["flip"] / _nt, self._nav["mnorm"] / _nt)
                # YIELD_GATE_DIAG (Фрай 06.06, read-only): механизм-замер «доедай»-
                # гейта. yield_fire_rate=доля тиков recent_yield=True (загорается ли
                # гейт?). gather/eat/move/stay%=action-разбор. cf=carried_food (копит
                # ли в склад?). g/e=glucose/energy (у пола?). Проверяет гипотезу:
                # доминантный GATHER (в склад) + гейт ждёт delta_energy от passive-EAT
                # → гейт почти не активен → бег не гасится → cost держится.
                _g_snap = _e_snap = -1.0
                try:
                    for _bcv in self.biochem.values():
                        _e_snap = float(getattr(_bcv, "energy", -1.0))
                        _g_snap = float(getattr(_bcv, "glucose", -1.0))
                        break  # single-organism: один Адам
                except Exception:
                    pass
                _mb_snap = ""
                try:
                    for _bcv in self.biochem.values():
                        _mb_snap = str(getattr(_bcv, "mental_break", "")
                                       or getattr(_bcv, "mb", "") or "")
                        break
                except Exception:
                    pass
                _nmv = max(1, self._nav["nav_moves"])
                logger.info(
                    "YIELD_GATE ticks=%d yield_fire_rate=%.3f gather_pct=%.3f "
                    "eat_pct=%.3f move_pct=%.3f stay_pct=%.3f nav_hit_rate=%.3f "
                    "nav_moves=%d cf=%d cf_p40=%d g=%.1f e=%.1f mb=%s "
                    "park=%.0f voice=%.2f",
                    self._nav["ticks"], self._nav["yield_fire"] / _nt,
                    self._nav["gather"] / _nt, self._nav["eat"] / _nt,
                    self._nav["move"] / _nt, self._nav["stay"] / _nt,
                    self._nav["nav_hit"] / _nmv, self._nav["nav_moves"],
                    self._nav["cf_last"], self._nav["cf_p40_seen"],
                    _g_snap, _e_snap, _mb_snap or "-",
                    self._motor_park_test, self._motor_voice)
                # METAB_DIAG (Хьюберт ×2): skip_rate≈0.5 → подтверждает дубли
                # server-тика (handle_tick ×2). applies = реальные server-тики.
                _ma = max(1, self._metab_applies)
                _tick_n = getattr(self, "_metab_tick_n", 0)
                logger.info(
                    "METAB_DIAG applies=%d mean_rate=%.4f mean_dt=%.2f "
                    "mean_drain_per_apply=%.4f tick_mode=%d/%d",
                    self._metab_applies, self._metab_sc_sum / _ma,
                    self._metab_dt_sum / _ma,
                    (self._metab_sc_sum / _ma) * (self._metab_dt_sum / _ma),
                    _tick_n, self._metab_applies)
                # DAMAGE_DIAG (Фрай 07.06): predator-давление + применённый урон.
                # dmg_applied=Σ урона к energy за окно; pred_ticks=тиков под атакой;
                # mean_rate=средний raw damage_per_tick; factor=калибровка.
                _na = max(1, self._nav["attack"])
                _nfl = max(1, self._nav["flee"])
                logger.info(
                    "DAMAGE_DIAG dmg_applied=%.1f pred_ticks=%d mean_rate=%.4f "
                    "factor=%.3f key_seen=%d/%d attack=%d flee=%d "
                    "atk_pp=%.3f atk_contact=%d flee_pp=%.3f",
                    self._dmg_sum, self._dmg_apply_n,
                    self._dmg_rate_sum / max(1, self._dmg_apply_n),
                    self._damage_factor,
                    getattr(self, "_dmg_key_n", 0), self._metab_applies,
                    self._nav["attack"], self._nav["flee"],
                    self._nav["atk_pp_sum"] / _na, self._nav["atk_contact"],
                    self._nav["flee_pp_sum"] / _nfl)
                self._metab_applies = 0
                self._metab_dt_sum = 0.0
                self._metab_sc_sum = 0.0
                self._metab_tick_n = 0
                self._dmg_sum = 0.0
                self._dmg_rate_sum = 0.0
                self._dmg_apply_n = 0
                self._dmg_key_n = 0
            except Exception as e:
                logger.debug("WATER_CALIB log failed: %s", e)
            self._hyd_thirst_sum = 0.0
            self._hyd_drink_sum = 0.0
            self._hyd_calib_ticks = 0
            self._e_income_sum = 0.0
            self._e_cost_sum = 0.0
            self._e_infdrain_sum = 0.0
            # §3.5-ledger: новое окно стартует с текущей energy (net_true база).
            try:
                for _bcv in self.biochem.values():
                    self._e_window_e0 = float(getattr(_bcv, "energy", -1.0))
                    break
            except Exception:
                self._e_window_e0 = -1.0
            self._paralysis_window_n = 0
            self._e_srv_cost_sum = 0.0
            self._nav = {"ticks": 0, "onf": 0, "gather": 0, "gather_onf": 0,
                         "eat": 0, "flip": 0, "mnorm": 0.0, "p40_ate": 0,
                         "yield_fire": 0, "move": 0, "stay": 0, "cf_last": 0,
                         "cf_p40_seen": 0, "nav_hit": 0, "nav_moves": 0, "attack": 0, "flee": 0, "atk_pp_sum": 0.0, "atk_contact": 0, "flee_pp_sum": 0.0}

    def _apply_biochem_decay(self, cid: str) -> None:
        """Тиковый passive update 8 веществ + mental_break baseline-decay.

        Вызывается из handle_tick per-cid loop. Требует чтобы:
          1. `self.biochem[cid]` существует (zodchiy, инициализирован в
             add_creature → make_default).
          2. Server-side зависимости (energy, hydration, infected,
             infection_severity, pair_bond_strength, last_social_tick)
             синхронизированы из obs_batch в ws_client._handle_obs_batch.
             Без sync decay работает на дефолтах (energy=100, hydration=100)
             — biochem становится biased, но не ломает.

        Math equivalence с server: `environment.biochemistry.decay_step`
        вызывается напрямую, тот же код P40 и client.

        В Phase 2 commit 4 здесь же добавится apply_* для events_per_cid
        (PvP/feed/share/mate); commit 5 — compute_mental_break override.
        """
        bc = self.biochem.get(cid)
        if bc is None:
            return
        try:
            from environment.biochemistry import decay_step  # type: ignore
            from .biochemistry import _FakeWorld
        except Exception as e:
            logger.debug("biochem decay import failed cid=%s: %s", cid, e)
            return
        # Loner cortisol-gate (Фрай 04.06): у by-design одиночки oxytocin всегда→0
        # (растёт только от clan-proximity/mating, которых нет) → decay_step
        # генерит loner-cortisol по oxytocin<10 → ХРОНИЧЕСКИЙ фоновый стресс
        # (Адам никогда не un-alone → cort без разрешения). 0.12.1 гейтил force-STAY
        # эффект + mb-флаг, но cortisol-ГЕНЕРАЦИЯ осталась. Колониальный артефакт
        # (canonical colonial-cue → gate). Floor oxytocin ДО decay_step (≥20,
        # нейтрально: loner<10 И pair-bond>75 оба ложны) → условие loner ложно →
        # нет loner-cortisol. НЕ маскировка — для solo социо-изоляция бессмысленна.
        if self._single_organism and bc is not None:
            try:
                if float(getattr(bc, "oxytocin", 0.0)) < 20.0:
                    bc.oxytocin = 20.0
            except Exception:
                pass
        try:
            decay_step(bc, _FakeWorld())
        except Exception as e:
            logger.debug("biochem decay_step failed cid=%s: %s", cid, e)
        # Phase 4 fix 0.11.6 (29.05): glucose floor energy-coupled (вариант 1,
        # одобрен Фраем). Физиология: глюкоза поддерживается из энергозапасов
        # (гомеостаз). Если organism сыт (energy высокая, P40 держит ~500),
        # glucose не падает ниже baseline → нет ложного catatonic при
        # energy=500/glucose=0 рассогласовании.
        #
        # Floor = baseline (не max): glucose ВЫШЕ baseline decay'ит нормально
        # (короткоживущая динамика сохранена), floor лишь не даёт упасть
        # <baseline при high energy. Вытаскивает существующих catatonic
        # (genesis g=0 → baseline → mental_break recompute снимет catatonic).
        try:
            energy = float(getattr(bc, "energy", 0.0))
            if energy >= _GLUCOSE_FLOOR_ENERGY_THRESHOLD:
                baseline_g = float(getattr(bc, "baseline_glucose", 50.0))
                if float(getattr(bc, "glucose", 0.0)) < baseline_g:
                    bc.glucose = baseline_g
        except Exception as e:
            logger.debug("glucose floor cid=%s: %s", cid, e)
        # Phase 4 fix 0.11.7 (29.05, Шеф одобрил путь 1): cortisol-гомеостаз.
        # cortisol — единственный гормон без baseline-clearance → монотонно
        # копится (damage от Старших → +2, без recovery) → catatonic lock.
        # Симметрия с dopamine_decay (unconditional). Физиология: гормон
        # стресса очищается (печень/почки), не вечный.
        # 0.995 = half-life ~144 тика (F12). Equilibrium при continuous stress
        # (+0.2/tick): 0.2/0.005 = ~40 < catatonic threshold 80 → сигнал
        # стресса держится, но ложный lock уходит. Recover existing catatonic
        # к ~40 за ~144 тика.
        try:
            _cort_decay = _CORTISOL_HOMEOSTASIS_DECAY  # 0.995 базовый гомеостаз
            # BERSERK cortisol-relief exit (Фрай 08.06, инвариант «нет залипших
            # mental_break»): decay_step даёт relief ×0.98 ТОЛЬКО catatonic
            # (biochemistry.py:474) → berserk был ABSORBING (cortisol не уходил →
            # berserk вечен → форсил холостые ATTACK, srv_cost 300). Зеркалим
            # catatonic-relief на берсерк client-side: при mb=berserk усиленный
            # decay ×0.98 → cortisol спадает ниже berserk-порога → выходит, спам
            # прекращается сам. Источник пина (ЖАЖДА: hydration~16 server-side,
            # water-halo OFF Phase 2 — Хьюберт убирает параллельно) при выходимом
            # берсерке даёт cortisol реально упасть. berserk НЕ-absorbing, как все mb.
            if str(getattr(bc, "mental_break", "")) == "berserk":
                _cort_decay = 0.98
            bc.cortisol = float(getattr(bc, "cortisol", 0.0)) * _cort_decay
        except Exception as e:
            logger.debug("cortisol homeostasis cid=%s: %s", cid, e)

    # ── TZ B Phase 2 (26.05.2026, Бендер): per-role Hebbian metrics ─────

    def _record_hebbian_per_tissue_sample(self, cid: str) -> None:
        """Записать per-role learning sample для cid после Hebbian/SFNN update.

        Hook в `handle_tick` per-cid loop. Источники (приоритет SFNN над
        classic — у одной роли учится ровно один путь за update):
          1. SFNN basic: `self.basic_tissue_sfnn_trace[role][cid]` — Tensor
          2. SFNN higher (+ zodchiy): `self.higher_tissue_sfnn_trace[role][cid]`
             — dict synapse → Tensor; берём сумму norms.
          3. Classic Hebbian: `heb._tissue_info[].trace` для ролей вне SFNN
             (когда `basic_sfnn_off` — 9 базовых учатся классикой).

        Per-role: `n_total += 1`, `n_learning += 1` если |norm| > epsilon,
        `delta_sum += norm`. Алиасы ('hippocampus' и др. вне 20 known)
        тихо игнорируются. heb is None (NEAT-only creature) — classic путь
        пропущен, SFNN traces всё равно проверены.

        Safety: try/except — никогда не ломает handle_tick.
        """
        try:
            heb = self.hebbian.get(cid)
            seen_roles: set[str] = set()

            # 1) SFNN basic (10 ролей: один Tensor per role per cid).
            for role in _BASIC_SFNN_TISSUES:
                trace = self.basic_tissue_sfnn_trace.get(role, {}).get(cid)
                if trace is None:
                    continue
                try:
                    norm = float(trace.norm().item())
                except Exception:
                    continue
                self._heb_pt_n_total[role] += 1
                if norm > _HEB_PT_EPSILON:
                    self._heb_pt_n_learning[role] += 1
                self._heb_pt_delta_sum[role] += norm
                self._heb_pt_samples[role] += 1
                seen_roles.add(role)

            # 2) SFNN higher + zodchiy (10 ролей: dict synapse → Tensor).
            for role in _HIGHER_SFNN_TISSUES + _ZODCHIY_EXTRA_TISSUES:
                trace_dict = (self.higher_tissue_sfnn_trace
                              .get(role, {}).get(cid))
                if not isinstance(trace_dict, dict) or not trace_dict:
                    continue
                try:
                    norm_sum = sum(
                        float(t.norm().item()) for t in trace_dict.values()
                        if t is not None
                    )
                except Exception:
                    continue
                self._heb_pt_n_total[role] += 1
                if norm_sum > _HEB_PT_EPSILON:
                    self._heb_pt_n_learning[role] += 1
                self._heb_pt_delta_sum[role] += norm_sum
                self._heb_pt_samples[role] += 1
                seen_roles.add(role)

            # 3) Classic Hebbian fallback для ролей вне SFNN (например basic
            # когда `basic_sfnn_off`). `_tissue_info[].trace` — eligibility
            # trace, обновляемая внутри HebbianController.update().
            if heb is not None:
                for info in getattr(heb, "_tissue_info", []):
                    role = info.get('role')
                    if not role or role in seen_roles:
                        continue
                    if role not in self._heb_pt_n_total:
                        continue  # роль вне 20 known (например 'hippocampus')
                    trace = info.get('trace')
                    norm = 0.0
                    if trace is not None:
                        try:
                            norm = float(trace.norm().item())
                        except Exception:
                            norm = 0.0
                    self._heb_pt_n_total[role] += 1
                    if norm > _HEB_PT_EPSILON:
                        self._heb_pt_n_learning[role] += 1
                    self._heb_pt_delta_sum[role] += norm
                    self._heb_pt_samples[role] += 1
        except Exception as e:
            logger.debug("hebbian_per_tissue record failed cid=%s: %s", cid, e)

    def _build_hebbian_per_tissue_snapshot(self) -> dict:
        """Построить snapshot per-role metrics и сбросить accumulators.

        Возвращает `{role: {n_learning, n_total, delta_mean}}` для всех
        20 ролей. Если samples=0 в окне — все нули, но ключ присутствует
        (backward compat для merge-агрегации на VPS).

        После вызова все 4 accumulator-dict'а обнуляются — следующий emit
        цикл стартует с чистого листа (per-period semantics).
        """
        snap: dict[str, dict] = {}
        for role in _HEB_PT_ALL_ROLES:
            samples = self._heb_pt_samples[role]
            delta_mean = (round(self._heb_pt_delta_sum[role] / samples, 6)
                          if samples > 0 else 0.0)
            snap[role] = {
                "n_learning": int(self._heb_pt_n_learning[role]),
                "n_total": int(self._heb_pt_n_total[role]),
                "delta_mean": delta_mean,
            }
        # EEG-ring sample (#2, 01.06.2026): нормированная активность per role
        # → ring (один срез на snapshot). runmax с медленным декеем (0.97) →
        # вариативность волны (не флэтлайн на максимуме).
        try:
            sample = []
            for role in _HEB_PT_ALL_ROLES:
                dm = float(snap[role]["delta_mean"])
                rm = max(dm, self._tissue_act_runmax.get(role, 0.0) * 0.97)
                self._tissue_act_runmax[role] = rm
                sample.append(round(min(1.0, dm / rm), 4) if rm > 1e-9 else 0.0)
            self._tissue_activation_ring.append(sample)
        except Exception as e:
            logger.debug("tissue_activation_ring sample failed: %s", e)
        for role in _HEB_PT_ALL_ROLES:
            self._heb_pt_n_total[role] = 0
            self._heb_pt_n_learning[role] = 0
            self._heb_pt_delta_sum[role] = 0.0
            self._heb_pt_samples[role] = 0
        return snap

    def _build_tissue_activation_ring_payload(self) -> dict:
        """EEG-осциллограф payload для /stats (#2). {tissues, data,
        window_size} — нормированные [0,1] срезы активности тканей во времени."""
        return {
            "tissues": list(_HEB_PT_ALL_ROLES),
            "data": list(self._tissue_activation_ring),
            "window_size": self._tissue_activation_ring.maxlen,
        }

    # ── Diagnostics aggregation ──────────────────────────────────────────

    def _dump_state(self) -> dict:
        """Расширенная диагностика для admin: размеры всех state-dict'ов
        + sample по первому живому cid (предиктор params count, loss_ema,
        intrinsic_ema, prev_obs.shape).
        """
        sample = None
        if self.organisms:
            first_cid = next(iter(self.organisms.keys()))
            pred = self.predictor.get(first_cid)
            prev = self.prev_obs.get(first_cid)
            sample = {
                "cid": first_cid,
                "has_predictor": pred is not None,
                "predictor_params": (
                    sum(p.numel() for p in pred.parameters())
                    if pred is not None else 0
                ),
                "has_prev_obs": prev is not None,
                "prev_obs_shape": (
                    list(prev.shape) if prev is not None else None
                ),
                "loss_ema": float(self.loss_ema.get(first_cid, 0.0)),
                "intrinsic_ema": float(self.intrinsic_ema.get(first_cid, 0.0)),
                "entropy_ema": float(self.entropy_ema.get(first_cid, 0.0)),
                "trace_norm_ema": float(self.trace_norm_ema.get(first_cid, 0.0)),
            }
        return {
            "sizes": {
                "organisms": len(self.organisms),
                "predictor": len(self.predictor),
                "predictor_opt": len(self.predictor_opt),
                "prev_obs": len(self.prev_obs),
                "loss_ema": len(self.loss_ema),
                "intrinsic_ema": len(self.intrinsic_ema),
                "entropy_ema": len(self.entropy_ema),
                "trace_norm_ema": len(self.trace_norm_ema),
                "reward_var_ema": len(self.reward_var_ema),
                "hebbian": len(self.hebbian),
                "action_selectors": len(self.action_selectors),
            },
            "counters": {
                "predictor_steps": int(self.predictor_steps),
                "hebbian_updates": int(self.hebbian_updates),
            },
            "sample": sample,
        }

    def _growth_graph_counts(self, org=None) -> tuple:
        """§6 рост (09.06): (закреплено, отвергнуто) ИЗ ГРАФА — durable, не из
        волатильного счётчика _growth_kept (тот обнуляется на рестарте → врал «0»
        в /stats, хотя связи живы). Адам стартует topo=0 → ВСЕ topology_genes
        выращены петлёй: enabled = закреплённые связи, disabled = откатанные.
        Граф персистит в .pt → всегда отражает реальность, переживает сбои питания.
        org=None → сумма по всем организмам (агрегат); иначе по одному."""
        orgs = [org] if org is not None else list(self.organisms.values())
        kept = reverted = 0
        for o in orgs:
            for g in (getattr(o, "tissue_topology_genes", []) or []):
                if getattr(g, "enabled", False):
                    kept += 1
                else:
                    reverted += 1
        return kept, reverted

    def diagnostics(self) -> dict:  # noqa: C901
        """Снимок метрик обучения для /api/diagnostics/training.

        Поля, которые знает только клиент (P40 их не видит):
          - Phase 1/2/6 — Forward Model / intrinsic / self-observable
          - architecture — n_embd/n_layer/n_head гистограммы по организмам
          - learning_genes — lr_oja, lr_reward, trace_decay, % hebbian_enabled
          - phase4 — specialization_avg по ролям тканей (Hebbian attribution)
        """
        from core.action_selector import N_ACTIONS
        n = len(self.organisms)
        # client_tps — handle_tick rate (общий, не per-cid).
        client_tps = 0.0
        if len(self.tick_ts) >= 2:
            dt = self.tick_ts[-1] - self.tick_ts[0]
            if dt > 0:
                client_tps = round((len(self.tick_ts) - 1) / dt, 2)
        if n == 0:
            return {
                "n_alive": 0,
                "prediction_accuracy": 0.0,
                "prediction_loss_avg": 0.0,
                "intrinsic_reward_avg": 0.0,
                "intrinsic_reward_last_avg": 0.0,
                "entropy_avg": 0.0,
                "trace_norm_avg": 0.0,
                "reward_var_avg": 0.0,
                "hebbian_updates_total": int(self.hebbian_updates),
                "predictor_steps_total": int(self.predictor_steps),
                "client_tps": client_tps,
                "s2_beta_local_avg": 0.0,
                "s2_imag_mult_avg": 0.0,
                "s2_stress_avg": 0.0,
                "s2_planner_norm_avg": 0.0,
                "s2_dmn_floor_avg": 0.0,
                "s2_tom_acc_avg": 0.0,
                "s2_lang_acc_avg": 0.0,
                "s2_dopa_td_avg": 0.0,
                "s2_action_dist_avg": [0.0] * N_ACTIONS,
                "motor_delta_norm_avg": 0.0,
                "motor_sfnn_steps_total": int(self.motor_sfnn_steps),
                "motor_sfnn_eta_avg": 0.0,
                "motor_sfnn_A_avg": 0.0,
                "motor_sfnn_T_avg": 0.0,
                "motor_sfnn_enabled_pct": 0.0,
                # SFNN S3.diag (14.05.2026): per-tissue для 7 высших.
                # Z3.B (17.05.2026, Зодчий): расширено на 3 zodchiy-ткани
                # (cerebellum/amygdala/episodic). Для не-Зодчих cid'ов
                # `higher_tissue_sfnn_steps` по этим ключам остаётся 0.
                "higher_sfnn": {
                    "enabled_pct": 0.0,
                    **{t: {"steps_total": int(
                                self.higher_tissue_sfnn_steps.get(t, 0)),
                             "eta_avg": 0.0,
                             "A_avg": 0.0}
                       for t in (_HIGHER_SFNN_TISSUES
                                 + _ZODCHIY_EXTRA_TISSUES)},
                },
                # SFNN S6.8 (16.05.2026): per-role для 10 базовых.
                "basic_sfnn": {
                    "enabled_pct": 0.0,
                    **{r: {"steps_total": int(
                                self.basic_tissue_sfnn_steps.get(r, 0)),
                             "eta_avg": 0.0,
                             "A_avg": 0.0,
                             "tau_avg": 0.0,
                             "w_norm_avg": 0.0}
                       for r in _BASIC_SFNN_TISSUES},
                },
                # Z7.i.h (17.05.2026, Зодчий): per-tissue snapshot-аггрегаты
                # для 3 sidecar-тканей. n_active=0 пока нет живых Зодчих.
                "zodchiy_sidecar": {
                    "n_active": 0,
                    "cerebellum_delta_norm_avg": 0.0,
                    "amygdala_valence_avg": 0.0,
                    "episodic_recall_norm_avg": 0.0,
                },
                # TZ B Phase 2 (Бендер, 26.05.2026): per-role Hebbian metrics
                # для UI "Тренировка мозга". Пустой stub: 20 ролей с нулями
                # (ключи присутствуют — backward compat для VPS merge).
                "hebbian_per_tissue": {
                    r: {"n_learning": 0, "n_total": 0, "delta_mean": 0.0}
                    for r in _HEB_PT_ALL_ROLES
                },
                "tissue_activation_ring": self._build_tissue_activation_ring_payload(),
                # Body Migration Phase 2 (Бендер, 27.05.2026): client-side
                # биохимия snapshot. Пустой stub: счётчики 0, distribution
                # пуст.
                "biochem": {
                    "n_active": 0,
                    "mental_break_counts": {},
                    "cortisol_avg": 0.0,
                    "serotonin_avg": 0.0,
                    "dopamine_avg": 0.0,
                    "oxytocin_avg": 0.0,
                    "adrenaline_avg": 0.0,
                    "glucose_avg": 0.0,
                    "fatigue_avg": 0.0,
                    "histamine_avg": 0.0,
                },
                "tom_steps_total": int(self.tom_steps),
                "lang_steps_total": int(self.lang_steps),
            }
        # Phase 1 — predictor accuracy.
        loss_vals = []
        for cid, hist in self.pred_loss_history.items():
            if hist:
                loss_vals.append(sum(hist) / len(hist))
        if loss_vals:
            avg_loss = sum(loss_vals) / len(loss_vals)
            pred_acc = round(float(math.exp(-avg_loss)), 4)
            pred_loss_avg = round(float(avg_loss), 5)
        else:
            pred_acc = 0.0
            pred_loss_avg = 0.0
        # Phase 2 — intrinsic.
        intr_emas = list(self.intrinsic_ema.values())
        intr_lasts = list(self.intrinsic_last.values())
        # Phase 6.
        ents = list(self.entropy_ema.values())
        tns = list(self.trace_norm_ema.values())
        rvs = list(self.reward_var_ema.values())

        # Architecture — гистограммы n_embd/n_layer/n_head по организмам.
        # Внутри одной ткани все cells имеют одинаковый genome — берём первую.
        n_embd_hist: dict[int, int] = {}
        n_layer_hist: dict[int, int] = {}
        n_head_hist: dict[int, int] = {}
        for org in list(self.organisms.values()):
            tissues = getattr(org, "tissues", None)
            if not tissues:
                continue
            try:
                first_tissue = next(iter(tissues.values()))
                first_cell = next(iter(first_tissue.cells.values()))
                ne = int(first_cell.n_embd)
                nl = int(first_cell.n_layer)
                nh = int(first_cell.n_head)
            except (StopIteration, AttributeError):
                continue
            n_embd_hist[ne] = n_embd_hist.get(ne, 0) + 1
            n_layer_hist[nl] = n_layer_hist.get(nl, 0) + 1
            n_head_hist[nh] = n_head_hist.get(nh, 0) + 1

        # Learning genes — средние по HebbianController.config.
        lr_ojas: list[float] = []
        lr_rewards: list[float] = []
        trace_decays: list[float] = []
        n_heb_enabled = 0
        for ctrl in self.hebbian.values():
            if ctrl is None:
                continue
            n_heb_enabled += 1
            cfg = getattr(ctrl, "config", None)
            if cfg is None:
                continue
            lr_ojas.append(float(getattr(cfg, "lr_oja", 0.0)))
            lr_rewards.append(float(getattr(cfg, "lr_reward", 0.0)))
            trace_decays.append(float(getattr(cfg, "eligibility_decay", 0.0)))

        def _avg(vals: list[float]) -> float:
            return round(sum(vals) / len(vals), 6) if vals else 0.0

        # Phase 4 — specialization_avg агрегат по ролям.
        # S6.11+ (16.05.2026): с `basic_tissue_sfnn_enabled=True` и
        # `higher_tissue_sfnn_enabled=True` классический heb.update пропускает
        # все 10 базовых + 7 высших + 3 sidecar Зодчего, traces в
        # _tissue_info[trace] не растут и tissue_specialization() возвращает
        # {}. Реальные traces теперь живут в basic/higher_tissue_sfnn_trace.
        # Считаем долю каждой роли по сумме L2-норм SFNN-trace по всем cids,
        # с fallback на классический trace для совместимости (если SFNN off).
        spec_sums: dict[str, float] = {}
        # 1) Basic tissues (10 базовых ролей organism graph).
        for role, per_cid in self.basic_tissue_sfnn_trace.items():
            for trace in per_cid.values():
                if trace is None:
                    continue
                try:
                    spec_sums[role] = (spec_sums.get(role, 0.0)
                                        + float(trace.norm().item()))
                except Exception:
                    continue
        # 2) Higher tissues (7 высших + 3 Зодчий sidecar): trace per synapse.
        for role, per_cid in self.higher_tissue_sfnn_trace.items():
            for trace_dict in per_cid.values():
                if not isinstance(trace_dict, dict):
                    continue
                for trace in trace_dict.values():
                    if trace is None:
                        continue
                    try:
                        spec_sums[role] = (spec_sums.get(role, 0.0)
                                            + float(trace.norm().item()))
                    except Exception:
                        continue
        # 3) Fallback на классику (если SFNN off — собираем как раньше).
        if not spec_sums:
            counts: dict[str, int] = {}
            for ctrl in self.hebbian.values():
                if ctrl is None or not hasattr(ctrl, "tissue_specialization"):
                    continue
                try:
                    shares = ctrl.tissue_specialization()
                except Exception:
                    continue
                for role, share in shares.items():
                    spec_sums[role] = spec_sums.get(role, 0.0) + float(share)
                    counts[role] = counts.get(role, 0) + 1
            specialization_avg = {
                role: round(spec_sums[role] / counts[role], 4)
                for role in spec_sums
                if counts[role] > 0
            }
        else:
            # Нормализуем в доли (сумма == 1.0 при наличии traces).
            total = sum(spec_sums.values())
            if total < 1e-9:
                specialization_avg = {r: 0.0 for r in spec_sums}
            else:
                specialization_avg = {
                    r: round(v / total, 4) for r, v in spec_sums.items()
                }

        # Brain migration: средние по высшим тканям S2.E/G/A/F.
        beta_vals = list(self.last_beta_local.values())
        imag_vals = list(self.last_imag_mult.values())
        stress_vals = list(self.last_stress.values())
        planner_norms: list[float] = []
        for delta in self.last_planner_delta.values():
            try:
                planner_norms.append(float(delta.norm().item()))
            except Exception:
                continue
        s2_beta_avg = round(sum(beta_vals) / len(beta_vals), 4) if beta_vals else 0.0
        s2_imag_avg = round(sum(imag_vals) / len(imag_vals), 4) if imag_vals else 0.0
        s2_stress_avg = round(sum(stress_vals) / len(stress_vals), 4) if stress_vals else 0.0
        s2_planner_avg = round(sum(planner_norms) / len(planner_norms), 4) if planner_norms else 0.0
        dmn_vals = list(self.last_dmn_floor.values())
        s2_dmn_avg = round(sum(dmn_vals) / len(dmn_vals), 6) if dmn_vals else 0.0
        # Phase 7 — motor_policy delta norms.
        motor_norms: list[float] = []
        for delta in self.last_motor_delta.values():
            try:
                motor_norms.append(float(delta.norm().item()))
            except Exception:
                continue
        motor_delta_avg = (round(sum(motor_norms) / len(motor_norms), 4)
                            if motor_norms else 0.0)
        # S2.B — theory_of_mind accuracy avg.
        tom_vals = list(self.last_tom_acc.values())
        s2_tom_avg = (round(sum(tom_vals) / len(tom_vals), 4)
                       if tom_vals else 0.0)
        # S2.C — language accuracy avg.
        lang_vals = list(self.last_lang_acc.values())
        s2_lang_avg = (round(sum(lang_vals) / len(lang_vals), 4)
                        if lang_vals else 0.0)
        # Phase 5d — TD-modulation avg (NEOL reward-gated Hebbian).
        td_vals = list(self.dopamine_td.values())
        s2_dopa_td_avg = (round(sum(td_vals) / len(td_vals), 6)
                           if td_vals else 0.0)
        action_dists: dict[str, list[float]] = {}
        bias_max: dict[str, float] = {}
        for cid, sel in self.action_selectors.items():
            stats = sel.get_stats()
            action_dists[cid] = [
                round(float(v), 4) for v in stats["action_distribution"]
            ]
            bias_max[cid] = round(float(stats.get("bias_max_abs", 0.0)), 4)
        if action_dists:
            sums = [0.0] * N_ACTIONS
            for dist in action_dists.values():
                for i, v in enumerate(dist):
                    sums[i] += v
            n_sel = len(action_dists)
            s2_action_dist_avg = [round(v / n_sel, 4) for v in sums]
        else:
            s2_action_dist_avg = [0.0] * N_ACTIONS
        s2_bias_max_avg = (
            round(sum(bias_max.values()) / len(bias_max), 4)
            if bias_max else 0.0
        )

        # SFNN S1.3 — средние η и A по 6 типам синапсов motor_policy
        # каждой особи, агрегированные по колонии. Растущая дисперсия η
        # между особями = эволюция правил активна (см. tz_sfnn_migration.md).
        sfnn_etas: list[float] = []
        sfnn_As: list[float] = []
        sfnn_Ts: list[float] = []
        sfnn_enabled_n = 0
        for c, rule in self.motor_sfnn_rule.items():
            if rule is None:
                continue
            try:
                sfnn_etas.append(float(rule.mean_eta()))
                sfnn_As.append(float(rule.mean_A()))
                sfnn_Ts.append(float(rule.temperature))
            except Exception:
                continue
            org_c = self.organisms.get(c)
            if bool(getattr(getattr(org_c, "genome", None),
                              "sfnn_enabled", False)):
                sfnn_enabled_n += 1
        motor_sfnn_eta_avg = (round(sum(sfnn_etas) / len(sfnn_etas), 6)
                                if sfnn_etas else 0.0)
        motor_sfnn_A_avg = (round(sum(sfnn_As) / len(sfnn_As), 4)
                              if sfnn_As else 0.0)
        # S5 (15.05.2026): средняя T pre-tanh делителя motor_forward.
        motor_sfnn_T_avg = (round(sum(sfnn_Ts) / len(sfnn_Ts), 4)
                              if sfnn_Ts else 0.0)
        motor_sfnn_enabled_pct = round(sfnn_enabled_n / max(1, n), 3)

        # SFNN S3.diag (14.05.2026): per-tissue агрегация для 7 высших.
        # Те же 3 метрики что у motor_sfnn (steps/eta_avg/A_avg), но bucketed
        # по 7 типам тканей. enabled_pct — единый флаг для всех 7.
        higher_sfnn_enabled_n = 0
        for c in self.organisms:
            org_c = self.organisms.get(c)
            if bool(getattr(getattr(org_c, "genome", None),
                              "higher_tissue_sfnn_enabled", False)):
                higher_sfnn_enabled_n += 1
        # Z3.B (17.05.2026, Зодчий): обходим 7 высших + 3 zodchiy-ткани.
        # rule_store / steps по `_ZODCHIY_EXTRA_TISSUES` инициализированы для
        # всех cid'ов через `_ALL_HIGHER` (строки 269+), но реально заполнены
        # только у тех, кому в `add_creature(lineage="zodchiy")` пришёл
        # `SFNNRule.for_role(_t)`. Для не-Зодчих etas/As останутся пустыми →
        # eta_avg/A_avg = 0.0, steps_total = 0.
        higher_sfnn_per_tissue: dict[str, dict] = {}
        for t in _HIGHER_SFNN_TISSUES + _ZODCHIY_EXTRA_TISSUES:
            etas: list[float] = []
            As: list[float] = []
            for rule in self.higher_tissue_sfnn_rule.get(t, {}).values():
                if rule is None:
                    continue
                try:
                    etas.append(float(rule.mean_eta()))
                    As.append(float(rule.mean_A()))
                except Exception:
                    continue
            higher_sfnn_per_tissue[t] = {
                "steps_total": int(self.higher_tissue_sfnn_steps.get(t, 0)),
                "eta_avg": (round(sum(etas) / len(etas), 6)
                              if etas else 0.0),
                "A_avg": (round(sum(As) / len(As), 4)
                            if As else 0.0),
            }
        higher_sfnn_block: dict = {
            "enabled_pct": round(higher_sfnn_enabled_n / max(1, n), 3),
            **higher_sfnn_per_tissue,
        }

        # SFNN S6.8 (16.05.2026): per-role агрегация для 10 базовых тканей.
        # eta_avg/A_avg по правилам, tau_avg отдельно, w_norm_avg — Frobenius
        # ‖W_sub‖ по cell.input_proj/output_proj (regression check на отлёт).
        basic_sfnn_enabled_n = 0
        for c in self.organisms:
            org_c = self.organisms.get(c)
            if bool(getattr(getattr(org_c, "genome", None),
                              "basic_tissue_sfnn_enabled", False)):
                basic_sfnn_enabled_n += 1
        # Сначала пройдём по hebbian._tissue_info чтобы достать W-нормы.
        # Структура: per-role список Frobenius norms.
        torch = self._torch
        basic_w_norms: dict[str, list[float]] = {
            r: [] for r in _BASIC_SFNN_TISSUES}
        for c, heb in self.hebbian.items():
            if heb is None:
                continue
            for info in getattr(heb, "_tissue_info", []):
                role = info.get('role')
                if role not in _BASIC_SFNN_TISSUES:
                    continue
                cell = info.get('cell')
                algo = info.get('algorithm')
                try:
                    if algo == 'oja_input' and hasattr(cell, 'input_proj'):
                        W = cell.input_proj.weight.data
                    elif algo == 'reward_output' and hasattr(cell, 'output_proj'):
                        W = cell.output_proj.weight.data[:16, :]
                    else:
                        continue
                    basic_w_norms[role].append(float(W.norm().item()))
                except Exception:
                    continue
        basic_sfnn_per_tissue: dict[str, dict] = {}
        for r in _BASIC_SFNN_TISSUES:
            etas: list[float] = []
            As: list[float] = []
            taus: list[float] = []
            for rule in self.basic_tissue_sfnn_rule.get(r, {}).values():
                if rule is None:
                    continue
                try:
                    etas.append(float(rule.mean_eta()))
                    As.append(float(rule.mean_A()))
                    taus.append(float(rule.tau))
                except Exception:
                    continue
            wnorms = basic_w_norms.get(r, [])
            basic_sfnn_per_tissue[r] = {
                "steps_total": int(self.basic_tissue_sfnn_steps.get(r, 0)),
                "eta_avg": (round(sum(etas) / len(etas), 6)
                              if etas else 0.0),
                "A_avg": (round(sum(As) / len(As), 4)
                            if As else 0.0),
                "tau_avg": (round(sum(taus) / len(taus), 3)
                              if taus else 0.0),
                "w_norm_avg": (round(sum(wnorms) / len(wnorms), 4)
                                 if wnorms else 0.0),
            }
        basic_sfnn_block: dict = {
            "enabled_pct": round(basic_sfnn_enabled_n / max(1, n), 3),
            **basic_sfnn_per_tissue,
        }

        # Z7.i.h (17.05.2026, Зодчий): colony-level аггрегаты по 3 sidecar-
        # тканям. n_active = количество cid'ов с заполненными snapshot'ами
        # (forward уже прошёл). Норму считаем L2 от вектора, valence — float.
        cer_norms: list[float] = []
        for cer_t in self.last_cerebellum_delta.values():
            try:
                cer_norms.append(float(cer_t.norm().item()))
            except Exception:
                continue
        amy_vals: list[float] = []
        for amy_v in self.last_amygdala_valence.values():
            try:
                amy_vals.append(float(amy_v))
            except Exception:
                continue
        epi_norms: list[float] = []
        for epi_t in self.last_episodic_recall.values():
            try:
                epi_norms.append(float(epi_t.norm().item()))
            except Exception:
                continue
        # n_active = объединение cid'ов, у которых заполнена хотя бы одна
        # Зодчий-снапшот-ткань. Обычно все три у одной особи.
        zodchiy_cids = (set(self.last_cerebellum_delta.keys())
                         | set(self.last_amygdala_valence.keys())
                         | set(self.last_episodic_recall.keys()))
        zodchiy_block: dict = {
            "n_active": len(zodchiy_cids),
            "cerebellum_delta_norm_avg": (
                round(sum(cer_norms) / len(cer_norms), 4)
                if cer_norms else 0.0),
            "amygdala_valence_avg": (
                round(sum(amy_vals) / len(amy_vals), 4)
                if amy_vals else 0.0),
            "episodic_recall_norm_avg": (
                round(sum(epi_norms) / len(epi_norms), 4)
                if epi_norms else 0.0),
        }

        # §6 рост (09.06): «закреплено»/«отвергнуто» из ГРАФА (durable), НЕ из
        # волатильного счётчика (тот обнулялся на рестарте → врал «0»). См.
        # _growth_graph_counts.
        _gk, _gr = self._growth_graph_counts()

        return {
            "n_alive": n,
            "n_predictors": len(self.predictor),
            "n_prev_obs": len(self.prev_obs),
            # Рост мозга при жизни (§6): закреплённых/откатанных связей + в полёте.
            "tissue_topology_mutations_total": _gk + _gr,
            "growth_kept": _gk,
            "growth_reverted": _gr,
            "growth_in_flight": len(self._growth_state),
            "prediction_accuracy": pred_acc,
            "prediction_loss_avg": pred_loss_avg,
            "intrinsic_reward_avg": (
                round(sum(intr_emas) / len(intr_emas), 6) if intr_emas else 0.0
            ),
            "intrinsic_reward_last_avg": (
                round(sum(intr_lasts) / len(intr_lasts), 6) if intr_lasts else 0.0
            ),
            "entropy_avg": round(sum(ents) / len(ents), 4) if ents else 0.0,
            "trace_norm_avg": round(sum(tns) / len(tns), 4) if tns else 0.0,
            "reward_var_avg": round(sum(rvs) / len(rvs), 6) if rvs else 0.0,
            "hebbian_updates_total": int(self.hebbian_updates),
            "predictor_steps_total": int(self.predictor_steps),
            "architecture": {
                "n_embd_hist": {str(k): v for k, v in n_embd_hist.items()},
                "n_layer_hist": {str(k): v for k, v in n_layer_hist.items()},
                "n_head_hist": {str(k): v for k, v in n_head_hist.items()},
            },
            "learning_genes": {
                "lr_oja_avg": _avg(lr_ojas),
                "lr_reward_avg": _avg(lr_rewards),
                "trace_decay_avg": _avg(trace_decays),
                "hebbian_enabled_pct": round(n_heb_enabled / max(1, n), 3),
            },
            "phase4": {
                "specialization_avg": specialization_avg,
            },
            "client_tps": client_tps,
            "s2_beta_local_avg": s2_beta_avg,
            "s2_imag_mult_avg": s2_imag_avg,
            "s2_stress_avg": s2_stress_avg,
            "s2_planner_norm_avg": s2_planner_avg,
            "s2_dmn_floor_avg": s2_dmn_avg,
            "motor_delta_norm_avg": motor_delta_avg,
            "motor_sfnn_steps_total": int(self.motor_sfnn_steps),
            "motor_sfnn_eta_avg": motor_sfnn_eta_avg,
            "motor_sfnn_A_avg": motor_sfnn_A_avg,
            "motor_sfnn_T_avg": motor_sfnn_T_avg,
            "motor_sfnn_enabled_pct": motor_sfnn_enabled_pct,
            # SFNN S3.diag (14.05.2026): per-tissue для 7 высших тканей.
            "higher_sfnn": higher_sfnn_block,
            # SFNN S6.8 (16.05.2026): per-role для 10 базовых тканей.
            "basic_sfnn": basic_sfnn_block,
            # Z7.i.h (17.05.2026, Зодчий): 3 sidecar-ткани, snapshot-аггрегаты.
            "zodchiy_sidecar": zodchiy_block,
            # TZ B Phase 2 (Бендер, 26.05.2026): per-role Hebbian metrics
            # для UI "Тренировка мозга" combined Mode E + Mode M. Build +
            # reset accumulators — следующий emit cycle с чистого листа.
            "hebbian_per_tissue": self._build_hebbian_per_tissue_snapshot(),
            "tissue_activation_ring": self._build_tissue_activation_ring_payload(),
            # Body Migration Phase 2 (Бендер, 27.05.2026): client-side
            # биохимия snapshot — для UI Brain панели (mental_break states
            # heatmap, chem averages) + Хьюберт merge на VPS если потребуется.
            "biochem": self._build_biochem_snapshot(),
            # Phase 4 этап G (Бендер, 28.05.2026): естественный отбор
            # tracking — pure observability. Client не вмешивается в жизнь,
            # только наблюдает кто weak. Server-side давление
            # (flora_density / TelomerePhase / safe_grove_mult) делает
            # отбор само. См. utopia_client/natural_selection.py.
            "natural_selection": self._build_natural_selection_snapshot(),
            "s2_tom_acc_avg": s2_tom_avg,
            "tom_steps_total": int(self.tom_steps),
            "s2_lang_acc_avg": s2_lang_avg,
            "lang_steps_total": int(self.lang_steps),
            "s2_dopa_td_avg": s2_dopa_td_avg,
            "s2_action_dist_avg": s2_action_dist_avg,
            "s2_bias_max_avg": s2_bias_max_avg,
            "creatures": self._per_creature_stats(
                action_dists=action_dists, bias_max=bias_max),
        }

    def _per_creature_stats(
        self,
        action_dists: dict[str, list[float]] | None = None,
        bias_max: dict[str, float] | None = None,
    ) -> list[dict]:
        """Per-organism breakdown — что клиент знает о каждой живой особи.

        Без position/clan_id/role/diet (это от P40 через colony reporter).
        Размер: ~21 особь × ~200 байт ≈ 4 КБ.
        """
        from core.action_selector import N_ACTIONS
        action_dists = action_dists or {}
        bias_max = bias_max or {}
        empty_dist = [0.0] * N_ACTIONS
        out: list[dict] = []
        for cid, org in list(self.organisms.items()):
            tissues = getattr(org, "tissues", None) or {}
            n_embd = n_layer = n_head = 0
            n_params = 0
            if tissues:
                try:
                    first_tissue = next(iter(tissues.values()))
                    first_cell = next(iter(first_tissue.cells.values()))
                    n_embd = int(first_cell.n_embd)
                    n_layer = int(first_cell.n_layer)
                    n_head = int(first_cell.n_head)
                except (StopIteration, AttributeError):
                    pass
                try:
                    n_params = sum(
                        sum(p.numel() for p in cell.parameters())
                        for tissue in tissues.values()
                        for cell in tissue.cells.values()
                    )
                except Exception:
                    n_params = 0
            ctrl = self.hebbian.get(cid)
            cfg = getattr(ctrl, "config", None) if ctrl is not None else None
            top_spec: list[list] = []
            if ctrl is not None and hasattr(ctrl, "tissue_specialization"):
                try:
                    shares = ctrl.tissue_specialization()
                    items = sorted(
                        ((r, float(s)) for r, s in shares.items()),
                        key=lambda kv: kv[1],
                        reverse=True,
                    )[:3]
                    top_spec = [[r, round(s, 4)] for r, s in items]
                except Exception:
                    pass
            loss_ema = float(self.loss_ema.get(cid, 0.0))
            # Stats UI /stats (01.06.2026): эволюция (species/topo/gen) + обучение
            # (age/inst/food). age/inst/food — только для трекаемых newborn.
            _bt = self._birth_tick.get(cid)
            _age = (max(0, int(self._last_world_tick) - int(_bt))
                    if _bt is not None else None)
            _inst = (round(max(0.0, 1.0 - _age / 500.0), 3)
                     if _age is not None else None)
            _food = (int(self._carried_food.get(cid, 0))
                     if _bt is not None else None)
            # §6 рост (09.06): «закреплено»/«отвергнуто» из ГРАФА (durable, не
            # волатильный счётчик — тот обнулялся на рестарте, врал «0»). См.
            # _growth_graph_counts.
            _topo_enabled, _topo_reverted = self._growth_graph_counts(org)
            _topo_total = _topo_enabled + _topo_reverted
            out.append({
                "cid": str(cid),
                "species_id": self.species_id.get(cid),
                "gen": int(getattr(org, "generation", 0) or 0),
                "topo": _topo_total,
                # topo_active/growth_kept = живые (enabled) выросшие связи;
                # growth_reverted = откатанные (disabled). Из графа → durable.
                "topo_active": _topo_enabled,
                "growth_kept": _topo_enabled,
                "growth_reverted": _topo_reverted,
                "tissue_topology_mutations_total": _topo_total,
                "age": _age,
                "inst": _inst,
                "food": _food,
                "n_embd": n_embd,
                "n_layer": n_layer,
                "n_head": n_head,
                "n_tissues": len(tissues),
                "n_params": int(n_params),
                "prediction_accuracy": round(math.exp(-loss_ema), 4)
                                        if loss_ema > 0 else 0.0,
                "loss_ema": round(loss_ema, 5),
                "intrinsic_ema": round(
                    float(self.intrinsic_ema.get(cid, 0.0)), 6),
                "intrinsic_last": round(
                    float(self.intrinsic_last.get(cid, 0.0)), 6),
                "entropy_ema": round(
                    float(self.entropy_ema.get(cid, 0.0)), 4),
                "trace_norm_ema": round(
                    float(self.trace_norm_ema.get(cid, 0.0)), 4),
                "reward_var_ema": round(
                    float(self.reward_var_ema.get(cid, 0.0)), 6),
                "hebbian_enabled": ctrl is not None,
                "lr_oja": round(float(getattr(cfg, "lr_oja", 0.0)), 6)
                         if cfg else 0.0,
                "lr_reward": round(float(getattr(cfg, "lr_reward", 0.0)), 6)
                            if cfg else 0.0,
                "trace_decay": round(
                    float(getattr(cfg, "eligibility_decay", 0.0)), 4)
                    if cfg else 0.0,
                "top_specialization": top_spec,
                "client_beta_local": round(
                    float(self.last_beta_local.get(cid, 0.0)), 4),
                "client_imag_mult": round(
                    float(self.last_imag_mult.get(cid, 1.0)), 4),
                "client_stress": round(
                    float(self.last_stress.get(cid, 0.0)), 4),
                "client_planner_norm": round(
                    float(self.last_planner_delta[cid].norm().item()), 4)
                    if cid in self.last_planner_delta else 0.0,
                "client_motor_norm": round(
                    float(self.last_motor_delta[cid].norm().item()), 4)
                    if cid in self.last_motor_delta else 0.0,
                "client_dmn_floor": round(
                    float(self.last_dmn_floor.get(cid, 0.0)), 6),
                "client_tom_acc": round(
                    float(self.last_tom_acc.get(cid, 0.0)), 4),
                "client_lang_acc": round(
                    float(self.last_lang_acc.get(cid, 0.0)), 4),
                "client_dopa_td": round(
                    float(self.dopamine_td.get(cid, 0.0)), 6),
                "client_action_dist": action_dists.get(cid, empty_dist),
                "client_bias_max": bias_max.get(cid, 0.0),
                # Z7.i.h (17.05.2026, Зодчий): per-cid snapshot-нормы 3 ткани.
                # Для wanderer/elder остаются 0.0 (snapshot store'ы пусты).
                "client_cerebellum_delta_norm": (
                    round(float(self.last_cerebellum_delta[cid].norm().item()), 4)
                    if cid in self.last_cerebellum_delta else 0.0),
                "client_amygdala_valence": round(
                    float(self.last_amygdala_valence.get(cid, 0.0)), 4),
                "client_episodic_recall_norm": (
                    round(float(self.last_episodic_recall[cid].norm().item()), 4)
                    if cid in self.last_episodic_recall else 0.0),
            })
        return out
