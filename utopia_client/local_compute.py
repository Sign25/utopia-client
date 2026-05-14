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
# Phase 7 motor_policy: delta = scale·tanh(out[:16]) ∈ [-1, 1]^16.
_MOTOR_POLICY_N_ACTIONS = 16
_MOTOR_POLICY_SCALE = 1.0
_MOTOR_POLICY_LR = 1e-4
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
# Phase S2.F insula: 64 obs + 7 интероцепции = 71.
_INSULA_DATA_DIM = 71
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
        # Phase 7 — motor_policy REINFORCE-sidecar (бит 17 с 13.05.2026,
        # перенумерован с 11 для освобождения канонического бита под S2.B).
        # supervised: log_prob[a_t] · (r_imm_total - baseline), Adam lr=1e-4.
        self.motor_policy: dict = {}        # cid → Tissue (Phase 7)
        self.motor_policy_opt: dict = {}    # cid → torch.optim.Adam
        # SFNN S1.1 (14.05.2026) — per-tissue SFNN-правило обучения для
        # motor_policy. Под флагом genome.sfnn_enabled (дефолт False) сейчас
        # только storage + наследование, фактическое применение в S1.2.
        # Дефолтное правило (A=1, B=C=D=0, η=1e-3) эквивалентно Phase 5d Hebb.
        self.motor_sfnn_rule: dict = {}     # cid → SFNNRule
        # SFNN S1.2b (14.05.2026) — forward-hooks pre/post активаций для
        # 6 Linear в motor_policy Tissue. Захватываются на каждом forward
        # через _motor_forward; используются на следующем тике в
        # _motor_sfnn_update_step для ΔW. Очищаются в remove_creature.
        self.motor_sfnn_hook_handles: dict = {}   # cid → list[hook handle]
        self.motor_sfnn_acts: dict = {}            # cid → {synapse_type → (pre, post)}
        # SFNN S3.0 (14.05.2026) — те же SFNN-правила, но для 7 высших тканей.
        # Внутренний dict per tissue: cid → SFNNRule. Под флагом
        # genome.higher_tissue_sfnn_enabled (дефолт False) сейчас только
        # storage + Y50-наследование, активное применение — S3.1+.
        # predictor сюда не входит — остаётся на Adam (supervised).
        self.higher_tissue_sfnn_rule: dict[str, dict] = {
            t: {} for t in _HIGHER_SFNN_TISSUES
        }
        # Per-tissue счётчик SFNN-апдейтов (заполняется в S3.1+ при включении
        # дисптача). Используется в diagnostics для отслеживания активности.
        self.higher_tissue_sfnn_steps: dict[str, int] = {
            t: 0 for t in _HIGHER_SFNN_TISSUES
        }
        # SFNN S3.1 (14.05.2026) — forward-hooks pre/post активаций 6 Linear
        # для каждой из 7 высших тканей. Регистрируются в add_creature только
        # для активных-в-S3.x тканей; cleanup в remove_creature. Per tissue:
        # {cid → list[hook handle]} и {cid → {synapse_type → (pre, post)}}.
        self.higher_tissue_sfnn_hook_handles: dict[str, dict] = {
            t: {} for t in _HIGHER_SFNN_TISSUES
        }
        self.higher_tissue_sfnn_acts: dict[str, dict] = {
            t: {} for t in _HIGHER_SFNN_TISSUES
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
        # pending_log_prob — scalar tensor с grad-графом от motor_delta предыдущего
        # тика; REINFORCE-loss применяется на следующем тике, когда приходят events.
        self.pending_log_prob: dict = {}    # cid → torch.Tensor (scalar, requires_grad)
        self.pending_action: dict = {}      # cid → int (для диагностики)
        self.motor_reinforce_steps: int = 0
        # SFNN S1.2 (14.05.2026) — счётчик SFNN-апдейтов motor_policy под
        # флагом genome.sfnn_enabled. В S1.2a skeleton увеличивает счётчик,
        # но веса не трогает (no-op для регрессии). S1.2b включит реальное
        # правило ΔW = η·(1+r)·(A·post·preᵀ + B·post + C·pre + D).
        self.motor_sfnn_steps: int = 0
        # Last-snapshot для get_phase_emas → push в actions_batch.
        self.last_beta_local: dict = {}    # cid → float ∈ [0, 1/φ]
        self.last_imag_mult: dict = {}     # cid → float ∈ [1, 2]
        self.last_planner_delta: dict = {} # cid → torch.Tensor [16]
        self.last_motor_delta: dict = {}   # cid → torch.Tensor [16] (Phase 7)
        self.last_stress: dict = {}        # cid → float ∈ [0, 1]
        self.last_dmn_floor: dict = {}     # cid → float ∈ [0, _DEFAULT_MODE_FLOOR_MAX]
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

        # SFNN S3.activate: дефолт higher_tissue_sfnn_enabled для новых особей.
        # set_higher_sfnn() флипает поверх + применяет к existing organisms.
        self._higher_sfnn_default: bool = False
        # SFNN S3.activate-motor: дефолт sfnn_enabled (motor_policy ветка)
        # для новых особей. set_motor_sfnn() флипает поверх.
        self._motor_sfnn_default: bool = False

        logger.info("LocalColonyCompute device=%s", self.device)

    # ── Регистрация особей ───────────────────────────────────────────────

    def add_creature(self, cid: str, organism, *, hebbian_enabled: bool = True,
                     learning_rate: float = 1e-4, trace_decay: float = 0.9) -> None:
        """Зарегистрировать особь. Organism — CompositeOrganism из seed_loader."""
        from core.action_selector import ActionSelector

        if hasattr(organism, "to"):
            organism.to(self.device)
        if hasattr(organism, "eval"):
            organism.eval()
        self.organisms[cid] = organism
        # SFNN S3.activate: CompositeOrganism не несёт genome — приклеиваем
        # SimpleNamespace с текущим дефолтом флага. set_higher_sfnn() позже
        # переписывает атрибут у всех существующих особей.
        if not hasattr(organism, "genome"):
            organism.genome = types.SimpleNamespace(
                higher_tissue_sfnn_enabled=self._higher_sfnn_default,
                sfnn_enabled=self._motor_sfnn_default,
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
        # Phase 7 — motor_policy: Tissue 21/3/1 + Adam(lr=1e-4) для REINFORCE.
        motor = self._make_higher_tissue("motor_policy")
        if motor is not None:
            self.motor_policy[cid] = motor
            self.motor_policy_opt[cid] = self._torch.optim.Adam(
                motor.parameters(), lr=_MOTOR_POLICY_LR)
            # SFNN S1.1: дефолтное правило (Phase 5d-эквивалент). Используется
            # только при genome.sfnn_enabled=True (S1.2+); в S1.1 — storage
            # + наследование через Y50.
            from core.sfnn_rule import SFNNRule
            self.motor_sfnn_rule[cid] = SFNNRule.default()
            # SFNN S1.2b: hooks для захвата pre/post 6 Linear motor_policy.
            # Регистрируются всегда (overhead — копия (pre,post) на forward),
            # используются только при genome.sfnn_enabled=True.
            self._register_motor_sfnn_hooks(cid, motor)
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
                    self.higher_tissue_sfnn_rule[_name][cid] = SFNNRule.default()
        except Exception as e:
            logger.debug("add_creature %s higher_tissue_sfnn_rule init: %s", cid, e)
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
        logger.info(
            "add_creature %s n_tissues=%d predictor=%s S2=%s",
            cid, getattr(organism, "n_tissues", 0), pred is not None,
            all(self.dopamine.get(cid) is not None for _ in [0]))

    def remove_creature(self, cid: str) -> None:
        self.organisms.pop(cid, None)
        self.action_selectors.pop(cid, None)
        self.hebbian.pop(cid, None)
        self.predictor.pop(cid, None)
        self.predictor_opt.pop(cid, None)
        self.prev_obs.pop(cid, None)
        self.loss_ema.pop(cid, None)
        self.pred_loss_history.pop(cid, None)
        self.intrinsic_last.pop(cid, None)
        self.intrinsic_ema.pop(cid, None)
        self.entropy_ema.pop(cid, None)
        self.trace_norm_ema.pop(cid, None)
        self.reward_var_ema.pop(cid, None)
        self.reward_history.pop(cid, None)
        # Brain migration (10.05.2026): высшие ткани S2.E/G/A/F.
        self.dopamine.pop(cid, None)
        self.imagination.pop(cid, None)
        self.planner.pop(cid, None)
        self.insula.pop(cid, None)
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
        # Phase 7 — motor_policy.
        self.motor_policy.pop(cid, None)
        self.motor_policy_opt.pop(cid, None)
        self.motor_sfnn_rule.pop(cid, None)
        # SFNN S1.2b: снять forward-hooks и забыть кешированные активации.
        for h in self.motor_sfnn_hook_handles.pop(cid, []) or []:
            try:
                h.remove()
            except Exception:
                pass
        self.motor_sfnn_acts.pop(cid, None)
        self.pending_log_prob.pop(cid, None)
        self.pending_action.pop(cid, None)
        self.last_motor_delta.pop(cid, None)
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
        for _t in _HIGHER_SFNN_TISSUES:
            self.higher_tissue_sfnn_rule.get(_t, {}).pop(cid, None)
        # SFNN S3.1: снять forward-hooks и забыть кешированные активации
        # всех зарегистрированных в S3.x тканей. Пока активна dopamine;
        # cleanup для остальных no-op до S3.2+.
        for _t in _HIGHER_SFNN_TISSUES:
            for h in (self.higher_tissue_sfnn_hook_handles.get(_t, {})
                      .pop(cid, []) or []):
                try:
                    h.remove()
                except Exception:
                    pass
            self.higher_tissue_sfnn_acts.get(_t, {}).pop(cid, None)

    def reset_all(self) -> int:
        n = len(self.organisms)
        for cid in list(self.organisms.keys()):
            self.remove_creature(cid)
        self.hebbian_updates = 0
        self.predictor_steps = 0
        self.motor_sfnn_steps = 0
        # SFNN S3.0: сбросить счётчики апдейтов всех 7 высших тканей.
        for _t in _HIGHER_SFNN_TISSUES:
            self.higher_tissue_sfnn_steps[_t] = 0
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
        # Phase 1 — Y50 наследование predictor от родителя.
        pred_sd = payload.get("predictor")
        if pred_sd is not None and self.predictor.get(cid) is not None:
            try:
                # Сначала просто загружаем веса родителя.
                self.predictor[cid].load_state_dict(pred_sd)
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
        for key, store in (
            ("dopamine", self.dopamine),
            ("imagination", self.imagination),
            ("planner", self.planner),
            ("insula", self.insula),
            ("default_mode", self.default_mode),
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
        # Phase 7 — motor_policy: Y50 + свежий Adam (как predictor).
        m_sd = payload.get("motor_policy")
        motor = self.motor_policy.get(cid)
        if m_sd is not None and motor is not None:
            try:
                motor.load_state_dict(m_sd)
                self._apply_y50_to_tissue(motor)
                self.motor_policy_opt[cid] = self._torch.optim.Adam(
                    motor.parameters(), lr=_MOTOR_POLICY_LR)
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
        for key, store in (
            ("dopamine", self.dopamine),
            ("imagination", self.imagination),
            ("planner", self.planner),
            ("insula", self.insula),
            ("motor_policy", self.motor_policy),
            ("default_mode", self.default_mode),
            ("theory_of_mind", self.theory_of_mind),
            ("language", self.language),
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
        higher_rules_dump: dict = {}
        for _t in _HIGHER_SFNN_TISSUES:
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
        for key, store in (
            ("dopamine", self.dopamine),
            ("imagination", self.imagination),
            ("planner", self.planner),
            ("insula", self.insula),
            ("motor_policy", self.motor_policy),
            ("default_mode", self.default_mode),
            ("theory_of_mind", self.theory_of_mind),
            ("language", self.language),
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
        for cid, org in self.organisms.items():
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
        for cid, org in self.organisms.items():
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
        return out or None

    # ── Tick ─────────────────────────────────────────────────────────────

    def handle_tick(self, obs_per_cid: dict,
                    events_per_cid: Optional[dict] = None,
                    intero_per_cid: Optional[dict] = None) -> dict:
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
        out: dict = {}
        torch = self._torch
        for cid, organism in self.organisms.items():
            obs = obs_per_cid.get(cid)
            if obs is None:
                out[cid] = {"action": STAY, "target_id": None}
                continue
            try:
                obs_arr = np.asarray(obs, dtype=np.float32).reshape(-1)
                # P40 присылает 80 (DATA_DIM=64 + STATE_DIM=16). forward берёт первые 64.
                obs64 = obs_arr[:64]
                obs_tensor = torch.from_numpy(obs64).to(self.device).unsqueeze(0)

                heb = self.hebbian.get(cid)
                if heb is not None:
                    try:
                        heb.capture_activations(obs_tensor)
                    except Exception as e:
                        logger.debug("hebbian capture %s: %s", cid, e)

                with torch.no_grad():
                    logits = organism.forward(obs_tensor)

                # Phase 1 — predictor supervised step + Phase 2 intrinsic.
                # Идёт ДО motor REINFORCE update, чтобы intrinsic подмешать
                # в r_imm_total как baseline-сигнал.
                intrinsic_now = self._predictor_train_step(cid, obs_tensor)

                # Phase 7 — REINFORCE update от прошлого тика. Сначала
                # применяем pending градиент по pending_log_prob, ПОТОМ
                # делаем свежий motor forward с grad для текущего тика.
                # SFNN S1.2 (14.05.2026): развилка по genome.sfnn_enabled.
                # При True — локальное правило пластичности (без Adam,
                # без REINFORCE), при False — legacy путь Phase 5d.
                org = self.organisms.get(cid)
                sfnn_on = bool(getattr(getattr(org, "genome", None),
                                        "sfnn_enabled", False))
                if sfnn_on:
                    self._motor_sfnn_update_step(cid, events_per_cid,
                                                   intrinsic_now)
                else:
                    self._motor_reinforce_step(cid, events_per_cid,
                                                 intrinsic_now)

                # Phase 7 — motor_policy forward с grad → motor_delta [16].
                # Применяется к первым 16 элементам logits (action-space)
                # ДО selector. grad через motor_delta нужен для log_prob[a_t]
                # — на следующем тике REINFORCE возьмёт его в loss. Если
                # ткани нет (legacy state) — fallback к raw_logits.
                motor_delta = self._motor_forward(cid, obs_tensor)
                selector = self.action_selectors[cid]
                if motor_delta is not None:
                    # logits = [1, 64] no_grad → action_slice [N_ACTIONS] с
                    # прикреплённым grad-графом через motor_delta.
                    action_slice = (logits[0, :N_ACTIONS].detach()
                                     + motor_delta)
                    action = int(selector.select(
                        action_slice.detach(), n_actions=N_ACTIONS))
                    # log_prob[a_t] с grad через motor_delta → pending для
                    # REINFORCE-update на следующем тике.
                    self._stash_pending_log_prob(cid, action_slice, action)
                else:
                    action = int(selector.select(logits, n_actions=N_ACTIONS))
                out[cid] = {"action": action, "target_id": None}

                # Brain migration (10.05.2026): forward S2.E/G/A/F (no_grad).
                intero_tensor = None
                if intero_per_cid is not None:
                    intero_arr = intero_per_cid.get(cid)
                    if intero_arr is not None:
                        try:
                            intero_tensor = torch.from_numpy(
                                np.asarray(intero_arr, dtype=np.float32)
                            ).to(self.device).unsqueeze(0)
                        except Exception:
                            intero_tensor = None
                self._compute_higher_tissues(cid, obs_tensor, intero_tensor)

                # SFNN S3.1 (14.05.2026): локальное правило пластичности
                # для dopamine под флагом genome.higher_tissue_sfnn_enabled.
                # r=0.0 — чистая Hebb, без reward-модуляции (см. S3 решение).
                # Forward dopamine произошёл строкой выше → hooks обновили
                # higher_tissue_sfnn_acts["dopamine"][cid]. update_step
                # читает их и в одном шаге применяет ΔW.
                higher_sfnn_on = bool(getattr(
                    getattr(org, "genome", None),
                    "higher_tissue_sfnn_enabled", False))
                if higher_sfnn_on:
                    self._higher_tissue_sfnn_update_step(
                        "dopamine", cid, r=0.0)
                    # SFNN S3.2 (14.05.2026): imagination — вторая активная
                    # высшая ткань. r=0.0 как и dopamine.
                    self._higher_tissue_sfnn_update_step(
                        "imagination", cid, r=0.0)
                    # SFNN S3.3 (14.05.2026): planner — третья активная.
                    self._higher_tissue_sfnn_update_step(
                        "planner", cid, r=0.0)
                    # SFNN S3.4 (14.05.2026): insula — четвёртая. No-op если
                    # intero_tensor не пришёл (acts пустой).
                    self._higher_tissue_sfnn_update_step(
                        "insula", cid, r=0.0)
                    # SFNN S3.5 (14.05.2026): default_mode — пятая.
                    self._higher_tissue_sfnn_update_step(
                        "default_mode", cid, r=0.0)

                # S2.B (13.05.2026) — theory_of_mind supervised step.
                # Использует WorldStateCache.tom_neighbors_view; если кеша
                # нет / соседей нет — no-op без побочных эффектов.
                # SFNN S3.6 (14.05.2026): под флагом эта функция делает
                # forward без Adam-шага → hooks стреляют → update_step
                # применяет ΔW локальным правилом.
                self._compute_theory_of_mind(cid)
                if higher_sfnn_on:
                    self._higher_tissue_sfnn_update_step(
                        "theory_of_mind", cid, r=0.0)

                # S2.C (13.05.2026) — language supervised step.
                # prev_context (сигналы соседей с прошлого тика) → текущий
                # event_class (ate/damage/killed/idle, idle skip).
                # SFNN S3.7 (14.05.2026): под флагом forward выполняется
                # без Adam → hooks стреляют → update_step применяет ΔW.
                lang_event = (events_per_cid.get(cid)
                              if events_per_cid is not None else None)
                self._compute_language(cid, lang_event)
                if higher_sfnn_on:
                    self._higher_tissue_sfnn_update_step(
                        "language", cid, r=0.0)

                # Phase 6 — entropy EMA по action-distribution.
                self._update_entropy_ema(cid, logits)

                # Phase F3.2.b: Hebbian update по локальному R3 reward.
                if heb is not None and events_per_cid is not None:
                    event = events_per_cid.get(cid)
                    if event is not None:
                        r_imm = self._compute_immediate_reward(event)
                        # Phase 2 — подмешать intrinsic в immediate.
                        r_imm_total = r_imm + intrinsic_now
                        # Phase 5d (NEOL): TD-модулятор η для reward_output.
                        # td = β − EMA(β, α=0.01), mult = 1 + clip(td, ±0.5).
                        # Если S2.E off (нет beta → td=0.0) → mult=1.0 (no-op).
                        # Если HebbianController старой версии (без kwarg) —
                        # TypeError → fallback на безмодуляторный вызов.
                        td = self.dopamine_td.get(cid, 0.0)
                        td_mult = 1.0 + max(-0.5, min(0.5, td))
                        try:
                            heb.update(logits,
                                       {"immediate": r_imm_total,
                                        "medium": 0.0, "long": 0.0},
                                       dopa_td_mult=td_mult)
                            self.hebbian_updates += 1
                        except TypeError:
                            try:
                                heb.update(logits,
                                           {"immediate": r_imm_total,
                                            "medium": 0.0, "long": 0.0})
                                self.hebbian_updates += 1
                            except Exception as e:
                                logger.debug("hebbian update %s: %s", cid, e)
                        except Exception as e:
                            logger.debug("hebbian update %s: %s", cid, e)
                        # Phase 6 — reward_var_ema по последним 10 r_imm.
                        self._update_reward_var_ema(cid, r_imm_total)
                # Phase 6 — trace_norm_ema по Hebbian-traces.
                self._update_trace_norm_ema(cid, heb)
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
        higher_rules_dump: dict = {}
        for _t in _HIGHER_SFNN_TISSUES:
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

    @staticmethod
    def _compute_immediate_reward(event: dict) -> float:
        """R3 immediate из событий тика.

        14.05.2026: усилен ate (1.0→5.0, равно killed) и δenergy (×10).
        Why: action-space collapse — REINFORCE не двигал политику, потому
        что r_imm для ate-тика тонул в метаболическом шуме (δenergy·0.05 ≈
        -0.0025), advantage около baseline=intrinsic_ema. reward_var_ema
        наблюдался 1.7e-5 на live колонии.
        """
        delta_energy = float(event.get("delta_energy", 0.0))
        ate = bool(event.get("ate", False))
        killed = bool(event.get("killed", False))
        damage_taken = float(event.get("damage_taken", 0.0))
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

    # ── Phase 7 — motor_policy REINFORCE-sidecar ─────────────────────────

    def _motor_forward(self, cid: str, obs_tensor):
        """Forward motor_policy с grad. Возвращает delta torch.Tensor [16]
        с прикреплённым grad-графом или None если ткани нет / ошибка.

        Грейд нужен здесь, чтобы при выборе действия с применённой delta
        log_prob[a_t] нёс градиент в motor_policy weights (REINFORCE).
        """
        torch = self._torch
        tissue = self.motor_policy.get(cid)
        if tissue is None:
            return None
        try:
            tissue.train()
            with torch.enable_grad():
                out = tissue({"input": obs_tensor.detach()})["output"]
                delta = (torch.tanh(out[0, :_MOTOR_POLICY_N_ACTIONS])
                          * _MOTOR_POLICY_SCALE)
            # Snapshot для push на сервер (detached cpu).
            self.last_motor_delta[cid] = delta.detach().cpu()
            return delta
        except Exception as e:
            logger.debug("motor_policy forward %s: %s", cid, e)
            return None

    def _stash_pending_log_prob(self, cid: str, action_slice,
                                  action: int) -> None:
        """Сохранить log_prob[a_t] (с grad через motor_delta) для REINFORCE
        на следующем тике. action_slice — [N_ACTIONS] 1D-тензор raw_logits +
        motor_delta с прикреплённым grad-графом через motor_delta.
        """
        torch = self._torch
        try:
            n = int(action_slice.shape[-1])
            if not (0 <= action < n):
                return
            log_probs = torch.log_softmax(action_slice, dim=-1)
            self.pending_log_prob[cid] = log_probs[action]
            self.pending_action[cid] = int(action)
        except Exception as e:
            logger.debug("stash_pending_log_prob %s: %s", cid, e)

    def _motor_reinforce_step(self, cid: str, events_per_cid,
                                intrinsic_now: float) -> None:
        """REINFORCE update от прошлого тика: loss = -log_prob · (r_imm_total
        - baseline). baseline = intrinsic_ema[cid] (running mean intrinsic).
        Применяется и опт-шаг, и градиент очищается через opt.zero_grad().
        Pending_log_prob — scalar tensor с requires_grad, созданный на
        предыдущем _stash_pending_log_prob.

        14.05.2026: дополнительно вызывает selector.reinforce(advantage,
        prev_action) — быстрый SFNN-стиль bias на ActionSelector. Bias
        реагирует немедленно (1 шаг = +lr·reward к выбранному действию),
        в отличие от motor_policy который обучается через градиент Adam.
        """
        pending = self.pending_log_prob.pop(cid, None)
        prev_action = self.pending_action.pop(cid, None)
        if pending is None:
            return
        if events_per_cid is None:
            return
        event = events_per_cid.get(cid)
        if event is None:
            return
        opt = self.motor_policy_opt.get(cid)
        if opt is None:
            return
        try:
            r_imm = self._compute_immediate_reward(event)
            r_imm_total = float(r_imm + intrinsic_now)
            baseline = float(self.intrinsic_ema.get(cid, 0.0))
            advantage = r_imm_total - baseline
            # REINFORCE: maximize log_prob · A → minimize -log_prob · A.
            loss = -pending * advantage
            opt.zero_grad()
            loss.backward()
            opt.step()
            self.motor_reinforce_steps += 1
            # SFNN-стиль reward-modulated bias на ActionSelector.
            sel = self.action_selectors.get(cid)
            if sel is not None and prev_action is not None:
                sel.reinforce(advantage, int(prev_action))
        except Exception as e:
            logger.debug("motor_reinforce %s: %s", cid, e)

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

        Заменяет `_motor_reinforce_step` (REINFORCE + Adam). Для каждого
        из 6 типов синапсов Tissue 21/3/1 motor_policy применяется:

            ΔW = η · (1 + r_imm_total) · (A·outer(post, pre)
                                            + B·post + C·preᵀ + D)
            ΔW ← clip(ΔW, ±0.01)            # защита от взрыва
            W  ← W + ΔW

        Oja-нормализация (W ← W/‖W‖) намеренно НЕ применяется в S1.2b:
        Frobenius-делитель быстро коллапсирует масштаб весов; row-wise
        Oja сложнее и обещан в S1.2c. На малом шаге (eta=1e-3, clip=0.01)
        веса не должны разъезжаться за несколько часов наблюдения.

        pre/post захватываются forward-hooks (`_register_motor_sfnn_hooks`)
        при предыдущем `_motor_forward`. Если активаций нет (первый тик
        или ткани нет) — обновление пропускается.

        Reward-модуляция использует тот же сигнал что REINFORCE: r_imm
        (immediate reward от событий) + intrinsic_now (surprise predictor).
        """
        # Чистим pending_log_prob/pending_action — в SFNN они не нужны.
        self.pending_log_prob.pop(cid, None)
        self.pending_action.pop(cid, None)
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
        try:
            r_imm = self._compute_immediate_reward(event)
            r_imm_total = float(r_imm + intrinsic_now)
            reward_gain = 1.0 + r_imm_total
            clip_val = self._MOTOR_SFNN_DW_CLIP
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
                    eta = float(coef.eta)
                    A = float(coef.A)
                    B = float(coef.B)
                    C = float(coef.C)
                    D = float(coef.D)
                    # ΔW = η·(1+r)·(A·outer(post,pre) + B·post[:,None] + C·pre[None,:] + D)
                    dW = A * torch.outer(post, pre)
                    if B != 0.0:
                        dW = dW + B * post.unsqueeze(1).expand_as(W)
                    if C != 0.0:
                        dW = dW + C * pre.unsqueeze(0).expand_as(W)
                    if D != 0.0:
                        dW = dW + D
                    dW = dW * (eta * reward_gain)
                    dW.clamp_(-clip_val, clip_val)
                    W.data.add_(dW)
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
                                          r: float = 0.0) -> None:
        """SFNN S3.1 (14.05.2026): локальное правило пластичности для одной
        высшей ткани под флагом `genome.higher_tissue_sfnn_enabled=True`.

        Идентично `_motor_sfnn_update_step`, кроме reward-модуляции:
        в S3.1 для всех 7 тканей r=0.0 (чистая Hebb без модуляции).
        Поэтому reward_gain = 1.0, и формула вырождается в
            ΔW = η · (A·outer(post, pre) + B·post + C·pre + D)
            ΔW ← clip(ΔW, ±0.01)
            W  ← W + ΔW

        pre/post захватываются forward-hooks при предыдущем
        `_compute_higher_tissues` (если ткань активна — был forward,
        активации обновились). Если активаций нет (первый тик /
        ткани нет / hook не зарегистрирован) — обновление пропускается.
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
        try:
            reward_gain = 1.0 + float(r)
            clip_val = self._MOTOR_SFNN_DW_CLIP
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
                    dW = A * torch.outer(post, pre)
                    if B != 0.0:
                        dW = dW + B * post.unsqueeze(1).expand_as(W)
                    if C != 0.0:
                        dW = dW + C * pre.unsqueeze(0).expand_as(W)
                    if D != 0.0:
                        dW = dW + D
                    dW = dW * (eta * reward_gain)
                    dW.clamp_(-clip_val, clip_val)
                    W.data.add_(dW)
            self.higher_tissue_sfnn_steps[tissue_name] = (
                self.higher_tissue_sfnn_steps.get(tissue_name, 0) + 1)
        except Exception as e:
            logger.debug("higher_sfnn_update %s/%s: %s",
                          tissue_name, cid, e)

    def _predictor_train_step(self, cid: str, obs_tensor) -> float:
        """Phase 1+2: один MSE-шаг predictor + intrinsic reward.

        Идентично _predictor_train_step на P40 (routes_world.py:1149).

        Возвращает intrinsic_last (β·max(0, loss_ema_prev - loss_curr)).
        Если predictor нет или prev_obs пустой — 0.0 (но prev_obs обновится).
        """
        torch = self._torch
        pred = self.predictor.get(cid)
        opt = self.predictor_opt.get(cid)
        prev = self.prev_obs.get(cid)
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
            self.prev_obs[cid] = obs_tensor.detach()
        return intrinsic

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
                "motor_reinforce_steps_total": int(self.motor_reinforce_steps),
                "motor_sfnn_steps_total": int(self.motor_sfnn_steps),
                "motor_sfnn_eta_avg": 0.0,
                "motor_sfnn_A_avg": 0.0,
                "motor_sfnn_enabled_pct": 0.0,
                # SFNN S3.diag (14.05.2026): per-tissue для 7 высших.
                "higher_sfnn": {
                    "enabled_pct": 0.0,
                    **{t: {"steps_total": int(
                                self.higher_tissue_sfnn_steps.get(t, 0)),
                             "eta_avg": 0.0,
                             "A_avg": 0.0}
                       for t in _HIGHER_SFNN_TISSUES},
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
        for org in self.organisms.values():
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
        spec_sums: dict[str, float] = {}
        spec_counts: dict[str, int] = {}
        for ctrl in self.hebbian.values():
            if ctrl is None or not hasattr(ctrl, "tissue_specialization"):
                continue
            try:
                shares = ctrl.tissue_specialization()
            except Exception:
                continue
            for role, share in shares.items():
                spec_sums[role] = spec_sums.get(role, 0.0) + float(share)
                spec_counts[role] = spec_counts.get(role, 0) + 1
        specialization_avg = {
            role: round(spec_sums[role] / spec_counts[role], 4)
            for role in spec_sums
            if spec_counts[role] > 0
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
        sfnn_enabled_n = 0
        for c, rule in self.motor_sfnn_rule.items():
            if rule is None:
                continue
            try:
                sfnn_etas.append(float(rule.mean_eta()))
                sfnn_As.append(float(rule.mean_A()))
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
        higher_sfnn_per_tissue: dict[str, dict] = {}
        for t in _HIGHER_SFNN_TISSUES:
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

        return {
            "n_alive": n,
            "n_predictors": len(self.predictor),
            "n_prev_obs": len(self.prev_obs),
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
            "motor_reinforce_steps_total": int(self.motor_reinforce_steps),
            "motor_sfnn_steps_total": int(self.motor_sfnn_steps),
            "motor_sfnn_eta_avg": motor_sfnn_eta_avg,
            "motor_sfnn_A_avg": motor_sfnn_A_avg,
            "motor_sfnn_enabled_pct": motor_sfnn_enabled_pct,
            # SFNN S3.diag (14.05.2026): per-tissue для 7 высших тканей.
            "higher_sfnn": higher_sfnn_block,
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
        for cid, org in self.organisms.items():
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
            out.append({
                "cid": str(cid),
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
            })
        return out
