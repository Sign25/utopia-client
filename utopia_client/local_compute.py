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
_INSULA_BIT = 15        # S2.F
_IMAGINATION_BIT = 16   # S2.G
# Phase S2.A planner — N_ACTIONS из routes_world.
_PLANNER_N_ACTIONS = 16
_PLANNER_SCALE = 1.0
# Phase S2.E dopamine: β_local ∈ [0, 1/φ ≈ 0.618].
_PHI = 1.6180339887498949
# Phase S2.F insula: 64 obs + 7 интероцепции = 71.
_INSULA_DATA_DIM = 71
# Y50 для высших тканей (тот же scale, что у predictor).
_HIGHER_TISSUE_Y50_SCALE = 0.0902


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
        # Last-snapshot для get_phase_emas → push в actions_batch.
        self.last_beta_local: dict = {}    # cid → float ∈ [0, 1/φ]
        self.last_imag_mult: dict = {}     # cid → float ∈ [1, 2]
        self.last_planner_delta: dict = {} # cid → torch.Tensor [16]
        self.last_stress: dict = {}        # cid → float ∈ [0, 1]

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
        self.last_beta_local.pop(cid, None)
        self.last_imag_mult.pop(cid, None)
        self.last_planner_delta.pop(cid, None)
        self.last_stress.pop(cid, None)

    def reset_all(self) -> int:
        n = len(self.organisms)
        for cid in list(self.organisms.keys()):
            self.remove_creature(cid)
        self.hebbian_updates = 0
        self.predictor_steps = 0
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
            ):
                if key in payload:
                    try:
                        getattr(self, target)[cid] = float(payload[key])
                    except Exception:
                        pass
        # Brain migration (10.05.2026): Y50 для высших тканей S2.E/G/A/F.
        for key, store in (
            ("dopamine", self.dopamine),
            ("imagination", self.imagination),
            ("planner", self.planner),
            ("insula", self.insula),
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
        for src, key in (
            (self.last_beta_local, "client_beta_local"),
            (self.last_imag_mult, "client_imag_mult"),
            (self.last_stress, "client_stress"),
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

                selector = self.action_selectors[cid]
                action = int(selector.select(logits, n_actions=N_ACTIONS))
                out[cid] = {"action": action, "target_id": None}

                # Phase 1 — predictor supervised step + Phase 2 intrinsic.
                # Должен идти ДО Hebbian update, чтобы intrinsic подмешать в r_imm.
                intrinsic_now = self._predictor_train_step(cid, obs_tensor)

                # Brain migration (10.05.2026): forward S2.E/G/A/F.
                # Результаты лежат в last_beta_local/last_imag_mult/
                # last_planner_delta/last_stress, отгружаются в actions_batch
                # через get_phase_emas. Сервер применит через fastpath
                # _compute_*. Forward после predictor — nothing depends.
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

                # Phase 6 — entropy EMA по action-distribution.
                self._update_entropy_ema(cid, logits)

                # Phase F3.2.b: Hebbian update по локальному R3 reward.
                # immediate-only — medium/long требуют state (ema, repro
                # tracking) и помечены как долг.
                if heb is not None and events_per_cid is not None:
                    event = events_per_cid.get(cid)
                    if event is not None:
                        r_imm = self._compute_immediate_reward(event)
                        # Phase 2 — подмешать intrinsic в immediate.
                        r_imm_total = r_imm + intrinsic_now
                        try:
                            heb.update(logits,
                                       {"immediate": r_imm_total,
                                        "medium": 0.0, "long": 0.0})
                            self.hebbian_updates += 1
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
        # Brain migration (10.05.2026): высшие ткани S2.E/G/A/F.
        for key, store in (
            ("dopamine", self.dopamine),
            ("imagination", self.imagination),
            ("planner", self.planner),
            ("insula", self.insula),
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
        """R3 immediate из событий тика. Вес как на P40 (Phase H1+):

            r_imm = δenergy·0.05 + 1·ate + 5·killed − 0.1·damage

        В реальной среде значения примерно: ate +1.0, killed +5.0,
        обычный метаболизм δenergy≈-0.05 → r_imm≈0 (нейтрально).
        """
        delta_energy = float(event.get("delta_energy", 0.0))
        ate = bool(event.get("ate", False))
        killed = bool(event.get("killed", False))
        damage_taken = float(event.get("damage_taken", 0.0))
        return (delta_energy * 0.05
                + (1.0 if ate else 0.0)
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
        n = len(self.organisms)
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
            "creatures": self._per_creature_stats(),
        }

    def _per_creature_stats(self) -> list[dict]:
        """Per-organism breakdown — что клиент знает о каждой живой особи.

        Без position/clan_id/role/diet (это от P40 через colony reporter).
        Размер: ~21 особь × ~200 байт ≈ 4 КБ.
        """
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
            })
        return out
