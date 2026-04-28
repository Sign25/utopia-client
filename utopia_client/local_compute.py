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
        logger.info("add_creature %s n_tissues=%d predictor=%s", cid,
                    getattr(organism, "n_tissues", 0), pred is not None)

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

    @property
    def n_alive(self) -> int:
        return len(self.organisms)

    # ── Tick ─────────────────────────────────────────────────────────────

    def handle_tick(self, obs_per_cid: dict,
                    events_per_cid: Optional[dict] = None) -> dict:
        """Forward + ActionSelector + Hebbian update для всех cid.

        Args:
            obs_per_cid: {cid: np.ndarray[80] float32} — env-наблюдения от P40.
            events_per_cid: {cid: {ate, killed, damage_taken, delta_energy}} —
                события прошлого тика. Если None — Hebbian update пропускается.

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
                payload["tissues_state_dict"] = {
                    tid: t.state_dict() for tid, t in org.tissues.items()
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

    def diagnostics(self) -> dict:  # noqa: C901
        # debug: лог при каждом snapshot чтобы видеть динамику
        logger.info(
            "diagnostics: n_alive=%d n_pred=%d n_prev_obs=%d steps=%d hebs=%d",
            self.n_alive, len(self.predictor),
            len(self.prev_obs), self.predictor_steps,
            self.hebbian_updates)
        """Снимок метрик обучения для /api/diagnostics/training (Phase 1/2/6)."""
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
        }
