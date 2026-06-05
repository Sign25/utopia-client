"""MotorSlowTrainer — порт серверного core/world_trainer.py на клиент (Фрай 05.06).

MIGRATION GAP fix: при переходе Адама в client-authoritative выпал МЕДЛЕННЫЙ
канал обучения политики (P40 не тренит owned, на клиент не портирован). Клиент
имел только БЫСТРЫЙ per-tick Hebbian — он представленческий нормализатор, НЕ
учитель политики. Все симптомы дня (lock-in, сатурация, SIGNAL_DANGER-коллапс,
skill-melt) = отсутствие медленного канала.

ПОРТ ВЕРНЫЙ (серверный проверенный конфиг, БЕЗ «улучшений» — переизобретение
быстрого канала стоило дня):
  • buffer 200 (приоритет |reward|>0), batch 16, train_every 10
  • advantage = (reward − baseline), НОРМАЛИЗАЦИЯ по std (НЕ clamp)
  • adaptive entropy_coef 0.15→2.0 при entropy<target 0.4 (анти-коллапс)
  • REINFORCE backprop через motor_policy ткань + Adam + grad-clip

Адаптация для Адама: политика = softmax(base_logits + own·motor_delta), где
base_logits = прайор-шейпинг (detached, не обучаемый), motor_delta =
tanh(motor_policy(obs)[:16]/T)·SCALE (обучаемая часть). Backprop только в
motor_policy. Быстрый Hebbian остаётся (оба канала, как сервер).
"""
from __future__ import annotations

import logging
import random
from collections import deque
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_N_ACTIONS = 16


@dataclass
class MotorSlowConfig:
    """Серверный проверенный конфиг (core/world_trainer.py:WorldTrainerConfig)."""
    buffer_size: int = 200
    batch_size: int = 16
    train_every: int = 10
    lr: float = 3e-5                  # серверный lr обученных тканей
    entropy_coef: float = 0.15
    entropy_coef_min: float = 0.10
    entropy_coef_max: float = 2.0
    entropy_target: float = 0.4
    max_grad_norm: float = 1.0
    baseline_decay: float = 0.95
    min_buffer_for_train: int = 8
    reward_clip: float = 10.0


@dataclass
class _Exp:
    obs: object          # torch.Tensor [data_dim] (detached cpu)
    action: int          # 0..15 (финальное взятое действие)
    reward: float
    base: object         # torch.Tensor [16] base-logits прайора (detached cpu)


class MotorSlowTrainer:
    """Per-cid медленный REINFORCE-канал для motor_policy ткани Адама.

    Использование (из local_compute):
        tr = MotorSlowTrainer(torch, motor_policy_tissue, scale, cfg)
        tr.record(obs, action, reward, base_logits)        # каждый тик
        if tr.should_train(): tr.train_step(own, temp)      # каждые train_every
    """

    def __init__(self, torch_mod, tissue, motor_scale: float,
                 config: Optional[MotorSlowConfig] = None):
        self.torch = torch_mod
        self.tissue = tissue
        self.scale = float(motor_scale)
        self.config = config or MotorSlowConfig()
        self._buffer: deque = deque(maxlen=self.config.buffer_size)
        self._optimizer = torch_mod.optim.Adam(
            tissue.parameters(), lr=self.config.lr)
        self._baseline: float = 0.0
        self._entropy_coef: float = self.config.entropy_coef
        self.total_ticks: int = 0
        self.total_trains: int = 0
        self.last_loss: float = 0.0
        self.last_entropy: float = 0.0

    def record(self, obs, action: int, reward: float, base_logits) -> None:
        """Записать опыт (obs, финальное действие, reward, base-логиты прайора)."""
        try:
            o = obs.detach().to("cpu").reshape(-1)
            b = base_logits.detach().to("cpu").reshape(-1)[:_N_ACTIONS]
        except Exception:
            return
        r = max(-self.config.reward_clip,
                min(self.config.reward_clip, float(reward)))
        self._buffer.append(_Exp(o, int(action), r, b))
        self._baseline = (self.config.baseline_decay * self._baseline
                          + (1.0 - self.config.baseline_decay) * r)
        self.total_ticks += 1

    def should_train(self) -> bool:
        return (self.total_ticks > 0
                and self.total_ticks % self.config.train_every == 0
                and len(self._buffer) >= self.config.min_buffer_for_train)

    def train_step(self, own: float, temp: float) -> float:
        """sample batch → adv-norm → REINFORCE через motor_policy → backward → step.

        Политика = softmax(base + own·tanh(motor_out[:16]/temp)·scale). Backprop
        только в motor_policy (base detached). own/temp передаются из live-флагов.
        """
        torch = self.torch
        F = torch.nn.functional
        if len(self._buffer) < self.config.batch_size:
            return 0.0
        try:
            device = next(iter(self.tissue.parameters())).device
        except StopIteration:
            return 0.0
        batch = self._sample_batch()
        obs_b = torch.stack([e.obs for e in batch]).to(device)         # [B, D]
        base_b = torch.stack([e.base for e in batch]).to(device)       # [B, 16]
        actions = torch.tensor([e.action for e in batch], device=device)
        rewards = torch.tensor([e.reward for e in batch],
                               dtype=torch.float32, device=device)
        # Advantage = reward − baseline, нормализация по std (серверный, НЕ clamp)
        adv = rewards - self._baseline
        if float(adv.std().item()) > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # Forward с градиентами: motor_delta = tanh(motor_out[:16]/T)·scale
        self.tissue.train()
        out = self.tissue({"input": obs_b})["output"]                  # [B, 64]
        motor_delta = torch.tanh(out[:, :_N_ACTIONS] / max(temp, 1e-6)) * self.scale
        logits = base_b + float(own) * motor_delta                     # [B, 16]
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        policy_loss = -(adv * action_log_probs).mean()
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1).mean()
        self._adapt_entropy_coef(float(entropy.item()))
        loss = policy_loss - self._entropy_coef * entropy
        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.tissue.parameters(), self.config.max_grad_norm)
        self._optimizer.step()
        self.tissue.eval()
        self.total_trains += 1
        self.last_loss = float(loss.item())
        self.last_entropy = float(entropy.item())
        if self.total_trains % 20 == 0:
            logger.info(
                "MOTOR_SLOW_TRAIN trains=%d loss=%.4f entropy=%.4f "
                "ent_coef=%.3f baseline=%.3f buffer=%d",
                self.total_trains, self.last_loss, self.last_entropy,
                self._entropy_coef, self._baseline, len(self._buffer))
        return self.last_loss

    def _adapt_entropy_coef(self, cur: float) -> None:
        """Серверный адаптивный entropy (анти-коллапс) — core/world_trainer.py."""
        t = self.config.entropy_target
        if cur < t * 0.25:
            self._entropy_coef = min(self._entropy_coef * 2.0,
                                     self.config.entropy_coef_max)
        elif cur < t:
            deficit = (t - cur) / t
            self._entropy_coef = min(self._entropy_coef + 0.02 * deficit,
                                     self.config.entropy_coef_max)
        elif cur > t * 2.0:
            self._entropy_coef = max(self._entropy_coef * 0.95,
                                     self.config.entropy_coef_min)

    def _sample_batch(self) -> list:
        """50% значимых (|reward|>0.1) + 50% случайных (серверный приоритет)."""
        bl = list(self._buffer)
        half = self.config.batch_size // 2
        sig = [e for e in bl if abs(e.reward) > 0.1]
        mund = [e for e in bl if abs(e.reward) <= 0.1]
        batch = []
        if sig:
            batch.extend(random.sample(sig, min(half, len(sig))))
        rem = self.config.batch_size - len(batch)
        pool = mund if mund else bl
        if len(pool) >= rem:
            batch.extend(random.sample(pool, rem))
        else:
            batch.extend(pool)
            extra = self.config.batch_size - len(batch)
            if extra > 0 and bl:
                batch.extend(random.choices(bl, k=extra))
        return batch[:self.config.batch_size]

    def stats(self) -> dict:
        return {"trains": self.total_trains, "ticks": self.total_ticks,
                "loss": round(self.last_loss, 5),
                "entropy": round(self.last_entropy, 4),
                "ent_coef": round(self._entropy_coef, 4),
                "baseline": round(self._baseline, 4),
                "buffer": len(self._buffer)}
