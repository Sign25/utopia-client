"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.80"  # Track 2 (этап 4): self-observable obs — predictor читает obs68 (env64 + entropy/trace-norm/reward-var/paralyzed) через input_proj [I|0] (non-destructive, math-equivalence доказана). Адам начинает моделировать своё мета-состояние → основа для обучения active EAT. paralyzed-бит закрывает §3 обучающую половину
