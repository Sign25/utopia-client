"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.75"  # FIX §3 recovery-starvation: death_suppressed(starved) каждый тик ре-армил паралич до recovery → energy вечно 0 (419 start/1 recovery). _enter_paralysis idempotent при истёкшем дедлайне + игнор starv-причин (домен триггера 1)
