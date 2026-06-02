"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.84"  # ЗАПУСК (б) insula-temp моста (go Фрая 02.06): деплой плумбинга client_flags.insula_temp. После self-update set_insula_temp читает флаг; near-identity старт (T_mod=1.0, no-op) → доход не прыгает, мост учится медленно (lr 3e-4). Tripwire income 123→45 → откат флагом (insula_temp=false, без деплоя). Baseline до флипа: income 141 (settling >база 132)
