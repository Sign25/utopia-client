"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.64"  # ИЗМЕРЕНИЕ ×2: skip_rate=0 (дублей world_tick НЕТ → guard no-op, ×2 не от дублей). Меряю реальную причину: METAB_DIAG +mean_step_cost (фактический rate от P40 — 0.272 или 0.53?) + mean_dworld (гранулярность world_tick vs apply). #2 подтверждён: p40_ate=0 при eat=6-15
