"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.68"  # ДИАГНОСТИКА: FOOD_SEEK не логировал → berry_pos пуст (ранний return). FLORA_KINDS лог (grass/berry/fruit/other в cache.flora) ВСЕГДА — отличить «нет berry рядом (тундра grass-only)» от «kind в кэше битый». cost растёт 4427→6540, income~0 — berry-seek не зацепился
