"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.65"  # Contract per-sec (Хьюберт): метаболизм energy -= rate × wall-clock_dt между applies, не per-apply. Убирает tick-mismatch ×2 НАВСЕГДА независимо от client TPS (dworld был ~36). Все 4 оси (step_cost/thirst/telomere/infection + dehydration). dt клемп 3с (reconnect). КООРДИНАЦИЯ: активировать ПОСЛЕ per-sec deploy Хьюберта
