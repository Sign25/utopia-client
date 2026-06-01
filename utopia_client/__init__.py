"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.66"  # #1 per-sec transition-safe: *_per_sec→wall-clock dt-интеграция, *_now (legacy)→dt=1 (per-apply, БЕЗ over-drain) — авто-апгрейд когда P40 завершит rename, нет окна rate=0. #2 EAT desync fix: instinct GATHER/EAT по P40 authoritative on_flora+carried_food (9f8d99d) вместо stale cache.flora+зеркало → убирает p40_ate=0
