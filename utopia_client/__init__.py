"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.77"  # FIX bootstrap-race: на рестарте флаг single_organism применяется на ~8с позже restore → persisted energy=0 → колониальный death-check → _dead_cids → ЗАМОРОЗКА. set_single_organism теперь оживляет dead-marked (un-mark + recovery energy)
