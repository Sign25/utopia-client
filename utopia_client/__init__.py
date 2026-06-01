"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.60"  # bias_scale curriculum (порт server loop.py:603-636 + routes_world). КОРЕНЬ подтверждён NAV-данными: motor_policy перебивал shaping (flip_rate 0.6, motor_norm≈shaping) → не доходили до флоры (onf_rate 0.09). Кроссфейд own_contribution=max(0,1-bias_scale)×motor_delta: untrained→motor подавлен→флора-градиент ведёт→доходят→едят→обучаются→bias decay→motor автономен. BS в shaping динамический
