"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.88"  # Ступень 2: motor renorm growth-cap (Фрай) — рекалибровка renorm-супрессора через client_flags motor_renorm_cap (числовой, мгновенно). cap=1.0 дефолт=текущий пин (no-op), cap>1=веса заостряются до target×cap (анти-взрыв сохранён). Тест renorm-гипотезы: cap↑ → flip падает = renorm был супрессором
