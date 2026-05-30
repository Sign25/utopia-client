"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.17"  # perf: rate-limit handle_tick (cap ~6.7Hz) — GIL-передышка ws-loop, рубит catch-up бёрсты
