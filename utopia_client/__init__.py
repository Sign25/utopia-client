"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.16"  # perf: offload handle_tick на asyncio.to_thread — ws-loop не блокируется (142 орг → keepalive)
