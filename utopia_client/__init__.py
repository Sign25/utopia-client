"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.61"  # bias_scale energy-health guard: декей bias только когда колония self-sustaining (income≥cost), не «по населению». Баг 0.11.60: колония закреплена на cap (elite-restore) но голодает → population-ratio=1.0 → bias декеил 1.0→0.30 → untrained motor возвращался → самоподрыв. Кроссфейд работает (gather_onf 1→21 при bias≈1.0), но income≪cost ×8 — энергобаланс отдельный блокер
