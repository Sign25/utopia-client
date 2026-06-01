"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.62"  # bias_scale ведём по энергобалансу (health=income/cost), не населению: health<1 (голод)→bias+0.1 (shaping↑), >=1 (self-sustaining)→-0.05 (motor↑). Фикс 0.11.61: bias застрял на 0 (декейнул багом, guard не поднимал). + UI: build_creature_stats (species_id/topo/inst keyed by cid) на heartbeat-канале (diag-push ловит 502)
