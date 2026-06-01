"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.58"  # Stats UI данные: build_colony_summary (выживание/эволюция/обучение + downsampled history) → public_meta.extra.colony_summary через heartbeat (без правки VPS, routes_me пробрасывает extra). Per-creature поля (species_id/gen/topo/age/inst/food) в _per_creature_stats → diagnostics.creatures → фронт. deaths-by-cause счётчик
