"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.54"  # Z2.b межтканевой NEAT-overlay подключён (Фрай): mate-flow зовёт crossover_org_topology_for_zodchiy → genes наполняются + apply_topology_overlay_to_org переписывает connections; genes persist (save_state/restore) → resume/elite не теряет divergence; speciation оживёт сама (assign_species видит непустые genes). motor_policy исключён из overlay (sidecar-policy)
