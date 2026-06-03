"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.93"  # КОРЕНЬ ИСПРАВЛЕН (Фрай одобрил): инстинкт-развязка food/prey/predator-направления от bias_scale. set_single_organism морозил bias_scale=0 → _shape_action_logits зануляло food-direction (прекондишн навыка). Под single_organism direction=×_instinct_dir_strength (инстинкт-приор, всегда on, умеренный), не ×bias_scale. client_flags instinct_dir_strength (дефолт 0=нейтрально, активирую после ОК Шефа+Фрая). context-boosts+φ остаются curriculum
