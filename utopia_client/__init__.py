"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.89"  # Ступень 2 (a): motor Oja-scale (Фрай) — ослабление Oja-стабилизатора (−post²·W) через client_flags motor_oja_scale (число [0,1], мгновенно). С renorm_cap>1 «освобождает магнитуду» → тест в спокойствии: flip падает (заостряется) или стоит (fundamental)? oja_scale=1.0 дефолт=текущее (no-op). MOTOR_LEARN_DEBUG логирует oja_scale
