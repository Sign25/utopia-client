"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.97"  # ЭКОНОМИКА (Фрай data-finding): glucose→energy конверсия. Адам ест на макс (glucose 99.7, dopamine 98.8) но energy↓ −0.74/сек — income(P40 yield) < drain, излишек glucose ПРОПАДАЛ. Фикс: излишек glucose(>50)→energy в _apply_metabolism → плотная еда=net-positive, бедная=net-negative (skill важен/достижим). client_flags glucose_energy_rate (дефолт 0, калибрую). Координирую с Хьюбертом (P40 yield/drain). Фикс энергии ДО чистого Phase-1 alignment-замера
