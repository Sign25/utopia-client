"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.13.19"  # §4 PREDATOR DEFENSE (predator_defense.md, Фрай 07.06): ATTACK_REACH вскрыл — Адам бьёт в воздух (atk_contact~0), ATTACK завязан на ДОБЫЧУ а не хищника, под single_organism BS=0 зануляло старые бусты. §4: DS-scaled рефлекс по obs[61] — хищник ВПЛОТНУЮ(≥0.8) bias ATTACK[5], ПРИБЛИЖАЕТСЯ(0.15-0.8) bias FLEE[10] рывок; flee-MOVE только при приближении (на контакте стоять-бить); prey-ATTACK ВЫКЛ травоядному (diet≤0.5). Цель строго хищник. Смоук 4/4. §6 trait-growth следом. | 0.13.18 ATTACK_REACH
