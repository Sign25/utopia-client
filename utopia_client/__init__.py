"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.83"  # ОТКАТ self-obs→action головы: live-эксперимент показал деградацию foraging (income 123→45 за 1ч, 2 точки — высоковариативный REINFORCE поверх motor-политики шумел). Голова за флагом _self_obs_head_enabled=False (код сохранён для rework). Predictor self-observable остаётся (non-destructive). Адам восстанавливается
