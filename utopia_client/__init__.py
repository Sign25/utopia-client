"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.87"  # Ступень 2 диагностика: MOTOR_LEARN_DEBUG (Фрай — исключить delivery-bug reward→policy ПЕРВЫМ). Логирует motor_sfnn_steps/baseline/adv_ema(|advantage|)/dw_ema(‖ΔW‖). Разводит (i) delivery-bug (dw≈0/steps плоский) / (ii) credit-fail (всё≠0 + flip 0.99) / (iii) slow-learn. Поведение НЕ меняется (только телеметрия)
