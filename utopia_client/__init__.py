"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.91"  # Ступень 2: LOGIT_DEBUG — локализация uniformity policy-выхода Адама (base organism+shaping vs final после motor; энтропии, motor_delta uniformity, argmax-сдвиг). entropy_loss/world_trainer подтверждено СЕРВЕРНЫМ (0 в utopia_client) → не на client-пути Адама; клиент motor=SFNN без entropy-бонуса. Диагностика: base пикует+final flat=motor смазывает; обе flat=reward не выучил preference. Только лог
