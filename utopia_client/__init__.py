"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.12.7"  # cleanup-фикс (code-review): remove_creature чистит ΔW-инструментирование dict'ы (_motor_dw_last/_radial_ema/_cos_ema) — паттерн как у соседних motor-EMA, _motor_dw_last держал tensor per-cid. | 0.12.6: ΔW-инструментирование output_proj
