"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.92"  # LOGIT_DEBUG fix: torch=self._torch (был NameError → тихий fail, count=0) + захват в else-ветке + own= в логе (применяется ли motor: _own=max(0,1-bias_scale)). Локализация uniformity: base(organism+shaping) vs final(после motor)
