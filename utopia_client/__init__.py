"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.12.8"  # output_proj-specific Oja/renorm развязка (Фрай, верифиц. dw_radial≈1=(a)): Oja-член радиален → свампит reward-направление на policy-выходе → ΔW радиален → renorm режет → policy залочена. Фикс: motor_oja_out (Oja-scale ТОЛЬКО output_proj, первично) + motor_renorm_cap_out (вторично). input/attn/mlp Oja+renorm нормальны. Снижение Oja_out → ΔW тангенциальнее → reward шейпит → разлок forage-learning. Верификация: dw_radial падает + m_attack движется + поворот к forage. | 0.12.7: cleanup-фикс
