"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.63"  # Contract ×2 fix (Хьюберт audit): server step_cost=0.272/server-тик, client дренил ×2 (handle_tick чаще server-тика, дубли world_tick). Guard: метаболизм раз на РАЗЛИЧНЫЙ server-тик (per-cid last tick). METAB_DIAG skip_rate (verify ×2). + инструмент #2: NAV_DIAG p40_ate (P40 ground-truth eat) vs gather_onf (client зеркало) — вскрыть рассинхрон EAT
