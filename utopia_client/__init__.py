"""Utopia Client — клиент-приложение распределённой эволюции."""

__version__ = "0.11.81"  # Track 2 фикс: чистка stale prev_obs при upgrade (убрать транзиентный shape-mismatch 64 vs 68) + restore-robustness (_load_predictor_sd: upgrade-before-load → сохранённый расширенный predictor переживает рестарт, обученный self-observable не теряется)
