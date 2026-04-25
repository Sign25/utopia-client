"""Бенчмарк ПК — оценка производительности под колонию.

MVP: замер CPU (numpy matmul) + базовая инфа о памяти/GPU.
Полноценный замер с PyTorch — после установки torch (Фаза C/D).
"""

from __future__ import annotations

import platform
import time

import psutil


def cpu_score() -> float:
    """Простой CPU-бенчмарк: 5×5 матумножения 2k×2k."""
    try:
        import numpy as np
    except ImportError:
        return 0.0
    a = np.random.rand(1024, 1024).astype("float32")
    b = np.random.rand(1024, 1024).astype("float32")
    # Прогрев
    _ = a @ b
    t0 = time.perf_counter()
    for _ in range(5):
        _ = a @ b
    elapsed = time.perf_counter() - t0
    # GFLOPS = 2*N^3 / time
    gflops = (2 * 1024**3 * 5) / elapsed / 1e9
    return round(gflops, 2)


def gpu_info() -> dict:
    info: dict = {"available": False}
    try:
        import torch  # type: ignore
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["available"] = True
            info["device"] = torch.cuda.get_device_name(0)
            info["vram_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**3, 1
            )
    except ImportError:
        info["torch"] = None
    return info


def system_info() -> dict:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": platform.processor() or platform.machine(),
        "cpu_count": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1024**3, 1),
    }


def run_full() -> dict:
    return {
        "system": system_info(),
        "cpu_gflops": cpu_score(),
        "gpu": gpu_info(),
    }


def estimate_population(score: dict) -> int:
    """Реалистичный потолок особей в реалтайме (TPS ≥ 2).

    Калибровка по P40: 12 особей → ~360 мс/тик. CPU forward на трансформерах
    в 30-50× медленнее GPU FP32, поэтому без CUDA — единицы особей.
    """
    gpu = score["gpu"]
    if gpu.get("available"):
        vram = gpu.get("vram_gb", 1)
        return max(5, min(50, int(vram * 2)))   # ~2 особи на ГБ VRAM
    gflops = score.get("cpu_gflops", 0.0)
    return max(1, min(5, int(gflops / 100)))    # ~1 особь на 100 GFLOPS
