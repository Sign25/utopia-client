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
    """Грубая оценка max организмов под этот ПК.

    Базис: 256 особей на 12 ГБ RAM при FP32 + 8 ГБ свободного.
    Без GPU — половина.
    """
    ram = score["system"]["ram_gb"]
    base = max(50, int(ram * 20))  # ~20 особей на ГБ RAM
    if not score["gpu"].get("available"):
        base //= 2
    return base
