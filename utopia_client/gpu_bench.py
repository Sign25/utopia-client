"""Phase F3.0: честный GPU forward-бенч на 3060 Ti.

Прогоняет StemCell forward на CUDA при разных N/dtype/режимах, печатает
ms/тик и потолок TPS. Цель — понять, сколько особей реально считать
локально при 30 TPS Мира.

Запуск:
    utopia-client.bat bench-gpu
"""

from __future__ import annotations

import time
from typing import Optional

import torch


def _make_cells(n: int, dtype: torch.dtype, device: torch.device) -> list:
    from core.stem_cell import StemCell
    from core.organism import Genome
    g = Genome()
    g_dict = g.to_dict() if hasattr(g, "to_dict") else dict(g.__dict__)
    cells = []
    for _ in range(n):
        m = StemCell(g_dict).to(device=device, dtype=dtype)
        m.eval()
        cells.append(m)
    return cells


def _bench_serial(n: int, dtype: torch.dtype, device: torch.device,
                   steps: int = 100) -> float:
    """N forward-проходов в цикле (как было бы без батчинга)."""
    cells = _make_cells(n, dtype, device)
    xs = [torch.randn(1, 80, device=device, dtype=dtype) for _ in range(n)]
    with torch.no_grad():
        for _ in range(5):
            for c, x in zip(cells, xs):
                _ = c(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            for c, x in zip(cells, xs):
                _ = c(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1000.0


def _bench_batched(n: int, dtype: torch.dtype, device: torch.device,
                    steps: int = 100) -> float:
    """Один батчевый forward (N, 80) — нижняя граница времени."""
    cells = _make_cells(1, dtype, device)
    cell = cells[0]
    x = torch.randn(n, 80, device=device, dtype=dtype)
    with torch.no_grad():
        for _ in range(5):
            _ = cell(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(steps):
            _ = cell(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1000.0


def run_bench(n_list: Optional[list[int]] = None) -> dict:
    info: dict = {"cuda": torch.cuda.is_available()}
    if not info["cuda"]:
        info["error"] = "CUDA недоступна"
        return info
    dev = torch.device("cuda")
    info["device"] = torch.cuda.get_device_name(0)
    info["torch"] = torch.__version__
    n_list = n_list or [16, 32, 64, 128]
    info["table"] = []
    # FP16 для StemCell требует autocast — без него RMSNorm теряет dtype.
    # В этом бенче — только FP32 (нижняя/верхняя граница без mixed-precision).
    for n in n_list:
        try:
            ms_serial = _bench_serial(n, torch.float32, dev)
        except Exception as e:
            ms_serial = float("nan")
            info.setdefault("errors", []).append(
                f"serial N={n}: {type(e).__name__}: {e}")
        try:
            ms_batched = _bench_batched(n, torch.float32, dev)
        except Exception as e:
            ms_batched = float("nan")
            info.setdefault("errors", []).append(
                f"batched N={n}: {type(e).__name__}: {e}")
        tps_serial = (1000.0 / ms_serial) if ms_serial > 0 else 0.0
        tps_batched = (1000.0 / ms_batched) if ms_batched > 0 else 0.0
        info["table"].append({
            "n": n,
            "dtype": "fp32",
            "ms_serial": round(ms_serial, 3),
            "tps_serial": round(tps_serial, 1),
            "ms_batched": round(ms_batched, 3),
            "tps_batched": round(tps_batched, 1),
        })
    return info


def print_report(info: dict) -> None:
    if not info.get("cuda"):
        print("CUDA недоступна:", info.get("error"))
        return
    print(f"GPU:    {info['device']}")
    print(f"Torch:  {info['torch']}")
    print()
    print(f"{'N':>4} {'dtype':>5} | {'serial':>10} {'TPS':>7} | "
          f"{'batched':>10} {'TPS':>7}")
    print("-" * 60)
    for row in info["table"]:
        print(f"{row['n']:>4} {row['dtype']:>5} | "
              f"{row['ms_serial']:>7.2f} ms {row['tps_serial']:>7.1f} | "
              f"{row['ms_batched']:>7.2f} ms {row['tps_batched']:>7.1f}")
    if info.get("errors"):
        print("\nErrors:")
        for e in info["errors"]:
            print(" -", e)


def cmd_bench_gpu(args=None) -> int:
    info = run_bench()
    print_report(info)
    return 0


if __name__ == "__main__":
    cmd_bench_gpu()
