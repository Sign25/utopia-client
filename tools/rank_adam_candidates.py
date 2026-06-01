"""Этап 1 (single-organism pivot, ТЗ e3cc81b §5.1): ранжировка кандидатов в Адама.

Read-only офлайн-скан persisted-мозгов колонии (.pt в colony_state_dir / elite).
Считает композит по §5.5 БЕЗ живого мира — все нужные EMA лежат в .pt:
  predictor_loss_ema, intrinsic_ema, entropy_ema, trace_norm_ema, reward_var_ema.

Инвариант (Фрай): accuracy ДОМИНИРУЕТ; intrinsic+reward_var (приспособляемость),
entropy (богатство), trace_norm («учусь») — строго тай-брейкеры СРЕДИ уже
точных, НЕ самостоятельные максимизируемые цели (entropy≈1.0 = шум, не богатство).

Финальный pick Адама утверждает Шеф — скрипт только показывает ранжировку
с разбивкой по компонентам, не выбирает сам.

Запуск:
    python -m tools.rank_adam_candidates <dir> [--top N]
Без <dir> — сканирует cheef.legacy* в ~/.utopia-client/colonies.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import sys
from pathlib import Path

import torch

# Windows-консоль по умолчанию cp1251 — кириллица/«≈» в выводе ломают charmap.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

_RANK_KEYS = ("predictor_loss_ema", "intrinsic_ema", "entropy_ema",
              "trace_norm_ema", "reward_var_ema")


def _load_metrics(pt_path: str) -> dict | None:
    """Извлечь метрики ранжировки из одного .pt (None если не мозг)."""
    try:
        p = torch.load(pt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  [skip] {os.path.basename(pt_path)}: load failed {e}")
        return None
    if "predictor_loss_ema" not in p:
        return None  # не обученный predictor — кандидатом быть не может
    loss = float(p.get("predictor_loss_ema", 99.0))
    return {
        "cid": Path(pt_path).stem,
        "accuracy": math.exp(-loss),               # §5.5 главная
        "loss_ema": loss,
        "intrinsic": float(p.get("intrinsic_ema", 0.0)),
        "reward_var": float(p.get("reward_var_ema", 0.0)),
        "entropy": float(p.get("entropy_ema", 0.0)),
        "trace_norm": float(p.get("trace_norm_ema", 0.0)),
    }


def rank(metrics: list[dict]) -> list[dict]:
    """Сортировка по инварианту: accuracy доминирует. Тай-брейкер (для near-ties
    по accuracy) — приспособляемость+обучение, с ПЕНАЛЬТИ за max-энтропию-шум."""
    for m in metrics:
        # Тай-брейкер ∈ ~[0,1], применяется ТОЛЬКО при равной accuracy.
        # entropy входит как «умеренная хорошо, крайности плохо»: 4·e·(1-e)
        # (пик при e=0.5, →0 при e→0 застыл и e→1 шум). Не максимизируем e.
        ent_mod = 4.0 * m["entropy"] * (1.0 - m["entropy"])
        m["tiebreak"] = (m["intrinsic"] + m["reward_var"]
                         + m["trace_norm"] + 0.25 * ent_mod)
    metrics.sort(key=lambda m: (round(m["accuracy"], 4), m["tiebreak"]),
                 reverse=True)
    return metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", nargs="?", default=None)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args(argv)

    dirs: list[str] = []
    if args.dir:
        dirs = [args.dir]
    else:
        base = Path.home() / ".utopia-client" / "colonies"
        dirs = [str(p) for p in sorted(base.glob("cheef*"))]
    files: list[str] = []
    for d in dirs:
        files += sorted(glob.glob(os.path.join(d, "*.pt")))
    print(f"scan {len(files)} .pt из {len(dirs)} dir(s): "
          f"{[os.path.basename(d) for d in dirs]}\n")

    metrics = [m for m in (_load_metrics(f) for f in files) if m]
    print(f"кандидатов с обученным predictor: {len(metrics)} / {len(files)}\n")
    if not metrics:
        print("НЕТ кандидатов — persist пуст или мозги без predictor.")
        return 1

    ranked = rank(metrics)
    hdr = (f"{'#':>3} {'cid':<10} {'accuracy':>9} {'loss_ema':>9} "
           f"{'intrins':>8} {'rew_var':>9} {'entropy':>8} {'trace_n':>9} "
           f"{'tiebrk':>7}")
    print(hdr)
    print("-" * len(hdr))
    for i, m in enumerate(ranked[:args.top], 1):
        print(f"{i:>3} {m['cid']:<10} {m['accuracy']:>9.4f} "
              f"{m['loss_ema']:>9.4f} {m['intrinsic']:>8.4f} "
              f"{m['reward_var']:>9.2e} {m['entropy']:>8.4f} "
              f"{m['trace_norm']:>9.2e} {m['tiebreak']:>7.4f}")

    accs = [m["accuracy"] for m in metrics]
    print(f"\naccuracy: max={max(accs):.4f} mean={sum(accs)/len(accs):.4f} "
          f"min={min(accs):.4f}")
    ents = [m["entropy"] for m in metrics]
    print(f"entropy:  max={max(ents):.4f} mean={sum(ents)/len(ents):.4f} "
          f"min={min(ents):.4f}  (≈1.0 = action-distribution шум, не богатство)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
