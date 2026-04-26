"""Reproduce-гибрид (Phase F3.5) — клиентская сторона.

Когда LocalColonyCompute видит, что организм A выбрал REPRODUCE-action,
вызывается `build_reproduce_envelope(A_cid, A.organism)`:
  1. снять state_dict тканей родителя
  2. наложить gaussian-шум σ=Y50 (1/φ⁵≈0.0902) — point mutations
  3. сжать `torch.save(...)` через zstd, упаковать в base64
  4. envelope `{type: "reproduce", parent_cid, child_weights_b64}`

P40 валидирует, наследует физические traits через `_inherit_traits` и
регистрирует newborn в Мире → отвечает `newborn_ack {parent_cid, child_cid}`.
Клиент по ack добавляет новый организм в LocalColonyCompute с теми же
мутированными весами.

Архитектурные мутации (n_embd Fibonacci, ablation_mask flip) — на стороне P40,
требуют world-context (clan size, niche carnivore quota).
"""

from __future__ import annotations

import base64
import io
import logging

logger = logging.getLogger("utopia_client.reproduce")

# Y50 σ = 1/φ⁵ ≈ 0.0902 — стандартное отклонение наследования весов.
_PHI = (1 + 5 ** 0.5) / 2
DEFAULT_SIGMA = _PHI ** -5  # ≈ 0.09017


def _extract_tissues_state_dict(organism) -> dict:
    """{tid: state_dict} всех тканей CompositeOrganism."""
    return {tid: t.state_dict() for tid, t in organism.tissues.items()}


def mutate_state_dict(tissues_sd: dict, *, sigma: float = DEFAULT_SIGMA,
                      generator=None) -> dict:
    """Y50: child_W = parent_W + σ · std(parent_W) · noise."""
    import torch

    out: dict = {}
    for tid, sd in tissues_sd.items():
        new_sd: dict = {}
        for k, v in sd.items():
            if isinstance(v, torch.Tensor) and v.dtype.is_floating_point:
                std = float(v.detach().float().std().item())
                if std > 0:
                    noise = torch.randn(v.shape, generator=generator,
                                        device=v.device, dtype=v.dtype)
                    new_sd[k] = v + sigma * std * noise
                else:
                    new_sd[k] = v.clone()
            else:
                new_sd[k] = v.clone() if isinstance(v, torch.Tensor) else v
        out[tid] = new_sd
    return out


def pack_zstd_b64(payload: dict) -> str:
    """torch.save → zstd → base64 (UTF-safe строка для JSON-envelope)."""
    import torch
    import zstandard as zstd

    buf = io.BytesIO()
    torch.save(payload, buf)
    raw = buf.getvalue()
    cctx = zstd.ZstdCompressor(level=3)
    compressed = cctx.compress(raw)
    return base64.b64encode(compressed).decode("ascii")


def unpack_zstd_b64(b64: str):
    """base64 → zstd → torch.load(payload).

    weights_only=True — payload содержит только tensors/dicts/strs (state_dict),
    защита от ACE если канал к P40 будет скомпрометирован.
    """
    import torch
    import zstandard as zstd

    compressed = base64.b64decode(b64)
    dctx = zstd.ZstdDecompressor()
    raw = dctx.decompress(compressed)
    return torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)


def build_reproduce_envelope(parent_cid: str, organism, *,
                              sigma: float = DEFAULT_SIGMA) -> dict:
    """Собрать envelope для отправки на P40.

    Возвращает dict готовый к json.dumps:
        {type: "reproduce", parent_cid, child_weights_b64, sigma}

    `child_weights_b64` — zstd-base64 от
        {"tissues_state_dict": {tid: state_dict}}
    """
    parent_sd = _extract_tissues_state_dict(organism)
    child_sd = mutate_state_dict(parent_sd, sigma=sigma)
    payload = {"tissues_state_dict": child_sd}
    return {
        "type": "reproduce",
        "parent_cid": str(parent_cid),
        "child_weights_b64": pack_zstd_b64(payload),
        "sigma": float(sigma),
    }


def apply_state_dict(organism, tissues_sd: dict) -> int:
    """Загрузить state_dict в ткани organism. Возвращает число загруженных."""
    n = 0
    for tid, sd in tissues_sd.items():
        t = organism.tissues.get(tid)
        if t is None:
            logger.warning("apply_state_dict: tissue %s missing in organism", tid)
            continue
        try:
            t.load_state_dict(sd)
            n += 1
        except Exception as e:
            logger.warning("apply_state_dict: load %s failed: %s", tid, e)
    return n
