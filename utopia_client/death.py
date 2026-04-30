"""Death → palaeo upload (Phase F3.6) — клиентская сторона.

Когда LocalColonyCompute видит, что особь B умерла (HP=0, energy<=0, age>max),
вызывается `build_death_envelope(B_cid, B.organism, reason, fitness)`:
  1. снять state_dict тканей умершей
  2. zstd-base64 (как и в reproduce)
  3. envelope `{type: "death", cid, reason, fitness, weights_b64}`

P40 сохраняет в `/data/palaeo/<cid>.pt.zst` + индексирует в `index.jsonl`,
удаляет особь из Мира → отвечает `death_ack {cid, ts}`.

Палеонтология — однонаправленная: P40 хранит вечно (TTL=∞), клиент после ack
освобождает organism локально.
"""

from __future__ import annotations

import logging

logger = logging.getLogger("utopia_client.death")


def build_death_envelope(cid: str, organism, *, reason: str = "hp_zero",
                          fitness: float = 0.0) -> dict:
    """Собрать envelope для отправки на P40.

    Returns dict готовый к json.dumps:
        {type: "death", cid, reason, fitness, weights_b64}
    """
    from utopia_client.reproduce import (
        _extract_tissues_by_role,
        pack_zstd_b64,
    )

    sd = _extract_tissues_by_role(organism)
    payload = {"tissues_by_role": sd}
    return {
        "type": "death",
        "cid": str(cid),
        "reason": str(reason),
        "fitness": float(fitness),
        "weights_b64": pack_zstd_b64(payload),
    }
