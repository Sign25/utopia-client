"""§3-зона drink-в-параличе (Фрай §18.10, pre-1b.2 gate) — water-seek обходит §3-STAY.

Закрывает §3-зону death-перехода (1b.2): когда hp→§3-порог (~max_hp/φ⁷) и нужда=
вода → §3-paralysis (force-STAY). Вопрос Фрая: Адам ДОСТАНЕТ воду в параличе (drink-
в-§3, non-absorbing) или force-STAY запрёт → спираль→hp=0→death?

Механизм (СУЩЕСТВУЕТ, Фрай/Хьюберт 11.06, hydration-аффорданс): `_apply_water_seek`
(ws_client, ПОСЛЕ _maybe_force_stay §3) ПЕРЕЗАПИСЫВАЕТ action на move-к-воде при
жажде (hyd<30) + флаг `life_critical` (hyd≤15) → P40 обходит §3-force-STAY → «ползёт
к воде в параличе → recovery». Hydration-keyed → независим от того, energy- или
hp-триггернут §3 (⇒ работает на 1b.2 hp-§3). Здесь — ПРЯМОЙ proof override.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utopia_client.ws_client import ColonyWSClient  # noqa: E402
from utopia_client.biochemistry import ClientCreatureBiochem  # noqa: E402


def _ws(hyd):
    """Минимальный ColonyWSClient (bypass __init__) + замоканный world_cache-нав."""
    ws = ColonyWSClient.__new__(ColonyWSClient)
    bc = ClientCreatureBiochem()
    bc.hydration = float(hyd)
    ws.compute = types.SimpleNamespace(biochem={"c0": bc})  # felt-drive отсутств → бинарный
    ws._resolve_pos = lambda cid, c: (5, 5)        # позиция есть
    ws._near_water = lambda r, c: False            # НЕ у воды → должен идти
    ws._water_seek_action = lambda r, c: 0         # ближайшая вода → N (action 0)
    return ws


def test_water_seek_overrides_s3_stay():
    """ГЛАВНЫЙ §3-zone proof: жажда (hyd≤critical) → water-seek ПЕРЕБИВАЕТ §3-STAY
    + life_critical=True (P40 обходит §3 → ползёт к воде в параличе → recovery)."""
    ws = _ws(hyd=10.0)                              # < critical 15
    out = [{"cid": "c0", "action": 4}]             # §3-paralysis поставил STAY=4
    ws._apply_water_seek([{"cid": "c0"}], out)
    assert out[0]["action"] == 0                    # STAY перебит на move-к-воде (drink-в-§3)
    assert out[0]["life_critical"] is True          # hyd≤15 → bypass §3-force-STAY


def test_life_critical_threshold():
    """Жажда но НЕ критично (15<hyd<30): override есть, life_critical=False."""
    ws = _ws(hyd=20.0)
    out = [{"cid": "c0", "action": 4}]
    ws._apply_water_seek([{"cid": "c0"}], out)
    assert out[0]["action"] == 0                    # override (жаждёт)
    assert out[0]["life_critical"] is False         # не критично → обычный override


def test_not_thirsty_keeps_stay():
    """Не жаждет (hyd≥30) → НЕ перебивает §3-STAY (рефлекс молчит)."""
    ws = _ws(hyd=80.0)
    out = [{"cid": "c0", "action": 4}]
    ws._apply_water_seek([{"cid": "c0"}], out)
    assert out[0]["action"] == 4                    # STAY сохранён (нет override)
    assert "life_critical" not in out[0] or out[0].get("life_critical") in (False, None)


def test_at_water_no_override():
    """У воды (near_water) → НЕ override (пьёт на месте, income восстановит)."""
    ws = _ws(hyd=10.0)
    ws._near_water = lambda r, c: True             # уже у воды
    out = [{"cid": "c0", "action": 4}]
    ws._apply_water_seek([{"cid": "c0"}], out)
    assert out[0]["action"] == 4                    # не дёргаем (пьёт на месте)
