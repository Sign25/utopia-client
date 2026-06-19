"""Нав-репеллент (Фрай 19.06): policy-пин active-Адама — move в стену (water/rock,
obs wall-канал) → редирект на проходимое. server-bounce не покрывает client-auth
Адама. Закрывает рекуррентные «замер» на краях/terrain. Флаг nav_repellent (OFF
dormant → bit-identical).

obs[0:32]=8 соседей × [empty/food/WALL/poison]; WALL=idx2/сосед. move 0=N→obs2,
1=S→obs18, 2=E→obs10, 3=W→obs26 (_MOVE_WALL_OBS, подтверждено observation_client
_NEIGHBOR_OFFSETS + client nav-hit).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute, STAY, _MOVE_WALL_OBS,
)


def _c():
    c = LocalColonyCompute(device="cpu")
    c._single_organism = True
    return c


def _obs(walls=()):
    """obs len 80; walls = set move-action'ов, у которых сосед-стена (wall-канал=1)."""
    o = [0.0] * 80
    for a in walls:
        o[_MOVE_WALL_OBS[a]] = 1.0
    return o


def test_mapping_constant():
    assert _MOVE_WALL_OBS == {0: 2, 1: 18, 2: 10, 3: 26}


def test_nav_repellent_setter():
    c = _c()
    assert c._nav_repellent_enabled is False           # dormant default
    assert c.set_nav_repellent(True) is True
    assert c._nav_repellent_enabled is True
    assert c.set_nav_repellent(False) is False


def test_off_no_redirect():
    """flag OFF → move в стену НЕ редиректится (bit-identical)."""
    c = _c()                                            # nav_repellent OFF
    out = {"c0": {"action": 0, "target_id": None}}      # N
    c._apply_nav_repellent("c0", out, _obs(walls={0}))  # N стена, но флаг off
    assert out["c0"]["action"] == 0                      # без изменений


def test_redirect_blocked_to_passable():
    """ON + выбранный move в стену → редирект на первое проходимое (N/S/E/W порядок)."""
    c = _c()
    c.set_nav_repellent(True)
    out = {"c0": {"action": 0, "target_id": None}}      # выбрал N
    # N(0) стена, S(1) стена, E(2) проходимо, W(3) проходимо → редирект на E(2)
    c._apply_nav_repellent("c0", out, _obs(walls={0, 1}))
    assert out["c0"]["action"] == 2                      # E (первое проходимое)


def test_passable_move_unchanged():
    """выбранный move проходим → не трогаем."""
    c = _c()
    c.set_nav_repellent(True)
    out = {"c0": {"action": 2, "target_id": None}}      # E проходим
    c._apply_nav_repellent("c0", out, _obs(walls={0}))  # только N стена
    assert out["c0"]["action"] == 2                      # без изменений


def test_all_blocked_keeps():
    """все 4 направления — стены (заперт) → оставляем (§3/server разрулят)."""
    c = _c()
    c.set_nav_repellent(True)
    out = {"c0": {"action": 0, "target_id": None}}
    c._apply_nav_repellent("c0", out, _obs(walls={0, 1, 2, 3}))
    assert out["c0"]["action"] == 0                      # без изменений (окружён)


def test_non_move_unchanged():
    """STAY/EAT (не move) → не трогаем (только move-действия)."""
    c = _c()
    c.set_nav_repellent(True)
    out = {"c0": {"action": STAY, "target_id": None}}   # STAY=4
    c._apply_nav_repellent("c0", out, _obs(walls={0, 1, 2, 3}))
    assert out["c0"]["action"] == STAY
    out2 = {"c0": {"action": 14, "target_id": None}}    # EAT=14
    c._apply_nav_repellent("c0", out2, _obs(walls={0, 1, 2, 3}))
    assert out2["c0"]["action"] == 14


def test_obs_none_safe():
    c = _c()
    c.set_nav_repellent(True)
    out = {"c0": {"action": 0, "target_id": None}}
    c._apply_nav_repellent("c0", out, None)             # нет obs → no-op
    assert out["c0"]["action"] == 0
