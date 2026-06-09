"""§3.2 (Фрай 09.06.2026): felt-thirst gradual drive + client-authoritative
intero. Адам ИНСТИНКТИВНО чувствует водный баланс (insula, бит 15) — новую
ткань НЕ добавляем. Два механизма:

  1. client-built intero[7] из self.biochem (P40-blind risk снят); slots 0,1,3,4,5
     точные, 2/6 (age/valence) — carry последнего P40-intero. Зеркало server-side
     _gather_interoception (нормировки совпадают, incl. raw-camel slot[1]).
  2. градуальный felt-drive: hydration<onset(φ=38.2) → felt=(onset−hyd)/onset
     масштабирует приоритет рефлекса A через детерминированный duty-cycle, вместо
     бинарного 30%. За флагом compute._felt_thirst_drive_enabled (kill-switch).

Регресс: client-intero READS cortisol/serotonin — НЕ мутирует (cortisol-гомеостаз
0.11.7 цел).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

import types  # noqa: E402

import utopia_client.ws_client as wsm  # noqa: E402
from utopia_client.local_compute import (  # noqa: E402
    LocalColonyCompute, _CLIENT_MAX_ENERGY, _CORTISOL_HOMEOSTASIS_DECAY,
)
from utopia_client.biochemistry import make_default  # noqa: E402


class _Cfg:
    def __init__(self, size): self.size = size


class _FakeWC:
    def __init__(self, size, water_cells, creature_pos=None):
        self.config = _Cfg(size)
        t = bytearray(size * size)
        for (r, c) in water_cells:
            t[r * size + c] = 1  # WATER
        self.terrain = bytes(t)
        self.creature_pos = dict(creature_pos or {})


def _client(wc=None):
    cli = wsm.ColonyWSClient(server="https://e.com", token="t",
                             colony_name="cheef", client_version="0.0.0",
                             estimated_population=0)
    cli.world_cache = wc
    return cli


def _compute_with(cid="c1", hydration=50.0, **bc_kw):
    c = LocalColonyCompute(device="cpu")
    bc = make_default()
    bc.hydration = hydration
    for k, v in bc_kw.items():
        setattr(bc, k, v)
    c.biochem[cid] = bc
    c.organisms[cid] = types.SimpleNamespace(generation=0)
    return c


# ── client-built intero ────────────────────────────────────────────────

def test_client_intero_slots_exact():
    c = _compute_with(hydration=50.0, energy=500.0, cortisol=20.0,
                      serotonin=50.0, infection_severity=0.3)
    c.traits["c1"] = {"camel": 10}
    v = c._build_client_intero("c1")
    assert v is not None and v.shape == (7,)
    assert abs(v[0] - 500.0 / _CLIENT_MAX_ENERGY) < 1e-6   # energy
    assert abs(v[1] - 50.0 / (100.0 * 10)) < 1e-6          # hydration/(100·camel)
    assert abs(v[3] - 20.0 / 100.0) < 1e-6                 # cortisol
    assert abs(v[4] - 50.0 / 100.0) < 1e-6                 # serotonin
    assert abs(v[5] - 0.3) < 1e-6                          # infection


def test_client_intero_camel_default_10_when_no_traits():
    c = _compute_with(hydration=50.0)
    v = c._build_client_intero("c1")
    assert abs(v[1] - 50.0 / (100.0 * 10.0)) < 1e-6        # default camel=10


def test_client_intero_carry_slots_2_6():
    import numpy as np
    c = _compute_with(hydration=50.0)
    # P40 прислал age=0.4 (slot2) / valence=-0.1 (slot6) → должны быть carried
    c._last_p40_intero["c1"] = np.array(
        [0.0, 0.0, 0.4, 0.0, 0.0, 0.0, -0.1], dtype=np.float32)
    v = c._build_client_intero("c1")
    assert abs(v[2] - 0.4) < 1e-6      # age carried
    assert abs(v[6] - (-0.1)) < 1e-6   # valence carried


def test_client_intero_slots_2_6_zero_without_carry():
    c = _compute_with(hydration=50.0)
    v = c._build_client_intero("c1")
    assert v[2] == 0.0 and v[6] == 0.0   # нет P40-carry → 0


def test_client_intero_none_without_biochem():
    c = LocalColonyCompute(device="cpu")
    assert c._build_client_intero("ghost") is None


def test_client_intero_readonly_cortisol_regression():
    # Регресс 0.11.7: построение intero ЧИТАЕТ cortisol/serotonin/hydration,
    # но НЕ мутирует их (иначе сломало бы cortisol-гомеостаз).
    c = _compute_with(hydration=44.0, cortisol=33.0, serotonin=55.0)
    bc = c.biochem["c1"]
    c._build_client_intero("c1")
    c._build_client_intero("c1")
    assert bc.cortisol == 33.0 and bc.serotonin == 55.0 and bc.hydration == 44.0


def test_cortisol_homeostasis_decay_intact():
    # Регресс: cortisol-гомеостаз (0.995/тик) не задет правками §3.2.
    # Требует server-package environment.biochemistry (decay_step); без него
    # _apply_biochem_decay рано выходит — пропускаем (dev-venv без neurocore).
    pytest.importorskip("environment.biochemistry")
    c = _compute_with(hydration=50.0, cortisol=80.0)
    bc = c.biochem["c1"]
    c._apply_biochem_decay("c1")
    assert abs(bc.cortisol - 80.0 * _CORTISOL_HOMEOSTASIS_DECAY) < 1e-6


# ── felt-thirst gradual drive (duty-cycle) ──────────────────────────────

def _seek_once(cli, cid="c1", reset_action=1):
    creatures = [{"cid": cid, "row": 10, "col": 10}]
    out = [{"cid": cid, "action": reset_action, "target_id": None}]
    cli._apply_water_seek(creatures, out)
    return out[0]["action"]


def test_binary_mode_default_overrides_thirsty():
    # Флаг OFF (дефолт) → бинарный 30%: hydration<30 всегда override (как было).
    c = _compute_with(hydration=10.0)
    cli = _client(_FakeWC(20, [(10, 16)]))  # вода восточнее → EAST=2
    cli.compute = c
    assert _seek_once(cli) == 2


def test_felt_drive_no_seek_above_onset():
    # Флаг ON, hydration ≥ onset(38.2) → не жаждет, override нет, accum чист.
    c = _compute_with(hydration=40.0)
    c.set_felt_thirst_drive(True)
    cli = _client(_FakeWC(20, [(10, 16)]))
    cli.compute = c
    assert _seek_once(cli) == 1            # не тронут
    assert "c1" not in cli._thirst_accum


def _override_fired(cli, cid="c1"):
    # sentinel 9 (не направление 0-3) → override фаирит ⇔ action сменился.
    # robust к stuck-ротации wd (направление меняется, факт override — нет).
    creatures = [{"cid": cid, "row": 10, "col": 10}]
    out = [{"cid": cid, "action": 9, "target_id": None}]
    cli._apply_water_seek(creatures, out)
    return out[0]["action"] != 9


def test_felt_drive_duty_cycle_matches_felt():
    # Флаг ON: rate override за окно ≈ felt (детерминированный duty-cycle).
    # felt = concave((onset-hyd)/onset)^φ⁻¹ — формула-driven, robust к степени.
    hyd = 19.1
    onset = wsm.ColonyWSClient._THIRST_ONSET
    power = wsm.ColonyWSClient._THIRST_FELT_POWER
    felt = ((onset - hyd) / onset) ** power
    c = _compute_with(hydration=hyd)
    c.set_felt_thirst_drive(True)
    cli = _client(_FakeWC(20, [(10, 16)]))
    cli.compute = c
    N = 50
    overrides = sum(1 for _ in range(N) if _override_fired(cli))
    assert abs(overrides - round(N * felt)) <= 1   # duty-cycle rate = felt


def test_felt_concave_boosts_mid():
    # Concave: felt^φ⁻¹ > linear в mid-зоне (строгое усиление пула жажды).
    onset = wsm.ColonyWSClient._THIRST_ONSET
    power = wsm.ColonyWSClient._THIRST_FELT_POWER
    hyd = 30.0                              # mid-зона
    lin = (onset - hyd) / onset             # ~0.215
    concave = lin ** power
    assert concave > lin                    # бустит
    assert 0.0 < power < 1.0                # φ⁻¹ concave


def test_felt_drive_high_compulsion_every_tick():
    # hydration→0 → felt≈1 → override каждый тик (компульсия).
    c = _compute_with(hydration=0.0)      # felt = 1.0
    c.set_felt_thirst_drive(True)
    cli = _client(_FakeWC(20, [(10, 16)]))
    cli.compute = c
    assert all(_seek_once(cli) == 2 for _ in range(5))


def test_felt_drive_stronger_when_thirstier():
    # Монотонность: больше жажда → выше rate override за фикс. окно тиков.
    def rate(hyd, ticks=20):
        c = _compute_with(hydration=hyd)
        c.set_felt_thirst_drive(True)
        cli = _client(_FakeWC(20, [(10, 16)]))
        cli.compute = c
        return sum(1 for _ in range(ticks) if _seek_once(cli) == 2)
    low = rate(34.0)    # felt ~0.11
    mid = rate(19.1)    # felt 0.5
    high = rate(4.0)    # felt ~0.9
    assert low < mid < high


def test_kill_switch_reverts_to_binary():
    # felt ON → потом OFF (kill-switch): hydration=10 (<30) снова бинарный override.
    c = _compute_with(hydration=10.0)
    c.set_felt_thirst_drive(True)
    c.set_felt_thirst_drive(False)
    cli = _client(_FakeWC(20, [(10, 16)]))
    cli.compute = c
    assert _seek_once(cli) == 2           # бинарная компульсия восстановлена
