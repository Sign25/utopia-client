"""stamina 1a-norm.2 (issue #22, Фрай §17) — client decay-нормировка /100→/1309.

Фикс PRE-EXISTING рассинхрона `BiochemTickContext.max_energy=100`: client тикал
cortisol/serotonin по /100 (×13 не там, где server /1309). Под флагом decay_norm_1309
→ ctx max_energy=1309 (паритет server).

HARD-блокер Фрая (§17.2): catatonic-петля (cort>80+ser<20→force-STAY) ДОКАЗАТЬ
RECOVERABLE — иначе absorbing → нарушает §3. Здесь — доказательство: decay
cortisol×0.98 (decay_step:487) ДОМИНИРУЕТ даже в худшем случае (голод+низкий
серотонин → max рост cortisol) → fixed-point ~10<80 → выход. force-STAY НЕ запирает.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_NEUROCORE = _ROOT.parent / "NeuroCore"
for _p in (str(_ROOT), str(_NEUROCORE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utopia_client.local_compute import LocalColonyCompute, _CLIENT_MAX_ENERGY  # noqa: E402
from utopia_client.biochemistry import ClientCreatureBiochem  # noqa: E402


def test_flag_default_off():
    c = LocalColonyCompute(device="cpu")
    assert c._decay_norm_enabled is False


def test_flag_setter():
    c = LocalColonyCompute(device="cpu")
    assert c.set_decay_norm(True) is True
    assert c._decay_norm_enabled is True
    assert c.set_decay_norm(False) is False
    assert c._decay_norm_enabled is False


def _bio():
    """Свежий ClientCreatureBiochem + decay_step/compute_mental_break (skip без env)."""
    bm = pytest.importorskip("environment.biochemistry")
    return bm


def _fresh(**kw):
    bc = ClientCreatureBiochem()
    for k, v in kw.items():
        setattr(bc, k, v)
    return bc


def test_catatonic_recoverable_under_1309():
    """HARD-блокер Фрая: catatonic RECOVERABLE под /1309 даже в ХУДШЕМ случае
    (голод energy<392 + низкий серотонин → максимальный рост cortisol). decay×0.98
    доминирует → cortisol выходит из catatonic-зоны (<80) → §3-доктрина соблюдена,
    НЕ absorbing. force-STAY (голод не лечится покоем) НЕ запирает."""
    bm = _bio()
    from utopia_client.biochemistry import _FakeWorld, BiochemTickContext
    # catatonic + ХУДШИЙ случай для recovery: голоден (cortisol-голод горит) +
    # серотонин<20 (low-ser cortisol горит). Оба гонят cortisol ВВЕРХ.
    bc = _fresh(cortisol=100.0, serotonin=10.0, mental_break="catatonic",
                energy=300.0, hydration=50.0)
    ctx = BiochemTickContext(max_energy=_CLIENT_MAX_ENERGY)
    for _ in range(60):                       # hysteresis catatonic = 50 тиков
        bm.decay_step(bc, _FakeWorld(ctx))
    assert bc.cortisol < 80.0, f"cortisol не вышел из catatonic-зоны: {bc.cortisol}"
    # compute_mental_break: cortisol<80 → больше НЕ catatonic → ВЫХОД (≤60 тиков)
    assert bm.compute_mental_break(bc, 0) != "catatonic"
    # сходимость к fixed-point ~10 (decay×0.98 доминирует над hunger+low-ser ростом
    # +0.2/тик → c=0.98(c+0.2) → 9.8) — НЕ застрял высоко, не absorbing
    for _ in range(140):
        bm.decay_step(bc, _FakeWorld(ctx))
    assert bc.cortisol < 15.0, f"не сошёлся к fixed-point: {bc.cortisol}"


def test_catatonic_recovery_monotone_decay():
    """Cortisol МОНОТОННО падает под catatonic×0.98 (нет роста-петли наверх)."""
    bm = _bio()
    from utopia_client.biochemistry import _FakeWorld, BiochemTickContext
    bc = _fresh(cortisol=100.0, serotonin=10.0, mental_break="catatonic",
                energy=300.0, hydration=50.0)
    ctx = BiochemTickContext(max_energy=_CLIENT_MAX_ENERGY)
    prev = bc.cortisol
    for _ in range(20):
        bm.decay_step(bc, _FakeWorld(ctx))
        assert bc.cortisol <= prev + 1e-9, "cortisol вырос — петля наверх (absorbing!)"
        prev = bc.cortisol


def test_norm_shifts_cortisol_hunger_onset():
    """Под /1309 cortisol-голод горит при energy<392 (vs <30 под /100). На energy=350
    /100 говорит «сыт» (ratio 3.5), /1309 — «голоден» (0.267<0.3) → cortisol выше."""
    bm = _bio()
    from utopia_client.biochemistry import _FakeWorld, BiochemTickContext
    a = _fresh(cortisol=50.0, serotonin=50.0, energy=350.0, hydration=60.0)
    b = _fresh(cortisol=50.0, serotonin=50.0, energy=350.0, hydration=60.0)
    for _ in range(10):
        bm.decay_step(a, _FakeWorld(BiochemTickContext(max_energy=100.0)))    # legacy
        bm.decay_step(b, _FakeWorld(BiochemTickContext(max_energy=_CLIENT_MAX_ENERGY)))
    assert b.cortisol > a.cortisol            # /1309 голод → cortisol выше
    assert b.serotonin < a.serotonin          # /1309 голод-дефицит → serotonin ниже


def test_off_path_inert_default_ctx():
    """OFF (decay_norm OFF): _apply_biochem использует дефолтный ctx (max_energy=100)
    → поведение как раньше (инертно). Проверяем что флаг гейтит выбор ctx."""
    c = LocalColonyCompute(device="cpu")
    assert c._decay_norm_enabled is False     # дефолт → ctx None → _FakeWorld default(100)
    c.set_decay_norm(True)
    assert c._decay_norm_enabled is True       # → ctx max_energy=1309
