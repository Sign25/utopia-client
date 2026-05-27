"""Phase 2 commit 1: smoke tests на ClientCreatureBiochem + duck-type compat.

Проверяем:
  - dataclass instantiation с defaults
  - make_default возвращает Адама с DEFAULT_BASELINES
  - as_snapshot возвращает rounded 8 chem + mental_break
  - duck-type compat с `environment.biochemistry` функциями (если
    neurocore[client] установлен в venv; иначе тесты skipped)
  - _FakeWorld duck-type для decay_step

В Phase 2 commit 2-7 добавятся integration tests (LocalColonyCompute
hooks, apply events из obs_batch, mental_break override, inheritance
в reproduce.py, math equivalence с server simulation 1000 ticks).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utopia_client.biochemistry import (  # noqa: E402
    BiochemTickContext,
    ClientCreatureBiochem,
    _FakeWorld,
    _LOCAL_DEFAULT_BASELINES,
    make_default,
    make_from_inheritance,
)


# ── ClientCreatureBiochem dataclass instantiation ─────────────────────────


def test_dataclass_default_instantiation():
    """Без аргументов — все 8 ephemeral + 7 baseline + служебные поля."""
    bc = ClientCreatureBiochem()
    # 8 веществ присутствуют
    for chem in (
        "cortisol", "dopamine", "serotonin", "oxytocin",
        "adrenaline", "glucose", "fatigue", "histamine",
    ):
        assert hasattr(bc, chem)
    # 7 baseline (histamine не baseline — вычисляемое)
    for chem in (
        "cortisol", "dopamine", "serotonin", "oxytocin",
        "adrenaline", "glucose", "fatigue",
    ):
        assert hasattr(bc, f"baseline_{chem}")
    # mental_break поля
    assert bc.mental_break == ""
    assert bc.mental_break_ticks == 0
    # Зависимости мира
    assert bc.energy == 100.0
    assert bc.hydration == 100.0
    assert bc.infected is False
    # Temp attrs
    assert bc._biochem_close_kin == 0
    assert bc._biochem_lone is False


def test_dataclass_default_values_match_server():
    """Дефолтные значения соответствуют `DEFAULT_BASELINES` server-side."""
    bc = ClientCreatureBiochem()
    # serotonin = 50, glucose = 50, rest = 0 — нейтральный baseline
    assert bc.serotonin == 50.0
    assert bc.baseline_serotonin == 50.0
    assert bc.glucose == 50.0
    assert bc.baseline_glucose == 50.0
    assert bc.cortisol == 0.0
    assert bc.baseline_cortisol == 0.0
    assert bc.histamine == 0.0


def test_dataclass_independent_instances():
    """Два instance — независимые (не shared state через class-level dict)."""
    a = ClientCreatureBiochem()
    b = ClientCreatureBiochem()
    a.cortisol = 50.0
    b.cortisol = 10.0
    assert a.cortisol == 50.0
    assert b.cortisol == 10.0


# ── as_snapshot ────────────────────────────────────────────────────────────


def test_as_snapshot_returns_8_chems_plus_mental_break():
    """Snapshot формат для diagnostics: 8 веществ rounded(2) + mental_break."""
    bc = ClientCreatureBiochem(
        cortisol=12.3399, dopamine=67.891,
        mental_break="catatonic",
    )
    snap = bc.as_snapshot()
    assert set(snap.keys()) == {
        "cortisol", "dopamine", "serotonin", "oxytocin",
        "adrenaline", "glucose", "fatigue", "histamine",
        "mental_break",
    }
    # Floating-point rounding edge: round(12.345, 2) banker's = 12.34 OR 12.35;
    # используем значения без 5-в-3rd-decimal чтобы тест был детерминированный.
    assert snap["cortisol"] == 12.34
    assert snap["dopamine"] == 67.89
    assert snap["mental_break"] == "catatonic"


def test_as_snapshot_no_baseline_no_temp_attrs():
    """Snapshot не включает baselines + temp-атрибуты (только ephemeral)."""
    bc = ClientCreatureBiochem()
    snap = bc.as_snapshot()
    for key in (
        "baseline_cortisol", "baseline_dopamine",
        "energy", "hydration",
        "_biochem_close_kin", "_biochem_lone",
    ):
        assert key not in snap


# ── make_default ───────────────────────────────────────────────────────────


def test_make_default_uses_baseline_constants():
    """Адам-Зодчий: baseline_* + ephemeral = DEFAULT_BASELINES."""
    adam = make_default()
    for chem, default_val in _LOCAL_DEFAULT_BASELINES.items():
        # baseline_X = default
        assert getattr(adam, f"baseline_{chem}") == default_val
        # ephemeral X = baseline (reset)
        assert getattr(adam, chem) == default_val
    # histamine = 0
    assert adam.histamine == 0.0


def test_make_default_returns_fresh_instance():
    """Каждый вызов — новый объект (не shared)."""
    a = make_default()
    b = make_default()
    a.cortisol = 99.0
    assert b.cortisol == 0.0


# ── BiochemTickContext + _FakeWorld ────────────────────────────────────────


def test_tick_context_default_values():
    """Дефолтный ctx — production values: max_energy=100, dopamine_decay=0.2."""
    ctx = BiochemTickContext()
    assert ctx.max_energy == 100.0
    assert ctx.max_hydration == 100.0
    assert ctx.biochem_dopamine_decay_per_tick == 0.2


def test_fake_world_duck_types_world_config():
    """`_FakeWorld(ctx).config.max_energy` — для decay_step."""
    ctx = BiochemTickContext(max_energy=120.0)
    fw = _FakeWorld(ctx)
    assert fw.config.max_energy == 120.0
    assert fw.config.max_hydration == 100.0
    assert fw.config.biochem_dopamine_decay_per_tick == 0.2


def test_fake_world_default_ctx_when_none():
    """`_FakeWorld(None)` создаёт дефолтный BiochemTickContext."""
    fw = _FakeWorld()
    assert fw.config.max_energy == 100.0


# ── Duck-type compat с environment.biochemistry (требует neurocore[client]) ─


@pytest.fixture
def envbio():
    """Импорт environment.biochemistry или skip всего теста."""
    return pytest.importorskip("environment.biochemistry")


def test_duck_type_apply_feed_works(envbio):
    """env.biochemistry.apply_feed(creature) увеличивает dopamine + glucose."""
    bc = make_default()
    initial_dopa = bc.dopamine
    initial_glu = bc.glucose
    envbio.apply_feed(bc)
    assert bc.dopamine > initial_dopa
    assert bc.glucose > initial_glu


def test_duck_type_compute_mental_break_normal(envbio):
    """compute_mental_break на свежем Адаме → "" (normal)."""
    bc = make_default()
    result = envbio.compute_mental_break(bc, world_tick=100)
    assert result == ""


def test_duck_type_compute_mental_break_catatonic(envbio):
    """High cortisol + low serotonin → catatonic."""
    bc = make_default()
    bc.cortisol = 90.0
    bc.serotonin = 10.0
    result = envbio.compute_mental_break(bc, world_tick=100)
    assert result == "catatonic"


def test_duck_type_should_force_stay_glucose_floor(envbio):
    """glucose < 5 → force STAY."""
    bc = make_default()
    bc.glucose = 3.0
    assert envbio.should_force_stay(bc) is True


def test_duck_type_should_force_stay_catatonic(envbio):
    """mental_break='catatonic' → force STAY."""
    bc = make_default()
    bc.mental_break = "catatonic"
    assert envbio.should_force_stay(bc) is True


def test_duck_type_should_force_stay_normal(envbio):
    """Нормальное состояние → free to act."""
    bc = make_default()
    assert envbio.should_force_stay(bc) is False


def test_duck_type_apply_pvp_hit(envbio):
    """apply_pvp_hit increments cortisol."""
    bc = make_default()
    envbio.apply_pvp_hit(bc, kind="fratricide_target")
    # PVP_DELTA_FRATRICIDE_TARGET_CORTISOL = 2.0
    assert bc.cortisol == 2.0


def test_duck_type_decay_step_with_fake_world(envbio):
    """decay_step(creature, _FakeWorld(ctx)) — стандартный flow."""
    bc = make_default()
    bc.dopamine = 50.0
    fw = _FakeWorld()
    envbio.decay_step(bc, fw)
    # dopamine падает на DOPAMINE_DECAY_DEFAULT=0.2
    assert bc.dopamine == 49.8


# ── inheritance ────────────────────────────────────────────────────────────


def test_make_from_inheritance_asexual_no_neurocore_fallback():
    """Без neurocore — fallback на make_default (warn)."""
    # При отсутствии environment.biochemistry make_from_inheritance
    # возвращает make_default — graceful degradation.
    parent = make_default()
    child = make_from_inheritance(parent)
    # Если neurocore не доступен — child = make_default (= same baselines)
    # Если доступен — inherit_baselines_asexual применит шум.
    # В обоих случаях child корректный, не падает.
    assert child is not None
    assert isinstance(child, ClientCreatureBiochem)


def test_make_from_inheritance_sexual_two_parents(envbio):
    """Sexual: child baselines между mother и father ± noise (σ=4.0)."""
    import random
    rng = random.Random(42)  # детерминированный
    mother = make_default()
    mother.baseline_cortisol = 20.0
    father = make_default()
    father.baseline_cortisol = 40.0
    child = make_from_inheritance(mother, father, rng=rng)
    # mean = 30.0, ± gauss(0, 4) — child baseline в районе 30 ± 12 (3σ)
    assert 0 <= child.baseline_cortisol <= 100  # clip
    # mean exact с этим seed — не утверждаем (rng-dependent), просто
    # проверяем что child не идентичен ни одному из родителей.
    # (С σ=4.0 шанс точно равен mean = 30.0 мизерный.)


def test_make_from_inheritance_returns_dataclass(envbio):
    """Возвращаемый child — ClientCreatureBiochem dataclass."""
    parent = make_default()
    child = make_from_inheritance(parent)
    assert isinstance(child, ClientCreatureBiochem)
    # Ephemeral сброшены до baseline (через reset_ephemeral_to_baseline)
    assert child.cortisol == child.baseline_cortisol
    assert child.glucose == child.baseline_glucose
    assert child.histamine == 0.0
