"""§10.3 Step-1 (Фрай 10.06): behavioral leave-one-out ПРОБА — ablate graduated-
узел (workbench выход→нули) для замера СИГНАЛА по измерениям самочувствия.
Обратимо мгновенно (граф не возмущаем), переиспользуемый субстрат Stage 3 GC.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402


def _c():
    return LocalColonyCompute(device="cpu")


def test_probe_default_empty():
    c = _c()
    assert c._behavioral_probe_role == ""


def test_probe_sets_and_clears_ablation():
    c = _c()
    org = types.SimpleNamespace(_ablated_roles=set(), _cached_levels=None)
    c.organisms["a"] = org
    assert c.set_behavioral_probe("grown133") == "grown133"
    assert "grown133" in org._ablated_roles
    assert c._behavioral_probe_role == "grown133"
    # снять
    c.set_behavioral_probe("")
    assert "grown133" not in org._ablated_roles
    assert c._behavioral_probe_role == ""


def test_probe_switches_role_without_leak():
    # смена пробы убирает прошлую grownN-абляцию (не копит)
    c = _c()
    org = types.SimpleNamespace(_ablated_roles=set(), _cached_levels=None)
    c.organisms["a"] = org
    c.set_behavioral_probe("grown10")
    c.set_behavioral_probe("grown20")
    assert org._ablated_roles == {"grown20"}


def test_probe_preserves_foreign_ablations():
    # чужие (не grownN) абляции — напр. P40 ablation_mask — не трогаем
    c = _c()
    org = types.SimpleNamespace(_ablated_roles={"motor_policy"}, _cached_levels=0)
    c.organisms["a"] = org
    c.set_behavioral_probe("grown5")
    assert org._ablated_roles == {"motor_policy", "grown5"}
    c.set_behavioral_probe("")
    assert org._ablated_roles == {"motor_policy"}      # чужая осталась


def test_probe_creates_ablated_set_if_missing():
    c = _c()
    org = types.SimpleNamespace()                       # без _ablated_roles
    c.organisms["a"] = org
    c.set_behavioral_probe("grown1")
    assert getattr(org, "_ablated_roles", None) == {"grown1"}
