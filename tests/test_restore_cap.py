"""Carrying-capacity cap для restore_colony_from_local (31.05.2026, Шеф).

Persist накапливал сотни .pt без чистки → restore всех → колония 20× сверх
ёмкости железа → cheef-PC не тикает → Мир замерзал. _cap_and_clean_pt грузит
только N свежайших (N = estimate_population), УДАЛЯЕТ устаревшие.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

from utopia_client.local_compute import LocalColonyCompute  # noqa: E402


def _make_pt_files(d: Path, n: int) -> list[Path]:
    """n dummy .pt с возрастающими mtime (файл i новее файла i-1)."""
    files = []
    for i in range(n):
        f = d / f"cid_{i:03d}.pt"
        f.write_bytes(b"x")
        # mtime: больший i = новее
        os.utime(f, (1000 + i, 1000 + i))
        files.append(f)
    return files


def test_cap_keeps_newest_deletes_rest(tmp_path):
    _make_pt_files(tmp_path, 50)
    keep, culled = LocalColonyCompute._cap_and_clean_pt(tmp_path, cap=16)
    assert len(keep) == 16
    assert len(culled) == 34  # culled cid'ы возвращены (для owned_bye)
    remaining = sorted(tmp_path.glob("*.pt"))
    assert len(remaining) == 16
    kept_names = {p.stem for p in remaining}
    assert "cid_049" in kept_names  # новейший
    assert "cid_034" in kept_names  # граница (50-16=34)
    assert "cid_033" not in kept_names  # за границей — удалён
    assert "cid_000" not in kept_names  # старейший — удалён
    # culled cid'ы = удалённые старейшие (для despawn на P40)
    assert "cid_000" in culled
    assert "cid_033" in culled
    assert "cid_034" not in culled  # сохранён, не в culled


def test_cap_under_count_keeps_all(tmp_path):
    _make_pt_files(tmp_path, 10)
    keep, culled = LocalColonyCompute._cap_and_clean_pt(tmp_path, cap=16)
    assert len(keep) == 10  # меньше cap → всё сохранено
    assert culled == []
    assert len(list(tmp_path.glob("*.pt"))) == 10


def test_cap_zero_deletes_all(tmp_path):
    _make_pt_files(tmp_path, 5)
    keep, culled = LocalColonyCompute._cap_and_clean_pt(tmp_path, cap=0)
    assert keep == []
    assert len(culled) == 5
    assert list(tmp_path.glob("*.pt")) == []


def test_empty_dir_noop(tmp_path):
    keep, culled = LocalColonyCompute._cap_and_clean_pt(tmp_path, cap=16)
    assert keep == []
    assert culled == []
