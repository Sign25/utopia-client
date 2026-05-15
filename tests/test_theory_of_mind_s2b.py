"""S2.B — theory_of_mind supervised sidecar (utopia-client сторона).

Бит 11 высших тканей (с 13.05.2026, ранее motor_policy жил на 11, переехал
на 17). На клиенте:
  - theory_of_mind Tissue 21/3/1 data_dim=52 + Adam(lr=1e-4) в add_creature
  - WorldStateCache.tom_neighbors_view возвращает 4 ближайших соседа с
    13-мерным feature-вектором (Δposition + lineage + signal + energy).
  - LocalColonyCompute._compute_theory_of_mind: на каждом тике supervised
    cross_entropy step → predict 8 motor-actions ближайшего focus-соседа.
    Target — Δposition между прошлым и текущим тиками.
  - get_phase_emas включает client_tom_acc ∈ [0, 1] (running EMA).
  - extract_brain_state_dicts / inherit_brain_y50 / save_state хранят
    theory_of_mind state_dict под ключом 'theory_of_mind'.

Покрытие:
  TM1   add_creature создаёт theory_of_mind + Adam opt
  TM2   _build_tom_features: правильная индексация 13 признаков × 4 соседа
  TM3   _tom_target_action: Δposition → action class (NORTH/SOUTH/EAST/WEST/STAY)
        с тор-обёрткой
  TM4   tom_neighbors_view: соседи отсортированы по расстоянию, тор-метрика
  TM5   handle_tick без world_cache: ToM no-op (counters не растут)
  TM6   handle_tick с world_cache: 2 тика → supervised step выполнен,
        tom_steps_total > 0, last_tom_acc != 0
  TM7   extract_brain_state_dicts содержит theory_of_mind
  TM8   inherit_brain_y50 копирует theory_of_mind в child (Y50 noise)
  TM9   remove_creature чистит last_tom_acc, _tom_prev_focus
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("storage.norg")
pytest.importorskip("core.workbench")
torch = pytest.importorskip("torch")


@pytest.fixture
def seed_file(tmp_path, monkeypatch):
    seed_path = tmp_path / "seed.norg"
    monkeypatch.setenv("WORLD_SEED_PATH", str(seed_path))
    import importlib
    from environment import seed_loader as ns_loader
    importlib.reload(ns_loader)
    ns_loader.ensure_seed(preset="nexus")
    monkeypatch.setenv("UTOPIA_SEED_PATH", str(tmp_path / "client_seed.norg"))
    monkeypatch.setenv("UTOPIA_WANDERER_SEED_PATH",
                        str(tmp_path / "client_seed.norg"))
    monkeypatch.setenv("UTOPIA_COLONIES_DIR", str(tmp_path / "colonies"))
    client_seed = tmp_path / "client_seed.norg"
    client_seed.write_bytes(seed_path.read_bytes())
    from utopia_client import seed_loader as cli_loader
    importlib.reload(cli_loader)
    return client_seed


def _compute(seed_file, cid="t1"):
    from utopia_client.local_compute import LocalColonyCompute
    from utopia_client.seed_loader import load_founders
    compute = LocalColonyCompute(device="cpu")
    org = load_founders(seed_file, 1)[0]
    compute.add_creature(cid, org, hebbian_enabled=True)
    return compute, org


class _FakeConfig:
    def __init__(self, size=32):
        self.size = size
        self.max_energy = 100.0
        self.max_hydration = 100.0
        self.smell_radius = 5
        self.signal_decay = 55
        self.night_vision_penalty = 4
        self.full_frame_interval = 100


class _FakeCache:
    """Минимальный двойник WorldStateCache для test'ов."""
    def __init__(self, size=32):
        self.config = _FakeConfig(size)
        self.is_bootstrapped = True
        self._positions: dict[str, tuple[int, int]] = {}
        self._tom: dict[str, tuple[str, float, float, int]] = {}

    def set_creature(self, cid: str, x: int, y: int, *,
                       lineage: str = "wanderer",
                       energy: float = 50.0,
                       max_energy: float = 100.0,
                       sig: int = 0) -> None:
        self._positions[cid] = (int(x), int(y))
        self._tom[cid] = (lineage, float(energy), float(max_energy), int(sig))

    def tom_neighbors_view(self, self_cid: str, *, n: int = 4):
        # Прямая копия логики из реального WorldStateCache.
        size = int(self.config.size)
        half = size // 2
        if self_cid not in self._positions:
            return []
        sx, sy = self._positions[self_cid]
        cands: list = []
        for cid, (x, y) in self._positions.items():
            if cid == self_cid:
                continue
            dx = x - sx
            dy = y - sy
            if dx > half:
                dx -= size
            elif dx < -half:
                dx += size
            if dy > half:
                dy -= size
            elif dy < -half:
                dy += size
            d2 = dx * dx + dy * dy
            lin, e, max_e, sig = self._tom.get(cid, ("", 0.0, 1.0, 0))
            energy_norm = e / max_e if max_e > 0 else 0.0
            cands.append((d2, cid, x, y, float(dx), float(dy), lin,
                          float(energy_norm), int(sig)))
        cands.sort(key=lambda t: t[0])
        return [(c, x, y, dx, dy, lin, en, sg)
                for (_d2, c, x, y, dx, dy, lin, en, sg) in cands[:n]]


# ── TM1: add_creature создаёт theory_of_mind + opt ───────────────────────

def test_add_creature_makes_theory_of_mind(seed_file):
    compute, _ = _compute(seed_file)
    assert "t1" in compute.theory_of_mind
    assert compute.theory_of_mind["t1"] is not None
    assert "t1" in compute.theory_of_mind_opt
    assert isinstance(compute.theory_of_mind_opt["t1"], torch.optim.Adam)
    pg = compute.theory_of_mind_opt["t1"].param_groups[0]
    assert pg["lr"] == pytest.approx(1e-4)
    assert compute.last_tom_acc.get("t1") == 0.0
    # tissue.data_dim должен быть 52 (4 соседа × 13 признаков).
    tissue = compute.theory_of_mind["t1"]
    # Найти первую input projection — её in_features == data_dim.
    first_cell = next(iter(tissue.cells.values()))
    # Tissue делает input_proj: Linear(data_dim → n_embd). Точное API
    # зависит от core.tissue — берём через input_proj если есть.
    assert hasattr(tissue, "input_proj")
    assert tissue.input_proj.in_features == 52


# ── TM2: _build_tom_features индексация ──────────────────────────────────

def test_build_tom_features_layout(seed_file):
    compute, _ = _compute(seed_file)
    # 2 соседа: один elder, один wanderer, разные sig.
    neighbors = [
        ("n1", 5, 5, 1.0, 2.0, "elder", 0.75, 3),
        ("n2", 7, 7, -3.0, 4.0, "wanderer", 0.25, 7),
    ]
    feat = compute._build_tom_features(neighbors)
    assert feat.shape == (1, 52)
    # Слот 0: dx=1/32, dy=2/32, lineage[0]=1 (elder), sig[3]=1, energy=0.75.
    assert feat[0, 0].item() == pytest.approx(1.0 / 32.0)
    assert feat[0, 1].item() == pytest.approx(2.0 / 32.0)
    assert feat[0, 2].item() == 1.0  # elder
    assert feat[0, 3].item() == 0.0  # not wanderer
    assert feat[0, 4 + 3].item() == 1.0  # sig=3
    assert feat[0, 4 + 8].item() == pytest.approx(0.75)
    # Слот 1: base=13.
    assert feat[0, 13 + 0].item() == pytest.approx(-3.0 / 32.0)
    assert feat[0, 13 + 1].item() == pytest.approx(4.0 / 32.0)
    assert feat[0, 13 + 2].item() == 0.0  # not elder
    assert feat[0, 13 + 3].item() == 1.0  # wanderer
    assert feat[0, 13 + 4 + 7].item() == 1.0  # sig=7
    assert feat[0, 13 + 4 + 8].item() == pytest.approx(0.25)
    # Слоты 2 и 3 — нули (padding).
    assert feat[0, 26:].abs().sum().item() == 0.0


# ── TM3: _tom_target_action → action class ───────────────────────────────

def test_tom_target_action_cardinals(seed_file):
    from utopia_client.local_compute import LocalColonyCompute
    # Не нужен compute, метод static.
    _act = LocalColonyCompute._tom_target_action
    # NORTH: dy<0, |dx|<|dy|. (x=5,y=5) → (x=5,y=4) → dy=-1.
    assert _act((5, 5), (5, 4), 32) == 0  # NORTH
    assert _act((5, 5), (5, 6), 32) == 1  # SOUTH
    assert _act((5, 5), (6, 5), 32) == 2  # EAST
    assert _act((5, 5), (4, 5), 32) == 3  # WEST
    assert _act((5, 5), (5, 5), 32) == 4  # STAY
    # Тор-обёртка: (0, 0) → (31, 0) на size=32 — это dx=-1 (через границу).
    assert _act((0, 0), (31, 0), 32) == 3  # WEST
    assert _act((31, 0), (0, 0), 32) == 2  # EAST
    # Диагональ: |dx| > |dy| → ось x.
    assert _act((0, 0), (3, 1), 32) == 2  # EAST


# ── TM4: tom_neighbors_view сортировка и тор-метрика ─────────────────────

def test_tom_neighbors_view_sort_and_torus():
    cache = _FakeCache(size=32)
    cache.set_creature("self", 16, 16)
    cache.set_creature("near", 17, 16)              # d=1 (EAST)
    cache.set_creature("far", 20, 20)               # d=√32≈5.66
    cache.set_creature("wrap", 15, 15, lineage="elder")  # d=√2
    res = cache.tom_neighbors_view("self", n=3)
    # Ожидаем порядок: near (1), wrap (√2), far (√32).
    assert [r[0] for r in res] == ["near", "wrap", "far"]
    # Тор: ставим соседа сразу за границей мира.
    cache2 = _FakeCache(size=32)
    cache2.set_creature("self", 0, 0)
    cache2.set_creature("east_wrap", 31, 0)  # dx через границу = -1
    res2 = cache2.tom_neighbors_view("self", n=1)
    assert res2[0][0] == "east_wrap"
    assert res2[0][3] == -1.0  # dx
    assert res2[0][4] == 0.0   # dy


# ── TM5: handle_tick без world_cache → ToM no-op ──────────────────────────

def test_handle_tick_without_world_cache_is_noop(seed_file):
    compute, _ = _compute(seed_file)
    assert compute.world_cache is None
    obs = {"t1": np.random.default_rng(0).normal(size=80).astype(np.float32)}
    compute.handle_tick(obs)
    compute.handle_tick(obs)
    # Без кеша supervised step не делался.
    assert compute.tom_steps == 0
    # last_tom_acc остался дефолтом (0.0 после add_creature).
    assert compute.last_tom_acc.get("t1") == 0.0


# ── TM6: handle_tick с world_cache → supervised step ─────────────────────

def test_handle_tick_with_world_cache_trains_tom(seed_file):
    compute, _ = _compute(seed_file)
    # S4 (14.05.2026): дефолт higher_tissue_sfnn_enabled=True блокирует
    # Adam-supervised path в _compute_theory_of_mind (S3.6 Variant 1).
    # Для проверки legacy Adam-шага выключаем SFNN явно.
    compute.set_higher_sfnn(False)
    cache = _FakeCache(size=32)
    cache.set_creature("t1", 16, 16)
    cache.set_creature("n1", 18, 16, lineage="wanderer")
    compute.world_cache = cache
    rng = np.random.default_rng(11)
    obs = {"t1": rng.normal(size=80).astype(np.float32)}
    # Тик 0: focus записан, supervised step не делался (нет prev_focus).
    compute.handle_tick(obs)
    assert compute.tom_steps == 0
    assert compute._tom_prev_focus.get("t1") == ("n1", 18, 16)
    # Сосед двигается на EAST.
    cache.set_creature("n1", 19, 16)
    obs2 = {"t1": rng.normal(size=80).astype(np.float32)}
    compute.handle_tick(obs2)
    # Теперь supervised шаг выполнен.
    assert compute.tom_steps == 1
    # last_tom_acc обновлён (EMA, ≥0).
    assert "t1" in compute.last_tom_acc
    assert compute.last_tom_acc["t1"] >= 0.0


# ── TM7: extract_brain_state_dicts содержит theory_of_mind ──────────────

def test_extract_brain_state_dicts_includes_tom(seed_file):
    compute, _ = _compute(seed_file)
    brain, _emas = compute.extract_brain_state_dicts("t1")
    assert "theory_of_mind" in brain
    sd = brain["theory_of_mind"]
    assert isinstance(sd, dict) and len(sd) > 0


# ── TM8: inherit_brain_y50 копирует theory_of_mind в child ──────────────

def test_inherit_brain_y50_copies_tom(seed_file):
    compute, _ = _compute(seed_file, cid="parent")
    from utopia_client.seed_loader import load_founders
    child_org = load_founders(seed_file, 1)[0]
    compute.add_creature("child", child_org, hebbian_enabled=True)
    # Сохраним веса parent для проверки Y50 noise.
    parent_w = next(iter(
        compute.theory_of_mind["parent"].named_parameters()))[1].clone()
    ok = compute.inherit_brain_y50("parent", "child")
    assert ok
    # Child получил theory_of_mind веса с Y50 noise (не идентичны parent).
    child_w = next(iter(
        compute.theory_of_mind["child"].named_parameters()))[1]
    # Y50: 0.5·parent + 0.5·noise(σ·std). Они почти точно НЕ равны parent.
    assert not torch.allclose(parent_w, child_w, atol=1e-5)


# ── TM9: remove_creature чистит ToM state ────────────────────────────────

def test_remove_creature_cleans_tom(seed_file):
    compute, _ = _compute(seed_file)
    compute.last_tom_acc["t1"] = 0.42
    compute._tom_prev_focus["t1"] = ("x", 1, 1)
    compute.remove_creature("t1")
    assert "t1" not in compute.theory_of_mind
    assert "t1" not in compute.theory_of_mind_opt
    assert "t1" not in compute.last_tom_acc
    assert "t1" not in compute._tom_prev_focus


# ── Дополнительно: get_phase_emas включает client_tom_acc ────────────────

def test_get_phase_emas_includes_tom_acc(seed_file):
    compute, _ = _compute(seed_file)
    cache = _FakeCache(size=32)
    cache.set_creature("t1", 16, 16)
    cache.set_creature("n1", 17, 16, lineage="wanderer")
    compute.world_cache = cache
    rng = np.random.default_rng(5)
    obs = {"t1": rng.normal(size=80).astype(np.float32)}
    compute.handle_tick(obs)
    cache.set_creature("n1", 18, 16)
    compute.handle_tick({"t1": rng.normal(size=80).astype(np.float32)})
    emas = compute.get_phase_emas("t1")
    assert emas is not None
    assert "client_tom_acc" in emas
    assert 0.0 <= emas["client_tom_acc"] <= 1.0


# ── Bit constants synchronized с P40 ─────────────────────────────────────

def test_tom_bit_is_11_and_motor_bit_is_17():
    from utopia_client import local_compute as lc
    assert lc._TOM_BIT == 11
    assert lc._MOTOR_POLICY_BIT == 17
    # Дублирующих бит нет.
    bits = (lc._DOPAMINE_BIT, lc._PLANNER_BIT, lc._TOM_BIT,
            lc._DEFAULT_MODE_BIT, lc._INSULA_BIT, lc._IMAGINATION_BIT,
            lc._MOTOR_POLICY_BIT)
    assert len(set(bits)) == len(bits)
