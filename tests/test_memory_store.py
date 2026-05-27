"""Phase 4 episodic memory persistence — pure helpers tests.

Тестируем `utopia_client.memory_store` без LocalColonyCompute:
  - layout (memory_dir, memory_file)
  - save/load roundtrip с torch tensors
  - version mismatch handling
  - missing file → None
  - invalid cid → ValueError
  - apply_memory_state_to_tissue с fake tissue
  - delete + list

Используем `tmp_path` через `base=` параметр (без monkeypatch на config).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")


# ────────────────────────────────────────────────────────────────────
# Fake tissue
# ────────────────────────────────────────────────────────────────────

class _FakeTissue:
    """Mock tissue с state_dict/load_state_dict для тестов."""

    def __init__(self, payload: dict | None = None):
        import torch
        self._state = payload or {
            "w": torch.zeros(4, 4),
            "b": torch.ones(4),
        }
        self.load_called_with: dict | None = None

    def state_dict(self) -> dict:
        return self._state

    def load_state_dict(self, sd: dict) -> None:
        self.load_called_with = sd
        self._state = sd


# ────────────────────────────────────────────────────────────────────
# Layout / paths
# ────────────────────────────────────────────────────────────────────

def test_memory_dir_creates_subfolder(tmp_path):
    from utopia_client.memory_store import memory_dir
    d = memory_dir(base=tmp_path)
    assert d == tmp_path / "memory"
    assert d.is_dir()


def test_memory_file_safe_cid(tmp_path):
    from utopia_client.memory_store import memory_file
    p = memory_file("abc-123_def", base=tmp_path)
    assert p.name == "abc-123_def.pt"


def test_memory_file_strips_unsafe_chars(tmp_path):
    from utopia_client.memory_store import memory_file
    # ../ → удалится → "etcpasswd"
    p = memory_file("../../etc/passwd", base=tmp_path)
    assert ".." not in p.name
    assert "/" not in p.name
    assert p.parent == tmp_path / "memory"


def test_memory_file_empty_cid_raises(tmp_path):
    from utopia_client.memory_store import memory_file
    with pytest.raises(ValueError):
        memory_file("", base=tmp_path)
    with pytest.raises(ValueError):
        memory_file("///", base=tmp_path)


# ────────────────────────────────────────────────────────────────────
# Save / load roundtrip
# ────────────────────────────────────────────────────────────────────

def test_save_creates_file(tmp_path):
    from utopia_client.memory_store import save_memory_state
    tissue = _FakeTissue()
    path = save_memory_state("cid-1", tissue, base=tmp_path)
    assert path is not None
    assert path.exists()
    assert path.stat().st_size > 0


def test_save_none_tissue_returns_none(tmp_path):
    from utopia_client.memory_store import save_memory_state
    assert save_memory_state("cid-1", None, base=tmp_path) is None


def test_load_missing_returns_none(tmp_path):
    from utopia_client.memory_store import load_memory_state
    assert load_memory_state("nonexistent", base=tmp_path) is None


def test_save_load_roundtrip_preserves_state(tmp_path):
    import torch
    from utopia_client.memory_store import save_memory_state, load_memory_state

    src_state = {
        "embedding.weight": torch.randn(8, 4),
        "bias": torch.tensor([0.1, 0.2, 0.3, 0.4]),
    }
    tissue = _FakeTissue(src_state)
    recall = torch.tensor([0.5, 0.6, 0.7])

    save_memory_state("cid-rt", tissue, recall, base=tmp_path)
    payload = load_memory_state("cid-rt", base=tmp_path)

    assert payload is not None
    assert payload["cid"] == "cid-rt"
    assert payload["version"] == 1
    loaded_sd = payload["episodic_state_dict"]
    assert set(loaded_sd.keys()) == set(src_state.keys())
    for k in src_state:
        assert torch.equal(loaded_sd[k], src_state[k]), \
            f"tensor mismatch at {k}"
    assert torch.equal(payload["last_episodic_recall"], recall)


def test_save_load_roundtrip_no_recall(tmp_path):
    from utopia_client.memory_store import save_memory_state, load_memory_state

    tissue = _FakeTissue()
    save_memory_state("cid-x", tissue, None, base=tmp_path)
    payload = load_memory_state("cid-x", base=tmp_path)
    assert payload is not None
    assert payload["last_episodic_recall"] is None


def test_save_load_roundtrip_via_torch_recall_detach(tmp_path):
    """recall сохраняется detach().cpu() — даже если был grad."""
    import torch
    from utopia_client.memory_store import save_memory_state, load_memory_state

    tissue = _FakeTissue()
    recall = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    save_memory_state("cid-grad", tissue, recall, base=tmp_path)
    payload = load_memory_state("cid-grad", base=tmp_path)
    assert payload is not None
    assert payload["last_episodic_recall"].requires_grad is False


# ────────────────────────────────────────────────────────────────────
# Version mismatch / corrupted
# ────────────────────────────────────────────────────────────────────

def test_load_wrong_version_returns_none(tmp_path):
    import torch
    from utopia_client.memory_store import load_memory_state, memory_file

    path = memory_file("cid-bad", base=tmp_path)
    torch.save({"version": 999, "cid": "cid-bad",
                "episodic_state_dict": {}}, path)
    payload = load_memory_state("cid-bad", base=tmp_path)
    assert payload is None


def test_load_garbage_file_returns_none(tmp_path):
    from utopia_client.memory_store import load_memory_state, memory_file

    path = memory_file("cid-junk", base=tmp_path)
    path.write_bytes(b"not a torch save file")
    payload = load_memory_state("cid-junk", base=tmp_path)
    assert payload is None


# ────────────────────────────────────────────────────────────────────
# Apply state to tissue
# ────────────────────────────────────────────────────────────────────

def test_apply_state_calls_load_state_dict(tmp_path):
    import torch
    from utopia_client.memory_store import (
        save_memory_state, load_memory_state, apply_memory_state_to_tissue,
    )

    src = _FakeTissue({"w": torch.eye(3)})
    save_memory_state("cid-a", src, base=tmp_path)
    payload = load_memory_state("cid-a", base=tmp_path)

    dst = _FakeTissue({"w": torch.zeros(3, 3)})
    ok = apply_memory_state_to_tissue(payload, dst)
    assert ok is True
    assert torch.equal(dst.load_called_with["w"], torch.eye(3))


def test_apply_state_none_payload_returns_false():
    from utopia_client.memory_store import apply_memory_state_to_tissue
    tissue = _FakeTissue()
    assert apply_memory_state_to_tissue(None, tissue) is False
    assert apply_memory_state_to_tissue({"foo": "bar"}, tissue) is False


def test_apply_state_none_tissue_returns_false():
    from utopia_client.memory_store import apply_memory_state_to_tissue
    assert apply_memory_state_to_tissue({"episodic_state_dict": {}}, None) is False


# ────────────────────────────────────────────────────────────────────
# Delete / list
# ────────────────────────────────────────────────────────────────────

def test_delete_memory_file(tmp_path):
    from utopia_client.memory_store import (
        save_memory_state, delete_memory_file, memory_file,
    )
    save_memory_state("cid-del", _FakeTissue(), base=tmp_path)
    assert memory_file("cid-del", base=tmp_path).exists()
    assert delete_memory_file("cid-del", base=tmp_path) is True
    assert not memory_file("cid-del", base=tmp_path).exists()


def test_delete_missing_returns_false(tmp_path):
    from utopia_client.memory_store import delete_memory_file
    assert delete_memory_file("never-existed", base=tmp_path) is False


def test_list_saved_cids(tmp_path):
    from utopia_client.memory_store import save_memory_state, list_saved_cids
    save_memory_state("alpha", _FakeTissue(), base=tmp_path)
    save_memory_state("beta", _FakeTissue(), base=tmp_path)
    cids = list_saved_cids(base=tmp_path)
    assert cids == ["alpha", "beta"]


def test_list_saved_cids_empty(tmp_path):
    from utopia_client.memory_store import list_saved_cids
    assert list_saved_cids(base=tmp_path) == []


# ────────────────────────────────────────────────────────────────────
# End-to-end: simulate restart
# ────────────────────────────────────────────────────────────────────

def test_simulated_restart_preserves_memory(tmp_path):
    """Симулируем «restart»: save→delete tissue→recreate→load."""
    import torch
    from utopia_client.memory_store import (
        save_memory_state, load_memory_state, apply_memory_state_to_tissue,
    )

    # Pre-restart: ткань с обученными весами
    learned_state = {"w": torch.randn(8, 8), "b": torch.randn(8)}
    pre = _FakeTissue(learned_state)
    save_memory_state("cid-restart", pre, base=tmp_path)
    del pre

    # Post-restart: fresh ткань с zero weights
    post = _FakeTissue({"w": torch.zeros(8, 8), "b": torch.zeros(8)})
    payload = load_memory_state("cid-restart", base=tmp_path)
    assert payload is not None
    ok = apply_memory_state_to_tissue(payload, post)
    assert ok is True
    # Веса восстановлены
    for k, v in learned_state.items():
        assert torch.equal(post._state[k], v)
