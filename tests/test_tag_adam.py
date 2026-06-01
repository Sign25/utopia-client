"""Single-organism pivot этап 1: api.tag_adam (POST /api/world/adam/tag).

Лёгкий тест без torch/seed — мок requests.post. Проверяем:
  - успех (200) → возврат JSON
  - пустой cid → None без вызова
  - не-200 → None
  - тело запроса = {"cid": ...}, URL правильный
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utopia_client.api import UtopiaAPI
from utopia_client import api as api_mod


class _Resp:
    def __init__(self, status, payload=None, text=""):
        self.status_code = status
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def test_tag_adam_success(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        return _Resp(200, {"adam_cid": "c103927", "previous_cid": None,
                           "passive_flora_eating": True})

    monkeypatch.setattr(api_mod.requests, "post", fake_post)
    api = UtopiaAPI("https://divisci.com", "tok")
    resp = api.tag_adam("http://192.168.0.7:8000", "c103927")
    assert resp["adam_cid"] == "c103927"
    assert resp["passive_flora_eating"] is True
    assert captured["url"] == "http://192.168.0.7:8000/api/world/adam/tag"
    assert captured["json"] == {"cid": "c103927"}


def test_tag_adam_empty_cid_no_call(monkeypatch):
    called = {"n": 0}

    def fake_post(*a, **k):
        called["n"] += 1
        return _Resp(200)

    monkeypatch.setattr(api_mod.requests, "post", fake_post)
    api = UtopiaAPI("https://divisci.com", "tok")
    assert api.tag_adam("http://192.168.0.7:8000", "") is None
    assert called["n"] == 0  # пустой cid — вызова нет


def test_tag_adam_http_error_returns_none(monkeypatch):
    monkeypatch.setattr(api_mod.requests, "post",
                        lambda *a, **k: _Resp(404, text="not owned"))
    api = UtopiaAPI("https://divisci.com", "tok")
    assert api.tag_adam("http://192.168.0.7:8000", "c103927") is None


def test_tag_adam_strips_trailing_slash(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        api_mod.requests, "post",
        lambda url, json=None, timeout=None: captured.update(url=url)
        or _Resp(200, {}))
    api = UtopiaAPI("https://divisci.com", "tok")
    api.tag_adam("http://192.168.0.7:8000/", "c103927")
    assert captured["url"] == "http://192.168.0.7:8000/api/world/adam/tag"
