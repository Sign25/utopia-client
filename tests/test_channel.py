"""Канал версий клиента (Шеф 17.06): alpha/beta/gamma.

Клиент шлёт свой канал в /api/client/info + /api/client/download → сервер отдаёт
версию ДЛЯ КАНАЛА (alpha=передовое, beta=эталон). Тест: URL содержит ?channel=X
когда канал задан; backward-compat (None → без param). Сетевой вызов замокан.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import utopia_client.api as apimod  # noqa: E402


class _Resp:
    status_code = 200

    def json(self):
        return {"version": "0.13.119"}


def _capture(monkeypatch):
    calls = {}

    def fake_get(url, *a, **k):
        calls["url"] = url
        return _Resp()

    monkeypatch.setattr(apimod.requests, "get", fake_get)
    return calls


def test_info_with_channel(monkeypatch):
    calls = _capture(monkeypatch)
    api = apimod.UtopiaAPI(server="https://e.com", token="t")
    api.get_client_info(channel="beta")
    assert calls["url"] == "https://e.com/api/client/info?channel=beta"


def test_info_no_channel_backward_compat(monkeypatch):
    calls = _capture(monkeypatch)
    api = apimod.UtopiaAPI(server="https://e.com", token="t")
    api.get_client_info()
    assert calls["url"] == "https://e.com/api/client/info"   # без ?channel


def test_info_alpha_channel(monkeypatch):
    calls = _capture(monkeypatch)
    api = apimod.UtopiaAPI(server="https://e.com", token="t")
    api.get_client_info(channel="alpha")
    assert "?channel=alpha" in calls["url"]
