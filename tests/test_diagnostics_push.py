"""Tests для UtopiaAPI.push_diagnostics (Part B+C ТЗ tz_owned_training_diagnostics).

Локальный HTTP server в клиенте отсутствует (daemon только опрашивает VPS).
Поэтому Part B сводится к push-методу + интеграции в daemon-loop.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_push_diagnostics_includes_colony_name():
    from utopia_client.api import UtopiaAPI

    api = UtopiaAPI("https://divisci.test", "TKN")
    diag = {"n_alive": 12, "prediction_accuracy": 0.81, "entropy_avg": 0.43}

    fake_resp = MagicMock(status_code=200, text="{}")
    with patch("utopia_client.api.requests.post", return_value=fake_resp) as mp:
        ok = api.push_diagnostics("home-3060ti", diag)

    assert ok is True
    call_kwargs = mp.call_args.kwargs
    body = call_kwargs["json"]
    assert body["colony_name"] == "home-3060ti"
    assert body["n_alive"] == 12
    assert body["prediction_accuracy"] == 0.81
    assert call_kwargs["headers"]["X-Push-Token"] == "TKN"
    assert mp.call_args.args[0].endswith("/api/colony/diagnostics")


def test_push_diagnostics_handles_http_error():
    from utopia_client.api import UtopiaAPI

    api = UtopiaAPI("https://divisci.test", "TKN")
    fake_resp = MagicMock(status_code=500, text="server down")
    with patch("utopia_client.api.requests.post", return_value=fake_resp):
        ok = api.push_diagnostics("any", {"n_alive": 0})
    assert ok is False


def test_push_diagnostics_handles_network_exception():
    from utopia_client.api import UtopiaAPI

    api = UtopiaAPI("https://divisci.test", "TKN")
    with patch("utopia_client.api.requests.post",
               side_effect=Exception("connection refused")):
        ok = api.push_diagnostics("any", {"n_alive": 0})
    assert ok is False


def test_push_diagnostics_does_not_mutate_input():
    from utopia_client.api import UtopiaAPI

    api = UtopiaAPI("https://divisci.test", "TKN")
    diag = {"n_alive": 5}
    fake_resp = MagicMock(status_code=200, text="{}")
    with patch("utopia_client.api.requests.post", return_value=fake_resp):
        api.push_diagnostics("c", diag)
    assert "colony_name" not in diag  # копия делается, оригинал чист
