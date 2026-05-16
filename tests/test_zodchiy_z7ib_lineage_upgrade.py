"""Z7.i.b — utopia-client poll lineage_upgrade_pending → P40 endpoint direct.

Покрытие:
  Z7ib.1   trigger_p40_lineage_upgrade успешный 200 OK
  Z7ib.2   trigger_p40_lineage_upgrade пустой user_id → None без вызова
  Z7ib.3   trigger_p40_lineage_upgrade 404 (нет серверного Странника) → None
  Z7ib.4   trigger_p40_lineage_upgrade network exception → None (не падаем)
  Z7ib.5   trigger_p40_lineage_upgrade timeout → None
  Z7ib.6   trigger_p40_lineage_upgrade URL формируется корректно с trailing slash

Тест на _fetch_colony_name (запись user_id в cfg) опущен: импорт main.py
тянет numpy/environment/neurocore-пакет, что не покрыто dev-venv'ом
утилитой; логика тривиальная (2 строки) и проверяется в live.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── Z7ib.1 ────────────────────────────────────────────────────────────────

def test_trigger_p40_lineage_upgrade_success():
    from utopia_client.api import UtopiaAPI

    api = UtopiaAPI("https://divisci.test", "TKN")
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {
        "ok": True,
        "creature_id": "abc-123",
        "lineage": "wanderer",
        "energy": 12.5,
        "user_id": "u1",
    }
    with patch("utopia_client.api.requests.post", return_value=fake_resp) as mp:
        result = api.trigger_p40_lineage_upgrade(
            "http://192.168.0.7:8000", "u1")

    assert result is not None
    assert result["ok"] is True
    assert result["creature_id"] == "abc-123"
    call_args = mp.call_args
    assert call_args.args[0].endswith(
        "/api/world/upgrade_lineage_to_zodchiy")
    assert call_args.kwargs["params"] == {"user_id": "u1"}
    assert call_args.kwargs["timeout"] == 5.0


# ── Z7ib.2 ────────────────────────────────────────────────────────────────

def test_trigger_p40_lineage_upgrade_empty_user_id():
    from utopia_client.api import UtopiaAPI

    api = UtopiaAPI("https://divisci.test", "TKN")
    with patch("utopia_client.api.requests.post") as mp:
        result = api.trigger_p40_lineage_upgrade("http://192.168.0.7:8000", "")
    assert result is None
    mp.assert_not_called()


# ── Z7ib.3 ────────────────────────────────────────────────────────────────

def test_trigger_p40_lineage_upgrade_404_no_wanderer():
    from utopia_client.api import UtopiaAPI

    api = UtopiaAPI("https://divisci.test", "TKN")
    fake_resp = MagicMock(
        status_code=404, text='{"detail":"no alive wanderer owned by u1"}')
    with patch("utopia_client.api.requests.post", return_value=fake_resp):
        result = api.trigger_p40_lineage_upgrade(
            "http://192.168.0.7:8000", "u1")
    assert result is None


# ── Z7ib.4 ────────────────────────────────────────────────────────────────

def test_trigger_p40_lineage_upgrade_network_exception():
    from utopia_client.api import UtopiaAPI

    api = UtopiaAPI("https://divisci.test", "TKN")
    with patch("utopia_client.api.requests.post",
               side_effect=Exception("connection refused")):
        result = api.trigger_p40_lineage_upgrade(
            "http://192.168.0.7:8000", "u1")
    assert result is None


# ── Z7ib.5 ────────────────────────────────────────────────────────────────

def test_trigger_p40_lineage_upgrade_timeout():
    import requests as _req

    from utopia_client.api import UtopiaAPI

    api = UtopiaAPI("https://divisci.test", "TKN")
    with patch("utopia_client.api.requests.post",
               side_effect=_req.exceptions.Timeout("timeout")):
        result = api.trigger_p40_lineage_upgrade(
            "http://192.168.0.7:8000", "u1", timeout=0.1)
    assert result is None


# ── Z7ib.6 ────────────────────────────────────────────────────────────────

def test_trigger_p40_lineage_upgrade_url_no_double_slash():
    from utopia_client.api import UtopiaAPI

    api = UtopiaAPI("https://divisci.test", "TKN")
    fake_resp = MagicMock(status_code=200)
    fake_resp.json.return_value = {}
    with patch("utopia_client.api.requests.post", return_value=fake_resp) as mp:
        api.trigger_p40_lineage_upgrade(
            "http://192.168.0.7:8000/", "u1")
    url = mp.call_args.args[0]
    assert url == "http://192.168.0.7:8000/api/world/upgrade_lineage_to_zodchiy"


