"""§10.9 ПОГОДА v0.1 (контракт Хьюберт/Фрай 09.06): temperature → obs[35]
(Adam-only слот). obs Адама строит КЛИЕНТ (owned skip_obs) → temp инжектится
client-side (server-патч obs[35] owned не достигает — прецедент nearest_flora,
который клиент zero-pad'ит в [62:63] и шлёт отдельным полем). obs[:64] фикс →
predictor target авто-включает temp@35 → prediction-давление. v0.1 perception-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

pytest.importorskip("torch")

from utopia_client.ws_client import ColonyWSClient  # noqa: E402

_inj = ColonyWSClient._apply_weather_to_obs


def test_temp_injected_at_slot_35():
    obs = [0.0] * 64
    obs[35] = 0.9                      # был steps_taken/5000
    out = _inj(obs, 0.73)
    assert out[35] == pytest.approx(0.73)   # перезаписан temperature
    # остальные слоты не тронуты
    assert all(out[i] == 0.0 for i in range(64) if i != 35)


def test_none_temp_is_noop():
    obs = [0.0] * 64
    obs[35] = 0.42
    out = _inj(obs, None)             # поле отсутствует (до деплоя weather.py)
    assert out[35] == 0.42            # steps_taken не тронут


def test_negative_and_clip_range():
    # temp ∈ [-1,1] — мороз тоже инжектится как есть (clip — серверная забота)
    obs = [0.0] * 64
    assert _inj(obs, -0.5)[35] == pytest.approx(-0.5)


def test_short_obs_no_crash():
    # деградировавший obs (короче 36) → не падаем (IndexError проглочен)
    short = [0.0] * 10
    out = _inj(short, 0.5)           # не бросает
    assert out is short


def test_bad_value_no_crash():
    obs = [0.0] * 64
    out = _inj(obs, "not-a-number")  # TypeError/ValueError проглочен
    assert out[35] == 0.0            # не тронут
