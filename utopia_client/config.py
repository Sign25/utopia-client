"""Конфиг клиента: ~/.utopia-client/config.json"""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path

DEFAULT_SERVER = "https://divisci.com"


def config_dir() -> Path:
    """Кросс-платформенная папка конфига."""
    if platform.system() == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home())) / "utopia-client"
    else:
        base = Path.home() / ".utopia-client"
    base.mkdir(parents=True, exist_ok=True)
    return base


def config_path() -> Path:
    return config_dir() / "config.json"


def load_config() -> dict:
    p = config_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(cfg: dict) -> None:
    config_path().write_text(
        json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def colonies_dir() -> Path:
    d = config_dir() / "colonies"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_or_prompt(field: str, prompt: str, secret: bool = False) -> str:
    """Вернуть поле из config или спросить у пользователя."""
    cfg = load_config()
    val = cfg.get(field)
    if val:
        return val
    if secret:
        import getpass
        val = getpass.getpass(prompt + ": ").strip()
    else:
        val = input(prompt + ": ").strip()
    if not val:
        raise SystemExit("Пустое значение, выход.")
    cfg[field] = val
    save_config(cfg)
    return val
