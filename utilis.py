from __future__ import annotations

import time
import json
from datetime import datetime, timedelta
from typing import Any, Dict

from config import IST, logger


def get_dte_for_expiry(expiry_str: str) -> float:
    expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").replace(
        hour=15, minute=30, second=0, tzinfo=IST
    )
    now = datetime.now(IST)
    seconds = max(0, (expiry_dt - now).total_seconds())
    return seconds / (365.0 * 24 * 3600)


class TTLCache:
    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[Any, Any] = {}
        self._ttl = ttl_seconds

    def get(self, key):
        value = self._cache.get(key)
        if not value:
            return None
        data, ts = value
        if time.time() - ts > self._ttl:
            self._cache.pop(key, None)
            return None
        return data

    def set(self, key, value):
        self._cache[key] = (value, time.time())

    def clear(self):
        self._cache.clear()


def load_json_state(path: str, default: Dict[str, Any] | None = None) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return default or {}
    except Exception as e:
        logger.error(f"Failed to load state from {path}: {e}")
        return default or {}


def save_json_state(path: str, state: Dict[str, Any]):
    try:
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Failed to save state to {path}: {e}")
