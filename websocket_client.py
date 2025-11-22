from __future__ import annotations

import json
import threading

import requests
import websocket  # pip install websocket-client

from config import UPSTOX_API_BASE, UPSTOX_ACCESS_TOKEN, logger


class UpstoxWebSocket:
    def __init__(self, access_token: str | None = None):
        self.access_token = access_token or UPSTOX_ACCESS_TOKEN
        if not self.access_token:
            raise ValueError("No access token for WebSocket")
        self.ws = None

    def authorize_feed(self) -> str:
        url = f"{UPSTOX_API_BASE}/v2/feed/market-data-feed/authorize"
        headers = {"Authorization": f"Bearer {self.access_token}"}
        r = requests.get(url, headers=headers)
        if r.status_code != 200:
            raise Exception(f"Feed authorization failed: {r.text}")
        return r.json()["data"]["authorized_redirect_uri"]

    def connect(self, instrument_keys: list[str]):
        ws_url = self.authorize_feed()
        logger.info(f"Connecting WS: {ws_url}")
        self.ws = websocket.WebSocketApp(
            ws_url,
            on_open=lambda ws: self._on_open(ws, instrument_keys),
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close,
        )
        t = threading.Thread(target=self.ws.run_forever)
        t.daemon = True
        t.start()
        return t

    def _on_open(self, ws, instrument_keys):
        logger.info("WS connected")
        sub_msg = {
            "guid": "volguard-sub",
            "method": "sub",
            "data": {"instrumentKeys": instrument_keys},
        }
        ws.send(json.dumps(sub_msg))

    def _on_message(self, ws, message: str):
        logger.debug(f"WS message: {message}")

    def _on_error(self, ws, error):
        logger.error(f"WS error: {error}")

    def _on_close(self, ws, status_code, msg):
        logger.warning(f"WS closed: {status_code} {msg}")
