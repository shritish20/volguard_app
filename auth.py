from __future__ import annotations

import webbrowser
from dataclasses import dataclass

import requests

from config import (
    UPSTOX_API_BASE,
    UPSTOX_CLIENT_ID,
    UPSTOX_CLIENT_SECRET,
    UPSTOX_REDIRECT_URI,
    logger,
)


@dataclass
class UpstoxToken:
    access_token: str
    refresh_token: str
    expires_in: int


class UpstoxAuth:
    def __init__(self):
        self.session = requests.Session()

    def request_login_token(self):
        url = f"{UPSTOX_API_BASE}/v3/login/auth/token/request/{UPSTOX_CLIENT_ID}"
        payload = {"redirect_uri": UPSTOX_REDIRECT_URI}
        r = self.session.post(url, json=payload)
        if r.status_code != 200:
            raise Exception(f"Token request failed: {r.text}")
        data = r.json()["data"]
        return data["login_uri"], data["request_token"]

    def open_login_dialog(self) -> str:
        login_url, request_token = self.request_login_token()
        logger.info("Opening Upstox login dialog in browser...")
        webbrowser.open(login_url)
        logger.info("After login, copy the `code` query param from redirect URL.")
        return request_token

    def exchange_code_for_token(self, code: str) -> UpstoxToken:
        url = f"{UPSTOX_API_BASE}/v2/login/authorization/token"
        payload = {
            "code": code,
            "redirect_uri": UPSTOX_REDIRECT_URI,
            "client_id": UPSTOX_CLIENT_ID,
            "client_secret": UPSTOX_CLIENT_SECRET,
            "grant_type": "authorization_code",
        }
        r = self.session.post(url, json=payload)
        if r.status_code != 200:
            raise Exception(f"Token exchange failed: {r.text}")
        data = r.json()["data"]
        return UpstoxToken(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_in=data["expires_in"],
        )

    def refresh_access_token(self, refresh_token: str) -> UpstoxToken:
        url = f"{UPSTOX_API_BASE}/v2/login/authorization/token"
        payload = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
            "client_id": UPSTOX_CLIENT_ID,
            "client_secret": UPSTOX_CLIENT_SECRET,
        }
        r = self.session.post(url, json=payload)
        if r.status_code != 200:
            raise Exception(f"Refresh failed: {r.text}")
        data = r.json()["data"]
        return UpstoxToken(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_in=data["expires_in"],
        )
