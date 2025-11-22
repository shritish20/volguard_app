from __future__ import annotations

import time
from typing import List, Dict, Any, Optional

import requests

from config import API_BASE_V2, API_BASE_V3, UPSTOX_ACCESS_TOKEN, logger


class UpstoxClient:
    def __init__(self, access_token: Optional[str] = None):
        self.access_token = access_token or UPSTOX_ACCESS_TOKEN
        if not self.access_token:
            raise ValueError("No access token set for UpstoxClient")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.access_token}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        self.last_request_time = 0.0
        self.min_request_interval = 0.2
        self.timeout = 10

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _request(
        self,
        method: str,
        url: str,
        params: Dict[str, Any] | None = None,
        data: Dict[str, Any] | None = None,
        retries: int = 3,
        timeout: Optional[int] = None,
    ) -> Dict[str, Any]:
        if timeout is None:
            timeout = self.timeout
        for attempt in range(retries):
            try:
                self._rate_limit()
                resp = self.session.request(
                    method=method, url=url, params=params, json=data, timeout=timeout
                )
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    wait_t = (2**attempt) + 1
                    logger.warning(f"Rate limited, waiting {wait_t}s")
                    time.sleep(wait_t)
                    continue
                else:
                    logger.error(f"API error {resp.status_code}: {resp.text}")
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout (attempt {attempt+1}/{retries})")
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")

            if attempt < retries - 1:
                time.sleep(1)

        return {"status": "error", "data": {}}

    # ===== Market quote =====
    def get_ltp(self, instrument_keys: List[str]) -> Dict[str, Any]:
        if not instrument_keys:
            return {"data": {}}
        params = {"instrument_key": ",".join(instrument_keys)}
        return self._request("GET", f"{API_BASE_V2}/market-quote/ltp", params=params)

    def get_full_quote(self, instrument_keys: List[str]) -> Dict[str, Any]:
        if not instrument_keys:
            return {"data": {}}
        params = {"instrument_key": ",".join(instrument_keys)}
        return self._request("GET", f"{API_BASE_V2}/market-quote/quotes", params=params)

    def get_option_greeks_v3(self, instrument_keys: List[str]) -> Dict[str, Any]:
        if not instrument_keys:
            return {"data": {}}
        params = {"instrument_key": ",".join(instrument_keys)}
        return self._request("GET", f"{API_BASE_V3}/market-quote/option-greek", params=params)

    # ===== Options =====
    def get_option_chain(self, symbol: str, expiry_date: str) -> Dict[str, Any]:
        params = {"symbol": symbol, "expiry_date": expiry_date}
        return self._request("GET", f"{API_BASE_V2}/option/chain", params=params, timeout=15)

    def get_option_contracts(self, symbol: str, expiry_date: str) -> Dict[str, Any]:
        params = {"underlying": symbol, "expiry": expiry_date}
        return self._request("GET", f"{API_BASE_V2}/option/contract", params=params, timeout=15)

    # ===== Orders =====
    def place_order_v2(self, order_data: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", f"{API_BASE_V2}/order/place", data=order_data, timeout=15)

    def place_multi_order_v2(self, orders_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        data = {"orders": orders_data}
        return self._request("POST", f"{API_BASE_V2}/order/multi/place", data=data, timeout=30)

    def get_order_details(self, order_id: str) -> Dict[str, Any]:
        params = {"order_id": order_id}
        return self._request("GET", f"{API_BASE_V2}/order/details", params=params)

    def get_positions(self) -> Dict[str, Any]:
        return self._request("GET", f"{API_BASE_V2}/portfolio/short-term-positions")

    def get_funds_and_margin(self) -> Dict[str, Any]:
        return self._request("GET", f"{API_BASE_V2}/user/get-funds-and-margin")
