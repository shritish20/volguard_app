from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import requests

from config import VIX_HISTORY_URL, NIFTY_HISTORY_URL, logger


class RealDataAnalytics:
    def __init__(self):
        self.vix_data = pd.DataFrame()
        self.nifty_data = pd.DataFrame()
        self._load_data()

    def _load_data(self):
        try:
            vix_df = pd.read_csv(VIX_HISTORY_URL)
            nifty_df = pd.read_csv(NIFTY_HISTORY_URL)

            vix_df["Date"] = pd.to_datetime(vix_df["Date"], format="%d-%b-%Y", errors="coerce")
            nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], format="%d-%b-%Y", errors="coerce")

            vix_df = vix_df.sort_values("Date").dropna(subset=["Date"])
            vix_df["Close"] = pd.to_numeric(vix_df["Close"], errors="coerce")
            self.vix_data = vix_df.dropna(subset=["Close"])

            nifty_df = nifty_df.sort_values("Date").dropna(subset=["Date"])
            nifty_df["Close"] = pd.to_numeric(nifty_df["Close"], errors="coerce")
            self.nifty_data = nifty_df.dropna(subset=["Close"])

            logger.info(
                f"Loaded {len(self.vix_data)} VIX records, {len(self.nifty_data)} Nifty records"
            )
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}")
            self._fallback_data()

    def _fallback_data(self):
        dates = pd.date_range(start="2020-01-01", end=datetime.today(), freq="D")
        vix = np.clip(np.random.normal(15, 5, len(dates)), 10, 40)
        self.vix_data = pd.DataFrame({"Date": dates, "Close": vix})

        prices = 10000 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, len(dates))))
        self.nifty_data = pd.DataFrame({"Date": dates, "Close": prices})
        logger.warning("Using synthetic fallback data")

    def calc_ivp(self, current_vix: float, lookback_days: int = 252) -> float:
        if self.vix_data.empty:
            return 50.0
        recent = self.vix_data.tail(lookback_days)["Close"]
        ivp = (recent < current_vix).mean() * 100.0
        return float(max(0.0, min(100.0, ivp)))

    def calc_iv_rv_spread(self, current_vix: float, lookback_days: int = 30) -> float:
        if self.nifty_data.empty or len(self.nifty_data) < lookback_days + 5:
            return 0.0
        prices = self.nifty_data.tail(lookback_days + 5)["Close"]
        returns = np.log(prices / prices.shift(1)).dropna()
        realized_vol = returns.std() * np.sqrt(252) * 100.0
        return float(current_vix - realized_vol)
