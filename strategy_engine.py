from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np

from config import IST, logger
from analytics import RealDataAnalytics
from utils import get_dte_for_expiry


class StrategyEngine:
    def __init__(self, analytics: RealDataAnalytics):
        self.analytics = analytics
        self.regime_history = deque(maxlen=100)
        self.metrics: Dict[str, Any] = {}
        self._last_strategy_time: datetime | None = None

    def _calc_pcr(self, chain_data: dict) -> float:
        data = chain_data.get("data", {})
        p_vol = 0
        c_vol = 0
        for _, opt in data.items():
            t = opt.get("option_type", "")
            vol = opt.get("volume", 0) or 0
            if t == "PE":
                p_vol += vol
            elif t == "CE":
                c_vol += vol
        return p_vol / max(c_vol, 1)

    def _calc_skew(self, chain_data: dict, spot: float) -> float:
        puts = []
        calls = []
        for o in chain_data.get("data", {}).values():
            k = float(o.get("strike", 0))
            iv = float(o.get("iv", 0))
            t = o.get("option_type", "")
            if not (5 < iv < 150):
                continue
            m = k / spot
            if t == "PE" and m < 0.98:
                puts.append(iv)
            elif t == "CE" and m > 1.02:
                calls.append(iv)
        if puts and calls:
            return (np.mean(puts) - np.mean(calls)) / spot * 1000
        return 0.0

    def _detect_regime(self, spot: float, vix: float, ivp: float, chain: dict, pcr: float) -> Dict[str, Any]:
        skew = self._calc_skew(chain, spot)
        if vix > 30 or ivp > 90:
            regime = "PANIC"
        elif vix < 12 and ivp < 20:
            regime = "CALM_COMPRESSION"
        elif vix > 22:
            regime = "FEAR_BACKWARDATION"
        elif vix > 18 and skew > 1.5:
            regime = "DEFENSIVE_SKEW"
        elif 15 <= vix <= 25 and ivp >= 60:
            regime = "BULL_EXPANSION"
        elif vix < 15 and ivp < 40:
            regime = "LOW_VOL_COMPRESSION"
        else:
            regime = "TRANSITION"

        if regime in ["BULL_EXPANSION", "TRANSITION"]:
            sub = "STRONG_UPTREND" if pcr < 0.8 else "SIDEWAYS"
        elif regime == "PANIC":
            sub = "RISK_OFF"
        else:
            sub = "NEUTRAL"

        conf = 0.7
        if regime == "PANIC" and (vix > 35 or ivp > 85):
            conf += 0.2
        regime_info = {
            "regime": regime,
            "sub_regime": sub,
            "confidence": min(conf, 0.95),
            "timestamp": datetime.now(IST),
            "skew": skew,
        }
        self.regime_history.append(regime_info)
        return regime_info

    def update_metrics(self, spot: float, vix: float, chain: dict):
        ivp = self.analytics.calc_ivp(vix)
        spread = self.analytics.calc_iv_rv_spread(vix)
        pcr = self._calc_pcr(chain)
        regime_info = self._detect_regime(spot, vix, ivp, chain, pcr)
        self.metrics = {
            "spot": spot,
            "vix": vix,
            "ivp": ivp,
            "iv_rv_spread": spread,
            "pcr": pcr,
            "regime": regime_info["regime"],
            "sub_regime": regime_info["sub_regime"],
            "regime_confidence": regime_info["confidence"],
            "skew": regime_info["skew"],
        }
        logger.info(
            f"Regime={regime_info['regime']} Sub={regime_info['sub_regime']} "
            f"Conf={regime_info['confidence']:.0%} IVP={ivp:.1f} PCR={pcr:.2f}"
        )

    def select_strategy(self, atm: float, expiry: str) -> Tuple[str, List[Dict[str, Any]]]:
        if not self.metrics:
            return "SKIP", []
        conf = self.metrics["regime_confidence"]
        if conf < 0.6:
            logger.warning("Low regime confidence – skip")
            return "SKIP", []
        regime = self.metrics["regime"]
        sub = self.metrics["sub_regime"]

        if regime == "PANIC":
            return (
                "WIDE_IRON_CONDOR",
                [
                    {"strike": atm + 500, "type": "CE", "side": "SELL", "expiry": expiry},
                    {"strike": atm + 700, "type": "CE", "side": "BUY", "expiry": expiry},
                    {"strike": atm - 500, "type": "PE", "side": "SELL", "expiry": expiry},
                    {"strike": atm - 700, "type": "PE", "side": "BUY", "expiry": expiry},
                ],
            )
        elif regime == "BULL_EXPANSION":
            if sub == "STRONG_UPTREND":
                return (
                    "BULL_PUT_SPREAD",
                    [
                        {"strike": atm - 100, "type": "PE", "side": "SELL", "expiry": expiry},
                        {"strike": atm - 300, "type": "PE", "side": "BUY", "expiry": expiry},
                    ],
                )
            else:
                return (
                    "IRON_CONDOR_BULLISH",
                    [
                        {"strike": atm + 200, "type": "CE", "side": "SELL", "expiry": expiry},
                        {"strike": atm + 400, "type": "CE", "side": "BUY", "expiry": expiry},
                        {"strike": atm - 300, "type": "PE", "side": "SELL", "expiry": expiry},
                        {"strike": atm - 500, "type": "PE", "side": "BUY", "expiry": expiry},
                    ],
                )
        elif regime == "CALM_COMPRESSION":
            return (
                "IRON_CONDOR_CALM",
                [
                    {"strike": atm + 150, "type": "CE", "side": "SELL", "expiry": expiry},
                    {"strike": atm + 300, "type": "CE", "side": "BUY", "expiry": expiry},
                    {"strike": atm - 150, "type": "PE", "side": "SELL", "expiry": expiry},
                    {"strike": atm - 300, "type": "PE", "side": "BUY", "expiry": expiry},
                ],
            )
        else:
            return (
                "IRON_CONDOR_NEUTRAL",
                [
                    {"strike": atm + 300, "type": "CE", "side": "SELL", "expiry": expiry},
                    {"strike": atm + 500, "type": "CE", "side": "BUY", "expiry": expiry},
                    {"strike": atm - 300, "type": "PE", "side": "SELL", "expiry": expiry},
                    {"strike": atm - 500, "type": "PE", "side": "BUY", "expiry": expiry},
                ],
            )
