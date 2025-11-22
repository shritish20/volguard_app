from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy import optimize
from scipy.stats import norm

from config import RISK_FREE_RATE, IST, logger
from models import GreeksSnapshot
from utils import get_dte_for_expiry


class RobustSABRModel:
    def __init__(self):
        self.alpha = 0.2
        self.beta = 0.5
        self.rho = -0.2
        self.nu = 0.3
        self.calibrated = False
        self.calibration_error = float("inf")
        self.last_calibration: Optional[datetime] = None

    def _sabr_vol(self, F: float, K: float, T: float) -> float:
        if F <= 0 or K <= 0 or T <= 0:
            return 20.0
        if abs(F - K) < 1e-6:
            return self.alpha / (F ** (1 - self.beta))
        try:
            z = (self.nu / self.alpha) * (F * K) ** ((1 - self.beta) / 2) * math.log(F / K)
            if abs(z) > 100:
                return self.alpha / (F ** (1 - self.beta))
            x = math.log(
                (math.sqrt(1 - 2 * self.rho * z + z * z) + z - self.rho) / (1 - self.rho)
            )
            num = self.alpha * (1 + (2 - 3 * self.rho**2) * self.nu**2 * T / 24)
            den = (F * K) ** ((1 - self.beta) / 2) * (
                1 + (1 - self.beta) ** 2 / 24 * math.log(F / K) ** 2
            )
            if abs(x) < 1e-6 or den == 0:
                return self.alpha / (F ** (1 - self.beta))
            return num / den * z / x
        except Exception:
            return self.alpha / (F ** (1 - self.beta))

    def sabr_volatility(self, F: float, K: float, T: float) -> float:
        v = self._sabr_vol(F, K, T)
        return max(5.0, min(150.0, v))

    def calibrate(self, strikes: List[float], ivs: List[float], F: float, T: float) -> bool:
        if len(strikes) < 5 or T <= 1 / 365:
            return False
        valid = [(K, iv) for K, iv in zip(strikes, ivs) if 5 < iv < 150 and K > 0]
        if len(valid) < 5:
            return False
        Ks, Vs = zip(*valid)

        def objective(params):
            a, b, r, n = params
            self.alpha, self.beta, self.rho, self.nu = a, b, r, n
            errs = []
            for K, iv in zip(Ks, Vs):
                m_iv = self.sabr_volatility(F, K, T)
                errs.append((m_iv - iv) ** 2)
            return math.sqrt(sum(errs) / len(errs))

        init = [0.2, 0.5, -0.2, 0.3]
        bounds = [(0.01, 1.0), (0.1, 0.9), (-0.99, 0.99), (0.1, 1.0)]
        res = optimize.minimize(objective, init, bounds=bounds, method="L-BFGS-B")
        if res.success:
            self.alpha, self.beta, self.rho, self.nu = res.x
            self.calibration_error = res.fun
            self.calibrated = True
            self.last_calibration = datetime.now(IST)
            logger.info(
                f"SABR calibrated α={self.alpha:.3f} β={self.beta:.3f} "
                f"ρ={self.rho:.3f} ν={self.nu:.3f} err={self.calibration_error:.3f}"
            )
            return True
        return False


class VolatilitySurface2D:
    def __init__(self):
        self.surface: Dict[str, RobustSABRModel] = {}
        self.expiries: List[str] = []
        self.spot: float = 0.0
        self.last_calibration: Optional[datetime] = None

    def _fallback_vol(self, strike: float, spot: float) -> float:
        m = strike / spot
        return 25.0 if m < 0.8 or m > 1.2 else 20.0

    def update(self, spot: float, chain_data: Dict[str, Any]) -> bool:
        self.spot = spot
        expiry_data = defaultdict(list)
        for opt in chain_data.get("data", {}).values():
            expiry = opt.get("expiry", "")
            strike = float(opt.get("strike", 0))
            iv = float(opt.get("iv", 0))
            if 5 < iv < 150 and strike > 0:
                expiry_data[expiry].append((strike, iv))

        success = 0
        for expiry, rows in expiry_data.items():
            if len(rows) < 8:
                continue
            Ks, Vs = zip(*rows)
            T = get_dte_for_expiry(expiry)
            m = RobustSABRModel()
            if m.calibrate(list(Ks), list(Vs), spot, T):
                self.surface[expiry] = m
                if expiry not in self.expiries:
                    self.expiries.append(expiry)
                success += 1

        self.expiries.sort()
        self.last_calibration = datetime.now(IST)
        logger.info(f"Vol surface updated: {success}/{len(expiry_data)} expiries calibrated")
        return success > 0

    def get_vol(self, strike: float, expiry: str) -> float:
        if expiry in self.surface and self.surface[expiry].calibrated:
            return self.surface[expiry].sabr_volatility(self.spot, strike, get_dte_for_expiry(expiry))
        return self._fallback_vol(strike, self.spot or strike)


class PricingEngine:
    def __init__(self, surface: VolatilitySurface2D):
        self.surface = surface

    @staticmethod
    def _d1(S, K, T, r, sigma):
        if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
            return 0.0
        return (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    def calculate_greeks(self, spot: float, strike: float, option_type: str, expiry: str) -> GreeksSnapshot:
        T = max(get_dte_for_expiry(expiry), 1.0 / 365)
        iv = self.surface.get_vol(strike, expiry)
        sigma = max(min(iv / 100.0, 2.0), 0.05)
        r = RISK_FREE_RATE

        d1 = self._d1(spot, strike, T, r, sigma)
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == "CE":
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1.0

        gamma = norm.pdf(d1) / (spot * sigma * math.sqrt(T)) if spot > 0 else 0
        vega = spot * norm.pdf(d1) * math.sqrt(T) / 100.0
        theta = (
            -spot * norm.pdf(d1) * sigma / (2 * math.sqrt(T))
            - r * strike * math.exp(-r * T) * norm.cdf(d2)
        ) / 365.0

        delta = max(min(delta, 1.0), -1.0)
        gamma = max(min(gamma, 0.1), 0.0)
        vega = max(min(vega, spot * 0.3), 0.0)

        return GreeksSnapshot(
            timestamp=datetime.now(IST),
            total_delta=delta,
            total_gamma=gamma,
            total_theta=theta,
            total_vega=vega,
            total_rho=0.0,
            staleness_seconds=0.0,
        )
