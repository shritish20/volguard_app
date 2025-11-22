from __future__ import annotations

from collections import deque
from typing import Deque, Tuple

import numpy as np

from config import (
    ACCOUNT_SIZE,
    DAILY_LOSS_LIMIT,
    SYSTEMATIC_MAX_RISK_PERCENT,
    MAX_PORTFOLIO_VEGA,
    MAX_PORTFOLIO_GAMMA,
    MAX_PORTFOLIO_ES,
    logger,
)
from models import PortfolioSnapshot


class CircuitBreaker:
    def __init__(self, max_daily_loss: float):
        self.max_daily_loss = max_daily_loss
        self.tripped = False

    def check(self, daily_pnl: float) -> bool:
        if self.tripped:
            return False
        if daily_pnl <= -self.max_daily_loss:
            self.tripped = True
            logger.critical(f"CIRCUIT BREAKER TRIPPED: daily_pnl={daily_pnl:,.0f}")
            return False
        return True

    def reset(self):
        self.tripped = False


class RiskManager:
    def __init__(self, account_size: float = ACCOUNT_SIZE):
        self.account_size = account_size
        self.daily_pnl = 0.0
        self.equity_now = account_size
        self.max_equity = account_size
        self.pnl_history: Deque[float] = deque(maxlen=1000)
        self.breaker = CircuitBreaker(DAILY_LOSS_LIMIT)

    def update_equity(self, open_pnl: float):
        self.equity_now = self.account_size + self.daily_pnl + open_pnl
        self.max_equity = max(self.max_equity, self.equity_now)

    def drawdown(self) -> float:
        if self.max_equity <= 0:
            return 0.0
        return max(0.0, (self.max_equity - self.equity_now) / self.max_equity)

    def record_pnl(self, pnl: float):
        self.pnl_history.append(pnl)

    def monte_carlo_var(
        self, portfolio_vega: float, current_vix: float, confidence: float = 0.99
    ) -> Tuple[float, float]:
        if len(self.pnl_history) < 100:
            return self.equity_now * 0.02, self.equity_now * 0.03
        num_sim = 1000
        vix_changes = np.random.normal(0, current_vix * 0.1, num_sim)
        pnl_vega = portfolio_vega * vix_changes
        hist = np.array(self.pnl_history)
        hist_std = hist.std()
        market_impact = np.random.normal(0, hist_std, num_sim)
        total = pnl_vega + 0.3 * market_impact
        var = np.percentile(total, (1 - confidence) * 100)
        es = total[total <= var].mean()
        return abs(var), abs(es)

    def can_open(self, snap: PortfolioSnapshot) -> bool:
        if not self.breaker.check(self.daily_pnl + snap.pnl_unrealized):
            return False
        if abs(snap.vega) > MAX_PORTFOLIO_VEGA:
            logger.warning("Vega limit reached")
            return False
        if abs(snap.gamma) > MAX_PORTFOLIO_GAMMA:
            logger.warning("Gamma limit reached")
            return False
        if abs(snap.ES_99) > MAX_PORTFOLIO_ES:
            logger.warning("ES limit reached")
            return False
        return True

    def position_size(self, max_loss_per_lot: float, current_vega: float) -> int:
        if max_loss_per_lot <= 0 or max_loss_per_lot == float("inf"):
            return 0
        max_risk = self.equity_now * SYSTEMATIC_MAX_RISK_PERCENT
        lots_loss = int(max_risk / max_loss_per_lot) if max_loss_per_lot > 0 else 0
        rem_vega = max(0.0, MAX_PORTFOLIO_VEGA - abs(current_vega))
        lots_vega = int(rem_vega / 500.0) if rem_vega > 0 else 0
        lots = min(lots_loss, lots_vega if lots_vega > 0 else lots_loss, 10)
        return max(1, lots) if lots > 0 else 0
