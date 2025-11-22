from __future__ import annotations

import time
from datetime import datetime, date, timedelta

from config import (
    IST,
    PAPER_TRADING,
    logger,
)
from analytics import RealDataAnalytics
from execution_engine import ExecutionEngine
from risk_manager import RiskManager
from strategy_engine import StrategyEngine
from upstox_client import UpstoxClient
from utils import get_dte_for_expiry
from volatility import VolatilitySurface2D


def get_weekly_expiry() -> str:
    now = datetime.now(IST)
    today = now.date()
    # Thursday expiry (weekday=3)
    days_ahead = (3 - today.weekday()) % 7
    if days_ahead == 0 and now.time().hour >= 15 and now.time().minute >= 30:
        days_ahead = 7
    exp = today + timedelta(days=days_ahead)
    return exp.strftime("%Y-%m-%d")


def main():
    logger.info("Starting VolGuard v9 (sync backend)")
    client = UpstoxClient()
    analytics = RealDataAnalytics()
    rm = RiskManager()
    engine = ExecutionEngine(client, rm)
    strategy_engine = StrategyEngine(analytics)

    symbol = "NIFTY"

    while True:
        try:
            # 1. Get spot & VIX
            ltp = client.get_ltp(["NSE_INDEX|Nifty 50", "NSE_INDEX|India VIX"])
            spot = float(ltp["data"]["NSE_INDEX|Nifty 50"]["last_price"])
            vix = float(ltp["data"]["NSE_INDEX|India VIX"]["last_price"])
            if spot <= 10000 or spot >= 50000:
                logger.warning(f"Invalid spot {spot}, skipping")
                time.sleep(5)
                continue

            expiry = get_weekly_expiry()
            atm = round(spot / 50) * 50

            # 2. Get option chain
            chain = client.get_option_chain(symbol, expiry)
            if "data" not in chain or not chain["data"]:
                logger.warning("Empty chain, retry")
                time.sleep(5)
                continue

            # 3. Update vol surface
            surf = engine.surface
            surf.update(spot, chain)

            # 4. Regime metrics
            strategy_engine.update_metrics(spot, vix, chain)
            strat_name, legs_def = strategy_engine.select_strategy(atm, expiry)
            if strat_name == "SKIP" or not legs_def:
                logger.info("No strategy selected this cycle")
                time.sleep(30)
                continue

            # Build dummy portfolio snapshot for risk decision (simple)
            snap = rm  # placeholder; in full system you aggregate actual portfolio greeks
            # For now, assume zero portfolio greeks (first trade of the day)
            from models import PortfolioSnapshot

            ps = PortfolioSnapshot(
                timestamp=datetime.now(IST),
                pnl_unrealized=0.0,
                delta=0.0,
                gamma=0.0,
                theta=0.0,
                vega=0.0,
                rho=0.0,
                iv=vix,
                ivp=strategy_engine.metrics["ivp"],
                regime=strategy_engine.metrics["regime"],
                VaR_99=0.0,
                ES_99=0.0,
                total_transaction_costs=0.0,
            )

            if not rm.can_open(ps):
                logger.warning("RiskManager blocked new trade")
                time.sleep(60)
                continue

            # 5. Position sizing
            # For a first pass, assume max_loss_per_lot ~= 10000 (rough)
            lots = rm.position_size(max_loss_per_lot=10000, current_vega=0.0)
            if lots == 0:
                logger.warning("Zero lots from risk sizing, skip")
                time.sleep(60)
                continue

            trade = engine.open_trade(symbol, legs_def, lots, expiry, spot)
            if trade:
                logger.info(
                    f"Trade opened: {strat_name} lots={lots} paper={PAPER_TRADING}"
                )
            else:
                logger.warning("Trade not opened")

            # wait before next cycle
            time.sleep(60)

        except KeyboardInterrupt:
            logger.info("Stopping VolGuard v9 (manual interrupt)")
            break
        except Exception as e:
            logger.error(f"Main loop error: {e}")
            time.sleep(10)


if __name__ == "__main__":
    main()
