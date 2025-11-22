from __future__ import annotations

import csv
from datetime import datetime
from typing import List, Optional

from config import (
    IST,
    PAPER_TRADING,
    TRADE_JOURNAL_FILE,
    BASE_SLIPPAGE,
    PANIC_EXIT_FACTOR,
    logger,
)
from models import (
    MultiLegTrade,
    TradeStatus,
    ExitReason,
    Position,
    OrderType,
    TransactionType,
    ProductType,
)
from risk_manager import RiskManager
from upstox_client import UpstoxClient
from volatility import PricingEngine, VolatilitySurface2D
from utils import TTLCache


class ExecutionEngine:
    def __init__(self, client: UpstoxClient, rm: RiskManager):
        self.client = client
        self.rm = rm
        self.surface = VolatilitySurface2D()
        self.pricing = PricingEngine(self.surface)
        self.instrument_cache = TTLCache(ttl_seconds=3600)

    def _log_trade(self, trade: MultiLegTrade, action: str, realized_pnl: float = 0.0):
        try:
            file_exists = False
            try:
                with open(TRADE_JOURNAL_FILE, "r"):
                    file_exists = True
            except FileNotFoundError:
                pass

            with open(TRADE_JOURNAL_FILE, "a", newline="") as f:
                w = csv.writer(f)
                if not file_exists:
                    w.writerow(
                        [
                            "timestamp",
                            "action",
                            "strategy",
                            "lots",
                            "net_prem_per_share",
                            "total_credit",
                            "vega",
                            "expiry",
                            "legs_count",
                            "realized_pnl",
                            "transaction_costs",
                        ]
                    )
                w.writerow(
                    [
                        datetime.now(IST).isoformat(),
                        action,
                        trade.strategy_type,
                        trade.lots,
                        trade.net_premium_per_share,
                        trade.total_credit(),
                        trade.trade_vega,
                        trade.expiry_date,
                        len(trade.legs),
                        realized_pnl,
                        trade.transaction_costs,
                    ]
                )
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def _get_instrument_key(self, symbol: str, expiry: str, strike: float, opt_type: str) -> str:
        key = f"{symbol}_{expiry}_{strike}_{opt_type}"
        cached = self.instrument_cache.get(key)
        if cached:
            return cached
        data = self.client.get_option_contracts(symbol, expiry)
        for c in data.get("data", []):
            try:
                k = float(c.get("strike", 0))
                t = c.get("option_type")
                if abs(k - strike) < 1e-3 and t == opt_type:
                    inst = c["instrument_key"]
                    self.instrument_cache.set(key, inst)
                    return inst
            except Exception:
                continue
        raise ValueError(f"Contract not found {symbol} {expiry} {strike} {opt_type}")

    def _compute_entry_price(self, last_price: float, sell: bool) -> float:
        sl = BASE_SLIPPAGE
        if sell:
            return last_price * (1 + sl)
        else:
            return last_price * (1 - sl)

    def open_trade(self, symbol: str, legs_def: List[dict], lots: int, expiry: str, spot: float) -> Optional[MultiLegTrade]:
        legs: List[Position] = []
        for leg_def in legs_def:
            strike = leg_def["strike"]
            opt_type = leg_def["type"]
            side = leg_def["side"]  # SELL/BUY
            inst_key = self._get_instrument_key(symbol, expiry, strike, opt_type)
            ltp_data = self.client.get_ltp([inst_key])
            last_price = float(
                ltp_data.get("data", {})
                .get(inst_key, {})
                .get("last_price", 0.0)
                or 0.0
            )
            if last_price <= 0:
                logger.warning(f"No LTP for {inst_key}, skipping")
                return None
            qty = lots * 50 * (1 if side == "BUY" else -1)
            greeks = self.pricing.calculate_greeks(spot, strike, opt_type, expiry)
            entry_px = self._compute_entry_price(last_price, sell=(qty < 0))
            pos = Position(
                symbol=symbol,
                instrument_key=inst_key,
                strike=strike,
                option_type=opt_type,
                quantity=qty,
                entry_price=entry_px,
                entry_time=datetime.now(IST),
                current_price=entry_px,
                current_greeks=greeks,
            )
            legs.append(pos)

        net_prem = sum(l.entry_price * l.quantity for l in legs) / (lots * 50)
        trade = MultiLegTrade(
            legs=legs,
            strategy_type="MULTI_LEG",
            net_premium_per_share=net_prem,
            entry_time=datetime.now(IST),
            lots=lots,
            trading_mode=TradingMode.SYSTEMATIC,  # type: ignore
            expiry_date=expiry,
            trade_vega=sum(l.current_greeks.total_vega * l.quantity for l in legs),
        )

        if PAPER_TRADING:
            logger.info(f"[PAPER] OPEN {trade.strategy_type} lots={lots}")
            self._log_trade(trade, "OPEN")
            return trade

        # build orders
        orders_payload = []
        for leg in legs:
            tx = "SELL" if leg.quantity < 0 else "BUY"
            orders_payload.append(
                {
                    "quantity": abs(leg.quantity),
                    "product": ProductType.MIS.value,
                    "validity": "DAY",
                    "price": round(leg.entry_price, 2),
                    "tag": "VOLGUARD_V9",
                    "instrument_key": leg.instrument_key,
                    "order_type": OrderType.LIMIT.value,
                    "transaction_type": tx,
                    "disclosed_quantity": 0,
                    "trigger_price": 0,
                }
            )
        resp = self.client.place_multi_order_v2(orders_payload)
        oids = resp.get("data", {}).get("order_ids", [])
        if not oids:
            logger.error("Failed to place multi order")
            return None
        trade.basket_order_id = str(oids)
        self._log_trade(trade, "OPEN")
        logger.info(f"[LIVE] OPEN lots={lots} basket={oids}")
        return trade

    def close_trade(self, trade: MultiLegTrade, reason: ExitReason) -> bool:
        pnl = trade.total_unrealized_pnl()
        if PAPER_TRADING:
            logger.info(f"[PAPER] CLOSE {trade.strategy_type} reason={reason.value} pnl={pnl:,.0f}")
            trade.status = TradeStatus.CLOSED
            self.rm.daily_pnl += pnl
            self._log_trade(trade, f"CLOSE_{reason.value}", realized_pnl=pnl)
            return True

        orders_payload = []
        for leg in trade.legs:
            side = "BUY" if leg.quantity < 0 else "SELL"
            ltp_data = self.client.get_ltp([leg.instrument_key])
            last_price = float(
                ltp_data.get("data", {})
                .get(leg.instrument_key, {})
                .get("last_price", leg.entry_price)
                or leg.entry_price
            )
            # panic-style exit slippage
            px = last_price * (1 + PANIC_EXIT_FACTOR) if side == "BUY" else last_price * (
                1 - PANIC_EXIT_FACTOR
            )
            orders_payload.append(
                {
                    "quantity": abs(leg.quantity),
                    "product": ProductType.MIS.value,
                    "validity": "DAY",
                    "price": round(px, 2),
                    "tag": "VOLGUARD_V9",
                    "instrument_key": leg.instrument_key,
                    "order_type": OrderType.LIMIT.value,
                    "transaction_type": side,
                    "disclosed_quantity": 0,
                    "trigger_price": 0,
                }
            )
        resp = self.client.place_multi_order_v2(orders_payload)
        oids = resp.get("data", {}).get("order_ids", [])
        logger.info(
            f"[LIVE] CLOSE {trade.strategy_type} reason={reason.value} "
            f"pnl={pnl:,.0f} basket={oids}"
        )
        trade.status = TradeStatus.CLOSED
        self.rm.daily_pnl += pnl
        self._log_trade(trade, f"CLOSE_{reason.value}", realized_pnl=pnl)
        return True
