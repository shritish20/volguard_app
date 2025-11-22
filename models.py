from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, List

from config import IST, LOT_SIZE, BROKERAGE_PER_ORDER, STT_RATE, GST_RATE, EXCHANGE_CHARGES


class TradingMode(Enum):
    SYSTEMATIC = "SYSTEMATIC"
    DISCRETIONARY = "DISCRETIONARY"
    MARKET_MAKING = "MARKET_MAKING"


class TradeStatus(Enum):
    FLAT = "FLAT"
    OPEN = "OPEN"
    CLOSED = "CLOSED"


class ExitReason(Enum):
    PROFIT_TARGET = "PROFIT_TARGET"
    STOP_LOSS = "STOP_LOSS"
    DAILY_LOSS_LIMIT = "DAILY_LOSS_LIMIT"
    VEGA_SHIELD = "VEGA_SHIELD"
    EOD_FLATTEN = "EOD_FLATTEN"
    IV_SHOCK = "IV_SHOCK"
    MANUAL = "MANUAL"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    HEALTH_CHECK_FAILED = "HEALTH_CHECK_FAILED"
    KILL_SWITCH = "KILL_SWITCH"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"


class ProductType(Enum):
    MIS = "MIS"
    NRML = "NRML"


class TransactionType(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(Enum):
    PENDING = "PENDING"
    PLACED = "PLACED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXPIRED = "EXPIRED"


@dataclass
class GreeksSnapshot:
    timestamp: datetime
    total_delta: float = 0.0
    total_gamma: float = 0.0
    total_theta: float = 0.0
    total_vega: float = 0.0
    total_rho: float = 0.0
    staleness_seconds: float = 0.0

    def is_stale(self, max_staleness: float = 60.0) -> bool:
        return (datetime.now(IST) - self.timestamp).total_seconds() > max_staleness


@dataclass
class Position:
    symbol: str
    instrument_key: str
    strike: float
    option_type: str  # "CE"/"PE"
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float
    current_greeks: GreeksSnapshot
    last_update_time: Optional[datetime] = None

    def __post_init__(self):
        if self.last_update_time is None:
            self.last_update_time = datetime.now(IST)

    def unrealized_pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity

    def update_price(self, new_price: float, greeks: GreeksSnapshot):
        self.current_price = new_price
        self.current_greeks = greeks
        self.last_update_time = datetime.now(IST)


@dataclass
class MultiLegTrade:
    legs: List[Position]
    strategy_type: str
    net_premium_per_share: float
    entry_time: datetime
    lots: int
    trading_mode: TradingMode
    status: TradeStatus = TradeStatus.OPEN
    expiry_date: str = ""
    trade_vega: float = 0.0
    basket_order_id: Optional[str] = None
    max_loss_per_lot: float = 0.0
    transaction_costs: float = 0.0

    def __post_init__(self):
        self.calculate_max_loss()
        self.calculate_transaction_costs()

    def calculate_max_loss(self):
        strikes = sorted({leg.strike for leg in self.legs})
        if len(strikes) >= 2:
            spread_width = strikes[-1] - strikes[0]
            max_loss_per_share = max(0.0, spread_width - self.net_premium_per_share)
            self.max_loss_per_lot = max_loss_per_share * LOT_SIZE
        else:
            self.max_loss_per_lot = float("inf")

    def calculate_transaction_costs(self):
        total_premium = abs(self.net_premium_per_share) * LOT_SIZE * self.lots
        brokerage = BROKERAGE_PER_ORDER * len(self.legs) * 2
        stt = total_premium * STT_RATE
        exchange_charges = total_premium * EXCHANGE_CHARGES
        gst = brokerage * GST_RATE
        self.transaction_costs = brokerage + stt + exchange_charges + gst

    def total_unrealized_pnl(self) -> float:
        return sum(leg.unrealized_pnl() for leg in self.legs) - self.transaction_costs

    def total_credit(self) -> float:
        credit_per_share = max(self.net_premium_per_share, 0.0)
        return credit_per_share * LOT_SIZE * self.lots


@dataclass
class PortfolioSnapshot:
    timestamp: datetime
    pnl_unrealized: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float
    iv: float
    ivp: float
    regime: str
    VaR_99: float = 0.0
    ES_99: float = 0.0
    total_transaction_costs: float = 0.0


@dataclass
class Order:
    order_id: str
    instrument_key: str
    quantity: int
    price: float
    order_type: OrderType
    transaction_type: TransactionType
    product_type: ProductType
    status: OrderStatus
    placed_time: datetime
    filled_quantity: int = 0
    average_price: float = 0.0
    last_update_time: Optional[datetime] = None

    def __post_init__(self):
        if self.last_update_time is None:
            self.last_update_time = self.placed_time

    def is_complete(self) -> bool:
        return self.status in (
            OrderStatus.FILLED,
            OrderStatus.REJECTED,
            OrderStatus.CANCELLED,
            OrderStatus.EXPIRED,
        )

    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED

    def is_rejected(self) -> bool:
        return self.status == OrderStatus.REJECTED
