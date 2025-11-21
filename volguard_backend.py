
```python
from __future__ import annotations

# ============================================================
# STANDARD LIBRARY IMPORTS
# ============================================================
import os
import json
import time
import logging
import warnings
import math
import csv
import calendar
import asyncio
import threading
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple, Any, Deque
from collections import defaultdict, deque
from datetime import datetime, timedelta, time as dtime
from concurrent.futures import ThreadPoolExecutor
from threading import Lock, RLock
from enum import Enum
import hashlib

# ============================================================
# THIRD-PARTY IMPORTS WITH ERROR HANDLING
# ============================================================
try:
    import pytz
    import requests
    import numpy as np
    import pandas as pd
    from scipy import optimize
    from scipy.stats import norm
    from arch import arch_model
except ImportError as e:
    print(f"CRITICAL: Missing required dependency - {e}")
    print("Please install: pip install pytz requests numpy pandas scipy arch")
    exit(1)

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CONFIGURATION - VOLGUARD v8.1 (PRODUCTION-READY)
# ============================================================

# Timezone
IST = pytz.timezone("Asia/Kolkata")

# API Configuration
API_BASE_V2 = "https://api.upstox.com/v2"
UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "PASTE_YOUR_UPSTOX_ACCESS_TOKEN_HERE")

# Trading Mode
LIVE_FLAG = os.getenv("VOLGUARD_LIVE", "0") == "1"
PAPER_TRADING = (not LIVE_FLAG) or ("PASTE_YOUR_UPSTOX_ACCESS_TOKEN_HERE" in UPSTOX_ACCESS_TOKEN)

# Alert Configuration
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Data Sources
VIX_HISTORY_URL = "https://raw.githubusercontent.com/shritish20/VolGuard/refs/heads/main/atmiv.csv"
NIFTY_HISTORY_URL = "https://raw.githubusercontent.com/shritish20/VolGuard/refs/heads/main/nifty_50.csv"

# Account Configuration
ACCOUNT_SIZE = 500_000.0  # START CONSERVATIVE
LOT_SIZE = 50

# File Paths
STATE_FILE = "volguard_state_v8_1.json"
TRADE_LOG_FILE = "volguard_trade_log_v8_1.txt"
TRADE_JOURNAL_FILE = "volguard_trade_journal_v8_1.csv"
HEALTH_METRICS_FILE = "volguard_health_metrics_v8_1.json"

# Risk Parameters (CONSERVATIVE)
SYSTEMATIC_MAX_RISK_PERCENT = 0.01
MAX_PORTFOLIO_VEGA = 1000.0
MAX_PORTFOLIO_GAMMA = 2.0
MAX_PORTFOLIO_ES = ACCOUNT_SIZE * 0.02
DAILY_LOSS_LIMIT = ACCOUNT_SIZE * 0.03

# Volatility Regime Parameters
IVP_PANIC = 85.0
IVP_CALM = 35.0

# Exit Rules
PROFIT_TARGET_PCT = 0.35
STOP_LOSS_MULTIPLE = 2.0

# Slippage & Transaction Costs
BASE_SLIPPAGE = 0.0005
VOLATILITY_SLIPPAGE_MULTIPLIER = 2.0
LIQUIDITY_SLIPPAGE_FACTOR = 0.001
PANIC_EXIT_FACTOR = 0.10
BROKERAGE_PER_ORDER = 20.0
STT_RATE = 0.0005
GST_RATE = 0.18
EXCHANGE_CHARGES = 0.00002

# Market Parameters
RISK_FREE_RATE = 0.05
MAX_ORDER_RETRIES = 3
ORDER_TIMEOUT_SECONDS = 30
ORDER_FILL_TIMEOUT = 10

# Market Holidays 2025
MARKET_HOLIDAYS_2025 = [
    "2025-01-26", "2025-03-07", "2025-03-25", "2025-04-11",
    "2025-04-14", "2025-04-17", "2025-05-01", "2025-06-26",
    "2025-08-15", "2025-09-05", "2025-10-02", "2025-10-22",
    "2025-11-04", "2025-11-14", "2025-12-25"
]

# Trading Times
EOD_FLAT_TIME = dtime(15, 15)
EXPIRY_FLAT_TIME = dtime(14, 30)
MARKET_OPEN_TIME = dtime(9, 15)
MARKET_CLOSE_TIME = dtime(15, 30)

# ============================================================
# ENHANCED LOGGING
# ============================================================

class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    
    FORMATS = {
        logging.DEBUG: grey + "%(asctime)s - %(levelname)s - %(message)s" + reset,
        logging.INFO: grey + "%(asctime)s - %(levelname)s - %(message)s" + reset,
        logging.WARNING: yellow + "%(asctime)s - %(levelname)s - %(message)s" + reset,
        logging.ERROR: red + "%(asctime)s - %(levelname)s - %(message)s" + reset,
        logging.CRITICAL: bold_red + "%(asctime)s - %(levelname)s - %(message)s" + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

logger = logging.getLogger("VolGuard")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(TRADE_LOG_FILE)
stream_handler = logging.StreamHandler()

file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
stream_handler.setFormatter(CustomFormatter())

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

# ============================================================
# ENUMS & DATA MODELS
# ============================================================

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
    option_type: str
    quantity: int
    entry_price: float
    entry_time: datetime
    current_price: float
    current_greeks: GreeksSnapshot
    last_update_time: datetime = None
    
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
        if "SPREAD" in self.strategy_type or "CONDOR" in self.strategy_type:
            strikes = sorted({leg.strike for leg in self.legs})
            if len(strikes) >= 2:
                spread_width = strikes[-1] - strikes[0]
                max_loss_per_share = max(0.0, spread_width - self.net_premium_per_share)
                self.max_loss_per_lot = max_loss_per_share * LOT_SIZE
                return
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
        credit_per_share = max(self.net_premium_per_share, 0)
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
    last_update_time: datetime = None
    
    def __post_init__(self):
        if self.last_update_time is None:
            self.last_update_time = self.placed_time
    
    def is_complete(self) -> bool:
        return self.status in [OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.CANCELLED, OrderStatus.EXPIRED]
    
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    def is_rejected(self) -> bool:
        return self.status == OrderStatus.REJECTED

# ============================================================
# ALERTING SYSTEM
# ============================================================

class AlertSystem:
    def __init__(self):
        self.last_alert_time = {}
        self.alert_cooldown = 300
        
    def send_email_alert(self, subject: str, message: str):
        if not ALERT_EMAIL or not EMAIL_PASSWORD:
            logger.warning("Email alerts not configured")
            return
            
        try:
            msg = MimeMultipart()
            msg['From'] = ALERT_EMAIL
            msg['To'] = ALERT_EMAIL
            msg['Subject'] = f"VOLGUARD ALERT: {subject}"
            
            body = f"""
            VolGuard System Alert
            
            Time: {datetime.now(IST).strftime('%Y-%m-%d %H:%M:%S')}
            
            {message}
            
            ---
            This is an automated alert from VolGuard v8.1
            """
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(ALERT_EMAIL, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email alert sent: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
    
    def circuit_breaker_alert(self, daily_pnl: float, limit: float):
        message = f"""
        CIRCUIT BREAKER TRIPPED!
        
        Daily PnL: ₹{daily_pnl:,.0f}
        Daily Limit: ₹{limit:,.0f}
        
        All trading has been stopped.
        Manual intervention required.
        """
        self.send_email_alert("CIRCUIT BREAKER TRIPPED", message)
    
    def health_check_failed_alert(self, issues: List[str]):
        message = f"""
        HEALTH CHECK FAILED!
        
        Issues detected:
        {chr(10).join(f'- {issue}' for issue in issues)}
        
        Trading has been paused until resolved.
        """
        self.send_email_alert("HEALTH CHECK FAILED", message)
    
    def order_fill_failed_alert(self, order_ids: List[str], strategy: str):
        message = f"""
        ORDER FILL FAILURE!
        
        Strategy: {strategy}
        Failed Orders: {order_ids}
        
        Manual intervention may be required.
        """
        self.send_email_alert("ORDER FILL FAILURE", message)
    
    def kill_switch_activated_alert(self):
        message = """
        KILL SWITCH ACTIVATED!
        
        All positions are being closed.
        No new trades will be entered.
        
        Manual reset required.
        """
        self.send_email_alert("KILL SWITCH ACTIVATED", message)

# ============================================================
# KILL SWITCH MECHANISM
# ============================================================

class KillSwitch:
    def __init__(self):
        self.activated = False
        self.activation_time = None
        self.activation_reason = ""
        self._lock = Lock()
        
    def activate(self, reason: str = "Manual activation"):
        with self._lock:
            self.activated = True
            self.activation_time = datetime.now(IST)
            self.activation_reason = reason
            logger.critical(f"KILL SWITCH ACTIVATED: {reason}")
    
    def deactivate(self):
        with self._lock:
            self.activated = False
            self.activation_time = None
            self.activation_reason = ""
            logger.info("Kill switch deactivated")
    
    def is_active(self) -> bool:
        with self._lock:
            return self.activated
    
    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "activated": self.activated,
                "activation_time": self.activation_time.isoformat() if self.activation_time else None,
                "activation_reason": self.activation_reason
            }

# ============================================================
# ROBUST SABR MODEL
# ============================================================

class RobustSABRModel:
    def __init__(self):
        self.alpha = 0.2
        self.beta = 0.5
        self.rho = -0.2
        self.nu = 0.3
        self.calibrated = False
        self.calibration_error = float('inf')
        self.last_calibration = None
        
    def _sabr_volatility_safe(self, F: float, K: float, T: float) -> float:
        if F <= 0 or K <= 0 or T <= 0:
            return 0.2
            
        if abs(F - K) < 1e-6:
            return self.alpha / (F ** (1 - self.beta))
        
        try:
            z = (self.nu / self.alpha) * (F * K) ** ((1 - self.beta) / 2) * math.log(F / K)
            if abs(z) > 100:
                return self.alpha / (F ** (1 - self.beta))
                
            x = math.log((math.sqrt(1 - 2 * self.rho * z + z * z) + z - self.rho) / (1 - self.rho))
            
            numerator = self.alpha * (1 + 
                ((1 - self.beta) ** 2 / 24) * (self.alpha ** 2 / (F * K) ** (1 - self.beta)) + 
                (1 - self.beta) ** 4 / 1920 * (self.alpha ** 4 / (F * K) ** (2 - 2 * self.beta)) + 
                ((1 - self.beta) ** 2 / 24) * (self.rho * self.beta * self.nu * self.alpha) / ((F * K) ** ((1 - self.beta) / 2)) + 
                (2 - 3 * self.rho ** 2) * self.nu ** 2 / 24) * T
            
            denominator = (F * K) ** ((1 - self.beta) / 2) * (1 + (1 - self.beta) ** 2 / 24 * math.log(F / K) ** 2 + 
                          (1 - self.beta) ** 4 / 1920 * math.log(F / K) ** 4)
            
            if denominator == 0:
                return self.alpha / (F ** (1 - self.beta))
            
            if abs(x) < 1e-6:
                return numerator / denominator
                
            return numerator / denominator * z / x
            
        except (ValueError, ZeroDivisionError):
            return self.alpha / (F ** (1 - self.beta))

    def sabr_volatility(self, F: float, K: float, T: float) -> float:
        vol = self._sabr_volatility_safe(F, K, T)
        return max(5.0, min(150.0, vol))

    def calibrate(self, strikes: List[float], ivs: List[float], F: float, T: float) -> bool:
        if len(strikes) < 5 or T <= 1/365:
            logger.warning(f"Insufficient data for SABR calibration: {len(strikes)} strikes, T={T:.4f}")
            return False
            
        valid_data = [(K, iv) for K, iv in zip(strikes, ivs) 
                     if K > 0 and iv > 5 and iv < 150 and 0.5 * F < K < 2.0 * F]
        
        if len(valid_data) < 5:
            logger.warning("Not enough valid data points after filtering")
            return False
            
        strikes_clean, ivs_clean = zip(*valid_data)
        
        try:
            def objective(params):
                alpha, beta, rho, nu = params
                self.alpha, self.beta, self.rho, self.nu = alpha, beta, rho, nu
                errors = []
                for K, market_iv in zip(strikes_clean, ivs_clean):
                    model_iv = self.sabr_volatility(F, K, T)
                    errors.append((model_iv - market_iv) ** 2)
                return np.sqrt(np.mean(errors)) if errors else 1.0

            initial_guess = [0.2, 0.5, -0.2, 0.3]
            bounds = [(0.01, 1.0), (0.1, 0.9), (-0.99, 0.99), (0.1, 1.0)]
            
            result = optimize.minimize(objective, initial_guess, bounds=bounds, 
                                     method="L-BFGS-B", options={'maxiter': 100})
            
            if result.success:
                self.alpha, self.beta, self.rho, self.nu = result.x
                self.calibration_error = result.fun
                self.calibrated = True
                self.last_calibration = datetime.now(IST)
                
                if not (0.01 <= self.alpha <= 1.0 and 
                       0.1 <= self.beta <= 0.9 and 
                       -0.99 <= self.rho <= 0.99 and 
                       0.1 <= self.nu <= 1.0):
                    logger.warning("SABR calibration produced unrealistic parameters")
                    self.calibrated = False
                    return False
                    
                logger.info(f"SABR calibrated: α={self.alpha:.3f}, β={self.beta:.3f}, ρ={self.rho:.3f}, ν={self.nu:.3f}, error={self.calibration_error:.3f}")
                return True
                
        except Exception as e:
            logger.error(f"SABR calibration failed: {e}")
            
        return False

# ============================================================
# VOLATILITY SURFACE
# ============================================================

class RobustVolatilitySurface2D:
    def __init__(self):
        self.surface: Dict[str, RobustSABRModel] = {}
        self.expiries: List[str] = []
        self.spot: float = 0.0
        self.last_calibration: Optional[datetime] = None
        self._calibration_lock = Lock()
        
    def _get_fallback_volatility(self, strike: float, spot: float) -> float:
        moneyness = strike / spot
        if moneyness < 0.8 or moneyness > 1.2:
            return 25.0
        else:
            return 20.0

    def update_surface(self, spot: float, chain_data: dict) -> bool:
        with self._calibration_lock:
            self.spot = spot
            expiry_data = defaultdict(list)
            
            for opt_data in chain_data.get("data", {}).values():
                expiry = opt_data.get("expiry", "")
                strike = float(opt_data.get("strike", 0))
                iv = float(opt_data.get("iv", 0))
                if 5 < iv < 150 and strike > 0:
                    expiry_data[expiry].append((strike, iv))

            success_count = 0
            for expiry, data in expiry_data.items():
                if len(data) < 8:
                    continue
                    
                strikes, ivs = zip(*data)
                T = get_dte_for_expiry(expiry)
                
                sabr = RobustSABRModel()
                if sabr.calibrate(list(strikes), list(ivs), spot, T):
                    self.surface[expiry] = sabr
                    if expiry not in self.expiries:
                        self.expiries.append(expiry)
                    success_count += 1

            self.expiries.sort()
            self.last_calibration = datetime.now(IST)
            
            logger.info(f"Vol surface updated: {success_count}/{len(expiry_data)} expiries calibrated")
            return success_count > 0

    def get_volatility(self, strike: float, expiry: str) -> float:
        try:
            if expiry in self.surface:
                sabr = self.surface[expiry]
                if sabr.calibrated:
                    return sabr.sabr_volatility(self.spot, strike, get_dte_for_expiry(expiry))

            if len(self.expiries) >= 2:
                before = after = None
                for exp in self.expiries:
                    if exp <= expiry:
                        before = exp
                    else:
                        after = exp
                        break
                        
                if before and after and before in self.surface and after in self.surface:
                    vol_before = self.surface[before].sabr_volatility(self.spot, strike, get_dte_for_expiry(before))
                    vol_after = self.surface[after].sabr_volatility(self.spot, strike, get_dte_for_expiry(after))
                    T_before = get_dte_for_expiry(before)
                    T_after = get_dte_for_expiry(after)
                    T_target = get_dte_for_expiry(expiry)
                    
                    if T_after != T_before:
                        w = (T_after - T_target) / (T_after - T_before)
                        return vol_before * w + vol_after * (1 - w)
                        
        except Exception as e:
            logger.warning(f"Volatility interpolation failed: {e}")
            
        return self._get_fallback_volatility(strike, self.spot)

# ============================================================
# PRICING ENGINE
# ============================================================

class InstitutionalPricingEngine:
    def __init__(self, vol_surface: RobustVolatilitySurface2D):
        self.vol_surface = vol_surface
        self._greeks_cache: Dict[Tuple[float, float, str, str], GreeksSnapshot] = {}
        self._cache_lock = Lock()
        
    @staticmethod
    def _bsm_d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 1e-6 or sigma <= 1e-6 or S <= 1e-6 or K <= 1e-6:
            return 0.0
        try:
            return (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        except Exception:
            return 0.0

    @staticmethod
    def _bsm_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str):
        T = max(T, 1.0 / 365.0)
        sigma = max(min(sigma, 2.0), 0.05)
        d1 = InstitutionalPricingEngine._bsm_d1(S, K, T, r, sigma)
        d2 = d1 - sigma * math.sqrt(T)

        if option_type == "CE":
            delta = norm.cdf(d1)
        else:
            delta = norm.cdf(d1) - 1.0

        gamma = norm.pdf(d1) / (S * sigma * math.sqrt(T)) if S > 0 else 0.0
        vega = S * norm.pdf(d1) * math.sqrt(T) / 100.0
        theta = (-S * norm.pdf(d1) * sigma / (2 * math.sqrt(T)) -
                 r * K * math.exp(-r * T) * norm.cdf(d2)) / 365.0

        delta = max(min(delta, 1.0), -1.0)
        gamma = max(min(gamma, 0.1), 0.0)
        vega = max(min(vega, S * 0.3), 0.0)
        return delta, gamma, theta, vega, 0.0

    def calculate_greeks(self, spot: float, strike: float, option_type: str, expiry_date: str) -> GreeksSnapshot:
        cache_key = (spot, strike, option_type, expiry_date)
        
        with self._cache_lock:
            if cache_key in self._greeks_cache:
                cached = self._greeks_cache[cache_key]
                if not cached.is_stale():
                    return cached
        
        T = get_dte_for_expiry(expiry_date)
        iv = self.vol_surface.get_volatility(strike, expiry_date)
        sigma = iv / 100.0
        r = RISK_FREE_RATE
        
        delta, gamma, theta, vega, rho = self._bsm_greeks(spot, strike, T, r, sigma, option_type)
        
        greeks = GreeksSnapshot(
            timestamp=datetime.now(IST),
            total_delta=delta,
            total_gamma=gamma,
            total_theta=theta,
            total_vega=vega,
            total_rho=rho,
            staleness_seconds=0.0
        )
        
        with self._cache_lock:
            self._greeks_cache[cache_key] = greeks
            self._greeks_cache = {k: v for k, v in self._greeks_cache.items() 
                                if not v.is_stale(max_staleness=300)}
        
        return greeks

# ============================================================
# REAL DATA ANALYTICS
# ============================================================

class RobustRealDataAnalytics:
    def __init__(self):
        self.vix_data = pd.DataFrame()
        self.nifty_data = pd.DataFrame()
        self._data_lock = Lock()
        self._load_historical_data()
        
    def _load_historical_data(self):
        try:
            vix_df = pd.read_csv(VIX_HISTORY_URL)
            nifty_df = pd.read_csv(NIFTY_HISTORY_URL)
            
            vix_df["Date"] = pd.to_datetime(vix_df["Date"], format="%d-%b-%Y", errors="coerce")
            vix_df = vix_df.sort_values("Date").dropna(subset=["Date"])
            vix_df["Close"] = pd.to_numeric(vix_df["Close"], errors="coerce")
            self.vix_data = vix_df.dropna(subset=["Close"])
            
            nifty_df["Date"] = pd.to_datetime(nifty_df["Date"], format="%d-%b-%Y", errors="coerce")
            nifty_df = nifty_df.sort_values("Date").dropna(subset=["Date"])
            nifty_df["Close"] = pd.to_numeric(nifty_df["Close"], errors="coerce")
            self.nifty_data = nifty_df.dropna(subset=["Close"])
            
            logger.info(f"Loaded {len(self.vix_data)} VIX and {len(self.nifty_data)} Nifty records")
            
        except Exception as e:
            logger.error(f"Historical data load failed: {e}")
            self._create_fallback_data()
    
    def _create_fallback_data(self):
        dates = pd.date_range(start='2020-01-01', end=datetime.now().strftime('%Y-%m-%d'), freq='D')
        vix_values = np.random.normal(15, 5, len(dates))
        nifty_values = 10000 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, len(dates))))
        
        self.vix_data = pd.DataFrame({'Date': dates, 'Close': np.clip(vix_values, 10, 40)})
        self.nifty_data = pd.DataFrame({'Date': dates, 'Close': nifty_values})
        logger.warning("Using synthetic fallback data")

    def calculate_real_ivp(self, current_vix: float, lookback_days: int = 252) -> float:
        with self._data_lock:
            if self.vix_data.empty or len(self.vix_data) < 30:
                return min(100.0, max(0.0, (current_vix / 20.0) * 50.0))
                
            try:
                recent = self.vix_data.tail(lookback_days)["Close"]
                ivp = (recent < current_vix).mean() * 100.0
                return float(max(0.0, min(100.0, ivp)))
            except Exception:
                return 50.0

    def calculate_iv_rv_spread(self, current_vix: float, lookback_days: int = 30) -> float:
        with self._data_lock:
            if self.nifty_data.empty or len(self.nifty_data) < lookback_days + 5:
                return 1.0
                
            try:
                prices = self.nifty_data.tail(lookback_days + 5)["Close"]
                returns = np.log(prices / prices.shift(1)).dropna()
                realized_vol = returns.std() * np.sqrt(252) * 100.0
                return float(current_vix - realized_vol)
            except Exception:
                return 0.0

    def regime_from_ivp(self, vix: float, ivp: float) -> str:
        if ivp >= IVP_PANIC or vix >= 25:
            return "PANIC"
        elif ivp <= IVP_CALM or vix <= 12:
            return "CALM"
        else:
            return "EXPANSION"

# ============================================================
# ADVANCED REGIME DETECTION
# ============================================================

class AdvancedRegimeDetector:
    def __init__(self, market_data: InstitutionalMarketData):
        self.market_data = market_data
        self.regime_history = deque(maxlen=100)
        
    def calculate_volatility_skew(self, chain_data: dict, spot: float) -> float:
        try:
            put_ivs = []
            call_ivs = []
            
            for opt_data in chain_data.get("data", {}).values():
                strike = float(opt_data.get("strike", 0))
                iv = float(opt_data.get("iv", 0))
                opt_type = opt_data.get("option_type", "")
                
                if 5 < iv < 150:
                    moneyness = strike / spot
                    if opt_type == "PE" and moneyness < 0.98:
                        put_ivs.append(iv)
                    elif opt_type == "CE" and moneyness > 1.02:
                        call_ivs.append(iv)
            
            if put_ivs and call_ivs:
                avg_put_iv = np.mean(put_ivs)
                avg_call_iv = np.mean(call_ivs)
                return (avg_put_iv - avg_call_iv) / spot * 1000
                
            return 0.0
            
        except Exception:
            return 0.0
    
    def detect_regime(self, spot: float, vix: float, ivp: float, 
                     chain_data: dict, pcr: float) -> Dict[str, Any]:
        
        skew = self.calculate_volatility_skew(chain_data, spot)
        
        # Advanced regime logic
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
        
        # Sub-regime detection
        if regime in ["BULL_EXPANSION", "TRANSITION"]:
            sub_regime = "STRONG_UPTREND" if pcr < 0.8 else "SIDEWAYS"
        elif regime == "BEAR_EXPANSION":
            sub_regime = "STRONG_DOWNTREND" if pcr > 1.2 else "RANGING"
        else:
            sub_regime = "NEUTRAL"
        
        confidence = 0.7
        if regime == "PANIC" and (vix > 35 or ivp > 85):
            confidence += 0.2
        elif regime == "CALM_COMPRESSION" and (vix < 15 and ivp < 30):
            confidence += 0.15
            
        regime_info = {
            'regime': regime,
            'confidence': min(confidence, 0.95),
            'sub_regime': sub_regime,
            'timestamp': datetime.now(IST),
            'skew': skew
        }
        
        self.regime_history.append(regime_info)
        return regime_info

# ============================================================
# ADVANCED STRATEGY ENGINE
# ============================================================

class AdvancedStrategyEngine:
    def __init__(self, market_data: InstitutionalMarketData):
        self.market_data = market_data
        self.regime_detector = AdvancedRegimeDetector(market_data)
        self._last_strategy_time = None
        
    def update_metrics(self, spot: float, vix: float, chain: dict, analytics: RobustRealDataAnalytics):
        ivp = analytics.calculate_real_ivp(vix)
        spread = analytics.calculate_iv_rv_spread(vix)
        pcr = self.calculate_pcr(chain)
        
        regime_info = self.regime_detector.detect_regime(spot, vix, ivp, chain, pcr)
        
        self.metrics = {
            "spot": spot,
            "vix": vix,
            "ivp": ivp,
            "iv_rv_spread": spread,
            "pcr": pcr,
            "regime": regime_info['regime'],
            "sub_regime": regime_info['sub_regime'],
            "regime_confidence": regime_info['confidence'],
            "timestamp": datetime.now(IST),
            "skew": regime_info['skew']
        }
        
        logger.info(f"Advanced Metrics: {regime_info['regime']} (Conf: {regime_info['confidence']:.1%}) | "
                   f"Sub: {regime_info['sub_regime']} | IVP: {ivp:.1f}% | PCR: {pcr:.2f}")
    
    def calculate_pcr(self, chain_data: dict) -> float:
        try:
            data = chain_data.get("data", {})
            put_volume = 0
            call_volume = 0
            
            for _, opt in data.items():
                opt_type = opt.get("option_type", "")
                volume = opt.get("volume", 0)
                if opt_type == "PE":
                    put_volume += volume
                elif opt_type == "CE":
                    call_volume += volume
                    
            return put_volume / max(call_volume, 1)
        except Exception:
            return 1.0
    
    def select_strategy(self) -> Tuple[str, List[Dict[str, Any]]]:
        if not self.metrics:
            return "SKIP", []
            
        now = datetime.now(IST)
        if self._last_strategy_time and (now - self._last_strategy_time).total_seconds() < 300:
            return "SKIP", []
            
        regime = self.metrics["regime"]
        sub_regime = self.metrics["sub_regime"]
        ivp = self.metrics["ivp"]
        pcr = self.metrics["pcr"]
        spot = self.metrics["spot"]
        confidence = self.metrics["regime_confidence"]
        
        if confidence < 0.6:
            logger.warning(f"Low regime confidence ({confidence:.1%}), skipping trade")
            return "SKIP", []
        
        expiry = self.market_data.get_weekly_expiry()
        atm = round(spot / 50) * 50
        
        # Advanced strategy selection
        if regime == "PANIC":
            return self._panic_regime_strategy(spot, atm, expiry, ivp)
        elif regime == "FEAR_BACKWARDATION":
            return self._fear_backwardation_strategy(spot, atm, expiry, pcr)
        elif regime == "DEFENSIVE_SKEW":
            return self._defensive_skew_strategy(spot, atm, expiry, pcr)
        elif regime == "BULL_EXPANSION":
            return self._bull_expansion_strategy(spot, atm, expiry, sub_regime, pcr)
        elif regime == "CALM_COMPRESSION":
            return self._calm_compression_strategy(spot, atm, expiry, ivp)
        elif regime == "LOW_VOL_COMPRESSION":
            return self._low_vol_strategy(spot, atm, expiry)
        else:
            return self._transition_strategy(spot, atm, expiry, pcr)
    
    def _panic_regime_strategy(self, spot: float, atm: float, expiry: str, ivp: float) -> Tuple[str, List[Dict]]:
        logger.info("PANIC regime - defensive strategies")
        
        if ivp > 85:
            return (
                "PUT_RATIO_SPREAD",
                [
                    {"strike": atm - 400, "type": "PE", "side": "BUY", "expiry": expiry},
                    {"strike": atm - 200, "type": "PE", "side": "SELL", "expiry": expiry},
                    {"strike": atm - 200, "type": "PE", "side": "SELL", "expiry": expiry},
                ],
            )
        else:
            return (
                "WIDE_IRON_CONDOR",
                [
                    {"strike": atm + 500, "type": "CE", "side": "SELL", "expiry": expiry},
                    {"strike": atm + 700, "type": "CE", "side": "BUY", "expiry": expiry},
                    {"strike": atm - 500, "type": "PE", "side": "SELL", "expiry": expiry},
                    {"strike": atm - 700, "type": "PE", "side": "BUY", "expiry": expiry},
                ],
            )
    
    def _fear_backwardation_strategy(self, spot: float, atm: float, expiry: str, pcr: float) -> Tuple[str, List[Dict]]:
        logger.info("FEAR_BACKWARDATION - short-dated premium selling")
        return (
            "SHORT_STRANGLE",
            [
                {"strike": atm + 300, "type": "CE", "side": "SELL", "expiry": expiry},
                {"strike": atm - 300, "type": "PE", "side": "SELL", "expiry": expiry},
            ],
        )
    
    def _defensive_skew_strategy(self, spot: float, atm: float, expiry: str, pcr: float) -> Tuple[str, List[Dict]]:
        logger.info("DEFENSIVE_SKEW - skew-based strategies")
        return (
            "SKEW_ADVANTAGE_SPREAD",
            [
                {"strike": atm - 200, "type": "PE", "side": "SELL", "expiry": expiry},
                {"strike": atm - 400, "type": "PE", "side": "BUY", "expiry": expiry},
                {"strike": atm + 300, "type": "CE", "side": "BUY", "expiry": expiry},
            ],
        )
    
    def _bull_expansion_strategy(self, spot: float, atm: float, expiry: str, 
                                sub_regime: str, pcr: float) -> Tuple[str, List[Dict]]:
        logger.info(f"BULL_EXPANSION ({sub_regime}) - bullish credit strategies")
        
        if sub_regime == "STRONG_UPTREND":
            return (
                "BULL_PUT_SPREAD_AGGRESSIVE",
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
    
    def _calm_compression_strategy(self, spot: float, atm: float, expiry: str, ivp: float) -> Tuple[str, List[Dict]]:
        logger.info("CALM_COMPRESSION - premium selling strategies")
        
        if ivp < 25:
            return (
                "SHORT_STRANGLE_AGGRESSIVE",
                [
                    {"strike": atm + 200, "type": "CE", "side": "SELL", "expiry": expiry},
                    {"strike": atm - 200, "type": "PE", "side": "SELL", "expiry": expiry},
                ],
            )
        else:
            return (
                "IRON_CONDOR_CALM",
                [
                    {"strike": atm + 150, "type": "CE", "side": "SELL", "expiry": expiry},
                    {"strike": atm + 300, "type": "CE", "side": "BUY", "expiry": expiry},
                    {"strike": atm - 150, "type": "PE", "side": "SELL", "expiry": expiry},
                    {"strike": atm - 300, "type": "PE", "side": "BUY", "expiry": expiry},
                ],
            )
    
    def _low_vol_strategy(self, spot: float, atm: float, expiry: str) -> Tuple[str, List[Dict]]:
        logger.info("LOW_VOL_COMPRESSION - defined risk strategies")
        return (
            "CREDIT_SPREAD_COMBO",
            [
                {"strike": atm + 100, "type": "CE", "side": "SELL", "expiry": expiry},
                {"strike": atm + 250, "type": "CE", "side": "BUY", "expiry": expiry},
                {"strike": atm - 100, "type": "PE", "side": "SELL", "expiry": expiry},
                {"strike": atm - 250, "type": "PE", "side": "BUY", "expiry": expiry},
            ],
        )
    
    def _transition_strategy(self, spot: float, atm: float, expiry: str, pcr: float) -> Tuple[str, List[Dict]]:
        logger.info("TRANSITION regime - neutral strategies")
        return (
            "IRON_CONDOR_NEUTRAL",
            [
                {"strike": atm + 300, "type": "CE", "side": "SELL", "expiry": expiry},
                {"strike": atm + 500, "type": "CE", "side": "BUY", "e
