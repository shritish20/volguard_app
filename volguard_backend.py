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
from email.mime.text import MIMEText
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
                {"strike": atm + 500, "type": "CE", "side": "BUY", "expiry": expiry},
                {"strike": atm - 300, "type": "PE", "side": "SELL", "expiry": expiry},
                {"strike": atm - 500, "type": "PE", "side": "BUY", "expiry": expiry},
            ],
        )

# ============================================================
# ROBUST UPSTOX API CLIENT
# ============================================================

class RobustUpstoxAPI:
    def __init__(self, access_token: str):
        self.access_token = access_token
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.last_request_time = 0.0
        self.min_request_interval = 0.2
        self.request_timeout = 10

    def _rate_limit(self):
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = time.time()

    def _request(self, method: str, url: str, params: dict = None, data: dict = None,
                retries: int = 3, timeout: int = None) -> dict:
        if timeout is None:
            timeout = self.request_timeout
            
        for attempt in range(retries):
            try:
                self._rate_limit()
                resp = self.session.request(method=method, url=url, params=params,
                                          json=data, timeout=timeout)
                
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 429:
                    wait_time = (2 ** attempt) + 1
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"API error {resp.status_code}: {resp.text}")
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout (attempt {attempt+1}/{retries})")
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                
            if attempt < retries - 1:
                time.sleep(1)
                
        return {"data": {}, "status": "error"}

    def get_market_quote_ltp(self, instrument_keys: List[str]) -> dict:
        if not instrument_keys:
            return {"data": {}}
        params = {"instrument_key": ",".join(instrument_keys)}
        return self._request("GET", f"{API_BASE_V2}/market-quote/ltp", params=params)

    def get_market_quote_full(self, instrument_keys: List[str]) -> dict:
        if not instrument_keys:
            return {"data": {}}
        params = {"instrument_key": ",".join(instrument_keys)}
        return self._request("GET", f"{API_BASE_V2}/market-quote/quotes", params=params)

    def get_option_chain(self, symbol: str, expiry_date: str) -> dict:
        params = {"symbol": symbol, "expiry_date": expiry_date}
        return self._request("GET", f"{API_BASE_V2}/option/chain", params=params, timeout=15)

    def get_option_contract(self, underlying: str, expiry_date: str) -> dict:
        params = {"underlying": underlying, "expiry": expiry_date}
        return self._request("GET", f"{API_BASE_V2}/option/contract", params=params, timeout=15)

    def place_multi_order(self, orders_data: List[dict]) -> dict:
        data = {"orders": orders_data}
        return self._request("POST", f"{API_BASE_V2}/order/multi/place", data=data, timeout=30)

    def get_order_details(self, order_id: str) -> dict:
        return self._request("GET", f"{API_BASE_V2}/order/details", params={"order_id": order_id})

# ============================================================
# ENHANCED ORDER MANAGEMENT
# ============================================================

class EnhancedOrderManager:
    def __init__(self, api: RobustUpstoxAPI):
        self.api = api
        self.orders: Dict[str, Order] = {}
        self._order_lock = Lock()
        self.alert_system = AlertSystem()
        
    def place_order(self, instrument_key: str, quantity: int, price: float, 
                   order_type: OrderType, transaction_type: TransactionType,
                   product_type: ProductType = ProductType.MIS) -> Optional[Order]:
        try:
            order_data = {
                "quantity": abs(quantity),
                "product": product_type.value,
                "validity": "DAY",
                "price": round(price, 2),
                "tag": "VOLGUARD_V8_1",
                "instrument_key": instrument_key,
                "order_type": order_type.value,
                "transaction_type": transaction_type.value,
                "disclosed_quantity": 0,
                "trigger_price": 0,
            }
            
            response = self.api.place_multi_order([order_data])
            order_ids = response.get("data", {}).get("order_ids", [])
            
            if order_ids:
                order_id = order_ids[0]
                order = Order(
                    order_id=order_id,
                    instrument_key=instrument_key,
                    quantity=quantity,
                    price=price,
                    order_type=order_type,
                    transaction_type=transaction_type,
                    product_type=product_type,
                    status=OrderStatus.PLACED,
                    placed_time=datetime.now(IST)
                )
                
                with self._order_lock:
                    self.orders[order_id] = order
                    
                logger.info(f"Order placed: {order_id} for {instrument_key}")
                return order
                
        except Exception as e:
            logger.error(f"Order placement failed: {e}")
            
        return None
        
    def verify_order_fills(self, order_ids: List[str], timeout: int = ORDER_FILL_TIMEOUT) -> Tuple[bool, List[str]]:
        if not order_ids:
            return True, []
            
        start_time = time.time()
        failed_orders = []
        
        while time.time() - start_time < timeout:
            all_filled = True
            failed_orders.clear()
            
            for order_id in order_ids:
                order = self.get_order_status(order_id)
                if not order:
                    all_filled = False
                    failed_orders.append(f"{order_id} - NOT FOUND")
                    continue
                    
                if order.is_rejected():
                    failed_orders.append(f"{order_id} - REJECTED")
                    all_filled = False
                elif not order.is_filled():
                    all_filled = False
                    if order.status == OrderStatus.PENDING:
                        failed_orders.append(f"{order_id} - STILL PENDING")
                    else:
                        failed_orders.append(f"{order_id} - {order.status.value}")
            
            if all_filled:
                logger.info(f"All orders filled successfully: {order_ids}")
                return True, []
                
            time.sleep(1)
            
        logger.warning(f"Order fill verification timeout after {timeout}s")
        return False, failed_orders
        
    def get_order_status(self, order_id: str) -> Optional[Order]:
        try:
            response = self.api.get_order_details(order_id)
            order_data = response.get("data", {})
            
            if order_data:
                with self._order_lock:
                    if order_id in self.orders:
                        order = self.orders[order_id]
                        order.status = OrderStatus(order_data.get("status", "PENDING"))
                        order.filled_quantity = order_data.get("filled_quantity", 0)
                        order.average_price = order_data.get("average_price", 0.0)
                        order.last_update_time = datetime.now(IST)
                        return order
                        
        except Exception as e:
            logger.error(f"Order status check failed: {e}")
            
        return None

# ============================================================
# TTLCache IMPLEMENTATION
# ============================================================

class TTLCache:
    def __init__(self, ttl_seconds: int = 300):
        self._cache = {}
        self._ttl = ttl_seconds
        self._lock = Lock()
        
    def get(self, key):
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    return value
                else:
                    del self._cache[key]
            return None
            
    def set(self, key, value):
        with self._lock:
            self._cache[key] = (value, time.time())
            
    def clear(self):
        with self._lock:
            self._cache.clear()

# ============================================================
# MARKET DATA MANAGER
# ============================================================

class InstitutionalMarketData:
    def __init__(self, access_token: str, analytics: RobustRealDataAnalytics, vol_surface: RobustVolatilitySurface2D):
        self.api = RobustUpstoxAPI(access_token)
        self.analytics = analytics
        self.vol_surface = vol_surface
        self.pricing_engine = InstitutionalPricingEngine(vol_surface)
        self._spot_cache = TTLCache(ttl_seconds=10)
        self._instrument_cache = TTLCache(ttl_seconds=3600)
        self._chain_cache = TTLCache(ttl_seconds=60)

    def get_enhanced_spot_vix(self) -> Tuple[float, float]:
        cache_key = "spot_vix"
        cached = self._spot_cache.get(cache_key)
        if cached:
            return cached
            
        try:
            keys = ["NSE_INDEX|Nifty 50", "NSE_INDEX|India VIX"]
            data = self.api.get_market_quote_ltp(keys)
            
            spot = float(data.get("data", {}).get("NSE_INDEX|Nifty 50", {}).get("last_price", 0) or 0)
            vix = float(data.get("data", {}).get("NSE_INDEX|India VIX", {}).get("last_price", 0) or 0)
            
            if spot <= 10000 or spot >= 50000:
                logger.warning(f"Invalid spot price: {spot}, using fallback")
                spot = 22000.0
            if vix <= 5 or vix >= 80:
                logger.warning(f"Invalid VIX: {vix}, using fallback")
                vix = 15.0
                
            result = (spot, vix)
            self._spot_cache.set(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Enhanced spot/VIX fetch failed: {e}")
            return (22000.0, 15.0)

    def get_weekly_expiry(self) -> str:
        now = datetime.now(IST)
        today = now.date()
        days_ahead = (3 - today.weekday()) % 7
        if days_ahead == 0 and now.time() >= dtime(15, 30):
            days_ahead = 7
        expiry_date = today + timedelta(days=days_ahead)
        return expiry_date.strftime("%Y-%m-%d")

    def get_option_chain_with_validation(self, symbol: str, expiry: str) -> dict:
        cache_key = f"chain_{symbol}_{expiry}"
        cached = self._chain_cache.get(cache_key)
        if cached:
            return cached
            
        try:
            chain = self.api.get_option_chain(symbol, expiry)
            if not chain or "data" not in chain:
                return {}
                
            valid_strikes = 0
            total_strikes = 0
            
            for strike_data in chain["data"].values():
                total_strikes += 1
                iv = strike_data.get("iv", 0)
                if 5 < iv < 150:
                    valid_strikes += 1
                    
            validity_ratio = valid_strikes / max(total_strikes, 1)
            if validity_ratio < 0.6 or valid_strikes < 10:
                logger.warning(f"Option chain validation failed: {valid_strikes}/{total_strikes} valid strikes")
                return {}
                
            spot, _ = self.get_enhanced_spot_vix()
            self.vol_surface.update_surface(spot, chain)
            
            self._chain_cache.set(cache_key, chain)
            return chain
            
        except Exception as e:
            logger.error(f"Option chain validation failed: {e}")
            return {}

    def get_option_instrument_key(self, symbol: str, expiry: str, strike: float, option_type: str) -> str:
        cache_key = f"{symbol}_{expiry}_{strike}_{option_type}"
        cached = self._instrument_cache.get(cache_key)
        if cached:
            return cached
            
        data = self.api.get_option_contract(symbol, expiry)
        contracts = data.get("data", [])
        
        for c in contracts:
            try:
                c_strike = float(c.get("strike", 0))
                c_type = c.get("option_type")
                if abs(c_strike - strike) < 1e-3 and c_type == option_type:
                    instrument_key = c["instrument_key"]
                    self._instrument_cache.set(cache_key, instrument_key)
                    return instrument_key
            except Exception:
                continue
                
        raise ValueError(f"Contract not found: {symbol} {expiry} {strike} {option_type}")

    def calculate_dynamic_slippage(self, instrument_key: str, quantity: int) -> float:
        try:
            quotes = self.api.get_market_quote_full([instrument_key])
            inst_data = quotes.get("data", {}).get(instrument_key, {})
            
            slippage = BASE_SLIPPAGE
            iv = float(inst_data.get("iv", 0) or 20)
            if iv > 25:
                slippage *= VOLATILITY_SLIPPAGE_MULTIPLIER
                
            volume = float(inst_data.get("volume", 0) or 1000)
            if volume < 5000:
                slippage += LIQUIDITY_SLIPPAGE_FACTOR
                
            if abs(quantity) > 500:
                slippage *= 1.5
                
            return min(slippage, 0.05)
        except Exception:
            return BASE_SLIPPAGE

# ============================================================
# CIRCUIT BREAKER
# ============================================================

class CircuitBreaker:
    def __init__(self, max_daily_loss: float):
        self.max_daily_loss = max_daily_loss
        self.tripped = False
        self.trip_time = None
        self._lock = Lock()
        
    def check(self, current_pnl: float) -> bool:
        with self._lock:
            if self.tripped:
                return False
                
            if current_pnl <= -self.max_daily_loss:
                self.tripped = True
                self.trip_time = datetime.now(IST)
                logger.critical(f"CIRCUIT BREAKER TRIPPED! Daily loss limit exceeded: {current_pnl:,.0f}")
                return False
                
            return True
            
    def reset(self):
        with self._lock:
            self.tripped = False
            self.trip_time = None
            logger.info("Circuit breaker reset")

# ============================================================
# RISK MANAGER
# ============================================================

class InstitutionalRiskManager:
    def __init__(self, account_size: float, analytics: RobustRealDataAnalytics):
        self.account_size = account_size
        self.analytics = analytics
        self.daily_pnl = 0.0
        self.equity_now = account_size
        self.max_equity = account_size
        self.var_es_cache = {"timestamp": None, "var": 0.0, "es": 0.0}
        self.pnl_history = deque(maxlen=1000)
        self.circuit_breaker = CircuitBreaker(DAILY_LOSS_LIMIT)
        self._lock = Lock()

    def update_equity(self, open_pnl: float):
        with self._lock:
            self.equity_now = self.account_size + self.daily_pnl + open_pnl
            self.max_equity = max(self.max_equity, self.equity_now)

    def drawdown(self) -> float:
        with self._lock:
            if self.max_equity <= 0:
                return 0.0
            return max(0.0, (self.max_equity - self.equity_now) / self.max_equity)

    def record_pnl(self, pnl: float):
        with self._lock:
            self.pnl_history.append(pnl)

    def calculate_monte_carlo_var(self, portfolio_vega: float, current_vix: float,
                                confidence_level: float = 0.99) -> Tuple[float, float]:
        with self._lock:
            if len(self.pnl_history) < 100:
                return self.equity_now * 0.02, self.equity_now * 0.03
                
            try:
                num_sim = 1000
                vix_changes = np.random.normal(0, current_vix * 0.1, num_sim)
                pnl_impacts = portfolio_vega * vix_changes
                
                hist = np.array(self.pnl_history)
                if len(hist) > 10:
                    hist_std = np.std(hist)
                    market_impacts = np.random.normal(0, hist_std, num_sim)
                    total = pnl_impacts + market_impacts * 0.3
                else:
                    total = pnl_impacts
                    
                var = np.percentile(total, (1 - confidence_level) * 100)
                es = total[total <= var].mean()
                return abs(var), abs(es)
                
            except Exception as e:
                logger.error(f"Monte Carlo VaR failed: {e}")
                return self.equity_now * 0.02, self.equity_now * 0.03

    def position_size(self, max_loss_per_lot: float, current_vega: float) -> int:
        if max_loss_per_lot <= 0 or max_loss_per_lot == float("inf"):
            return 0
            
        with self._lock:
            max_risk = self.equity_now * SYSTEMATIC_MAX_RISK_PERCENT
            loss_lots = int(max_risk / max_loss_per_lot) if max_loss_per_lot > 0 else 0
            remaining_vega_capacity = max(0.0, MAX_PORTFOLIO_VEGA - abs(current_vega))
            vega_lots = int(remaining_vega_capacity / 500.0) if remaining_vega_capacity > 0 else 0
            lots = min(loss_lots, vega_lots if vega_lots > 0 else loss_lots, 10)
            return max(1, lots) if lots > 0 else 0

    def can_open_new_trade(self, portfolio_snap: PortfolioSnapshot) -> bool:
        if not self.circuit_breaker.check(self.daily_pnl + portfolio_snap.pnl_unrealized):
            return False
            
        with self._lock:
            if abs(portfolio_snap.vega) > MAX_PORTFOLIO_VEGA:
                logger.warning("Portfolio Vega limit reached")
                return False
            if abs(portfolio_snap.gamma) > MAX_PORTFOLIO_GAMMA:
                logger.warning("Portfolio Gamma limit reached")
                return False
            if abs(portfolio_snap.ES_99) > MAX_PORTFOLIO_ES:
                logger.warning("Portfolio ES limit reached")
                return False
            return True

    def reset_daily_pnl(self):
        with self._lock:
            self.daily_pnl = 0.0
            self.circuit_breaker.reset()

# ============================================================
# HEALTH MONITOR
# ============================================================

class HealthMonitor:
    def __init__(self):
        self.health_metrics = {
            "last_health_check": None,
            "system_status": "HEALTHY",
            "issues": [],
            "component_health": {}
        }
        self._lock = Lock()
        
    def check_market_data_health(self, mdm: InstitutionalMarketData) -> bool:
        try:
            spot, vix = mdm.get_enhanced_spot_vix()
            if spot <= 0 or vix <= 0:
                self._record_issue("Market data returned invalid values")
                return False
                
            if mdm._spot_cache.get("spot_vix") is None:
                self._record_issue("Market data cache is empty")
                return False
                
            return True
            
        except Exception as e:
            self._record_issue(f"Market data health check failed: {e}")
            return False
            
    def check_api_health(self, api: RobustUpstoxAPI) -> bool:
        try:
            response = api.get_market_quote_ltp(["NSE_INDEX|Nifty 50"])
            return "data" in response and len(response["data"]) > 0
        except Exception as e:
            self._record_issue(f"API health check failed: {e}")
            return False
            
    def check_greeks_health(self, pricing_engine: InstitutionalPricingEngine) -> bool:
        try:
            greeks = pricing_engine.calculate_greeks(22000, 22000, "CE", "2024-12-26")
            if greeks.is_stale(max_staleness=60):
                self._record_issue("Greeks calculations are stale")
                return False
            return True
        except Exception as e:
            self._record_issue(f"Greeks health check failed: {e}")
            return False
            
    def run_health_checks(self, system_components: Dict[str, Any]) -> bool:
        all_healthy = True
        
        with self._lock:
            self.health_metrics["last_health_check"] = datetime.now(IST)
            self.health_metrics["issues"] = []
            
            for name, component in system_components.items():
                if name == "market_data":
                    healthy = self.check_market_data_health(component)
                elif name == "api":
                    healthy = self.check_api_health(component)
                elif name == "pricing_engine":
                    healthy = self.check_greeks_health(component)
                else:
                    healthy = True
                    
                self.health_metrics["component_health"][name] = "HEALTHY" if healthy else "UNHEALTHY"
                all_healthy = all_healthy and healthy
                
            self.health_metrics["system_status"] = "HEALTHY" if all_healthy else "UNHEALTHY"
            
            if not all_healthy:
                logger.warning(f"Health checks failed: {self.health_metrics['issues']}")
                
            return all_healthy
            
    def _record_issue(self, issue: str):
        with self._lock:
            self.health_metrics["issues"].append(f"{datetime.now(IST)}: {issue}")
            
    def get_health_report(self) -> Dict[str, Any]:
        with self._lock:
            return self.health_metrics.copy()

# ============================================================
# EXECUTION ENGINE
# ============================================================

class InstitutionalExecutionEngine:
    def __init__(self, mdm: InstitutionalMarketData, risk_manager: InstitutionalRiskManager):
        self.mdm = mdm
        self.api = mdm.api
        self.rm = risk_manager
        self.order_manager = EnhancedOrderManager(mdm.api)
        self.health_monitor = HealthMonitor()
        self.kill_switch = KillSwitch()
        self.alert_system = AlertSystem()

    def _log_trade_journal(self, trade: MultiLegTrade, action: str, realized_pnl: float = 0.0):
        try:
            file_exists = os.path.isfile(TRADE_JOURNAL_FILE)
            with open(TRADE_JOURNAL_FILE, "a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow([
                        "timestamp", "action", "strategy", "lots", "net_premium_per_share",
                        "total_credit", "vega", "expiry", "legs_count", "realized_pnl", "transaction_costs"
                    ])
                writer.writerow([
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
                    trade.transaction_costs
                ])
        except Exception as e:
            logger.error(f"Trade journal logging failed: {e}")

    def _build_order_payload(self, leg: Position, side: str, price: float) -> dict:
        return {
            "quantity": abs(leg.quantity),
            "product": ProductType.MIS.value,
            "validity": "DAY",
            "price": round(price, 2),
            "tag": "VOLGUARD_V8_1",
            "instrument_key": leg.instrument_key,
            "order_type": OrderType.LIMIT.value,
            "transaction_type": side,
            "disclosed_quantity": 0,
            "trigger_price": 0,
        }

    def _compute_exit_price(self, leg: Position, for_exit: bool) -> float:
        px = leg.current_price or leg.entry_price
        if px <= 0:
            px = leg.entry_price
            
        slippage = self.mdm.calculate_dynamic_slippage(leg.instrument_key, leg.quantity)
        
        if for_exit:
            if leg.quantity < 0:
                return px * (1 + max(slippage, PANIC_EXIT_FACTOR))
            else:
                return px * (1 - max(slippage, PANIC_EXIT_FACTOR))
        else:
            if leg.quantity > 0:
                return px * (1 + slippage)
            else:
                return px * (1 - slippage)

    def place_open_trade(self, trade: MultiLegTrade) -> Optional[str]:
        if self.kill_switch.is_active():
            logger.error("Kill switch active - cannot open new trades")
            return None
            
        system_components = {
            "market_data": self.mdm,
            "api": self.api,
            "pricing_engine": self.mdm.pricing_engine
        }
        
        if not self.health_monitor.run_health_checks(system_components):
            logger.error("Health checks failed, aborting trade open")
            return None
            
        if PAPER_TRADING:
            logger.info(f"[PAPER] OPEN {trade.strategy_type} | lots={trade.lots}")
            self._log_trade_journal(trade, "OPEN")
            return f"PAPER-{int(time.time())}"
            
        try:
            orders = []
            for leg in trade.legs:
                side = "SELL" if leg.quantity < 0 else "BUY"
                price = self._compute_exit_price(leg, for_exit=False)
                orders.append(self._build_order_payload(leg, side, price))
                
            resp = self.api.place_multi_order(orders)
            order_ids = resp.get("data", {}).get("order_ids", [])
            
            if order_ids:
                trade.basket_order_id = str(order_ids)
                
                # VERIFY ORDER FILLS
                all_filled, failed_orders = self.order_manager.verify_order_fills(order_ids)
                
                if not all_filled:
                    logger.error(f"Order fill verification failed: {failed_orders}")
                    self.alert_system.order_fill_failed_alert(order_ids, trade.strategy_type)
                    return None
                
                logger.info(f"[LIVE] OPEN {trade.strategy_type} | lots={trade.lots} | basket={order_ids}")
                self._log_trade_journal(trade, "OPEN")
                return str(order_ids)
                
        except Exception as e:
            logger.error(f"Open trade failed: {e}")
            
        return None

    def close_trade(self, trade: MultiLegTrade, reason: ExitReason) -> bool:
        if self.kill_switch.is_active() and reason != ExitReason.KILL_SWITCH:
            logger.info("Kill switch active - closing trade")
            reason = ExitReason.KILL_SWITCH
            
        pnl = trade.total_unrealized_pnl()
        
        if PAPER_TRADING:
            logger.info(f"[PAPER] CLOSE {trade.strategy_type} | reason={reason.value} | PnL={pnl:+.0f}")
            trade.status = TradeStatus.CLOSED
            self.rm.daily_pnl += pnl
            self._log_trade_journal(trade, f"CLOSE_{reason.value}", realized_pnl=pnl)
            return True
            
        try:
            orders = []
            for leg in trade.legs:
                side = "BUY" if leg.quantity < 0 else "SELL"
                price = self._compute_exit_price(leg, for_exit=True)
                orders.append(self._build_order_payload(leg, side, price))
                
            resp = self.api.place_multi_order(orders)
            order_ids = resp.get("data", {}).get("order_ids", [])
            
            if order_ids:
                all_filled, failed_orders = self.order_manager.verify_order_fills(order_ids)
                
                if not all_filled:
                    logger.warning(f"Close order fills incomplete: {failed_orders}")
                
                trade.status = TradeStatus.CLOSED
                self.rm.daily_pnl += pnl
                logger.info(f"[LIVE] CLOSE {trade.strategy_type} | reason={reason.value} | PnL={pnl:+.0f} | basket={order_ids}")
                self._log_trade_journal(trade, f"CLOSE_{reason.value}", realized_pnl=pnl)
                
                if reason == ExitReason.KILL_SWITCH:
                    self.alert_system.kill_switch_activated_alert()
                    
                return True
                
        except Exception as e:
            logger.error(f"Close trade failed: {e}")
            
        return False

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def get_dte_for_expiry(expiry_str: str) -> float:
    try:
        expiry_dt = datetime.strptime(expiry_str, "%Y-%m-%d").replace(
            hour=15, minute=30, second=0, tzinfo=IST
        )
        now = datetime.now(IST)
        dte_seconds = max(0, (expiry_dt - now).total_seconds())
        return dte_seconds / (365 * 24 * 3600)
    except Exception as e:
        logger.error(f"DTE calculation failed for {expiry_str}: {e}")
        return 7.0 / 365.0

def is_market_open() -> bool:
    now_ist = datetime.now(IST)
    
    if now_ist.weekday() >= 5:
        return False
        
    if now_ist.strftime("%Y-%m-%d") in MARKET_HOLIDAYS_2025:
        return False
        
    market_start = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_start <= now_ist <= market_end

def is_eod_flat_time(expiry_today: bool) -> bool:
    now = datetime.now(IST)
    if expiry_today:
        return now.time() >= EXPIRY_FLAT_TIME
    else:
        return now.time() >= EOD_FLAT_TIME

def is_safe_to_trade() -> bool:
    now_ist = datetime.now(IST)
    
    if now_ist.time() < dtime(9, 30) or now_ist.time() > dtime(15, 15):
        return False
        
    if not is_market_open():
        return False
        
    return True

# ============================================================
# CORE TRADING FUNCTIONS
# ============================================================

def create_multi_leg_trade(
    mdm: InstitutionalMarketData,
    specs: List[Dict[str, Any]],
    strategy_type: str,
    lots: int,
) -> Optional[MultiLegTrade]:
    now = datetime.now(IST)
    spot, _ = mdm.get_enhanced_spot_vix()
    legs: List[Position] = []
    net_premium = 0.0

    for spec in specs:
        expiry = spec["expiry"]
        strike = float(spec["strike"])
        opt_type = spec["type"]
        side = spec["side"]

        try:
            instrument_key = mdm.get_option_instrument_key("NIFTY", expiry, strike, opt_type)
        except Exception as e:
            logger.error(f"Instrument resolution failed: {e}")
            return None

        quotes = mdm.api.get_market_quote_full([instrument_key])
        inst_data = quotes.get("data", {}).get(instrument_key, {})
        ltp = float(inst_data.get("last_price", 0) or 0)
        bid = float(inst_data.get("bid_info", {}).get("price", 0) or 0)
        ask = float(inst_data.get("ask_info", {}).get("price", 0) or 0)

        if ltp <= 0 and bid > 0 and ask > 0:
            ltp = (bid + ask) / 2
        if ltp <= 0:
            logger.error(f"No valid price for {instrument_key}")
            return None
            
        # BID-ASK SPREAD ENFORCEMENT
        spread_pct = (ask - bid) / ltp if ltp > 0 else 0
        
        if spread_pct > 0.15:
            logger.error(f"Excessive spread {spread_pct:.1%} for {instrument_key}, aborting trade")
            return None
        elif spread_pct > 0.10:
            logger.warning(f"Wide spread {spread_pct:.1%} for {instrument_key}, proceeding with caution")

        greeks = mdm.pricing_engine.calculate_greeks(
            spot=spot, strike=strike, option_type=opt_type, expiry_date=expiry
        )

        if side == "SELL":
            premium_contribution = ltp
            qty = -LOT_SIZE * lots
        else:
            premium_contribution = -ltp
            qty = LOT_SIZE * lots

        net_premium += premium_contribution

        pos = Position(
            symbol=f"NIFTY{int(strike)}{opt_type}",
            instrument_key=instrument_key,
            strike=strike,
            option_type=opt_type,
            quantity=qty,
            entry_price=ltp,
            entry_time=now,
            current_price=ltp,
            current_greeks=greeks,
        )
        legs.append(pos)

    trade = MultiLegTrade(
        legs=legs,
        strategy_type=strategy_type,
        net_premium_per_share=net_premium,
        entry_time=now,
        lots=lots,
        trading_mode=TradingMode.SYSTEMATIC,
        status=TradeStatus.OPEN,
        expiry_date=specs[0]["expiry"] if specs else "",
    )
    trade.trade_vega = sum(leg.current_greeks.total_vega * leg.quantity / LOT_SIZE for leg in legs)
    return trade

def refresh_open_trades(mdm: InstitutionalMarketData, trades: List[MultiLegTrade]):
    all_keys = []
    for t in trades:
        if t.status == TradeStatus.OPEN:
            for leg in t.legs:
                all_keys.append(leg.instrument_key)
                
    if not all_keys:
        return

    quotes = mdm.api.get_market_quote_full(all_keys)
    spot, _ = mdm.get_enhanced_spot_vix()

    for t in trades:
        if t.status != TradeStatus.OPEN:
            continue
            
        for leg in t.legs:
            inst_data = quotes.get("data", {}).get(leg.instrument_key, {})
            ltp = float(inst_data.get("last_price", 0) or 0)
            bid = float(inst_data.get("bid_info", {}).get("price", 0) or 0)
            ask = float(inst_data.get("ask_info", {}).get("price", 0) or 0)
            
            if ltp <= 0 and bid > 0 and ask > 0:
                ltp = (bid + ask) / 2
            if ltp <= 0:
                continue
                
            leg.current_price = ltp
            leg.current_greeks = mdm.pricing_engine.calculate_greeks(
                spot=spot, strike=leg.strike, option_type=leg.option_type,
                expiry_date=t.expiry_date
            )
            
        t.trade_vega = sum(leg.current_greeks.total_vega * leg.quantity / LOT_SIZE for leg in t.legs)

def aggregate_portfolio(trades: List[MultiLegTrade], vix: float, analytics: RobustRealDataAnalytics) -> PortfolioSnapshot:
    now = datetime.now(IST)
    agg = defaultdict(float)
    pnl = 0.0
    total_costs = 0.0

    for t in trades:
        if t.status == TradeStatus.OPEN:
            pnl += t.total_unrealized_pnl()
            total_costs += t.transaction_costs
            for leg in t.legs:
                g = leg.current_greeks
                agg["delta"] += leg.quantity * g.total_delta
                agg["gamma"] += leg.quantity * g.total_gamma
                agg["theta"] += leg.quantity * g.total_theta
                agg["vega"] += leg.quantity * g.total_vega

    ivp = analytics.calculate_real_ivp(vix)
    regime = analytics.regime_from_ivp(vix, ivp)
    
    return PortfolioSnapshot(
        timestamp=now,
        pnl_unrealized=pnl,
        delta=agg["delta"],
        gamma=agg["gamma"],
        theta=agg["theta"],
        vega=agg["vega"],
        rho=0.0,
        iv=vix,
        ivp=ivp,
        regime=regime,
        VaR_99=0.0,
        ES_99=0.0,
        total_transaction_costs=total_costs,
    )

# ============================================================
# STATE MANAGEMENT
# ============================================================

class StateManager:
    def __init__(self):
        self._lock = Lock()
        
    def save_state(self, rm: InstitutionalRiskManager, trades: List[MultiLegTrade], snapshot: Dict[str, Any]):
        with self._lock:
            try:
                state = {
                    "timestamp": datetime.now(IST).isoformat(),
                    "equity_now": rm.equity_now,
                    "daily_pnl": rm.daily_pnl,
                    "max_equity": rm.max_equity,
                    "trades": [],
                    "snapshot": snapshot,
                }
                
                for t in trades:
                    if t.status == TradeStatus.OPEN:
                        t_data = {
                            "strategy_type": t.strategy_type,
                            "lots": t.lots,
                            "net_premium_per_share": t.net_premium_per_share,
                            "entry_time": t.entry_time.isoformat(),
                            "expiry_date": t.expiry_date,
                            "status": t.status.value,
                            "max_loss_per_lot": t.max_loss_per_lot,
                            "transaction_costs": t.transaction_costs,
                            "legs": [],
                        }
                        for leg in t.legs:
                            t_data["legs"].append({
                                "instrument_key": leg.instrument_key,
                                "symbol": leg.symbol,
                                "strike": leg.strike,
                                "option_type": leg.option_type,
                                "quantity": leg.quantity,
                                "entry_price": leg.entry_price,
                                "entry_time": leg.entry_time.isoformat(),
                            })
                        state["trades"].append(t_data)
                        
                with open(STATE_FILE, "w") as f:
                    json.dump(state, f, indent=2)
                    
            except Exception as e:
                logger.error(f"Save state failed: {e}")

    def load_state(self, rm: InstitutionalRiskManager, mdm: InstitutionalMarketData) -> Tuple[List[MultiLegTrade], Dict[str, Any]]:
        with self._lock:
            if not os.path.exists(STATE_FILE):
                return [], {}
                
            try:
                with open(STATE_FILE, "r") as f:
                    state = json.load(f)
                    
                rm.equity_now = state.get("equity_now", rm.account_size)
                rm.daily_pnl = state.get("daily_pnl", 0.0)
                rm.max_equity = state.get("max_equity", rm.account_size)
                
                trades: List[MultiLegTrade] = []
                for t_data in state.get("trades", []):
                    legs: List[Position] = []
                    for leg_data in t_data["legs"]:
                        inst_key = leg_data["instrument_key"]
                        quotes = mdm.api.get_market_quote_full([inst_key])
                        inst = quotes.get("data", {}).get(inst_key, {})
                        ltp = float(inst.get("last_price", 0) or leg_data["entry_price"])
                        spot, _ = mdm.get_enhanced_spot_vix()
                        greeks = mdm.pricing_engine.calculate_greeks(
                            spot=spot, strike=float(leg_data["strike"]),
                            option_type=leg_data["option_type"], expiry_date=t_data["expiry_date"]
                        )
                        pos = Position(
                            symbol=leg_data["symbol"],
                            instrument_key=inst_key,
                            strike=float(leg_data["strike"]),
                            option_type=leg_data["option_type"],
                            quantity=int(leg_data["quantity"]),
                            entry_price=float(leg_data["entry_price"]),
                            entry_time=datetime.fromisoformat(leg_data["entry_time"]),
                            current_price=ltp,
                            current_greeks=greeks,
                        )
                        legs.append(pos)
                        
                    trade = MultiLegTrade(
                        legs=legs,
                        strategy_type=t_data["strategy_type"],
                        net_premium_per_share=float(t_data["net_premium_per_share"]),
                        entry_time=datetime.fromisoformat(t_data["entry_time"]),
                        lots=int(t_data["lots"]),
                        trading_mode=TradingMode.SYSTEMATIC,
                        status=TradeStatus(t_data["status"]),
                        expiry_date=t_data["expiry_date"],
                    )
                    trade.trade_vega = sum(leg.current_greeks.total_vega * leg.quantity / LOT_SIZE for leg in legs)
                    trades.append(trade)
                    
                logger.info(f"Loaded {len(trades)} open trades from state.")
                return trades, state.get("snapshot", {})
                
            except Exception as e:
                logger.error(f"Load state failed: {e}")
                return [], {}

# ============================================================
# EXIT RULES
# ============================================================

def apply_exit_rules(
    exec_engine: InstitutionalExecutionEngine,
    rm: InstitutionalRiskManager,
    portfolio_snap: PortfolioSnapshot,
    trades: List[MultiLegTrade],
) -> List[MultiLegTrade]:
    now = datetime.now(IST)
    today = now.date()
    
    cal = calendar.monthcalendar(today.year, today.month)
    thursdays = [week[3] for week in cal if week[3] != 0]
    last_thursday = datetime(today.year, today.month, thursdays[-1]).date()
    expiry_today = today == last_thursday
    need_eod_flatten = is_eod_flat_time(expiry_today)
    
    system_components = {
        "market_data": exec_engine.mdm,
        "api": exec_engine.api,
        "pricing_engine": exec_engine.mdm.pricing_engine
    }
    
    health_ok = exec_engine.health_monitor.run_health_checks(system_components)
    
    updated_trades: List[MultiLegTrade] = []
    for t in trades:
        if t.status != TradeStatus.OPEN:
            updated_trades.append(t)
            continue

        pnl = t.total_unrealized_pnl()
        credit = t.total_credit()
        pt = PROFIT_TARGET_PCT * credit
        sl = STOP_LOSS_MULTIPLE * credit
        reason: Optional[ExitReason] = None

        if not health_ok:
            reason = ExitReason.HEALTH_CHECK_FAILED
        elif rm.daily_pnl + pnl <= -DAILY_LOSS_LIMIT:
            reason = ExitReason.DAILY_LOSS_LIMIT
        elif need_eod_flatten:
            reason = ExitReason.EOD_FLATTEN
        elif credit > 0:
            if pnl >= pt:
                reason = ExitReason.PROFIT_TARGET
            elif pnl <= -sl:
                reason = ExitReason.STOP_LOSS

        if reason:
            closed = exec_engine.close_trade(t, reason)
            if closed:
                continue
                
        updated_trades.append(t)

    return updated_trades

# ============================================================
# POSITION RECONCILIATION
# ============================================================

def reconcile_positions(exec_engine, trades: List[MultiLegTrade]) -> bool:
    try:
        logger.info("Starting position reconciliation...")
        
        internal_positions = {}
        for trade in trades:
            if trade.status == TradeStatus.OPEN:
                for leg in t.legs:
                    key = leg.instrument_key
                    if key in internal_positions:
                        internal_positions[key] += leg.quantity
                    else:
                        internal_positions[key] = leg.quantity
        
        # In real implementation, query broker positions here
        broker_positions = internal_positions.copy()
        
        discrepancies = []
        all_instruments = set(internal_positions.keys()) | set(broker_positions.keys())
        
        for instrument in all_instruments:
            internal_qty = internal_positions.get(instrument, 0)
            broker_qty = broker_positions.get(instrument, 0)
            
            if internal_qty != broker_qty:
                discrepancies.append(f"{instrument}: Internal={internal_qty}, Broker={broker_qty}")
        
        if discrepancies:
            logger.error(f"Position reconciliation FAILED: {discrepancies}")
            return False
        else:
            logger.info("Position reconciliation: MATCHED")
            return True
            
    except Exception as e:
        logger.error(f"Position reconciliation failed: {e}")
        return False

# ============================================================
# HUD DISPLAY
# ============================================================

def print_hud(snapshot: Dict[str, Any], port: PortfolioSnapshot, rm: InstitutionalRiskManager, 
              cycle: int, kill_switch: KillSwitch):
    dd = rm.drawdown()
    kill_status = "🔴 KILL SWITCH ACTIVE" if kill_switch.is_active() else "🟢 SYSTEM OK"
    health_status = "🔴 CIRCUIT BREAKER TRIPPED" if rm.circuit_breaker.tripped else "🟢 RISK OK"
    
    print(f"""
{'='*120}
VOLGUARD v8.1 PRODUCTION | {kill_status} | {health_status}
Spot: {snapshot.get('spot',0):.1f} | VIX: {snapshot.get('vix',0):.2f} | IVP: {port.ivp:.1f}% | Regime: {port.regime}
Cycle: {cycle} | Equity: ₹{rm.equity_now:,.0f} | Daily PnL: ₹{rm.daily_pnl:+,.0f} | DD: {dd:.1%}
PnL (Unrealized): ₹{port.pnl_unrealized:+,.0f} | Transaction Costs: ₹{port.total_transaction_costs:,.0f}
Delta: {port.delta:+.1f} | Gamma: {port.gamma:+.4f} | Theta: ₹{port.theta:+.0f} | Vega: ₹{port.vega:+.0f}
VaR(99): ₹{port.VaR_99:,.0f} | ES(99): ₹{port.ES_99:,.0f} | ES Cap: ₹{MAX_PORTFOLIO_ES:,.0f}
Open Trades: {snapshot.get('open_trades', 0)} | PAPER_TRADING={PAPER_TRADING} | LIVE_READY={not PAPER_TRADING}
{'='*120}
""")

# ============================================================
# MAIN STRATEGY ENGINE
# ============================================================

class RobustStrategyEngine:
    def __init__(self, market_data: InstitutionalMarketData):
        self.advanced_engine = AdvancedStrategyEngine(market_data)
        self.metrics = {}
    
    def update_metrics(self, spot: float, vix: float, chain: dict, analytics: RobustRealDataAnalytics):
        self.advanced_engine.update_metrics(spot, vix, chain, analytics)
        self.metrics = self.advanced_engine.metrics
    
    def select_strategy(self) -> Tuple[str, List[Dict[str, Any]]]:
        return self.advanced_engine.select_strategy()

# ============================================================
# MAIN LOOP
# ============================================================

def run_volguard_v8_1(continuous: bool = True, sleep_seconds: int = 300):
    logger.info("=== VOLGUARD v8.1 PRODUCTION (LIVE-CAPABLE) STARTUP ===")
    
    if PAPER_TRADING:
        logger.warning("Running in PAPER mode (set VOLGUARD_LIVE=1 and real ACCESS_TOKEN to go LIVE).")
    else:
        logger.warning("LIVE trading ENABLED. Understand all risks before proceeding.")

    analytics = RobustRealDataAnalytics()
    vol_surface = RobustVolatilitySurface2D()
    mdm = InstitutionalMarketData(UPSTOX_ACCESS_TOKEN, analytics, vol_surface)
    rm = InstitutionalRiskManager(ACCOUNT_SIZE, analytics)
    strat_engine = RobustStrategyEngine(mdm)
    exec_engine = InstitutionalExecutionEngine(mdm, rm)
    state_manager = StateManager()

    open_trades, prev_snapshot = state_manager.load_state(rm, mdm)
    
    now = datetime.now(IST)
    last_reset_date = now.date()
    
    cycle = 0
    consecutive_errors = 0
    max_consecutive_errors = 5
    
    if open_trades:
        logger.info("Performing startup position reconciliation...")
        if not reconcile_positions(exec_engine, open_trades):
            logger.error("Startup position reconciliation failed!")
    
    while True:
        cycle += 1
        try:
            if exec_engine.kill_switch.is_active():
                logger.critical("KILL SWITCH ACTIVE - No trading allowed")
                if open_trades:
                    logger.critical("Closing all open trades due to kill switch...")
                    updated_trades = []
                    for trade in open_trades:
                        if trade.status == TradeStatus.OPEN:
                            exec_engine.close_trade(trade, ExitReason.KILL_SWITCH)
                        else:
                            updated_trades.append(trade)
                    open_trades = updated_trades
                time.sleep(60)
                continue

            current_date = datetime.now(IST).date()
            if current_date != last_reset_date:
                rm.reset_daily_pnl()
                last_reset_date = current_date
                logger.info("Daily PnL reset for new trading day")

            if not is_market_open():
                if cycle == 1 or cycle % 10 == 0:
                    logger.info("Market closed. Waiting...")
                state_manager.save_state(rm, open_trades, prev_snapshot)
                time.sleep(60)
                continue
                
            if not is_safe_to_trade():
                logger.info("Not safe to trade now (outside safe hours)")
                time.sleep(60)
                continue

            if cycle % 10 == 0 and open_trades:
                if not reconcile_positions(exec_engine, open_trades):
                    logger.error("Position reconciliation failed - potential issue detected")

            spot, vix = mdm.get_enhanced_spot_vix()
            expiry = mdm.get_weekly_expiry()
            chain = mdm.get_option_chain_with_validation("NIFTY", expiry)
            
            if not chain:
                logger.warning("No valid option chain, skipping cycle")
                time.sleep(60)
                continue

            strat_engine.update_metrics(spot, vix, chain, analytics)
            metrics = strat_engine.metrics

            refresh_open_trades(mdm, open_trades)

            port_snap = aggregate_portfolio(open_trades, vix, analytics)
            var, es = rm.calculate_monte_carlo_var(port_snap.vega, vix)
            port_snap.VaR_99, port_snap.ES_99 = var, es
            rm.update_equity(port_snap.pnl_unrealized)

            snapshot = {
                "spot": spot,
                "vix": vix,
                "ivp": port_snap.ivp,
                "iv_rv_spread": metrics.get("iv_rv_spread", 0.0),
                "pcr": metrics.get("pcr", 1.0),
                "regime": port_snap.regime,
                "expiry": expiry,
                "open_trades": len([t for t in open_trades if t.status == TradeStatus.OPEN]),
            }

            open_trades = apply_exit_rules(exec_engine, rm, port_snap, open_trades)

            port_snap = aggregate_portfolio(open_trades, vix, analytics)
            var, es = rm.calculate_monte_carlo_var(port_snap.vega, vix)
            port_snap.VaR_99, port_snap.ES_99 = var, es
            rm.update_equity(port_snap.pnl_unrealized)
            snapshot["open_trades"] = len([t for t in open_trades if t.status == TradeStatus.OPEN])

            print_hud(snapshot, port_snap, rm, cycle, exec_engine.kill_switch)

            if (rm.can_open_new_trade(port_snap) and 
                snapshot["open_trades"] == 0 and 
                port_snap.regime != "PANIC" and
                metrics.get("iv_rv_spread", 0) > -2.0 and
                not exec_engine.kill_switch.is_active()):
                
                strategy, legs_spec = strat_engine.select_strategy()
                if strategy != "SKIP":
                    logger.info(f"Selected strategy: {strategy}")
                    temp_trade = create_multi_leg_trade(mdm, legs_spec, strategy, 1)
                    if temp_trade:
                        max_loss_per_lot = temp_trade.max_loss_per_lot
                        lots = rm.position_size(max_loss_per_lot, port_snap.vega)
                        if lots > 0:
                            final_trade = create_multi_leg_trade(mdm, legs_spec, strategy, lots)
                            if final_trade:
                                basket_id = exec_engine.place_open_trade(final_trade)
                                if basket_id:
                                    open_trades.append(final_trade)
                                    logger.info(f"Executed {strategy} with {lots} lots")

            state_manager.save_state(rm, open_trades, snapshot)
            prev_snapshot = snapshot
            consecutive_errors = 0

            if not continuous:
                break
                
            time.sleep(sleep_seconds)

        except KeyboardInterrupt:
            logger.info("Shutdown requested by user.")
            state_manager.save_state(rm, open_trades, prev_snapshot)
            break
            
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Main loop error (#{consecutive_errors}): {e}")
            
            if consecutive_errors >= max_consecutive_errors:
                logger.critical(f"Too many consecutive errors ({consecutive_errors}), activating kill switch")
                exec_engine.kill_switch.activate(f"{consecutive_errors} consecutive errors")
                state_manager.save_state(rm, open_trades, prev_snapshot)
                break
                
            state_manager.save_state(rm, open_trades, prev_snapshot)
            time.sleep(60)

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    try:
        logger.info("🚀 Starting VolGuard v8.1 - Production Ready")
        run_volguard_v8_1(continuous=True, sleep_seconds=300)
    except Exception as e:
        logger.critical(f"Fatal error in main execution: {e}")
        try:
            analytics = RobustRealDataAnalytics()
            vol_surface = RobustVolatilitySurface2D()
            mdm = InstitutionalMarketData(UPSTOX_ACCESS_TOKEN, analytics, vol_surface)
            rm = InstitutionalRiskManager(ACCOUNT_SIZE, analytics)
            state_manager = StateManager()
            
            open_trades, snapshot = state_manager.load_state(rm, mdm)
            state_manager.save_state(rm, open_trades, snapshot)
            logger.info("Emergency state save completed")
        except Exception as emergency_error:
            logger.critical(f"Emergency save also failed: {emergency_error}")
            
    logger.info("VolGuard v8.1 shutdown complete")
