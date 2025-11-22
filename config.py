from __future__ import annotations

import os
import logging
import warnings
from datetime import time as dtime

import pytz

warnings.filterwarnings("ignore", category=FutureWarning)

# Timezone
IST = pytz.timezone("Asia/Kolkata")

# API base
API_BASE_V2 = "https://api.upstox.com/v2"
API_BASE_V3 = "https://api.upstox.com/v3"
UPSTOX_API_BASE = "https://api.upstox.com"

# OAuth / Auth – either set env or use auth.py to generate tokens
UPSTOX_CLIENT_ID = os.getenv("UPSTOX_CLIENT_ID", "")
UPSTOX_CLIENT_SECRET = os.getenv("UPSTOX_CLIENT_SECRET", "")
UPSTOX_REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI", "https://localhost")

UPSTOX_ACCESS_TOKEN = os.getenv("UPSTOX_ACCESS_TOKEN", "")
UPSTOX_REFRESH_TOKEN = os.getenv("UPSTOX_REFRESH_TOKEN", "")

LIVE_FLAG = os.getenv("VOLGUARD_LIVE", "0") == "1"
PAPER_TRADING = not LIVE_FLAG or not UPSTOX_ACCESS_TOKEN

# Alert / Email
ALERT_EMAIL = os.getenv("ALERT_EMAIL", "")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD", "")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587

# Data sources
VIX_HISTORY_URL = "https://raw.githubusercontent.com/shritish20/VolGuard/refs/heads/main/atmiv.csv"
NIFTY_HISTORY_URL = "https://raw.githubusercontent.com/shritish20/VolGuard/refs/heads/main/nifty_50.csv"

# Account / files
ACCOUNT_SIZE = float(os.getenv("VOLGUARD_ACCOUNT_SIZE", "500000"))
LOT_SIZE = 50

STATE_FILE = "volguard_state_v9.json"
TRADE_LOG_FILE = "volguard_trade_log_v9.txt"
TRADE_JOURNAL_FILE = "volguard_trade_journal_v9.csv"
HEALTH_METRICS_FILE = "volguard_health_metrics_v9.json"

# Risk parameters
SYSTEMATIC_MAX_RISK_PERCENT = 0.01
MAX_PORTFOLIO_VEGA = 1000.0
MAX_PORTFOLIO_GAMMA = 2.0
MAX_PORTFOLIO_ES = ACCOUNT_SIZE * 0.02
DAILY_LOSS_LIMIT = ACCOUNT_SIZE * 0.03

IVP_PANIC = 85.0
IVP_CALM = 35.0

PROFIT_TARGET_PCT = 0.35
STOP_LOSS_MULTIPLE = 2.0

# Slippage & charges
BASE_SLIPPAGE = 0.0005
VOLATILITY_SLIPPAGE_MULTIPLIER = 2.0
LIQUIDITY_SLIPPAGE_FACTOR = 0.001
PANIC_EXIT_FACTOR = 0.10
BROKERAGE_PER_ORDER = 20.0
STT_RATE = 0.0005
GST_RATE = 0.18
EXCHANGE_CHARGES = 0.00002

RISK_FREE_RATE = 0.05
MAX_ORDER_RETRIES = 3
ORDER_TIMEOUT_SECONDS = 30
ORDER_FILL_TIMEOUT = 10

MARKET_HOLIDAYS_2025 = [
    "2025-01-26", "2025-03-07", "2025-03-25", "2025-04-11",
    "2025-04-14", "2025-04-17", "2025-05-01", "2025-06-26",
    "2025-08-15", "2025-09-05", "2025-10-02", "2025-10-22",
    "2025-11-04", "2025-11-14", "2025-12-25",
]

EOD_FLAT_TIME = dtime(15, 15)
EXPIRY_FLAT_TIME = dtime(14, 30)
MARKET_OPEN_TIME = dtime(9, 15)
MARKET_CLOSE_TIME = dtime(15, 30)

# Logging
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
        logging.CRITICAL: bold_red + "%(asctime)s - %(levelname)s - %(message)s" + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


logger = logging.getLogger("VolGuard")
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(TRADE_LOG_FILE)
stream_handler = logging.StreamHandler()

file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_handler.setFormatter(file_formatter)
stream_handler.setFormatter(CustomFormatter())

if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
