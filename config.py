"""Central configuration for the momentum strategy.

Every tunable lives here so the strategy can be re-parameterised without
touching logic. Import `config as C` elsewhere and read `C.SOMETHING`.
"""

from pathlib import Path

# --- Universe ---------------------------------------------------------------
# The 11 SPDR sector ETFs. Stage 1 ranks these by momentum.
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Health Care",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

BENCHMARK = "SPY"          # buy-and-hold comparison on the equity-curve chart
TOP_HOLDINGS_N = 10        # constituents pulled per ETF (yfinance top holdings)

# --- Strategy parameters ----------------------------------------------------
N_SECTORS = 3              # Stage 1: number of top sectors to hold
N_STOCKS_MIN = 10          # Stage 2: equal-weight book size (lower bound)
N_STOCKS_MAX = 15          # Stage 2: equal-weight book size (upper bound)

# Momentum lookbacks, in months.
MOM_LONG = 12              # "12-1" long leg
MOM_SKIP = 1              # "12-1" skip the most recent month
MOM_MED = 6               # 6-month momentum feature
MOM_SHORT = 3             # 3-month momentum feature

VOL_WINDOW_DAYS = 60       # trailing volatility feature (daily-return std)

# --- Fundamentals / point-in-time -------------------------------------------
# yfinance reports fiscal PERIOD-END dates, not filing dates. Companies file
# weeks-to-months later, so we lag each quarterly report before it is allowed
# to influence a decision. 2 quarters is conservative (safe against slow
# filers); lower it to 1 for a more aggressive (slightly optimistic) test.
FUND_LAG_QUARTERS = 2

# --- Model ------------------------------------------------------------------
MIN_TRAIN_MONTHS = 24      # months of history required before first prediction
GBM_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)

# --- Backtest ---------------------------------------------------------------
BACKTEST_YEARS = 5         # history window to download / test over
START_CAPITAL = 100_000.0  # simulation only; no real money, no broker
COST_PER_TRADE = 0.0005    # 0.05% commission + slippage, charged on turnover
RISK_FREE_ANNUAL = 0.0     # used in the Sharpe ratio

# --- Monte Carlo ------------------------------------------------------------
MC_RUNS = 1000             # monthly-return reshuffles
MC_SEED = 7

# --- Guardrails -------------------------------------------------------------
SHARPE_WARN = 2.5          # above this, warn: likely overfit / lookahead bug

# --- Position sizing (conviction, not equal weight) -------------------------
# weight_i proportional to confidence_i / volatility_i, then capped + normalised.
MAX_WEIGHT = 0.25          # no single name above 25% of the book
INVEST_FRACTION = 0.98     # keep a small cash buffer; never lever
CONF_LABEL = "P(beats cross-sectional median next month)"

# --- Live execution (Alpaca PAPER only) -------------------------------------
REBALANCE_BAND = 0.03      # only trade a name if target vs current weight
                           # drifts more than 3% of equity (anti-churn)
MIN_ORDER_USD = 10.0       # skip dust orders below this notional
RECOMPUTE_HOURS = 1        # how often run_loop recomputes during market hours
ENABLED = True             # master switch; set False to halt all trading
PAPER_HOST_MARKER = "paper-api"  # broker refuses any endpoint without this

# --- State (trade log, dry-run flag, kill-switch) ---------------------------
STATE_DIR = Path(__file__).parent / "data" / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
DRYRUN_FLAG = STATE_DIR / "dryrun_done.flag"   # first run is forced dry-run
STOP_FILE = STATE_DIR / "STOP"                  # create this to halt trading
TRADE_LOG = STATE_DIR / "trade_log.jsonl"

# --- Caching ----------------------------------------------------------------
CACHE_DIR = Path(__file__).parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
