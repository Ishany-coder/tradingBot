"""Point-in-time (~early 2020) top-10 holdings of the 11 SPDR sector ETFs.

Used ONLY by the 2020 backtest to remove the survivorship bias of testing
today's winners back in time. These are the cap-weighted sector mega-caps as of
~Q1 2020, reconstructed from public records (SEC NPORT-P / 497K filings and
contemporaneous fund fact sheets). The LIVE trader still uses *current* holdings
(correct for trading today) — this module never touches it.

Caveats:
- Approximate to ~early 2020; exact rank/membership of the 10th name may differ.
- Tickers reflect 2020 identities where they still resolve in yfinance:
  BRK-B (Berkshire), META (ex-FB, history intact). ATVI (Activision) legitimately
  delists in 2023 on the Microsoft acquisition — that is correct point-in-time
  behaviour (it drops out when acquired), not a bug.
- This is still a *fixed* 2020 snapshot held constant to today, not full annual
  re-constitution — but it removes the dominant "today's winners" lookahead.
"""

HOLDINGS_2020 = {
    "XLK": ["AAPL", "MSFT", "V", "MA", "INTC", "CSCO", "NVDA", "ADBE", "CRM", "ACN"],
    "XLF": ["BRK-B", "JPM", "BAC", "WFC", "C", "USB", "GS", "MS", "BLK", "AXP"],
    "XLV": ["JNJ", "UNH", "MRK", "PFE", "ABBV", "ABT", "TMO", "AMGN", "MDT", "LLY"],
    "XLY": ["AMZN", "HD", "MCD", "NKE", "LOW", "SBUX", "BKNG", "TJX", "TGT", "GM"],
    "XLP": ["PG", "KO", "PEP", "WMT", "COST", "MO", "MDLZ", "PM", "CL", "KMB"],
    "XLE": ["XOM", "CVX", "COP", "EOG", "SLB", "KMI", "PSX", "VLO", "MPC", "WMB"],
    "XLI": ["BA", "HON", "UNP", "UPS", "MMM", "CAT", "LMT", "GE", "RTX", "DE"],
    "XLB": ["LIN", "APD", "SHW", "ECL", "NEM", "FCX", "DD", "DOW", "PPG", "NUE"],
    "XLU": ["NEE", "DUK", "D", "SO", "AEP", "EXC", "XEL", "SRE", "ED", "WEC"],
    "XLRE": ["AMT", "PLD", "CCI", "EQIX", "DLR", "PSA", "SPG", "WELL", "AVB", "EQR"],
    "XLC": ["GOOGL", "GOOG", "META", "T", "VZ", "DIS", "NFLX", "CMCSA", "CHTR", "ATVI"],
}
