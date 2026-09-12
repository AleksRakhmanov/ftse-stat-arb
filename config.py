from pathlib import Path

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
# 50 large, liquid FTSE 100 constituents (Yahoo Finance tickers, ".L" suffix).
# Index membership drifts over time; this is a representative snapshot rather
# than a live feed of the index committee's decisions.
TICKERS = [
    "AZN.L", "SHEL.L", "HSBA.L", "ULVR.L", "BP.L",
    "RIO.L", "GSK.L", "DGE.L", "BATS.L", "REL.L",
    "NG.L", "LSEG.L", "GLEN.L", "AAL.L", "BARC.L",
    "LLOY.L", "VOD.L", "PRU.L", "CPG.L", "RKT.L",
    "STAN.L", "TSCO.L", "NWG.L", "IMB.L", "BT-A.L",
    "SSE.L", "CRH.L", "III.L", "AHT.L", "SGE.L",
    "EXPN.L", "SMIN.L", "WTB.L", "BNZL.L", "CCH.L",
    "SN.L", "IHG.L", "ITRK.L", "PSON.L", "SPX.L",
    "HLN.L", "ANTO.L", "MNDI.L", "UU.L", "SVT.L",
    "JD.L", "KGF.L", "LAND.L", "SGRO.L", "BDEV.L",
]

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
HISTORY_PERIOD = "2y"
MIN_OBS = 400
DATA_DIR = Path(__file__).resolve().parent / "data"
PRICES_CSV = DATA_DIR / "adj_close_prices.csv"

# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------
ADF_PVALUE_THRESHOLD = 0.05

# ---------------------------------------------------------------------------
# Signal generation
# ---------------------------------------------------------------------------
ZSCORE_WINDOW = 30
ENTRY_Z = 2.0
EXIT_Z = 0.0

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
TRANSACTION_COST_RATE = 0.001
CAPITAL_PER_PAIR = 100_000.0
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.0

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
CHARTS_DIR = OUTPUT_DIR / "charts"
TRADE_LOGS_DIR = OUTPUT_DIR / "trade_logs"
EXCEL_REPORT_PATH = OUTPUT_DIR / "pairs_trading_dashboard.xlsx"

# Cap the number of pairs actually backtested (cointegrated pairs are ranked
# by ADF p-value and the strongest N are traded). Set to None to trade all.
MAX_PAIRS_TO_TRADE = 20
