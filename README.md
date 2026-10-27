# FTSE 100 Statistical Arbitrage — Pairs Trading System

A cointegration-based pairs trading backtest over 50 FTSE 100 constituents, with an
interactive Streamlit dashboard that separates in-sample pair selection from
out-of-sample trading.

**Live demo:** _coming soon_ <!-- replace with https://<your-app>.streamlit.app -->

![Dashboard screenshot](docs/screenshot.png) <!-- add a screenshot at docs/screenshot.png -->

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at http://localhost:8501. Adjust thresholds, z-score window,
ADF cutoff, costs, pair count and the in-sample / out-of-sample split in the
sidebar, then press **Run backtest**.

- Prices come from the bundled snapshot `data/prices_snapshot.csv`, so the app
  works offline. Tick **Refresh live data (yfinance)** to download fresh prices
  instead; if the download fails the app falls back to the snapshot.
- Pairs and hedge ratios are fitted on in-sample data only, then traded
  unchanged out-of-sample. Metrics are reported separately for each period.
- `data/pair_scan_snapshot.csv` caches the pair scan for the snapshot at the
  default split so the app starts quickly. Regenerate it after replacing the
  snapshot with `python -m ui.engine`.

App code lives in `app.py` (layout) and `ui/` (data loading, in/out-of-sample
engine, Plotly charts). The analysis modules below are reused unchanged.

## Pipeline

1. **`data_loader.py`** — downloads 2 years of daily OHLCV via `yfinance` for the 50
   tickers in `config.TICKERS`, extracts adjusted close, caches to
   `data/adj_close_prices.csv`.
2. **`pair_selection.py`** — tests all C(50,2) = 1,225 ticker combinations: OLS
   regression (`price_A ~ price_B`) for the hedge ratio, then an Augmented
   Dickey-Fuller test on the residual spread. Pairs with ADF p-value < 0.05 are
   flagged cointegrated.
3. **`signals.py`** — builds `spread = price_A - beta * price_B`, its 30-day
   rolling z-score, and a position state machine: enter at |z| >= 2, exit when
   z reverts to 0.
4. **`backtest.py`** — simulates daily cash/holdings/portfolio value per pair
   with dollar-neutral sizing and a 0.1% transaction cost per rebalance.
5. **`metrics.py`** — annualised return, annualised Sharpe ratio (rf=0%), win
   rate (fraction of profitable round trips), max drawdown.
6. **`report.py`** — matplotlib PNG charts (per-pair spread/z-score/equity,
   aggregate portfolio curve, drawdown) and an openpyxl Excel dashboard.
7. **`main.py`** — orchestrates the full run.

## Batch pipeline (CLI)

```bash
pip install -r requirements.txt
python main.py
```

Outputs land in `output/`:
- `pairs_trading_dashboard.xlsx` — Summary / Pair Stats / Trade Log / Pair Charts sheets
- `charts/*.png` — all rendered charts
- `trade_logs/trade_log.csv`, `trade_logs/round_trips.csv` — raw CSV trade data

Delete `data/adj_close_prices.csv` to force a fresh yfinance download (the
pipeline uses the cache by default so repeated runs don't re-hit the network).

## Key parameters (`config.py`)

| Parameter | Value |
|---|---|
| History window | 2y daily |
| ADF p-value cutoff | 0.05 |
| Z-score window | 30 days |
| Entry / exit threshold | ±2σ / 0σ |
| Transaction cost | 0.1% of notional per rebalance |
| Capital per pair | £100,000 |
| Max pairs traded | 20 (ranked by ADF p-value) |

## Notes / caveats

- The 50-ticker list is a representative snapshot of large, liquid FTSE 100
  names, not a live feed of index committee membership.
- Cointegration is tested in-sample over the full 2y window (no train/test
  split) — this is a methodology demo, not a walk-forward production system.
  A real deployment should re-test cointegration out-of-sample and roll the
  pair universe over time.
- Position sizing is dollar-neutral per leg (capital fully deployed in leg A,
  leg B sized at the OLS beta), not volatility- or risk-parity weighted.
