"""In-sample / out-of-sample wrapper around the existing pipeline modules.

run_pipeline() reads every parameter from config and selects pairs on the full
history, so the app drives the underlying modules directly instead. The
analysis code itself (OLS + ADF, z-score state machine, backtest, metrics) is
reused unchanged.
"""
import logging

import numpy as np
import pandas as pd
import streamlit as st

import backtest
import config
import metrics
import pair_selection
import signals

log = logging.getLogger(__name__)

# Pair scan for the bundled snapshot at its default split, so a cold start on
# Streamlit Cloud skips ~1,000 ADF tests. Regenerate with `python -m ui.engine`.
SCAN_CACHE_CSV = config.DATA_DIR / "pair_scan_snapshot.csv"

PERIODS = ("is", "oos")
PERIOD_LABELS = {"is": "In-sample", "oos": "Out-of-sample"}

TRADE_LOG_COLS = ["date", "ticker_a", "ticker_b", "from_position", "to_position", "trade_shares_a",
                  "trade_shares_b", "price_a", "price_b", "notional_traded", "transaction_cost"]
ROUND_TRIP_COLS = ["ticker_a", "ticker_b", "side", "open_date", "close_date", "pnl", "return_pct"]


def half_life(spread: pd.Series) -> float:
    """Ornstein-Uhlenbeck half-life in days from regressing d(spread) on lagged spread."""
    lagged = spread.shift(1).iloc[1:].to_numpy()
    delta = spread.diff().iloc[1:].to_numpy()
    slope = np.polyfit(lagged, delta, 1)[0]
    return float(-np.log(2) / slope) if slope < 0 else np.nan


def _scan(in_sample: pd.DataFrame) -> pd.DataFrame:
    stats = pair_selection.find_cointegrated_pairs(in_sample)
    # The module flags pairs at config's cutoff; the app applies the sidebar cutoff instead.
    stats = stats.drop(columns=["cointegrated"])
    stats["half_life"] = [
        half_life(in_sample[a] - beta * in_sample[b])
        for a, b, beta in zip(stats["ticker_a"], stats["ticker_b"], stats["beta"])
    ]
    return stats


@st.cache_data(show_spinner=False)
def scan_pairs(_prices: pd.DataFrame, data_key: str, split_date: pd.Timestamp) -> pd.DataFrame:
    """OLS hedge ratio + ADF test for every pair, fitted on in-sample prices only."""
    split_date = pd.Timestamp(split_date)
    if data_key == "snapshot" and SCAN_CACHE_CSV.exists():
        cached = pd.read_csv(SCAN_CACHE_CSV)
        if (cached["split_date"] == str(split_date.date())).all():
            return cached.drop(columns=["split_date"])
    return _scan(_prices[_prices.index < split_date])


def _signal_frame(prices, ticker_a, ticker_b, beta, window, entry_z, exit_z) -> pd.DataFrame:
    # Same steps as signals.build_signal_frame, which doesn't forward window/thresholds.
    spread = signals.compute_spread(prices, ticker_a, ticker_b, beta)
    zscore = signals.compute_zscore(spread, window)
    return pd.DataFrame({
        "price_a": prices[ticker_a],
        "price_b": prices[ticker_b],
        "spread": spread,
        "zscore": zscore,
        "position": signals.generate_positions(zscore, entry_z, exit_z),
    })


@st.cache_data(show_spinner=False, max_entries=512)
def backtest_pair(_prices, data_key, split_date, ticker_a, ticker_b, beta, window, entry_z, exit_z, tc_rate) -> dict:
    """Signals on the full history, then an independent backtest per period.

    The rolling z-score only looks backwards, so computing it on the full series
    leaks nothing and means out-of-sample trading doesn't wait for a fresh window.
    Each period starts flat with fresh capital so its metrics stand alone.
    """
    split_date = pd.Timestamp(split_date)
    frame = _signal_frame(_prices, ticker_a, ticker_b, beta, window, entry_z, exit_z)
    halves = {"is": frame[frame.index < split_date], "oos": frame[frame.index >= split_date]}
    results = {
        p: backtest.run_backtest(f, ticker_a, ticker_b, beta, transaction_cost_rate=tc_rate)
        for p, f in halves.items() if not f.empty
    }
    return {"signals": frame, "results": results}


def pair_metrics(bt: dict) -> dict:
    return {p: metrics.summarise(res.daily["portfolio_value"], res.round_trips) for p, res in bt["results"].items()}


def concat_frames(frames: list, columns: list) -> pd.DataFrame:
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)


@st.cache_data(show_spinner=False, max_entries=64)
def run_strategy(_prices, data_key, split_date, adf_cutoff, max_pairs, window, entry_z, exit_z, tc_rate) -> dict:
    scan = scan_pairs(_prices, data_key, split_date).copy()
    scan["cointegrated"] = scan["adf_pvalue"] < adf_cutoff
    traded = scan[scan["cointegrated"]].head(max_pairs)
    scan["traded"] = scan.index.isin(traded.index)

    collected = {p: {"values": [], "trade_logs": [], "round_trips": []} for p in PERIODS}
    pair_rows = []
    for _, row in traded.iterrows():
        bt = backtest_pair(_prices, data_key, split_date, row["ticker_a"], row["ticker_b"], row["beta"],
                           window, entry_z, exit_z, tc_rate)
        pair_row = {"ticker_a": row["ticker_a"], "ticker_b": row["ticker_b"]}
        for p, res in bt["results"].items():
            m = metrics.summarise(res.daily["portfolio_value"], res.round_trips)
            pair_row.update({f"{p}_sharpe": m["sharpe_ratio"], f"{p}_pnl": m["total_pnl"],
                             f"{p}_trades": m["num_round_trips"]})
            collected[p]["values"].append(res.daily["portfolio_value"])
            collected[p]["trade_logs"].append(res.trade_log.assign(period=PERIOD_LABELS[p]))
            collected[p]["round_trips"].append(res.round_trips.assign(period=PERIOD_LABELS[p]))
        pair_rows.append(pair_row)

    if pair_rows:
        scan = scan.merge(pd.DataFrame(pair_rows), on=["ticker_a", "ticker_b"], how="left")

    periods = {}
    for p in PERIODS:
        c = collected[p]
        equity = pd.concat(c["values"], axis=1).sum(axis=1) if c["values"] else pd.Series(dtype=float)
        round_trips = concat_frames(c["round_trips"], ROUND_TRIP_COLS + ["period"])
        periods[p] = {
            "equity": equity,
            "metrics": metrics.summarise(equity, round_trips),
            "trade_log": concat_frames(c["trade_logs"], TRADE_LOG_COLS + ["period"]),
            "round_trips": round_trips,
        }

    return {"scan": scan, "num_traded": len(traded), "periods": periods}


if __name__ == "__main__":
    from ui.data import SNAPSHOT_CSV, default_split_date

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prices = pd.read_csv(SNAPSHOT_CSV, index_col=0, parse_dates=True)
    split = default_split_date(prices)
    out = _scan(prices[prices.index < split])
    out["split_date"] = str(split.date())
    out.to_csv(SCAN_CACHE_CSV, index=False)
    log.info("Wrote %d pairs (split %s) -> %s", len(out), split.date(), SCAN_CACHE_CSV)
