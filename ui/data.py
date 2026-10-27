import logging

import pandas as pd
import streamlit as st

import config
import data_loader

log = logging.getLogger(__name__)

# Committed to the repo so the app never depends on yfinance, which is often
# rate-limited from shared cloud IPs. Live downloads go to config.PRICES_CSV
# (git-ignored) so a refresh can never overwrite the snapshot.
SNAPSHOT_CSV = config.DATA_DIR / "prices_snapshot.csv"


@st.cache_data(show_spinner=False)
def load_snapshot() -> pd.DataFrame:
    return pd.read_csv(SNAPSHOT_CSV, index_col=0, parse_dates=True)


@st.cache_data(show_spinner=False, ttl="6h")
def _fetch_live() -> pd.DataFrame:
    return data_loader.download_adjusted_close(use_cache=False)


def load_prices(live: bool) -> tuple[pd.DataFrame, str, str | None]:
    """Return (prices, source_key, warning). Falls back to the snapshot if yfinance fails.

    source_key identifies the dataset for downstream caches, so results computed
    on the snapshot are never reused for live data or vice versa.
    """
    if live:
        try:
            prices = _fetch_live()
            if prices.shape[1] < 2:
                raise RuntimeError("fewer than two tickers returned")
            return prices, f"live:{prices.index[-1].date()}", None
        except Exception as exc:  # noqa: BLE001 - any network/parse failure falls back
            log.warning("Live download failed: %s", exc)
            warning = f"Live yfinance download failed ({exc}). Showing the bundled snapshot instead."
            return load_snapshot(), "snapshot", warning
    return load_snapshot(), "snapshot", None


def default_split_date(prices: pd.DataFrame) -> pd.Timestamp:
    """60% of the history in-sample, 40% out-of-sample."""
    return prices.index[int(len(prices) * 0.6)]
