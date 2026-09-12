import logging

import pandas as pd
import yfinance as yf

import config

log = logging.getLogger(__name__)


def download_adjusted_close(
    tickers=None,
    period=None,
    use_cache=True,
) -> pd.DataFrame:
    """Download 2y daily OHLCV for `tickers` and return an adjusted-close matrix.

    Columns with too few observations (delisted/illiquid/failed downloads) are
    dropped. Results are cached to `config.PRICES_CSV` so repeated runs don't
    re-hit the network.
    """
    tickers = tickers or config.TICKERS
    period = period or config.HISTORY_PERIOD

    if use_cache and config.PRICES_CSV.exists():
        log.info("Loading cached prices from %s", config.PRICES_CSV)
        prices = pd.read_csv(config.PRICES_CSV, index_col=0, parse_dates=True)
        return prices

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Downloading %d tickers over %s from yfinance...", len(tickers), period)
    raw = yf.download(
        tickers,
        period=period,
        interval="1d",
        auto_adjust=False,   # keep raw Close and Adj Close distinct
        group_by="column",
        progress=False,
        threads=True,
    )

    if raw.empty:
        raise RuntimeError("yfinance returned no data for the requested tickers.")

    # With multiple tickers yfinance returns a MultiIndex column frame.
    if isinstance(raw.columns, pd.MultiIndex):
        adj_close = raw["Adj Close"]
    else:
        # Single-ticker fallback (shouldn't normally happen with 50 tickers).
        adj_close = raw[["Adj Close"]]
        adj_close.columns = tickers[:1]

    valid_cols = [
        col for col in adj_close.columns
        if adj_close[col].notna().sum() >= config.MIN_OBS
    ]
    dropped = sorted(set(adj_close.columns) - set(valid_cols))
    if dropped:
        log.warning("Dropping %d tickers with insufficient history: %s", len(dropped), dropped)

    adj_close = adj_close[valid_cols].sort_index()
    adj_close = adj_close.ffill().dropna(how="any")

    adj_close.to_csv(config.PRICES_CSV)
    log.info(
        "Cached adjusted close prices: %d rows x %d tickers -> %s",
        adj_close.shape[0], adj_close.shape[1], config.PRICES_CSV,
    )
    return adj_close


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = download_adjusted_close(use_cache=False)
    print(df.tail())
