import itertools
import logging
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

import config

# statsmodels currently emits a FutureWarning about the eventual default
# return type of adfuller(); we pin the legacy tuple shape explicitly below
# via result_object=False, so the warning is a false positive here.
warnings.filterwarnings("ignore", category=FutureWarning, module="statsmodels")

log = logging.getLogger(__name__)


def _ols_hedge_ratio(y: pd.Series, x: pd.Series):
    x_ = sm.add_constant(x.values)
    model = sm.OLS(y.values, x_).fit()
    alpha, beta = model.params
    residuals = y.values - (alpha + beta * x.values)
    return beta, alpha, pd.Series(residuals, index=y.index)


def test_pair(prices: pd.DataFrame, ticker_a: str, ticker_b: str) -> dict:
    """Run OLS + ADF for a single pair and return a result record."""
    y = prices[ticker_a]
    x = prices[ticker_b]

    beta, alpha, spread = _ols_hedge_ratio(y, x)
    adf_stat, adf_pvalue, *_ = adfuller(spread, autolag="AIC", result_object=False)

    return {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "beta": beta,
        "alpha": alpha,
        "adf_stat": adf_stat,
        "adf_pvalue": adf_pvalue,
        "spread_mean": spread.mean(),
        "spread_std": spread.std(),
    }


def find_cointegrated_pairs(prices: pd.DataFrame) -> pd.DataFrame:
    """Test all C(n, 2) combinations and return a DataFrame of results.

    The returned frame includes every pair tested (for the stats table) with
    a boolean `cointegrated` flag for those passing the ADF p-value cutoff.
    """
    tickers = list(prices.columns)
    combos = list(itertools.combinations(tickers, 2))
    log.info("Testing %d ticker pairs for cointegration...", len(combos))

    records = []
    for i, (a, b) in enumerate(combos, start=1):
        try:
            records.append(test_pair(prices, a, b))
        except Exception as exc:  # noqa: BLE001 - keep scanning on isolated failures
            log.debug("Pair %s/%s failed: %s", a, b, exc)
        if i % 200 == 0:
            log.info("  ...%d/%d pairs tested", i, len(combos))

    results = pd.DataFrame.from_records(records)
    results["cointegrated"] = results["adf_pvalue"] < config.ADF_PVALUE_THRESHOLD
    results = results.sort_values("adf_pvalue").reset_index(drop=True)

    n_coint = int(results["cointegrated"].sum())
    log.info(
        "Found %d cointegrated pairs (ADF p-value < %.2f) out of %d tested",
        n_coint, config.ADF_PVALUE_THRESHOLD, len(results),
    )
    return results


if __name__ == "__main__":
    import data_loader

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    prices = data_loader.download_adjusted_close()
    stats = find_cointegrated_pairs(prices)
    print(stats[stats["cointegrated"]].head(20))
