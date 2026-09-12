import pandas as pd

import config


def compute_spread(prices: pd.DataFrame, ticker_a: str, ticker_b: str, beta: float) -> pd.Series:
    spread = prices[ticker_a] - beta * prices[ticker_b]
    spread.name = "spread"
    return spread


def compute_zscore(spread: pd.Series, window: int = None) -> pd.Series:
    window = window or config.ZSCORE_WINDOW
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean) / rolling_std
    zscore.name = "zscore"
    return zscore


def generate_positions(
    zscore: pd.Series,
    entry_z: float = None,
    exit_z: float = None,
) -> pd.Series:
    """Turn a z-score series into a discrete spread position series.

    Position convention (position in the *spread*, i.e. long price_A / short
    beta*price_B when position == +1):
        +1  : long the spread   (entered when zscore <= -entry_z)
        -1  : short the spread  (entered when zscore >= +entry_z)
         0  : flat              (exited when zscore reverts to exit_z)

    A position, once opened, is held until the z-score reverts to `exit_z`
    (mean reversion exit) -- it does not flip directly from long to short.
    """
    entry_z = entry_z if entry_z is not None else config.ENTRY_Z
    exit_z = exit_z if exit_z is not None else config.EXIT_Z

    positions = pd.Series(0, index=zscore.index, dtype=int)
    current = 0

    for t, z in zscore.items():
        if pd.isna(z):
            positions[t] = current
            continue

        if current == 0:
            if z <= -entry_z:
                current = 1
            elif z >= entry_z:
                current = -1
        elif current == 1:
            if z >= exit_z:
                current = 0
        elif current == -1:
            if z <= exit_z:
                current = 0

        positions[t] = current

    positions.name = "position"
    return positions


def build_signal_frame(
    prices: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    beta: float,
) -> pd.DataFrame:
    spread = compute_spread(prices, ticker_a, ticker_b, beta)
    zscore = compute_zscore(spread)
    position = generate_positions(zscore)

    return pd.DataFrame({
        "price_a": prices[ticker_a],
        "price_b": prices[ticker_b],
        "spread": spread,
        "zscore": zscore,
        "position": position,
    })
