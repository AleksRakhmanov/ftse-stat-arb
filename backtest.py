from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config


@dataclass
class BacktestResult:
    ticker_a: str
    ticker_b: str
    beta: float
    daily: pd.DataFrame          # date-indexed: position, cash, holdings, portfolio_value
    trade_log: pd.DataFrame      # one row per executed rebalance
    round_trips: pd.DataFrame    # one row per completed open->close cycle


def _target_shares(position: int, capital: float, price_a: float, price_b: float, beta: float):
    if position == 0:
        return 0.0, 0.0
    if position == 1:  # long spread: long A, short beta * A-equivalent of B
        shares_a = capital / price_a
        shares_b = -beta * shares_a
    else:  # position == -1: short spread: short A, long B
        shares_a = -capital / price_a
        shares_b = -beta * shares_a
    return shares_a, shares_b


def run_backtest(
    signal_frame: pd.DataFrame,
    ticker_a: str,
    ticker_b: str,
    beta: float,
    capital: float = None,
    transaction_cost_rate: float = None,
) -> BacktestResult:
    capital = capital if capital is not None else config.CAPITAL_PER_PAIR
    tc_rate = transaction_cost_rate if transaction_cost_rate is not None else config.TRANSACTION_COST_RATE

    dates = signal_frame.index
    cash = capital
    shares_a = 0.0
    shares_b = 0.0
    prev_position = 0

    daily_records = []
    trade_records = []
    round_trips = []
    open_state = None  # dict tracking the currently open cycle

    for t in dates:
        row = signal_frame.loc[t]
        price_a, price_b = row["price_a"], row["price_b"]
        target_position = int(row["position"])

        if target_position != prev_position:
            target_shares_a, target_shares_b = _target_shares(target_position, capital, price_a, price_b, beta)
            trade_shares_a = target_shares_a - shares_a
            trade_shares_b = target_shares_b - shares_b
            notional_traded = abs(trade_shares_a) * price_a + abs(trade_shares_b) * price_b
            transaction_cost = notional_traded * tc_rate

            # Cash impact: buying (positive trade_shares) consumes cash, selling/shorting releases cash.
            cash -= trade_shares_a * price_a + trade_shares_b * price_b
            cash -= transaction_cost

            shares_a, shares_b = target_shares_a, target_shares_b

            trade_records.append({
                "date": t,
                "ticker_a": ticker_a,
                "ticker_b": ticker_b,
                "from_position": prev_position,
                "to_position": target_position,
                "trade_shares_a": trade_shares_a,
                "trade_shares_b": trade_shares_b,
                "price_a": price_a,
                "price_b": price_b,
                "notional_traded": notional_traded,
                "transaction_cost": transaction_cost,
            })

            portfolio_value_now = cash + shares_a * price_a + shares_b * price_b

            if prev_position == 0 and target_position != 0:
                # Opening a new round trip.
                open_state = {
                    "open_date": t,
                    "side": "long_spread" if target_position == 1 else "short_spread",
                    "open_equity": portfolio_value_now,
                }
            elif prev_position != 0 and target_position == 0 and open_state is not None:
                # Closing a round trip.
                pnl = portfolio_value_now - open_state["open_equity"]
                round_trips.append({
                    "ticker_a": ticker_a,
                    "ticker_b": ticker_b,
                    "side": open_state["side"],
                    "open_date": open_state["open_date"],
                    "close_date": t,
                    "pnl": pnl,
                    "return_pct": pnl / open_state["open_equity"] if open_state["open_equity"] else np.nan,
                })
                open_state = None

        prev_position = target_position
        portfolio_value = cash + shares_a * price_a + shares_b * price_b

        daily_records.append({
            "date": t,
            "position": target_position,
            "shares_a": shares_a,
            "shares_b": shares_b,
            "cash": cash,
            "portfolio_value": portfolio_value,
        })

    daily = pd.DataFrame(daily_records).set_index("date")
    trade_log = pd.DataFrame(trade_records)
    round_trip_df = pd.DataFrame(round_trips)

    return BacktestResult(
        ticker_a=ticker_a,
        ticker_b=ticker_b,
        beta=beta,
        daily=daily,
        trade_log=trade_log,
        round_trips=round_trip_df,
    )
