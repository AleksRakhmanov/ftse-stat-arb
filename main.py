import logging

import pandas as pd

import backtest
import config
import data_loader
import metrics
import pair_selection
import report
import signals

log = logging.getLogger(__name__)


def run_pipeline(use_cache: bool = True):
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    config.TRADE_LOGS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Data ----------------------------------------------------------------
    prices = data_loader.download_adjusted_close(use_cache=use_cache)
    log.info("Price matrix: %d days x %d tickers (%s -> %s)",
              prices.shape[0], prices.shape[1], prices.index[0].date(), prices.index[-1].date())

    # 2. Pair selection --------------------------------------------------------
    all_pair_stats = pair_selection.find_cointegrated_pairs(prices)
    cointegrated = all_pair_stats[all_pair_stats["cointegrated"]].copy()

    if cointegrated.empty:
        log.warning("No cointegrated pairs found at p<%.2f; nothing to backtest.", config.ADF_PVALUE_THRESHOLD)
        traded = cointegrated
    else:
        traded = cointegrated.head(config.MAX_PAIRS_TO_TRADE).copy()

    log.info("Backtesting %d pairs (out of %d cointegrated, %d tested)",
              len(traded), len(cointegrated), len(all_pair_stats))

    # 3. Backtest each traded pair -----------------------------------------------
    per_pair_metrics = []
    all_trade_logs = []
    all_round_trips = []
    daily_values = {}
    pair_chart_paths = {}

    for _, row in traded.iterrows():
        ticker_a, ticker_b, beta = row["ticker_a"], row["ticker_b"], row["beta"]
        pair_name = f"{ticker_a}_{ticker_b}"

        sig_frame = signals.build_signal_frame(prices, ticker_a, ticker_b, beta)
        result = backtest.run_backtest(sig_frame, ticker_a, ticker_b, beta)

        pair_metrics = metrics.summarise(result.daily["portfolio_value"], result.round_trips)
        pair_metrics.update({"ticker_a": ticker_a, "ticker_b": ticker_b, "beta": beta,
                              "adf_pvalue": row["adf_pvalue"]})
        per_pair_metrics.append(pair_metrics)

        if not result.trade_log.empty:
            all_trade_logs.append(result.trade_log)
        if not result.round_trips.empty:
            all_round_trips.append(result.round_trips)

        daily_values[pair_name] = result.daily["portfolio_value"]

        chart_path = config.CHARTS_DIR / f"{pair_name}.png"
        report.plot_pair_chart(sig_frame, result.daily, ticker_a, ticker_b, chart_path)
        pair_chart_paths[pair_name] = str(chart_path)

        log.info(
            "%s/%s beta=%.3f p=%.4f -> ann.ret=%.2f%% sharpe=%.2f win=%.1f%% mdd=%.2f%% trades=%d",
            ticker_a, ticker_b, beta, row["adf_pvalue"],
            pair_metrics["annualised_return"] * 100 if pd.notna(pair_metrics["annualised_return"]) else float("nan"),
            pair_metrics["sharpe_ratio"] if pd.notna(pair_metrics["sharpe_ratio"]) else float("nan"),
            pair_metrics["win_rate"] * 100 if pd.notna(pair_metrics["win_rate"]) else float("nan"),
            pair_metrics["max_drawdown"] * 100 if pd.notna(pair_metrics["max_drawdown"]) else float("nan"),
            pair_metrics["num_round_trips"],
        )

    # 4. Aggregate portfolio across all traded pairs ------------------------------
    if daily_values:
        aggregate_value = pd.concat(daily_values.values(), axis=1).sum(axis=1)
        aggregate_value.name = "portfolio_value"
    else:
        aggregate_value = pd.Series(dtype=float)

    combined_trade_log = pd.concat(all_trade_logs, ignore_index=True) if all_trade_logs else pd.DataFrame()
    combined_round_trips = pd.concat(all_round_trips, ignore_index=True) if all_round_trips else pd.DataFrame()

    aggregate_metrics = metrics.summarise(aggregate_value, combined_round_trips)
    aggregate_metrics.update({
        "num_pairs_tested": len(all_pair_stats),
        "num_pairs_cointegrated": len(cointegrated),
        "num_pairs_traded": len(traded),
    })

    log.info(
        "PORTFOLIO: ann.return=%.2f%% sharpe=%.2f win_rate=%.1f%% max_dd=%.2f%% total_pnl=%.2f",
        aggregate_metrics["annualised_return"] * 100 if pd.notna(aggregate_metrics["annualised_return"]) else float("nan"),
        aggregate_metrics["sharpe_ratio"] if pd.notna(aggregate_metrics["sharpe_ratio"]) else float("nan"),
        aggregate_metrics["win_rate"] * 100 if pd.notna(aggregate_metrics["win_rate"]) else float("nan"),
        aggregate_metrics["max_drawdown"] * 100 if pd.notna(aggregate_metrics["max_drawdown"]) else float("nan"),
        aggregate_metrics["total_pnl"],
    )

    # 5. Charts -----------------------------------------------------------------
    chart_paths = {"pair_charts": pair_chart_paths}
    if not aggregate_value.empty:
        chart_paths["portfolio_curve"] = report.plot_portfolio_curve(
            aggregate_value, config.CHARTS_DIR / "portfolio_curve.png")
        chart_paths["drawdown_curve"] = report.plot_drawdown(
            aggregate_value, config.CHARTS_DIR / "portfolio_drawdown.png")

    # 6. Pair stats table: full cointegration test results + traded-pair metrics ---
    perf_df = pd.DataFrame(per_pair_metrics)
    pair_stats_table = all_pair_stats.merge(
        perf_df.drop(columns=["adf_pvalue"], errors="ignore"),
        on=["ticker_a", "ticker_b"], how="left", suffixes=("", "_bt"),
    )
    if "beta_bt" in pair_stats_table.columns:
        pair_stats_table = pair_stats_table.drop(columns=["beta_bt"])
    pair_stats_table["traded"] = pair_stats_table["ticker_a"].isin(traded["ticker_a"]) & \
        pair_stats_table["ticker_b"].isin(traded["ticker_b"])

    # 7. Persist raw trade log / round trips as CSV too --------------------------
    if not combined_trade_log.empty:
        combined_trade_log.to_csv(config.TRADE_LOGS_DIR / "trade_log.csv", index=False)
    if not combined_round_trips.empty:
        combined_round_trips.to_csv(config.TRADE_LOGS_DIR / "round_trips.csv", index=False)

    # 8. Excel dashboard ----------------------------------------------------------
    report.build_excel_dashboard(
        summary_metrics=aggregate_metrics,
        pair_stats=pair_stats_table,
        trade_log=combined_trade_log,
        chart_paths=chart_paths,
        out_path=config.EXCEL_REPORT_PATH,
    )

    return {
        "all_pair_stats": all_pair_stats,
        "traded": traded,
        "aggregate_metrics": aggregate_metrics,
        "aggregate_value": aggregate_value,
        "combined_trade_log": combined_trade_log,
        "combined_round_trips": combined_round_trips,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_pipeline(use_cache=True)
