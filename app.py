import math

import pandas as pd
import streamlit as st

import config
from ui import charts, engine
from ui.data import default_split_date, load_prices

st.set_page_config(page_title="FTSE 100 Pairs Trading", page_icon="📈", layout="wide")

MIN_IS_DAYS = 120
MIN_OOS_DAYS = 40
TRADED_ROW_BG = "background-color: rgba(76, 154, 255, 0.16)"

DEFAULTS = {
    "live": False,
    "split_date": None,  # resolved against the loaded data
    "entry_z": config.ENTRY_Z,
    "exit_z": config.EXIT_Z,
    "window": config.ZSCORE_WINDOW,
    "adf_cutoff": config.ADF_PVALUE_THRESHOLD,
    "tc_bps": round(config.TRANSACTION_COST_RATE * 10_000),
    "max_pairs": config.MAX_PAIRS_TO_TRADE or 50,
}


def fmt_pct(x, signed=True):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x * 100:{'+' if signed else ''}.2f}%"


def fmt_num(x, spec=".2f"):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def fmt_gbp(x):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"£{x:+,.0f}"


def kpi_block(title: str, m: dict) -> None:
    with st.container(border=True):
        st.markdown(f"**{title}**")
        r1 = st.columns(3)
        r1[0].metric("Annualised return", fmt_pct(m["annualised_return"]))
        r1[1].metric("Sharpe ratio", fmt_num(m["sharpe_ratio"]))
        r1[2].metric("Win rate", fmt_pct(m["win_rate"], signed=False))
        r2 = st.columns(3)
        r2[0].metric("Max drawdown", fmt_pct(m["max_drawdown"]))
        r2[1].metric("Total P&L", fmt_gbp(m["total_pnl"]))
        r2[2].metric("Round trips", m["num_round_trips"])


def clamp_split(prices: pd.DataFrame, split) -> pd.Timestamp:
    lo, hi = prices.index[MIN_IS_DAYS], prices.index[-MIN_OOS_DAYS]
    if split is None or not lo <= pd.Timestamp(split) <= hi:
        return default_split_date(prices)
    return pd.Timestamp(split)


# --- Data + parameters -------------------------------------------------------
params = st.session_state.setdefault("params", dict(DEFAULTS))
prices, data_key, data_warning = load_prices(params["live"])
params["split_date"] = clamp_split(prices, params["split_date"])

with st.sidebar:
    st.header("Strategy settings")
    with st.form("settings"):
        live = st.toggle("Refresh live data (yfinance)", value=params["live"],
                         help="Off uses the bundled price snapshot. Falls back to it if the download fails.")
        split = st.date_input("In-sample / out-of-sample split", value=params["split_date"].date(),
                              min_value=prices.index[MIN_IS_DAYS].date(), max_value=prices.index[-MIN_OOS_DAYS].date(),
                              help="Pairs and hedge ratios are fitted only on data before this date.")
        entry_z = st.slider("Entry |z|", 1.0, 3.5, float(params["entry_z"]), 0.1)
        exit_z = st.slider("Exit z", 0.0, 1.5, float(params["exit_z"]), 0.1,
                           help="Positions close once the z-score reverts to this level.")
        window = st.slider("Rolling z-score window (days)", 10, 120, int(params["window"]), 5)
        adf_cutoff = st.select_slider("ADF p-value cutoff", [0.01, 0.025, 0.05, 0.10], value=params["adf_cutoff"])
        tc_bps = st.number_input("Transaction cost (bps per rebalance)", 0, 100, int(params["tc_bps"]), 1)
        max_pairs = st.slider("Max pairs traded", 1, 50, int(params["max_pairs"]))
        submitted = st.form_submit_button("Run backtest", type="primary", width="stretch")

    if submitted:
        if exit_z >= entry_z:
            st.error("Exit z must be below entry |z|.")
        else:
            st.session_state["params"] = dict(live=live, split_date=pd.Timestamp(split), entry_z=entry_z,
                                              exit_z=exit_z, window=window, adf_cutoff=adf_cutoff,
                                              tc_bps=tc_bps, max_pairs=max_pairs)
            st.rerun()

    st.caption(f"Capital per pair: £{config.CAPITAL_PER_PAIR:,.0f}  ·  Sizing: dollar-neutral per leg")

split_date = params["split_date"]
tc_rate = params["tc_bps"] / 10_000
strategy_args = (prices, data_key, split_date, params["adf_cutoff"], params["max_pairs"],
                 params["window"], params["entry_z"], params["exit_z"], tc_rate)

with st.spinner("Running backtest…"):
    result = engine.run_strategy(*strategy_args)
scan, periods = result["scan"], result["periods"]
scan["pair"] = scan["ticker_a"] + " / " + scan["ticker_b"]

# --- Header -------------------------------------------------------------------
st.title("FTSE 100 Statistical Arbitrage")
st.caption(
    f"Cointegration pairs trading on {prices.shape[1]} FTSE 100 stocks · "
    f"{prices.index[0]:%d %b %Y} → {prices.index[-1]:%d %b %Y} · "
    f"data: {'live yfinance' if data_key != 'snapshot' else 'bundled snapshot'}"
)
if data_warning:
    st.warning(data_warning)

n_coint = int(scan["cointegrated"].sum())
st.markdown(
    f"**{len(scan):,}** pairs tested in-sample · **{n_coint}** pass ADF p < {params['adf_cutoff']} · "
    f"**{result['num_traded']}** traded · split **{split_date:%d %b %Y}**"
)
if result["num_traded"] == 0:
    st.warning("No pairs pass the ADF cutoff on the in-sample period. Try a looser cutoff or a later split date.")

# --- 1. KPIs ------------------------------------------------------------------
is_range = f"{prices.index[0]:%b %Y} – {split_date:%b %Y}"
oos_range = f"{split_date:%b %Y} – {prices.index[-1]:%b %Y}"
k1, k2 = st.columns(2)
with k1:
    kpi_block(f"In-sample · {is_range}", periods["is"]["metrics"])
with k2:
    kpi_block(f"Out-of-sample · {oos_range}", periods["oos"]["metrics"])

# --- 2. Portfolio equity + drawdown --------------------------------------------
st.subheader("Portfolio")
if result["num_traded"]:
    st.plotly_chart(charts.equity_and_drawdown(periods, split_date), width="stretch")
    st.caption("Each period starts flat with fresh capital, so returns are measured from the period's own start.")
else:
    st.info("Nothing to plot: no pairs were traded.")

# --- 3. Pair table ------------------------------------------------------------
st.subheader("Pairs")
f1, f2 = st.columns([2, 3])
ticker_filter = f1.text_input("Filter by ticker", placeholder="e.g. RIO").strip().upper()
show = f2.radio("Show", ["All", "Cointegrated", "Traded"], horizontal=True)

view = scan
if ticker_filter:
    view = view[view["pair"].str.contains(ticker_filter, regex=False)]
if show == "Cointegrated":
    view = view[view["cointegrated"]]
elif show == "Traded":
    view = view[view["traded"]]

table_cols = ["pair", "adf_pvalue", "beta", "half_life", "cointegrated", "traded",
              "is_sharpe", "is_pnl", "oos_sharpe", "oos_pnl", "oos_trades"]
view = view.reindex(columns=table_cols).reset_index(drop=True)
styled = view.style.apply(lambda r: [TRADED_ROW_BG if r["traded"] else ""] * len(r), axis=1)
event = st.dataframe(
    styled, hide_index=True, width="stretch", height=360, key="pair_table",
    on_select="rerun", selection_mode="single-row",
    column_config={
        "pair": "Pair",
        "adf_pvalue": st.column_config.NumberColumn("ADF p-value", format="%.4f"),
        "beta": st.column_config.NumberColumn("Hedge ratio β", format="%.3f"),
        "half_life": st.column_config.NumberColumn("Half-life (d)", format="%.1f",
                                                   help="Mean-reversion half-life of the in-sample spread"),
        "cointegrated": "Coint.",
        "traded": "Traded",
        "is_sharpe": st.column_config.NumberColumn("IS Sharpe", format="%.2f"),
        "is_pnl": st.column_config.NumberColumn("IS P&L (£)", format="%.0f"),
        "oos_sharpe": st.column_config.NumberColumn("OOS Sharpe", format="%.2f"),
        "oos_pnl": st.column_config.NumberColumn("OOS P&L (£)", format="%.0f"),
        "oos_trades": st.column_config.NumberColumn("OOS trades", format="%d"),
    },
)
st.caption(f"{len(view):,} rows · highlighted rows are traded · click a row to inspect it below")

# --- 4. Pair drill-down -------------------------------------------------------
st.subheader("Pair drill-down")
pair_options = scan["pair"].tolist()
traded_pairs = set(scan.loc[scan["traded"], "pair"])

# A table click overrides the selectbox only when the click is new, so the
# selectbox stays usable afterwards.
rows = event.selection.rows
clicked = view.iloc[rows[0]]["pair"] if rows else None
if clicked and clicked != st.session_state.get("last_clicked"):
    st.session_state["last_clicked"] = clicked
    st.session_state["pair_select"] = clicked
if st.session_state.get("pair_select") not in pair_options:
    st.session_state["pair_select"] = next(iter(scan.loc[scan["traded"], "pair"]), pair_options[0])

selected = st.selectbox("Pair", pair_options, key="pair_select",
                        format_func=lambda p: f"{p}  ★ traded" if p in traded_pairs else p)
pair_row = scan.loc[scan["pair"] == selected].iloc[0]
bt = engine.backtest_pair(prices, data_key, split_date, pair_row["ticker_a"], pair_row["ticker_b"], pair_row["beta"],
                          params["window"], params["entry_z"], params["exit_z"], tc_rate)
pair_m = engine.pair_metrics(bt)

st.caption(
    f"β = {pair_row['beta']:.3f} · ADF p = {pair_row['adf_pvalue']:.4f} · "
    f"half-life = {fmt_num(pair_row['half_life'], '.1f')} days (in-sample fit)"
)
if selected not in traded_pairs:
    st.info("This pair isn't in the traded set at the current settings; shown for illustration.")

pm_cols = st.columns(len(pair_m) * 2)
for i, (p, m) in enumerate(pair_m.items()):
    label = engine.PERIOD_LABELS[p]
    pm_cols[2 * i].metric(f"{label} P&L", fmt_gbp(m["total_pnl"]), f"{m['num_round_trips']} round trips",
                          delta_color="off")
    pm_cols[2 * i + 1].metric(f"{label} Sharpe", fmt_num(m["sharpe_ratio"]))

if all(res.trade_log.empty for res in bt["results"].values()):
    st.info("No trades: the z-score never crossed the entry threshold for this pair.")
st.plotly_chart(charts.pair_drilldown(bt, pair_row["ticker_a"], pair_row["ticker_b"], params["entry_z"],
                                      params["exit_z"], split_date), width="stretch")

# --- 5. Trade log -------------------------------------------------------------
st.subheader("Trade log")
scope = st.radio("Scope", ["Selected pair", "All traded pairs"], horizontal=True, label_visibility="collapsed")
if scope == "Selected pair":
    round_trips = engine.concat_frames(
        [r.round_trips.assign(period=engine.PERIOD_LABELS[p]) for p, r in bt["results"].items()],
        engine.ROUND_TRIP_COLS + ["period"])
    executions = engine.concat_frames(
        [r.trade_log.assign(period=engine.PERIOD_LABELS[p]) for p, r in bt["results"].items()],
        engine.TRADE_LOG_COLS + ["period"])
    file_stub = selected.replace(" / ", "_")
else:
    round_trips = engine.concat_frames([periods[p]["round_trips"] for p in engine.PERIODS],
                                       engine.ROUND_TRIP_COLS + ["period"])
    executions = engine.concat_frames([periods[p]["trade_log"] for p in engine.PERIODS],
                                      engine.TRADE_LOG_COLS + ["period"])
    file_stub = "all_pairs"

tab_rt, tab_ex = st.tabs([f"Round trips ({len(round_trips)})", f"Executions ({len(executions)})"])
for tab, df, name in ((tab_rt, round_trips, "round_trips"), (tab_ex, executions, "executions")):
    with tab:
        if df.empty:
            st.info("No trades in this selection.")
            continue
        st.dataframe(df, hide_index=True, width="stretch", height=300)
        st.download_button(f"Download {name.replace('_', ' ')} CSV", df.to_csv(index=False).encode(),
                           file_name=f"{file_stub}_{name}.csv", mime="text/csv", key=f"dl_{name}")

# --- 6. Methodology -----------------------------------------------------------
with st.expander("Methodology"):
    st.markdown(f"""
**Pair selection (cointegration).** For every pair of stocks, the price of A is regressed on the price of B
using only in-sample data. The slope is the hedge ratio β, and the leftover spread `A − β·B` is tested with an
Augmented Dickey-Fuller test. A low p-value suggests the spread is stationary: it tends to drift back to its
average instead of wandering off. Pairs below the cutoff are ranked by p-value and the strongest are traded.

**Signals.** The spread is converted to a z-score using a rolling {params['window']}-day mean and standard
deviation. When it stretches beyond ±{params['entry_z']:.1f}, the strategy bets on a snap-back: short the spread
when it is high, long when it is low. The position is closed once the z-score reverts to {params['exit_z']:.1f}.

**In-sample vs out-of-sample.** Pairs and hedge ratios are chosen using data before the split date only, then
traded unchanged after it. The out-of-sample result is the honest test; the in-sample result is flattered
because the pairs were picked for looking good over that same period.

**Costs and sizing.** Each pair gets £{config.CAPITAL_PER_PAIR:,.0f}. Leg A is fully invested and leg B is sized by
β. Every rebalance pays {params['tc_bps']} bps of the traded notional.

**Limitations.**
- *Survivorship bias:* the universe is today's FTSE 100 names, which excludes companies that dropped out.
- *No slippage or market impact:* fills happen at the daily adjusted close.
- *No borrow costs* on the short leg, and no financing on cash.
- *Multiple testing:* with ~1,000 pairs tested at 5%, dozens pass by chance alone.
- P&L, win rate and trade counts include completed round trips only; positions still open at the end of a
  period are marked to market in the equity curve but not counted as trades.
""")
