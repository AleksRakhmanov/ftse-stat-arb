import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


COLORS = {"is": "#4C9AFF", "oos": "#F5A623", "a": "#4C9AFF", "b": "#B98CFF",
          "long": "#2ECC71", "short": "#FF5C5C", "exit": "#E0E0E0", "band": "#888888"}
SPLIT_COLOR = "rgba(255,255,255,0.45)"


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))
    return fig


def _mark_split(fig: go.Figure, split_date: pd.Timestamp, rows: int) -> None:
    for r in range(1, rows + 1):
        fig.add_vline(x=split_date, line=dict(color=SPLIT_COLOR, dash="dash", width=1), row=r, col=1)
    # add_vline's own annotation breaks on datetime axes, so annotate separately.
    fig.add_annotation(x=split_date, y=1, yref="paper", text="IS | OOS", showarrow=False,
                       yanchor="bottom", font=dict(size=11, color=SPLIT_COLOR))


def equity_and_drawdown(periods: dict, split_date: pd.Timestamp) -> go.Figure:
    """Cumulative return and drawdown per period. Each period restarts from fresh capital,
    so returns are shown relative to the period's own starting value."""
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.05,
                        subplot_titles=("Cumulative return (%)", "Drawdown (%)"))
    for p, label in (("is", "In-sample"), ("oos", "Out-of-sample")):
        equity = periods[p]["equity"]
        if equity.empty:
            continue
        cum = (equity / equity.iloc[0] - 1) * 100
        dd = (equity / equity.cummax() - 1) * 100
        fig.add_trace(go.Scatter(x=cum.index, y=cum, name=label, line=dict(color=COLORS[p], width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=f"{label} drawdown", showlegend=False, fill="tozeroy",
                                 line=dict(color=COLORS[p], width=1)), row=2, col=1)
    _mark_split(fig, split_date, rows=2)
    return _layout(fig, 520)


def pair_drilldown(bt: dict, ticker_a: str, ticker_b: str, entry_z: float, exit_z: float,
                   split_date: pd.Timestamp) -> go.Figure:
    frame = bt["signals"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.36, 0.28, 0.36],
                        specs=[[{"secondary_y": True}], [{}], [{}]],
                        subplot_titles=("Prices", "Spread (A − β·B)", "Z-score"))

    fig.add_trace(go.Scatter(x=frame.index, y=frame["price_a"], name=ticker_a, line=dict(color=COLORS["a"])),
                  row=1, col=1, secondary_y=False)
    fig.add_trace(go.Scatter(x=frame.index, y=frame["price_b"], name=ticker_b, line=dict(color=COLORS["b"])),
                  row=1, col=1, secondary_y=True)
    fig.add_trace(go.Scatter(x=frame.index, y=frame["spread"], name="Spread", showlegend=False,
                             line=dict(color="#CCCCCC", width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=frame.index, y=frame["zscore"], name="Z-score", showlegend=False,
                             line=dict(color="#CCCCCC", width=1.2)), row=3, col=1)

    for level, dash in ((entry_z, "dash"), (-entry_z, "dash"), (exit_z, "dot")):
        fig.add_hline(y=level, line=dict(color=COLORS["band"], dash=dash, width=1), row=3, col=1)
    if exit_z != 0:
        fig.add_hline(y=0, line=dict(color=COLORS["band"], width=0.5), row=3, col=1)

    trades = [res.trade_log for res in bt["results"].values() if not res.trade_log.empty]
    if trades:
        log = pd.concat(trades)
        z_at = frame["zscore"].reindex(pd.to_datetime(log["date"])).to_numpy()
        kinds = (("Long spread", log["to_position"] == 1, "triangle-up", COLORS["long"]),
                 ("Short spread", log["to_position"] == -1, "triangle-down", COLORS["short"]),
                 ("Exit", log["to_position"] == 0, "x", COLORS["exit"]))
        for name, mask, symbol, color in kinds:
            mask = mask.to_numpy()
            if mask.any():
                fig.add_trace(go.Scatter(x=log["date"][mask], y=z_at[mask], mode="markers", name=name,
                                         marker=dict(symbol=symbol, size=10, color=color)), row=3, col=1)

    fig.update_yaxes(title_text=ticker_a, row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text=ticker_b, row=1, col=1, secondary_y=True, showgrid=False)
    _mark_split(fig, split_date, rows=3)
    return _layout(fig, 720)
