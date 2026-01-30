"""
Bloomberg-style Seasonality Heatmap Dashboard (Plotly interactive)
with additional quant analysis panels.
Run:  streamlit run app.py
"""

import re, datetime as dt, numpy as np, pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Bloomberg-dark Plotly layout defaults reused across all figures.
BBG_LAYOUT = dict(
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="grey", size=11),
)

# ─────────────────────────────────────────────────────────────
# Core data helpers (unchanged math)
# ─────────────────────────────────────────────────────────────

def parse_ticker(raw: str) -> str:
    t = (raw or "").strip().upper()
    toks = re.split(r"[,\s;]+", t)
    return toks[0] if toks else ""


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_prices(ticker: str, start: str, end: str) -> pd.Series:
    """Daily close series via yfinance (auto_adjust=True)."""
    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    except Exception:
        return pd.Series(dtype=float)
    if df.empty:
        return pd.Series(dtype=float)
    if isinstance(df.columns, pd.MultiIndex):
        if "Close" not in df.columns.get_level_values(0):
            return pd.Series(dtype=float)
        px = df["Close"].iloc[:, 0] if isinstance(df["Close"], pd.DataFrame) else df["Close"]
    else:
        if "Close" not in df.columns:
            return pd.Series(dtype=float)
        px = df["Close"]
    try:
        px.index = px.index.tz_localize(None)
    except Exception:
        pass
    return px.dropna().rename(ticker)


def compute_monthly_series(
    ticker: str, metric: str, start: dt.date, end: dt.date
) -> pd.DataFrame:
    """Return DataFrame[year, month, value].

    For pct_return *value* is in PERCENT (×100) to match heatmap display.
    For delta_points / level *value* is in raw points.
    We fetch 2 extra months before *start* for pct_return and delta_points
    so the first month in the user range has a valid prior-month denominator.
    """
    needs_prior = metric in ("delta_points", "pct_return")
    if needs_prior:
        first_of_month = start.replace(day=1)
        fetch_start = (pd.Timestamp(first_of_month) - pd.DateOffset(months=2)).date()
    else:
        fetch_start = start

    px = fetch_prices(ticker, str(fetch_start), str(end))
    if px.empty:
        return pd.DataFrame(columns=["year", "month", "value"])

    monthly = px.resample("ME").last().dropna()
    if monthly.empty:
        return pd.DataFrame(columns=["year", "month", "value"])

    if metric == "level":
        vals = monthly.values.astype(float)
    elif metric == "delta_points":
        vals = monthly.diff().values.astype(float)
    else:  # pct_return — stored as percent for the heatmap
        vals = monthly.pct_change().values.astype(float) * 100.0

    out = pd.DataFrame({"year": monthly.index.year,
                         "month": monthly.index.month,
                         "value": vals})
    # Trim to user range
    out = out[(out["year"] > start.year) |
              ((out["year"] == start.year) & (out["month"] >= start.month))]
    out = out[(out["year"] < end.year) |
              ((out["year"] == end.year) & (out["month"] <= end.month))]
    return out.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# Seasonality matrix (heatmap)
# ─────────────────────────────────────────────────────────────

def build_seasonality_matrix(
    metric_df: pd.DataFrame,
    start_year: int, end_year: int,
    trailing_windows: list[int],
    full_years_trailing: bool, full_years_avg: bool,
):
    df = metric_df[(metric_df["year"] >= start_year) &
                   (metric_df["year"] <= end_year)].copy()
    pivot = df.pivot_table(index="year", columns="month",
                           values="value", aggfunc="last")
    pivot = pivot.reindex(columns=range(1, 13))
    all_years = sorted(pivot.index)
    full_year_set = set(pivot.dropna(thresh=12).index)

    rows: list[np.ndarray] = []
    labels: list[str] = []

    for n in trailing_windows:
        candidates = [y for y in all_years if y <= end_year]
        if full_years_trailing:
            candidates = [y for y in candidates if y in full_year_set]
        trail_years = candidates[-n:] if len(candidates) >= n else candidates
        if trail_years:
            row = pivot.loc[pivot.index.isin(trail_years)].mean(axis=0).values
            labels.append(f"{n} Yr Trailing Avg({trail_years[-1]})")
        else:
            row = np.full(12, np.nan)
            labels.append(f"{n} Yr Trailing Avg(N/A)")
        rows.append(row)

    avg_pool = pivot.loc[pivot.index.isin(full_year_set)] if full_years_avg else pivot
    rows.append(avg_pool.mean(axis=0).values)
    labels.append("Avg Month")

    for y in sorted(all_years, reverse=True):
        rows.append(pivot.loc[y].values)
        labels.append(str(y))

    return np.array(rows, dtype=float), labels


# ─────────────────────────────────────────────────────────────
# Heatmap renderer
# ─────────────────────────────────────────────────────────────

def render_heatmap_plotly(
    matrix: np.ndarray, row_labels: list[str],
    metric: str, ticker: str,
) -> go.Figure:
    """Interactive Bloomberg-style heatmap."""
    plot_matrix = matrix[::-1, :]
    plot_labels = row_labels[::-1]
    n_rows, n_cols = plot_matrix.shape
    vals = plot_matrix[np.isfinite(plot_matrix)]
    if len(vals) == 0:
        vals = np.array([0.0])

    pct = 80
    if metric == "level":
        med = float(np.median(vals))
        vmin_p = float(np.percentile(vals, 100 - pct))
        vmax_p = float(np.percentile(vals, pct))
        colorscale = [[0, "rgb(30,140,50)"], [.5, "rgb(25,25,25)"], [1, "rgb(190,30,30)"]]
        zmid, zmin, zmax = med, vmin_p, vmax_p
    elif metric == "delta_points":
        abs_v = np.abs(vals)
        cap = max(float(np.percentile(abs_v, pct)), 1e-9)
        colorscale = [[0, "rgb(30,140,50)"], [.5, "rgb(25,25,25)"], [1, "rgb(190,30,30)"]]
        zmid, zmin, zmax = 0.0, -cap, cap
    else:
        abs_v = np.abs(vals)
        cap = max(float(np.percentile(abs_v, pct)), 1e-9)
        colorscale = [[0, "rgb(190,30,30)"], [.5, "rgb(25,25,25)"], [1, "rgb(30,140,50)"]]
        zmid, zmin, zmax = 0.0, -cap, cap

    unit = "%" if metric == "pct_return" else "pts"
    text_matrix, hover_text = [], []
    for i in range(n_rows):
        tr, hr = [], []
        for j in range(n_cols):
            v = plot_matrix[i, j]
            tr.append(f"{v:.2f}" if np.isfinite(v) else "")
            vs = f"{v:.2f} {unit}" if np.isfinite(v) else "N/A"
            hr.append(f"<b>{plot_labels[i]}</b><br>{MONTHS[j]}<br>{vs}")
        text_matrix.append(tr)
        hover_text.append(hr)

    fig_h = max(400, n_rows * 28 + 100)
    fig = go.Figure(go.Heatmap(
        z=plot_matrix, x=MONTHS, y=plot_labels,
        zmin=zmin, zmax=zmax, zmid=zmid,
        colorscale=colorscale, showscale=False, xgap=2, ygap=2,
        text=text_matrix, texttemplate="%{text}",
        textfont=dict(color="white", size=11),
        hovertext=hover_text, hoverinfo="text",
        hoverlabel=dict(bgcolor="rgb(40,40,40)", font_size=12, font_color="white"),
    ))
    title_map = {"pct_return": "Monthly % Return",
                 "delta_points": "Δ Points (month-end)",
                 "level": "Level (month-end)"}
    fig.update_layout(
        **BBG_LAYOUT,
        title=dict(text=f"{ticker}  —  {title_map.get(metric, metric)}",
                   font=dict(color="white", size=15), x=0.5),
        height=fig_h,
        margin=dict(l=170, r=10, t=50, b=10),
        xaxis=dict(side="top", tickfont=dict(color="grey", size=11),
                   showgrid=False, fixedrange=True),
        yaxis=dict(tickfont=dict(color="grey", size=10), showgrid=False,
                   type="category", categoryorder="array",
                   categoryarray=plot_labels),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# TAB 1: Stability metrics
# ─────────────────────────────────────────────────────────────

def stability_metrics_table(
    mdf: pd.DataFrame, metric: str, downside_thr: float,
) -> pd.DataFrame:
    records = []
    for m in range(1, 13):
        s = mdf.loc[mdf["month"] == m, "value"].dropna()
        n = len(s)
        if n == 0:
            records.append(dict(Month=MONTHS[m - 1], Mean=np.nan, Median=np.nan,
                                HitRate=np.nan, DownsideFreq=np.nan,
                                IQR=np.nan, N=0))
            continue
        q1, q3 = np.percentile(s, 25), np.percentile(s, 75)
        records.append(dict(
            Month=MONTHS[m - 1],
            Mean=round(float(s.mean()), 4),
            Median=round(float(s.median()), 4),
            HitRate=round(float((s > 0).sum() / n) * 100, 1),
            DownsideFreq=round(float((s < -abs(downside_thr)).sum() / n) * 100, 1),
            IQR=round(float(q3 - q1), 4),
            N=n,
        ))
    return pd.DataFrame(records)


def render_stability_bar(
    tbl: pd.DataFrame, y_col: str, metric: str, ticker: str,
) -> go.Figure:
    unit = "%" if metric == "pct_return" else "pts"
    if y_col in ("HitRate", "DownsideFreq"):
        unit = "% obs"
    fig = go.Figure(go.Bar(
        x=tbl["Month"], y=tbl[y_col],
        marker_color=["rgb(30,140,50)" if v >= 0 else "rgb(190,30,30)"
                       for v in tbl[y_col].fillna(0)],
        hovertemplate="%{x}: %{y:.2f} " + unit + "<extra></extra>",
    ))
    fig.update_layout(
        **BBG_LAYOUT,
        title=dict(text=f"{ticker} — {y_col} by Month",
                   font=dict(color="white", size=14), x=0.5),
        height=370,
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(showgrid=False, tickfont=dict(color="grey")),
        yaxis=dict(showgrid=True, gridcolor="rgb(40,40,40)",
                   tickfont=dict(color="grey"),
                   title=dict(text=unit, font=dict(color="grey"))),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# TAB 2: Rolling seasonality regime
# ─────────────────────────────────────────────────────────────

def rolling_month_mean_series(
    mdf: pd.DataFrame, target_month: int, window_years: int,
    min_pct: float = 0.7,
) -> pd.DataFrame:
    sub = mdf.loc[mdf["month"] == target_month, ["year", "value"]].dropna()
    sub = sub.sort_values("year").set_index("year")["value"]
    min_obs = max(1, int(np.ceil(min_pct * window_years)))
    records = []
    for y in sub.index:
        window = sub.loc[(sub.index >= y - window_years + 1) & (sub.index <= y)]
        if len(window) >= min_obs:
            records.append(dict(year=y, rolling_mean=float(window.mean())))
    return pd.DataFrame(records)


def render_rolling_line(
    rdf: pd.DataFrame, month_name: str, window: int,
    metric: str, ticker: str,
) -> go.Figure:
    unit = "%" if metric == "pct_return" else "pts"
    fig = go.Figure(go.Scatter(
        x=rdf["year"], y=rdf["rolling_mean"],
        mode="lines+markers",
        line=dict(color="rgb(100,180,255)", width=2),
        marker=dict(size=5, color="rgb(100,180,255)"),
        hovertemplate="Year %{x}: %{y:.2f} " + unit + "<extra></extra>",
    ))
    if metric == "level":
        ref = float(rdf["rolling_mean"].median())
        fig.add_hline(y=ref, line_dash="dash", line_color="grey",
                      annotation_text=f"median {ref:.1f}",
                      annotation_font_color="grey")
    else:
        fig.add_hline(y=0, line_dash="dash", line_color="grey")

    fig.update_layout(
        **BBG_LAYOUT,
        title=dict(text=f"{ticker} — {month_name} Rolling {window}Y Mean",
                   font=dict(color="white", size=14), x=0.5),
        height=400,
        margin=dict(l=60, r=20, t=50, b=40),
        xaxis=dict(showgrid=False, tickfont=dict(color="grey"),
                   title=dict(text="End Year", font=dict(color="grey"))),
        yaxis=dict(showgrid=True, gridcolor="rgb(40,40,40)",
                   tickfont=dict(color="grey"),
                   title=dict(text=unit, font=dict(color="grey"))),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# TAB 3: Distribution boxplots
# ─────────────────────────────────────────────────────────────

def render_boxplots(
    mdf: pd.DataFrame, metric: str, ticker: str, show_points: bool,
) -> go.Figure:
    unit = "%" if metric == "pct_return" else "pts"
    fig = go.Figure()
    for m in range(1, 13):
        s = mdf.loc[mdf["month"] == m, "value"].dropna()
        fig.add_trace(go.Box(
            y=s, name=MONTHS[m - 1],
            boxpoints="all" if show_points else "outliers",
            jitter=0.4 if show_points else 0,
            pointpos=0,
            marker=dict(color="rgb(100,180,255)", size=3, opacity=0.5),
            line=dict(color="rgb(100,180,255)"),
            fillcolor="rgba(100,180,255,0.15)",
            hoverinfo="y+name",
        ))
    if metric != "level":
        fig.add_hline(y=0, line_dash="dash", line_color="grey")

    fig.update_layout(
        **BBG_LAYOUT,
        title=dict(text=f"{ticker} — Monthly Distribution ({unit})",
                   font=dict(color="white", size=14), x=0.5),
        height=450,
        margin=dict(l=60, r=20, t=50, b=40),
        showlegend=False,
        xaxis=dict(showgrid=False, tickfont=dict(color="grey")),
        yaxis=dict(showgrid=True, gridcolor="rgb(40,40,40)",
                   tickfont=dict(color="grey"),
                   title=dict(text=unit, font=dict(color="grey"))),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Seasonality Terminal", layout="wide")

# ---- Sidebar ----
with st.sidebar:
    st.header("Seasonality Terminal")

    ticker_raw = st.text_input("Ticker", value="SPY",
                               placeholder="e.g. SPY, ^VIX, QQQ")
    active_ticker = parse_ticker(ticker_raw)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start", dt.date(2010, 1, 1))
    with col2:
        end_date = st.date_input("End", dt.date.today())

    trail_input = st.text_input("Trailing windows (comma-sep)", "10")
    trailing_windows: list[int] = []
    for tok in trail_input.replace(" ", "").split(","):
        try:
            trailing_windows.append(int(tok))
        except ValueError:
            pass
    if not trailing_windows:
        trailing_windows = [10]
    # 10 always present and first
    if 10 in trailing_windows:
        trailing_windows.remove(10)
    trailing_windows = [10] + trailing_windows

    is_vix = active_ticker == "^VIX"
    if is_vix:
        vix_metric = st.selectbox(
            "Metric",
            ["Δ points (month-end)", "Level (month-end)"], index=0)
        metric = "delta_points" if "Δ" in vix_metric else "level"
    else:
        metric = "pct_return"

    st.markdown("---")
    full_years_trail = st.checkbox("Trailing avgs use full years only", value=True)
    full_years_avg = st.checkbox("Avg Month uses full years only", value=False)

# ---- Guard ----
if not active_ticker:
    st.info("Enter a ticker in the sidebar to begin.")
    st.stop()

with st.spinner(f"Fetching {active_ticker}…"):
    mdf = compute_monthly_series(active_ticker, metric, start_date, end_date)

if mdf.empty or mdf["value"].dropna().empty:
    st.error(f"No data for **{active_ticker}**. Check ticker symbol or date range.")
    st.stop()

# ---- Heatmap (always visible) ----
matrix, labels = build_seasonality_matrix(
    mdf, start_year=start_date.year, end_year=end_date.year,
    trailing_windows=trailing_windows,
    full_years_trailing=full_years_trail, full_years_avg=full_years_avg,
)
fig_hm = render_heatmap_plotly(matrix, labels, metric, active_ticker)
st.plotly_chart(fig_hm, use_container_width=True,
                config={"displayModeBar": True})

# ─────────────────────────────────────────────────────────────
# Quant panels (tabs)
# ─────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(
    ["Stability Metrics", "Rolling Regime", "Distributions"]
)

unit_label = "%" if metric == "pct_return" else "pts"

# ── TAB 1: Effect Size & Stability ──────────────────────────
with tab1:
    st.subheader("Effect Size & Stability by Month")
    c1, c2 = st.columns([1, 2])
    with c1:
        default_thr = 2.0
        thr_label = f"Downside threshold ({unit_label})"
        downside_thr = st.number_input(thr_label, min_value=0.0,
                                        value=default_thr, step=0.5,
                                        key="ds_thr")
        bar_y = st.selectbox("Bar chart metric",
                             ["Mean", "Median", "HitRate",
                              "DownsideFreq", "IQR"], key="bar_y")

    tbl = stability_metrics_table(mdf, metric, downside_thr)
    with c2:
        st.dataframe(
            tbl.style.format({
                "Mean": f"{{:.2f}}",
                "Median": f"{{:.2f}}",
                "HitRate": "{:.1f}",
                "DownsideFreq": "{:.1f}",
                "IQR": f"{{:.2f}}",
            }).set_properties(**{"text-align": "center"}),
            use_container_width=True, hide_index=True,
        )
    fig_bar = render_stability_bar(tbl, bar_y, metric, active_ticker)
    st.plotly_chart(fig_bar, use_container_width=True)

# ── TAB 2: Rolling Regime ───────────────────────────────────
with tab2:
    st.subheader("Rolling Seasonality Regime")
    c1, c2 = st.columns(2)
    with c1:
        roll_win = st.selectbox("Window (years)",
                                [5, 10, 15, 20], index=1, key="roll_w")
    with c2:
        sel_month_idx = st.selectbox("Month to track", range(12),
                                      format_func=lambda i: MONTHS[i],
                                      index=9, key="roll_m")
    sel_month = sel_month_idx + 1

    rdf = rolling_month_mean_series(mdf, sel_month, roll_win)
    if rdf.empty:
        st.warning("Not enough data for this rolling window / month.")
    else:
        fig_roll = render_rolling_line(rdf, MONTHS[sel_month - 1],
                                       roll_win, metric, active_ticker)
        st.plotly_chart(fig_roll, use_container_width=True)

# ── TAB 3: Distributions ────────────────────────────────────
with tab3:
    st.subheader("Monthly Distribution (Boxplots)")
    show_pts = st.checkbox("Show all data points (jitter)", value=False,
                           key="box_pts")
    fig_box = render_boxplots(mdf, metric, active_ticker, show_pts)
    st.plotly_chart(fig_box, use_container_width=True)
