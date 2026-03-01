from datetime import date, timedelta
import base64
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.interpolate import make_interp_spline

from models.financial_ml import forecast_statement_series, run_ml_models
from models.portfolio_optimization import optimize_portfolio
from models.valuation import run_dcf_valuation
from utils.data_loader import load_price_data
from utils.fundamentals import compute_financial_metrics, compute_health_score, fetch_ticker_bundle
from utils.indicators import add_indicators
from utils.risk import calculate_risk_metrics, return_series_stats
from utils.simulation import monte_carlo_paths
from utils.time_series import analyze_time_series, arima_forecast


st.set_page_config(page_title="AlphaForge Quant Terminal", layout="wide")


def inject_apple_glass_theme():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');
            .stApp {
                font-family: Inter, -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", sans-serif;
                background:
                    radial-gradient(60rem 40rem at 10% -10%, rgba(76, 111, 255, 0.25), transparent 55%),
                    radial-gradient(50rem 35rem at 90% 0%, rgba(147, 87, 255, 0.22), transparent 50%),
                    radial-gradient(45rem 30rem at 50% 100%, rgba(70, 160, 255, 0.16), transparent 50%),
                    #0A0A0B;
                color: #f5f6f7;
            }
            [data-testid="stSidebar"] {
                background: transparent;
            }
            [data-testid="stSidebar"] > div:first-child {
                background:
                    linear-gradient(180deg, rgba(255,255,255,0.10) 0%, rgba(255,255,255,0.06) 100%);
                backdrop-filter: blur(28px);
                -webkit-backdrop-filter: blur(28px);
                border: 1px solid rgba(255,255,255,0.16);
                border-radius: 28px;
                margin: 14px 10px 14px 12px;
                padding: 8px 10px 16px 10px;
                box-shadow:
                    0 22px 42px rgba(2, 7, 30, 0.35),
                    inset 0 1px 0 rgba(255,255,255,0.16);
            }
            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
                color: #d4dcf4;
            }
            [data-testid="stSidebar"] .stTextInput > div > div,
            [data-testid="stSidebar"] .stDateInput > div > div,
            [data-testid="stSidebar"] .stSelectbox > div > div,
            [data-testid="stSidebar"] .stNumberInput > div > div {
                background: rgba(12, 16, 28, 0.82);
                border: 1px solid rgba(150, 171, 225, 0.30);
                border-radius: 12px;
            }
            [data-testid="stSidebar"] .stTextInput input,
            [data-testid="stSidebar"] .stDateInput input,
            [data-testid="stSidebar"] .stSelectbox input,
            [data-testid="stSidebar"] .stNumberInput input {
                color: #edf2ff;
            }
            [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] {
                padding-top: 2px;
                padding-bottom: 8px;
            }
            [data-testid="stSidebar"] .stSlider [role="slider"] {
                box-shadow: 0 0 0 4px rgba(122,164,255,0.22);
            }
            [data-testid="stSidebar"] .stButton > button {
                border-radius: 12px;
                border: 1px solid rgba(145, 168, 230, 0.35);
                background: linear-gradient(180deg, rgba(120,148,230,0.30), rgba(94,119,201,0.24));
                color: #eff4ff;
            }
            [data-testid="stSidebar"] .stButton > button:hover {
                border-color: rgba(170, 191, 247, 0.65);
                background: linear-gradient(180deg, rgba(130,160,246,0.38), rgba(103,130,216,0.28));
            }
            @media (max-width: 900px) {
                [data-testid="stSidebar"] > div:first-child {
                    border-radius: 20px;
                    margin: 6px;
                    padding: 6px 6px 12px 6px;
                }
            }
            .glass-card {
                background: rgba(255,255,255,0.06);
                border: 1px solid rgba(255,255,255,0.14);
                backdrop-filter: blur(25px);
                -webkit-backdrop-filter: blur(25px);
                border-radius: 28px;
                padding: 24px;
                margin-bottom: 16px;
            }
            .hero-block {
                border-radius: 34px;
                padding: 34px 34px 30px 34px;
                margin: 4px 0 18px 0;
                min-height: 190px;
                display: flex;
                flex-direction: column;
                justify-content: flex-end;
                background-size: cover;
                background-position: center;
                border: 1px solid rgba(255,255,255,0.18);
                box-shadow:
                    0 26px 44px rgba(0,0,0,0.36),
                    inset 0 1px 0 rgba(255,255,255,0.16);
            }
            .hero-title {
                margin: 0;
                font-size: clamp(1.7rem, 3vw, 2.5rem);
                font-weight: 800;
                color: #f7f9ff;
                letter-spacing: -0.02em;
                text-shadow: 0 2px 12px rgba(0,0,0,0.35);
            }
            .hero-subtitle {
                margin: 8px 0 0 0;
                color: #d9e2ff;
                font-size: 0.98rem;
                font-weight: 500;
                text-shadow: 0 1px 8px rgba(0,0,0,0.30);
            }
            .balance-label { color: #a1a1aa; font-size: 0.9rem; font-weight: 500; }
            .balance-value {
                font-size: 3rem;
                line-height: 1.1;
                font-weight: 800;
                margin: 8px 0 4px 0;
                font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
                font-variant-numeric: tabular-nums;
            }
            .muted-small { color: #9ca3af; font-size: 0.85rem; }
            .txn-row {
                background: rgba(255,255,255,0.05);
                border-radius: 18px;
                padding: 14px 16px;
                margin-bottom: 10px;
            }
            .stButton > button {
                border-radius: 12px;
                border: 1px solid rgba(255,255,255,0.2);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _image_data_uri(image_path: str) -> str:
    p = Path(image_path)
    if not p.exists() or not p.is_file():
        return ""
    ext = p.suffix.lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"
    if ext not in {"png", "jpeg", "webp"}:
        return ""
    encoded = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/{ext};base64,{encoded}"


def render_header_hero(title: str, subtitle: str, image_path: str = "assets/hero-.jpg"):
    image_uri = _image_data_uri(image_path)
    if image_uri:
        bg_style = (
            "background-image: "
            "linear-gradient(120deg, rgba(8,12,26,0.72) 0%, rgba(10,10,14,0.45) 55%, rgba(16,24,44,0.65) 100%), "
            f"url('{image_uri}');"
        )
    else:
        bg_style = (
            "background-image: "
            "radial-gradient(40rem 20rem at 10% 10%, rgba(95,142,255,0.40), transparent 60%), "
            "radial-gradient(35rem 20rem at 90% 90%, rgba(74,208,190,0.28), transparent 60%), "
            "linear-gradient(125deg, rgba(16,19,31,0.96), rgba(9,11,17,0.94));"
        )
    st.markdown(
        f"""
        <div class="hero-block" style="{bg_style}">
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def parse_tickers(text):
    return [t.strip().upper() for t in text.split(",") if t.strip()]


def fmt_money(x):
    if pd.isna(x):
        return "N/A"
    return f"${x:,.2f}"


def fmt_num(x, digits=2):
    if pd.isna(x):
        return "N/A"
    return f"{x:.{digits}f}"


def fmt_pct(x, digits=2):
    if pd.isna(x):
        return "N/A"
    return f"{x * 100:.{digits}f}%"


def _first_valid(*values):
    for v in values:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        try:
            if pd.isna(v):
                continue
        except Exception:
            pass
        return v
    return np.nan


def build_market_display_info(info: dict, history: pd.DataFrame, close: pd.Series, dividends: pd.Series):
    out = dict(info or {})

    one_year = close.tail(252)
    if not one_year.empty:
        out["fiftyTwoWeekHigh"] = _first_valid(out.get("fiftyTwoWeekHigh"), float(one_year.max()))
        out["fiftyTwoWeekLow"] = _first_valid(out.get("fiftyTwoWeekLow"), float(one_year.min()))

    if isinstance(history, pd.DataFrame) and not history.empty and "Volume" in history.columns:
        avg_vol = pd.to_numeric(history["Volume"], errors="coerce").tail(252).mean()
        out["averageVolume"] = _first_valid(out.get("averageVolume"), avg_vol)

    if isinstance(dividends, pd.Series) and not dividends.empty and not close.empty:
        trailing_div = pd.to_numeric(dividends, errors="coerce").dropna().tail(4).sum()
        div_yield = trailing_div / float(close.iloc[-1]) if close.iloc[-1] > 0 else np.nan
        out["dividendYield"] = _first_valid(out.get("dividendYield"), div_yield)

    trailing_eps = _first_valid(out.get("trailingEps"), out.get("epsTrailingTwelveMonths"))
    if pd.notna(trailing_eps) and trailing_eps != 0 and not close.empty:
        pe_fallback = float(close.iloc[-1]) / float(trailing_eps)
        out["trailingPE"] = _first_valid(out.get("trailingPE"), pe_fallback)

    out["marketCap"] = _first_valid(out.get("marketCap"), out.get("market_cap"))
    out["currency"] = _first_valid(out.get("currency"), out.get("financialCurrency"))
    out["exchange"] = _first_valid(out.get("exchange"), out.get("exchangeName"))
    out["quoteType"] = _first_valid(out.get("quoteType"), out.get("type"))
    return out


def style_plotly(fig: go.Figure, title: str = "", height: int = 380):
    fig.update_layout(
        template="plotly_dark",
        title=title,
        height=height,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e9eefc"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def smooth_area_figure(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return go.Figure()
    x_vals = np.arange(len(s))
    y_vals = s.values.astype(float)
    if len(y_vals) >= 4:
        x_smooth = np.linspace(x_vals.min(), x_vals.max(), len(y_vals) * 4)
        spline = make_interp_spline(x_vals, y_vals, k=3)
        y_smooth = spline(x_smooth)
    else:
        x_smooth = x_vals
        y_smooth = y_vals

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode="lines",
            line=dict(color="#9EC2FF", width=3),
            fill="tozeroy",
            fillcolor="rgba(121,167,255,0.25)",
            name="Price",
        )
    )
    style_plotly(fig, height=320)
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, title="")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, title="")
    return fig


def compact_table(df: pd.DataFrame, max_rows=25):
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out = out.head(max_rows)
    for c in out.columns:
        if pd.api.types.is_numeric_dtype(out[c]):
            out[c] = out[c].round(4)
    return out


def smart_df_height(df: pd.DataFrame, min_height=120, max_height=900):
    rows = 0 if df is None else len(df.index)
    return int(min(max(min_height + rows * 38, min_height), max_height))


def show_table(df: pd.DataFrame, max_rows=25, hide_index=False):
    compact = compact_table(df, max_rows=max_rows)
    st.dataframe(
        compact,
        hide_index=hide_index,
        use_container_width=True,
        height=smart_df_height(compact),
    )


def plot_monte_carlo_3d(mc_paths: pd.DataFrame):
    if mc_paths.empty or mc_paths.shape[0] < 3 or mc_paths.shape[1] < 3:
        return None

    row_step = max(1, mc_paths.shape[0] // 80)
    col_step = max(1, mc_paths.shape[1] // 45)
    sample = mc_paths.iloc[::row_step, ::col_step].to_numpy()
    fig = go.Figure(
        data=[
            go.Surface(
                z=sample,
                colorscale="Cividis",
                showscale=True,
                opacity=0.95,
            )
        ]
    )
    style_plotly(fig, title="Monte Carlo Price Surface", height=520)
    fig.update_layout(
        scene=dict(
            xaxis_title="Day",
            yaxis_title="Simulation Slice",
            zaxis_title="Price",
            xaxis=dict(showbackground=True, backgroundcolor="rgba(10,12,20,0.8)", gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(showbackground=True, backgroundcolor="rgba(10,12,20,0.8)", gridcolor="rgba(255,255,255,0.08)"),
            zaxis=dict(showbackground=True, backgroundcolor="rgba(10,12,20,0.8)", gridcolor="rgba(255,255,255,0.08)"),
        )
    )
    return fig


def build_portfolio_cloud(price_df: pd.DataFrame, n_portfolios=3000, seed=42):
    returns = price_df.pct_change().dropna()
    if returns.empty or returns.shape[1] < 2:
        return pd.DataFrame()

    mu = returns.mean().values * 252
    cov = returns.cov().values * 252
    n_assets = returns.shape[1]

    rng = np.random.default_rng(seed)
    w = rng.random((n_portfolios, n_assets))
    w /= w.sum(axis=1, keepdims=True)

    exp_return = w @ mu
    variance = np.einsum("ij,jk,ik->i", w, cov, w)
    vol = np.sqrt(np.clip(variance, 0, None))
    sharpe = np.divide(exp_return, vol, out=np.zeros_like(exp_return), where=vol > 1e-12)

    return pd.DataFrame({"return": exp_return, "volatility": vol, "sharpe": sharpe})


def normalize_score(value: float, low: float, high: float, inverse=False) -> float:
    if pd.isna(value):
        score = 0.5
    elif high == low:
        score = 0.5
    else:
        score = float(np.clip((value - low) / (high - low), 0, 1))
    if inverse:
        score = 1 - score
    return float(np.clip(score, 0, 1))


def build_hedge_fund_scorecard(metrics: dict, risk: dict, dcf: dict, close: pd.Series) -> pd.DataFrame:
    indicators = add_indicators(close).dropna()
    latest_rsi = indicators["rsi_14"].iloc[-1] if not indicators.empty and "rsi_14" in indicators.columns else np.nan
    momentum_90 = close.pct_change(90).iloc[-1] if len(close) > 90 else np.nan

    growth_components = [
        normalize_score(metrics.get("revenue_cagr", np.nan), -0.05, 0.25),
        normalize_score(metrics.get("net_income_cagr", np.nan), -0.10, 0.30),
        normalize_score(metrics.get("fcf_cagr", np.nan), -0.10, 0.30),
        normalize_score(metrics.get("eps_growth_proxy", np.nan), -0.20, 0.40),
    ]
    profitability_components = [
        normalize_score(metrics.get("gross_margin", np.nan), 0.05, 0.70),
        normalize_score(metrics.get("operating_margin", np.nan), 0.00, 0.40),
        normalize_score(metrics.get("net_margin", np.nan), -0.05, 0.30),
        normalize_score(metrics.get("roe", np.nan), 0.00, 0.30),
        normalize_score(metrics.get("roa", np.nan), 0.00, 0.20),
        normalize_score(metrics.get("roic", np.nan), 0.00, 0.30),
    ]
    stability_components = [
        normalize_score(metrics.get("debt_to_equity", np.nan), 0.00, 2.50, inverse=True),
        normalize_score(metrics.get("debt_to_assets", np.nan), 0.00, 0.80, inverse=True),
        normalize_score(metrics.get("current_ratio", np.nan), 0.80, 3.00),
        normalize_score(metrics.get("quick_ratio", np.nan), 0.60, 2.50),
    ]
    valuation_components = [
        normalize_score(metrics.get("pe", np.nan), 8.00, 45.00, inverse=True),
        normalize_score(metrics.get("peg", np.nan), 0.70, 3.00, inverse=True),
        normalize_score(metrics.get("price_to_book", np.nan), 1.00, 12.00, inverse=True),
        normalize_score(metrics.get("ev_to_ebitda", np.nan), 6.00, 30.00, inverse=True),
        normalize_score(dcf.get("undervaluation_pct", np.nan), -0.40, 0.40),
    ]
    risk_components = [
        normalize_score(risk.get("annualized_volatility", np.nan), 0.10, 0.60, inverse=True),
        normalize_score(risk.get("max_drawdown", np.nan), 0.10, 0.70, inverse=True),
        normalize_score(risk.get("sharpe_ratio", np.nan), 0.00, 2.00),
        normalize_score(risk.get("sortino_ratio", np.nan), 0.00, 3.00),
    ]
    behavior_components = [
        normalize_score(momentum_90, -0.30, 0.50),
        normalize_score(latest_rsi, 30.0, 70.0),
    ]

    categories = [
        ("Growth", growth_components),
        ("Profitability", profitability_components),
        ("Financial Stability", stability_components),
        ("Valuation", valuation_components),
        ("Risk", risk_components),
        ("Market Behavior", behavior_components),
    ]

    rows = []
    for name, comps in categories:
        value = float(np.mean(comps) * 10)
        rows.append({"Category": name, "Score": value})

    out = pd.DataFrame(rows)
    out["Score"] = out["Score"].round(2)
    out["Weight"] = [0.20, 0.20, 0.15, 0.20, 0.15, 0.10]
    out["Weighted"] = (out["Score"] * out["Weight"]).round(2)
    return out


def scorecard_radar_figure(score_df: pd.DataFrame):
    theta = score_df["Category"].tolist()
    r = score_df["Score"].tolist()
    theta.append(theta[0])
    r.append(r[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            line=dict(color="#7AA4FF", width=3),
            fillcolor="rgba(122,164,255,0.30)",
            name="Category Score",
        )
    )
    style_plotly(fig, title="Hedge Fund Scorecard Radar", height=460)
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 10], gridcolor="rgba(255,255,255,0.10)"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False,
    )
    return fig


def research_pipeline_figure(score_df: pd.DataFrame):
    stage_names = [
        "Company Snapshot",
        "Growth",
        "Profitability",
        "Stability",
        "Risk",
        "Valuation",
        "Market Behavior",
    ]
    stage_values = [
        100,
        float(score_df.loc[score_df["Category"] == "Growth", "Score"].iloc[0] * 10),
        float(score_df.loc[score_df["Category"] == "Profitability", "Score"].iloc[0] * 10),
        float(score_df.loc[score_df["Category"] == "Financial Stability", "Score"].iloc[0] * 10),
        float(score_df.loc[score_df["Category"] == "Risk", "Score"].iloc[0] * 10),
        float(score_df.loc[score_df["Category"] == "Valuation", "Score"].iloc[0] * 10),
        float(score_df.loc[score_df["Category"] == "Market Behavior", "Score"].iloc[0] * 10),
    ]

    fig = go.Figure(
        go.Funnel(
            y=stage_names,
            x=stage_values,
            textinfo="value+percent initial",
            marker=dict(color=["#8EB4FF", "#7AA4FF", "#5E8BFF", "#5C7EF2", "#7F6BFF", "#9B6CFF", "#4CD4B0"]),
        )
    )
    style_plotly(fig, title="Research Pipeline Readiness", height=460)
    return fig


def statement_series(df: pd.DataFrame, candidates):
    if df.empty:
        return pd.Series(dtype=float)
    for key in candidates:
        if key in df.index:
            s = pd.to_numeric(df.loc[key], errors="coerce").dropna()
            if not s.empty:
                return s.sort_index()
    return pd.Series(dtype=float)


def normalize_statement_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    out = out.apply(pd.to_numeric, errors="coerce")
    out = out.dropna(how="all")
    if out.empty:
        return out

    dt = pd.to_datetime(out.columns, errors="coerce")
    if dt.notna().sum() >= 2:
        order = np.argsort(dt.values)
        out = out.iloc[:, order]
    return out


def format_statement_df(df: pd.DataFrame, scale_mode: str = "Billions", transpose=False) -> tuple[pd.DataFrame, str]:
    if df is None or df.empty:
        return pd.DataFrame(), ""
    out = df.copy()
    out = out.apply(pd.to_numeric, errors="coerce")

    scale_map = {"Raw": 1.0, "Millions": 1_000_000.0, "Billions": 1_000_000_000.0}
    divisor = scale_map.get(scale_mode, 1_000_000_000.0)
    unit_label = "" if scale_mode == "Raw" else f" ({scale_mode.lower()})"

    num_cols = out.columns
    out[num_cols] = out[num_cols] / divisor
    out = out.round(3)
    if transpose:
        out = out.T
    out.columns = out.columns.astype(str)
    return out, unit_label


def statement_visual_lab(df: pd.DataFrame, title: str, default_lines):
    data = normalize_statement_df(df)
    if data.empty:
        st.info(f"No {title} data available for visualization.")
        return

    period_count = st.slider(
        f"{title}: periods to display",
        min_value=2,
        max_value=max(2, data.shape[1]),
        value=min(6, data.shape[1]),
        key=f"{title}_period_count",
    )
    data = data.iloc[:, -period_count:]
    x_labels = [str(c)[:10] for c in data.columns]

    line_choices = data.index.tolist()
    default_picks = [x for x in default_lines if x in line_choices]
    if not default_picks:
        default_picks = line_choices[: min(4, len(line_choices))]

    selected = st.multiselect(
        f"{title}: line items",
        options=line_choices,
        default=default_picks,
        key=f"{title}_line_items",
    )

    if selected:
        fig = go.Figure()
        for item in selected:
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=data.loc[item].values,
                    mode="lines+markers",
                    name=item,
                )
            )
        style_plotly(fig, title=f"{title} Multi-Line Trend", height=420)
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig, use_container_width=True)

    latest_col = data.columns[-1]
    latest = data[latest_col].dropna()
    if not latest.empty:
        top = latest.reindex(latest.abs().sort_values(ascending=False).index).head(12)
        bar = go.Figure(
            data=[
                go.Bar(
                    x=top.values,
                    y=top.index,
                    orientation="h",
                    marker=dict(
                        color=np.where(top.values >= 0, "#6EE7B7", "#FCA5A5"),
                    ),
                )
            ]
        )
        style_plotly(bar, title=f"{title} Latest Period Contribution ({str(latest_col)[:10]})", height=420)
        bar.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        bar.update_yaxes(showgrid=False, automargin=True)
        st.plotly_chart(bar, use_container_width=True)

    heat_rows = data.abs().sum(axis=1).sort_values(ascending=False).head(12).index.tolist()
    heat_df = data.loc[heat_rows]
    if not heat_df.empty:
        zmax = float(np.nanmax(np.abs(heat_df.values))) if np.isfinite(heat_df.values).any() else 1.0
        heat = go.Figure(
            data=[
                go.Heatmap(
                    z=heat_df.values,
                    x=x_labels,
                    y=heat_df.index,
                    zmin=-zmax,
                    zmax=zmax,
                    colorscale="RdBu",
                    reversescale=True,
                    colorbar=dict(title="Value"),
                )
            ]
        )
        style_plotly(heat, title=f"{title} Cross-Period Heatmap", height=440)
        heat.update_xaxes(showgrid=False)
        heat.update_yaxes(showgrid=False, automargin=True)
        st.plotly_chart(heat, use_container_width=True)

        # 3D surface for cross-period structure across major line items.
        y_labels = heat_df.index.tolist()
        x_idx = list(range(len(x_labels)))
        y_idx = list(range(len(y_labels)))
        surface_mode = st.radio(
            f"{title}: 3D mode",
            ["Lightweight Mesh", "Surface"],
            horizontal=True,
            key=f"{title}_3d_mode",
        )
        if surface_mode == "Surface":
            trace = go.Surface(
                z=heat_df.values,
                x=x_idx,
                y=y_idx,
                colorscale="RdBu",
                reversescale=True,
                colorbar=dict(title="Value"),
            )
        else:
            xi, yi = np.meshgrid(x_idx, y_idx)
            trace = go.Mesh3d(
                x=xi.flatten(),
                y=yi.flatten(),
                z=heat_df.values.flatten(),
                intensity=heat_df.values.flatten(),
                colorscale="RdBu",
                reversescale=True,
                opacity=0.95,
                colorbar=dict(title="Value"),
            )
        surface = go.Figure(
            data=[trace]
        )
        style_plotly(surface, title=f"{title} 3D Structure Surface", height=560)
        surface.update_layout(
            scene=dict(
                xaxis=dict(
                    title="Period",
                    tickmode="array",
                    tickvals=x_idx,
                    ticktext=x_labels,
                    showbackground=True,
                    backgroundcolor="rgba(10,12,20,0.8)",
                    gridcolor="rgba(255,255,255,0.08)",
                ),
                yaxis=dict(
                    title="Line Item",
                    tickmode="array",
                    tickvals=y_idx,
                    ticktext=y_labels,
                    showbackground=True,
                    backgroundcolor="rgba(10,12,20,0.8)",
                    gridcolor="rgba(255,255,255,0.08)",
                ),
                zaxis=dict(
                    title="Value",
                    showbackground=True,
                    backgroundcolor="rgba(10,12,20,0.8)",
                    gridcolor="rgba(255,255,255,0.08)",
                ),
            )
        )
        st.plotly_chart(surface, use_container_width=True)
        st.caption("3D chart tip: drag to rotate, scroll to zoom. If your browser/GPU struggles, use Lightweight Mesh mode.")


def show_statement(
    df: pd.DataFrame,
    title: str,
    max_rows=30,
    scale_mode: str = "Billions",
    table_height: int = 760,
    transpose=False,
):
    st.markdown(f"**{title}**")
    if df.empty:
        st.info("No data available from source.")
        return
    table, unit_label = format_statement_df(df, scale_mode=scale_mode, transpose=transpose)
    if table.empty:
        st.info("No numeric data available from source.")
        return
    compact = compact_table(table, max_rows=max_rows)
    st.caption(f"Display unit: {scale_mode}{unit_label}")
    st.dataframe(
        compact,
        use_container_width=True,
        height=table_height,
    )


@st.cache_data(show_spinner=False)
def get_prices(tickers, start_dt, end_dt):
    return load_price_data(tickers=tickers, start=start_dt, end=end_dt)


@st.cache_data(show_spinner=False)
def get_bundle(ticker, schema_version="v2"):
    _ = schema_version
    return fetch_ticker_bundle(ticker)


def ensure_bundle_fields(bundle):
    df_fields = [
        "history",
        "actions",
        "financials",
        "balance_sheet",
        "cashflow",
        "quarterly_financials",
        "quarterly_balance_sheet",
        "quarterly_cashflow",
        "major_holders",
        "institutional_holders",
        "mutualfund_holders",
        "insider_transactions",
        "insider_roster_holders",
        "recommendations",
        "upgrades_downgrades",
        "earnings_dates",
    ]
    series_fields = ["dividends", "splits"]

    if not hasattr(bundle, "info") or not isinstance(getattr(bundle, "info"), dict):
        setattr(bundle, "info", {})

    for name in df_fields:
        if not hasattr(bundle, name) or not isinstance(getattr(bundle, name), pd.DataFrame):
            setattr(bundle, name, pd.DataFrame())
    for name in series_fields:
        if not hasattr(bundle, name) or not isinstance(getattr(bundle, name), pd.Series):
            setattr(bundle, name, pd.Series(dtype=float))

    return bundle


today = date.today()
default_start = today - timedelta(days=365 * 5)
inject_apple_glass_theme()
render_header_hero(
    title="AlphaForge Quant Terminal",
    subtitle="End-to-End Factor, Risk, and Valuation Intelligence from One Ticker.",
    image_path="assets/hero-.jpg",
)

st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Primary ticker", "AAPL").upper().strip()
start_dt = st.sidebar.date_input("Price start", default_start)
end_dt = st.sidebar.date_input("Price end", today)
portfolio_text = st.sidebar.text_input("Portfolio tickers", "AAPL,MSFT,GOOGL,AMZN")
max_table_rows = st.sidebar.slider("Max table rows", 10, 200, 40, 5)

if start_dt >= end_dt:
    st.error("Start date must be before end date.")
    st.stop()

bundle = ensure_bundle_fields(get_bundle(ticker, schema_version="v4"))
metrics = compute_financial_metrics(bundle)
health_score = compute_health_score(metrics)

prices = get_prices([ticker], start_dt, end_dt)
if prices.empty or ticker not in prices.columns:
    st.error("Unable to fetch price data for selected ticker and date range.")
    st.stop()

close = prices[ticker].dropna()
returns = close.pct_change().dropna()
current_price = float(close.iloc[-1])
info = build_market_display_info(bundle.info, bundle.history, close, bundle.dividends)
front_risk = calculate_risk_metrics(close, confidence=0.95, horizon_days=1)
front_dcf = run_dcf_valuation(
    cashflow=bundle.cashflow,
    shares_outstanding=metrics.get("shares_outstanding", np.nan),
    beta=info.get("beta", 1.0),
    current_price=current_price,
    forecast_years=5,
    terminal_growth=0.025,
    risk_free_rate=0.045,
    equity_risk_premium=0.055,
)
scorecard_df = build_hedge_fund_scorecard(metrics, front_risk, front_dcf, close)
overall_score = float(scorecard_df["Weighted"].sum()) if not scorecard_df.empty else np.nan

tabs = st.tabs(
    [
        "1) Executive Dashboard",
        "2) Market Intelligence",
        "3) Financial Statements",
        "4) Ratios & Health",
        "5) Risk Lab",
        "6) Valuation + ML",
        "7) Portfolio Studio",
        "8) Raw Data Vault",
    ]
)


with tabs[0]:
    st.subheader("Executive Dashboard")
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("<div class='balance-label'>Total Balance (Spot Proxy)</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='balance-value'>{fmt_money(current_price)}</div>", unsafe_allow_html=True)
        st.markdown(
            f"<div class='muted-small'>Market Cap: {fmt_money(info.get('marketCap', np.nan))} | Health Score: {health_score:.1f}/100</div>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(smooth_area_figure(close.tail(120)), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        st.markdown("#### Recent Activity")
        actions = bundle.actions.copy() if isinstance(bundle.actions, pd.DataFrame) else pd.DataFrame()
        if not actions.empty:
            recent = actions.reset_index().tail(5).iloc[::-1]
            for _, row in recent.iterrows():
                dt = row.iloc[0]
                div_val = row.get("Dividends", 0.0) if "Dividends" in recent.columns else 0.0
                split_val = row.get("Stock Splits", 0.0) if "Stock Splits" in recent.columns else 0.0
                detail = f"Dividend: {div_val:.4f}" if div_val else (f"Split: {split_val:.4f}" if split_val else "Corporate action")
                st.markdown(
                    f"<div class='txn-row'><div><strong>{ticker}</strong></div><div class='muted-small'>{dt}</div><div>{detail}</div></div>",
                    unsafe_allow_html=True,
                )
        else:
            st.info("No recent actions available.")
        st.markdown("</div>", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("52W High", fmt_money(info.get("fiftyTwoWeekHigh", np.nan)))
    c2.metric("52W Low", fmt_money(info.get("fiftyTwoWeekLow", np.nan)))
    c3.metric("Avg Volume", fmt_num(info.get("averageVolume", np.nan), 0))
    c4.metric("Dividend Yield", fmt_pct(info.get("dividendYield", np.nan)))
    c5.metric("Trailing PE", fmt_num(info.get("trailingPE", np.nan), 2))

    if not returns.empty:
        summary = {
            "1D Return": returns.iloc[-1],
            "30D Return": close.pct_change(30).iloc[-1],
            "90D Return": close.pct_change(90).iloc[-1],
            "YTD Return": (close.iloc[-1] / close[close.index.year == close.index[-1].year].iloc[0] - 1)
            if (close.index.year == close.index[-1].year).any()
            else np.nan,
        }
        s_df = pd.DataFrame(summary.items(), columns=["Metric", "Value"])
        s_df["Value"] = s_df["Value"].apply(fmt_pct)
        show_table(s_df, max_rows=20, hide_index=True)

    st.markdown("### Hedge Fund Research Desk Overview")
    if scorecard_df.empty:
        st.info("Scorecard unavailable due to insufficient data.")
    else:
        s1, s2, s3 = st.columns(3)
        s1.metric("Overall Desk Score", f"{overall_score:.2f} / 10")
        s2.metric("Growth Score", f"{scorecard_df.loc[scorecard_df['Category']=='Growth', 'Score'].iloc[0]:.2f} / 10")
        s3.metric("Risk Score", f"{scorecard_df.loc[scorecard_df['Category']=='Risk', 'Score'].iloc[0]:.2f} / 10")

        v1, v2 = st.columns(2, gap="large")
        with v1:
            st.plotly_chart(scorecard_radar_figure(scorecard_df), use_container_width=True)
        with v2:
            st.plotly_chart(research_pipeline_figure(scorecard_df), use_container_width=True)

        score_view = scorecard_df.copy()
        score_view["Score"] = score_view["Score"].map(lambda x: f"{x:.2f} / 10")
        score_view["Weight"] = score_view["Weight"].map(lambda x: f"{x * 100:.0f}%")
        score_view["Weighted"] = score_view["Weighted"].map(lambda x: f"{x:.2f}")
        show_table(score_view[["Category", "Score", "Weight", "Weighted"]], max_rows=12, hide_index=True)
        st.caption("Use tabs 2-7 to deep-dive each layer: financials, risk, valuation/ML, and portfolio impact.")


with tabs[1]:
    st.subheader("Market Intelligence")

    mi_tabs = st.tabs(
        [
            "Snapshot",
            "Corporate Actions",
            "Holders + Insider",
            "Analyst Activity",
            "Earnings",
        ]
    )

    profile_keys = [
        "longName",
        "sector",
        "industry",
        "country",
        "fullTimeEmployees",
        "website",
        "exchange",
        "currency",
        "quoteType",
    ]
    profile_fallback = {
        "longName": _first_valid(info.get("longName"), info.get("shortName"), ticker),
        "sector": _first_valid(info.get("sector"), "N/A"),
        "industry": _first_valid(info.get("industry"), "N/A"),
        "country": _first_valid(info.get("country"), "N/A"),
        "fullTimeEmployees": _first_valid(info.get("fullTimeEmployees"), "N/A"),
        "website": _first_valid(info.get("website"), "N/A"),
        "exchange": _first_valid(info.get("exchange"), info.get("exchangeName"), "N/A"),
        "currency": _first_valid(info.get("currency"), info.get("financialCurrency"), "N/A"),
        "quoteType": _first_valid(info.get("quoteType"), info.get("type"), "N/A"),
    }
    profile = pd.DataFrame(
        [{"Field": k, "Value": profile_fallback.get(k, "N/A")} for k in profile_keys]
    )

    with mi_tabs[0]:
        st.markdown("**Company Snapshot**")
        show_table(profile, max_rows=30, hide_index=True)

    with mi_tabs[1]:
        st.markdown("**Corporate Actions**")
        a1, a2 = st.columns(2)
        if not bundle.actions.empty:
            with a1:
                show_table(bundle.actions.reset_index(), max_rows=max_table_rows)
        else:
            a1.info("No actions available")

        div_df = bundle.dividends.tail(max_table_rows).rename("Dividend").to_frame()
        split_df = bundle.splits.tail(max_table_rows).rename("Split").to_frame()

        if not div_df.empty or not split_df.empty:
            merged = div_df.join(split_df, how="outer").reset_index()
            with a2:
                show_table(merged, max_rows=max_table_rows)
        else:
            a2.info("No dividends/splits available")

    with mi_tabs[2]:
        st.markdown("**Holders and Insider Data**")
        h1, h2 = st.columns(2)
        with h1:
            st.markdown("`major_holders`")
            show_table(bundle.major_holders, max_rows=max_table_rows)
        with h2:
            st.markdown("`institutional_holders`")
            show_table(bundle.institutional_holders, max_rows=max_table_rows)

        h3, h4 = st.columns(2)
        with h3:
            st.markdown("`mutualfund_holders`")
            show_table(bundle.mutualfund_holders, max_rows=max_table_rows)
        with h4:
            st.markdown("`insider_transactions`")
            show_table(bundle.insider_transactions, max_rows=max_table_rows)

    with mi_tabs[3]:
        st.markdown("**Analyst Activity**")
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("`recommendations`")
            show_table(bundle.recommendations, max_rows=max_table_rows)
        with r2:
            st.markdown("`upgrades_downgrades`")
            show_table(bundle.upgrades_downgrades, max_rows=max_table_rows)

    with mi_tabs[4]:
        st.markdown("**Earnings Calendar**")
        show_table(bundle.earnings_dates, max_rows=max_table_rows)


with tabs[2]:
    st.subheader("Financial Statements")

    mode = st.radio("Statement frequency", ["Annual", "Quarterly"], horizontal=True)
    cfs1, cfs2, cfs3 = st.columns([1, 1, 1.2])
    scale_mode = cfs1.selectbox("Statement scale", ["Raw", "Millions", "Billions"], index=2)
    table_height = cfs2.slider("Table height", 420, 1200, 820, 20)
    transpose_view = cfs3.toggle("Transpose tables", value=False)
    if mode == "Annual":
        inc, bal, cfs = bundle.financials, bundle.balance_sheet, bundle.cashflow
    else:
        inc, bal, cfs = (
            bundle.quarterly_financials,
            bundle.quarterly_balance_sheet,
            bundle.quarterly_cashflow,
        )

    show_statement(
        inc,
        "Income Statement",
        max_rows=max_table_rows,
        scale_mode=scale_mode,
        table_height=table_height,
        transpose=transpose_view,
    )
    st.divider()
    show_statement(
        bal,
        "Balance Sheet",
        max_rows=max_table_rows,
        scale_mode=scale_mode,
        table_height=table_height,
        transpose=transpose_view,
    )
    st.divider()
    show_statement(
        cfs,
        "Cash Flow",
        max_rows=max_table_rows,
        scale_mode=scale_mode,
        table_height=table_height,
        transpose=transpose_view,
    )

    rev = statement_series(inc, ["Total Revenue", "Revenue"])
    ni = statement_series(inc, ["Net Income", "Net Income Common Stockholders"])
    fcf = statement_series(cfs, ["Free Cash Flow"])

    if not rev.empty or not ni.empty or not fcf.empty:
        growth = pd.DataFrame({"Revenue": rev, "Net Income": ni, "FCF": fcf}).sort_index()
        fig = go.Figure()
        for col in growth.columns:
            if growth[col].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=growth.index.astype(str),
                        y=growth[col],
                        mode="lines+markers",
                        name=col,
                    )
                )
        style_plotly(fig, title=f"{mode} Fundamental Trend", height=380)
        fig.update_xaxes(tickangle=30, showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Interactive Statement Visuals")
    vis_tabs = st.tabs(["Income Statement Visuals", "Balance Sheet Visuals", "Cash Flow Visuals"])
    with vis_tabs[0]:
        statement_visual_lab(
            inc,
            "Income Statement",
            default_lines=[
                "Total Revenue",
                "Gross Profit",
                "Operating Income",
                "Net Income",
            ],
        )
    with vis_tabs[1]:
        statement_visual_lab(
            bal,
            "Balance Sheet",
            default_lines=[
                "Total Assets",
                "Stockholders Equity",
                "Total Debt",
                "Cash And Cash Equivalents",
            ],
        )
    with vis_tabs[2]:
        statement_visual_lab(
            cfs,
            "Cash Flow",
            default_lines=[
                "Operating Cash Flow",
                "Free Cash Flow",
                "Capital Expenditure",
                "Net Income",
            ],
        )


with tabs[3]:
    st.subheader("Ratios & Financial Health")

    ratio_rows = [
        ("Gross Margin", metrics.get("gross_margin"), "pct"),
        ("Operating Margin", metrics.get("operating_margin"), "pct"),
        ("Net Margin", metrics.get("net_margin"), "pct"),
        ("ROE", metrics.get("roe"), "pct"),
        ("ROA", metrics.get("roa"), "pct"),
        ("ROIC", metrics.get("roic"), "pct"),
        ("Current Ratio", metrics.get("current_ratio"), "num"),
        ("Quick Ratio", metrics.get("quick_ratio"), "num"),
        ("Cash Ratio", metrics.get("cash_ratio"), "num"),
        ("Debt / Equity", metrics.get("debt_to_equity"), "num"),
        ("Debt / Assets", metrics.get("debt_to_assets"), "pct"),
        ("Asset Turnover", metrics.get("asset_turnover"), "num"),
        ("Inventory Turnover", metrics.get("inventory_turnover"), "num"),
        ("Receivable Turnover", metrics.get("receivable_turnover"), "num"),
        ("Revenue CAGR", metrics.get("revenue_cagr"), "pct"),
        ("Net Income CAGR", metrics.get("net_income_cagr"), "pct"),
        ("FCF CAGR", metrics.get("fcf_cagr"), "pct"),
        ("PEG", metrics.get("peg"), "num"),
        ("Price / Book", metrics.get("price_to_book"), "num"),
        ("EV / EBITDA", metrics.get("ev_to_ebitda"), "num"),
    ]

    ratio_df = pd.DataFrame(ratio_rows, columns=["Metric", "Value", "Type"])
    ratio_df["Display"] = ratio_df.apply(
        lambda r: fmt_pct(r["Value"]) if r["Type"] == "pct" else fmt_num(r["Value"], 3), axis=1
    )
    show_table(ratio_df[["Metric", "Display"]], max_rows=40, hide_index=True)

    h1, h2, h3 = st.columns(3)
    h1.metric("Financial Health Score", f"{health_score:.1f} / 100")
    h2.metric("Revenue", fmt_money(metrics.get("revenue", np.nan)))
    h3.metric("Free Cash Flow", fmt_money(metrics.get("free_cash_flow", np.nan)))


with tabs[4]:
    st.subheader("Risk Lab")

    conf = st.slider("VaR confidence", 0.90, 0.99, 0.95, 0.01)
    horizon = st.slider("VaR horizon (days)", 1, 30, 1)
    mc_days = st.slider("Monte Carlo horizon (days)", 30, 504, 252, 1)
    mc_sims = st.slider("Monte Carlo simulations", 200, 5000, 1000, 100)

    risk = calculate_risk_metrics(close, confidence=conf, horizon_days=horizon)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Historical VaR", fmt_pct(risk["historical_var"]))
    r2.metric("Parametric VaR", fmt_pct(risk["parametric_var"]))
    r3.metric("Sharpe", fmt_num(risk["sharpe_ratio"], 3))
    r4.metric("Sortino", fmt_num(risk["sortino_ratio"], 3))

    r5, r6 = st.columns(2)
    r5.metric("Annualized Volatility", fmt_pct(risk["annualized_volatility"]))
    r6.metric("Max Drawdown", fmt_pct(risk["max_drawdown"]))

    ts_stats = return_series_stats(close, window=30)
    if not ts_stats.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ts_stats.index, y=ts_stats["rolling_mean"], mode="lines", name="Rolling Mean"))
        fig.add_trace(go.Scatter(x=ts_stats.index, y=ts_stats["rolling_vol"], mode="lines", name="Rolling Volatility"))
        style_plotly(fig, title="Rolling Return Stats", height=340)
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig, use_container_width=True)

    ts_diag = analyze_time_series(close)
    if "error" not in ts_diag:
        t1, t2 = st.columns(2)
        t1.metric("ADF Statistic", fmt_num(ts_diag["adf_stat"], 4))
        t2.metric("ADF p-value", fmt_num(ts_diag["adf_pvalue"], 6))

        fig = go.Figure(
            data=[
                go.Bar(
                    x=ts_diag["acf_lags"],
                    y=ts_diag["acf_values"],
                    marker=dict(color="#7AA4FF"),
                    name="ACF",
                )
            ]
        )
        style_plotly(fig, title="Autocorrelation of Returns", height=320)
        fig.update_xaxes(title="Lag", showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(title="ACF", showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig, use_container_width=True)

    mc_paths = monte_carlo_paths(close, n_sims=mc_sims, n_days=mc_days, seed=42)
    if not mc_paths.empty:
        terminal = mc_paths.iloc[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Median terminal", fmt_money(terminal.median()))
        m2.metric("5th pct", fmt_money(terminal.quantile(0.05)))
        m3.metric("95th pct", fmt_money(terminal.quantile(0.95)))

        fig = go.Figure()
        for i in range(min(50, mc_paths.shape[1])):
            fig.add_trace(
                go.Scatter(
                    x=mc_paths.index,
                    y=mc_paths[i],
                    mode="lines",
                    line=dict(width=1),
                    opacity=0.18,
                    showlegend=False,
                )
            )
        style_plotly(fig, title="Monte Carlo Paths", height=380)
        fig.update_xaxes(title="Day", showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(title="Price", showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig, use_container_width=True)

        mc_surface = plot_monte_carlo_3d(mc_paths)
        if mc_surface is not None:
            st.plotly_chart(mc_surface, use_container_width=True)


with tabs[5]:
    st.subheader("Valuation + ML")

    c1, c2, c3 = st.columns(3)
    rf_rate = c1.number_input("Risk-free rate", min_value=0.0, max_value=0.2, value=0.045, step=0.005)
    erp = c2.number_input("Equity risk premium", min_value=0.0, max_value=0.2, value=0.055, step=0.005)
    terminal_growth = c3.number_input("Terminal growth", min_value=0.0, max_value=0.1, value=0.025, step=0.005)

    dcf = run_dcf_valuation(
        cashflow=bundle.cashflow,
        shares_outstanding=metrics.get("shares_outstanding", np.nan),
        beta=info.get("beta", 1.0),
        current_price=current_price,
        forecast_years=5,
        terminal_growth=terminal_growth,
        risk_free_rate=rf_rate,
        equity_risk_premium=erp,
    )

    if "error" in dcf:
        st.warning(dcf["error"])
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Discount Rate", fmt_pct(dcf["discount_rate"]))
        d2.metric("Intrinsic Value/Share", fmt_money(dcf["intrinsic_per_share"]))
        d3.metric("Current Price", fmt_money(dcf["current_price"]))
        d4.metric("Undervaluation", fmt_pct(dcf["undervaluation_pct"]))

    arima = arima_forecast(close, steps=30, order=(1, 1, 1))
    if not arima.empty:
        hist = close.tail(100)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(hist))), y=hist.values, mode="lines", name="Recent Close"))
        fig.add_trace(
            go.Scatter(
                x=list(range(len(hist), len(hist) + len(arima))),
                y=arima["forecast"].values,
                mode="lines",
                name="ARIMA Forecast",
            )
        )
        style_plotly(fig, title="ARIMA Price Forecast", height=360)
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig, use_container_width=True)

    ml = run_ml_models(close)
    if "error" in ml:
        st.info(ml["error"])
    else:
        ml1, ml2, ml3, ml4 = st.columns(4)
        ml1.metric("RF RMSE", fmt_num(ml["rmse"], 4))
        ml2.metric("RF MAE", fmt_num(ml["mae"], 4))
        ml3.metric("Next Price", fmt_money(ml["next_price"]))
        ml4.metric("Up Probability", fmt_pct(ml["up_probability"]))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ml["y_test"].index, y=ml["y_test"].values, mode="lines", name="Actual"))
        fig.add_trace(go.Scatter(x=ml["y_test"].index, y=ml["reg_pred"], mode="lines", name="Predicted"))
        style_plotly(fig, title="ML Regression Fit", height=350)
        fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
        st.plotly_chart(fig, use_container_width=True)

    if not bundle.quarterly_financials.empty and "Total Revenue" in bundle.quarterly_financials.index:
        rev_fc = forecast_statement_series(bundle.quarterly_financials.loc["Total Revenue"], periods=4)
        if not rev_fc.empty:
            st.markdown("**Quarterly Revenue Forecast (Linear Regression)**")
            show_table(rev_fc, max_rows=12, hide_index=True)


with tabs[6]:
    st.subheader("Portfolio Studio")
    portfolio_tickers = parse_tickers(portfolio_text)
    if len(portfolio_tickers) < 2:
        st.info("Enter at least two tickers in sidebar for portfolio optimization.")
    else:
        objective = st.selectbox("Optimization objective", ["max_sharpe", "min_volatility"])
        p_prices = get_prices(portfolio_tickers, start_dt, end_dt)

        if p_prices.empty or p_prices.shape[1] < 2:
            st.warning("Unable to fetch enough portfolio price history.")
        else:
            try:
                weights, perf = optimize_portfolio(p_prices, objective)
                w = pd.Series(weights).sort_values(ascending=False)
                w_df = w.reset_index()
                w_df.columns = ["Ticker", "Weight"]
                fig = go.Figure(
                    data=[go.Bar(x=w_df["Ticker"], y=w_df["Weight"], marker=dict(color="#7AA4FF"))]
                )
                style_plotly(fig, title="Optimized Weights", height=360)
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.08)")
                st.plotly_chart(fig, use_container_width=True)

                p1, p2, p3 = st.columns(3)
                p1.metric("Expected Return", fmt_pct(perf[0]))
                p2.metric("Volatility", fmt_pct(perf[1]))
                p3.metric("Sharpe", fmt_num(perf[2], 3))

                corr = p_prices.pct_change().dropna().corr()
                fig = go.Figure(
                    data=[
                        go.Heatmap(
                            z=corr.values,
                            x=list(corr.columns),
                            y=list(corr.index),
                            zmin=-1,
                            zmax=1,
                            colorscale="RdBu",
                            reversescale=True,
                        )
                    ]
                )
                style_plotly(fig, title="Correlation Heatmap", height=460)
                fig.update_xaxes(tickangle=35)
                st.plotly_chart(fig, use_container_width=True)

                st.metric("Selected ticker portfolio weight", fmt_pct(w.get(ticker, 0.0)))

                cloud = build_portfolio_cloud(p_prices, n_portfolios=3000, seed=42)
                if not cloud.empty:
                    fig = go.Figure(
                        data=[
                            go.Scatter3d(
                                x=cloud["volatility"],
                                y=cloud["return"],
                                z=cloud["sharpe"],
                                mode="markers",
                                marker=dict(
                                    size=3,
                                    color=cloud["sharpe"],
                                    colorscale="Plasma",
                                    opacity=0.7,
                                    colorbar=dict(title="Sharpe"),
                                ),
                                name="Portfolios",
                            )
                        ]
                    )
                    style_plotly(fig, title="Portfolio Cloud (3D)", height=560)
                    fig.update_layout(
                        scene=dict(
                            xaxis_title="Volatility",
                            yaxis_title="Expected Return",
                            zaxis_title="Sharpe",
                            xaxis=dict(showbackground=True, backgroundcolor="rgba(10,12,20,0.8)", gridcolor="rgba(255,255,255,0.08)"),
                            yaxis=dict(showbackground=True, backgroundcolor="rgba(10,12,20,0.8)", gridcolor="rgba(255,255,255,0.08)"),
                            zaxis=dict(showbackground=True, backgroundcolor="rgba(10,12,20,0.8)", gridcolor="rgba(255,255,255,0.08)"),
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Portfolio optimization failed: {exc}")


with tabs[7]:
    st.subheader("Raw Data Vault")
    st.caption("Use this tab to inspect exactly what was extracted from source.")

    vault_options = {
        "info (flattened)": pd.DataFrame(
            [{"key": k, "value": v} for k, v in sorted(info.items(), key=lambda x: x[0])]
        ),
        "history": bundle.history,
        "actions": bundle.actions,
        "financials": bundle.financials,
        "balance_sheet": bundle.balance_sheet,
        "cashflow": bundle.cashflow,
        "quarterly_financials": bundle.quarterly_financials,
        "quarterly_balance_sheet": bundle.quarterly_balance_sheet,
        "quarterly_cashflow": bundle.quarterly_cashflow,
        "major_holders": bundle.major_holders,
        "institutional_holders": bundle.institutional_holders,
        "mutualfund_holders": bundle.mutualfund_holders,
        "insider_transactions": bundle.insider_transactions,
        "insider_roster_holders": bundle.insider_roster_holders,
        "recommendations": bundle.recommendations,
        "upgrades_downgrades": bundle.upgrades_downgrades,
        "earnings_dates": bundle.earnings_dates,
    }

    selection = st.selectbox("Dataset", list(vault_options.keys()))
    selected_df = vault_options[selection]
    if isinstance(selected_df, pd.Series):
        selected_df = selected_df.to_frame(name="value")
    show_table(selected_df, max_rows=max_table_rows)


st.divider()
trend_view = (
    "stable"
    if abs(metrics.get("revenue_cagr", 0) or 0) < 0.03
    else "growing"
    if (metrics.get("revenue_cagr", 0) or 0) > 0
    else "contracting"
)

risk = calculate_risk_metrics(close, confidence=0.95, horizon_days=1)
summary = (
    f"{ticker} screens as {trend_view}; health score {health_score:.1f}/100, "
    f"ROE {fmt_pct(metrics.get('roe'))}, debt/equity {fmt_num(metrics.get('debt_to_equity'), 2)}, "
    f"annualized volatility {fmt_pct(risk.get('annualized_volatility', np.nan))}."
)
st.markdown("**Analyst Summary**")
st.write(summary)
