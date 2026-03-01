from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.interpolate import make_interp_spline

from models.financial_ml import forecast_statement_series, run_ml_models
from models.portfolio_optimization import optimize_portfolio
from models.valuation import run_dcf_valuation
from utils.data_loader import load_price_data
from utils.fundamentals import compute_financial_metrics, compute_health_score, fetch_ticker_bundle
from utils.risk import calculate_risk_metrics, return_series_stats
from utils.simulation import monte_carlo_paths
from utils.time_series import analyze_time_series, arima_forecast


st.set_page_config(page_title="Quant AI Lab", layout="wide")
st.title("Quant AI Lab")
st.caption("Section-wise quant research cockpit from a single ticker")


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
                background: rgba(255,255,255,0.06);
                backdrop-filter: blur(25px);
                -webkit-backdrop-filter: blur(25px);
                border-right: 1px solid rgba(255,255,255,0.14);
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


def smooth_area_figure(series: pd.Series):
    y = series.dropna().values
    fig, ax = plt.subplots(figsize=(11, 4))
    if len(y) < 4:
        x = np.arange(len(y))
        ax.plot(x, y, color="#9EC2FF", linewidth=2.8)
        ax.fill_between(x, y, np.min(y), color="#79A7FF", alpha=0.18)
    else:
        x = np.arange(len(y))
        x_new = np.linspace(x.min(), x.max(), len(y) * 4)
        spline = make_interp_spline(x, y, k=3)
        y_new = spline(x_new)
        baseline = np.min(y_new)
        ax.plot(x_new, y_new, color="#9EC2FF", linewidth=3.0)
        ax.fill_between(x_new, y_new, baseline, color="#79A7FF", alpha=0.20)
    ax.grid(alpha=0.12)
    ax.set_facecolor((0, 0, 0, 0))
    fig.patch.set_alpha(0)
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


def statement_series(df: pd.DataFrame, candidates):
    if df.empty:
        return pd.Series(dtype=float)
    for key in candidates:
        if key in df.index:
            s = pd.to_numeric(df.loc[key], errors="coerce").dropna()
            if not s.empty:
                return s.sort_index()
    return pd.Series(dtype=float)


def show_statement(df: pd.DataFrame, title: str, max_rows=30):
    st.markdown(f"**{title}**")
    if df.empty:
        st.info("No data available from source.")
        return
    table = df.copy()
    table.columns = table.columns.astype(str)
    st.dataframe(compact_table(table, max_rows=max_rows), use_container_width=True)


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

st.sidebar.header("Inputs")
ticker = st.sidebar.text_input("Primary ticker", "AAPL").upper().strip()
start_dt = st.sidebar.date_input("Price start", default_start)
end_dt = st.sidebar.date_input("Price end", today)
portfolio_text = st.sidebar.text_input("Portfolio tickers", "AAPL,MSFT,GOOGL,AMZN")
max_table_rows = st.sidebar.slider("Max table rows", 10, 200, 40, 5)

if start_dt >= end_dt:
    st.error("Start date must be before end date.")
    st.stop()

bundle = ensure_bundle_fields(get_bundle(ticker, schema_version="v3"))
metrics = compute_financial_metrics(bundle)
health_score = compute_health_score(metrics)

prices = get_prices([ticker], start_dt, end_dt)
if prices.empty or ticker not in prices.columns:
    st.error("Unable to fetch price data for selected ticker and date range.")
    st.stop()

close = prices[ticker].dropna()
returns = close.pct_change().dropna()
current_price = float(close.iloc[-1])
info = bundle.info

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
        st.pyplot(smooth_area_figure(close.tail(120)))
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
        st.dataframe(s_df, hide_index=True, use_container_width=True)


with tabs[1]:
    st.subheader("Market Intelligence")

    st.markdown("**Company Snapshot**")
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
    profile = pd.DataFrame(
        [{"Field": k, "Value": info.get(k, "N/A")} for k in profile_keys]
    )
    st.dataframe(profile, hide_index=True, use_container_width=True)

    st.markdown("**Corporate Actions**")
    a1, a2 = st.columns(2)
    if not bundle.actions.empty:
        a1.dataframe(compact_table(bundle.actions.reset_index(), max_rows=max_table_rows), use_container_width=True)
    else:
        a1.info("No actions available")

    div_df = bundle.dividends.tail(max_table_rows).rename("Dividend").to_frame()
    split_df = bundle.splits.tail(max_table_rows).rename("Split").to_frame()

    if not div_df.empty or not split_df.empty:
        merged = div_df.join(split_df, how="outer").reset_index()
        a2.dataframe(merged, use_container_width=True)
    else:
        a2.info("No dividends/splits available")

    st.markdown("**Holders and Insider Data**")
    h1, h2 = st.columns(2)
    h1.markdown("`major_holders`")
    h1.dataframe(compact_table(bundle.major_holders, max_rows=max_table_rows), use_container_width=True)
    h2.markdown("`institutional_holders`")
    h2.dataframe(compact_table(bundle.institutional_holders, max_rows=max_table_rows), use_container_width=True)

    h3, h4 = st.columns(2)
    h3.markdown("`mutualfund_holders`")
    h3.dataframe(compact_table(bundle.mutualfund_holders, max_rows=max_table_rows), use_container_width=True)
    h4.markdown("`insider_transactions`")
    h4.dataframe(compact_table(bundle.insider_transactions, max_rows=max_table_rows), use_container_width=True)

    st.markdown("**Analyst Activity**")
    r1, r2 = st.columns(2)
    r1.markdown("`recommendations`")
    r1.dataframe(compact_table(bundle.recommendations, max_rows=max_table_rows), use_container_width=True)
    r2.markdown("`upgrades_downgrades`")
    r2.dataframe(compact_table(bundle.upgrades_downgrades, max_rows=max_table_rows), use_container_width=True)

    st.markdown("**Earnings Calendar**")
    st.dataframe(compact_table(bundle.earnings_dates, max_rows=max_table_rows), use_container_width=True)


with tabs[2]:
    st.subheader("Financial Statements")

    mode = st.radio("Statement frequency", ["Annual", "Quarterly"], horizontal=True)
    if mode == "Annual":
        inc, bal, cfs = bundle.financials, bundle.balance_sheet, bundle.cashflow
    else:
        inc, bal, cfs = (
            bundle.quarterly_financials,
            bundle.quarterly_balance_sheet,
            bundle.quarterly_cashflow,
        )

    s1, s2, s3 = st.columns(3)
    with s1:
        show_statement(inc, "Income Statement", max_rows=max_table_rows)
    with s2:
        show_statement(bal, "Balance Sheet", max_rows=max_table_rows)
    with s3:
        show_statement(cfs, "Cash Flow", max_rows=max_table_rows)

    rev = statement_series(inc, ["Total Revenue", "Revenue"])
    ni = statement_series(inc, ["Net Income", "Net Income Common Stockholders"])
    fcf = statement_series(cfs, ["Free Cash Flow"])

    if not rev.empty or not ni.empty or not fcf.empty:
        growth = pd.DataFrame({"Revenue": rev, "Net Income": ni, "FCF": fcf}).sort_index()
        fig, ax = plt.subplots(figsize=(11, 4))
        for col in growth.columns:
            if growth[col].notna().any():
                ax.plot(growth.index.astype(str), growth[col], marker="o", label=col)
        ax.set_title(f"{mode} Fundamental Trend")
        ax.legend()
        ax.tick_params(axis="x", rotation=30)
        st.pyplot(fig)


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
    st.dataframe(ratio_df[["Metric", "Display"]], hide_index=True, use_container_width=True)

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
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(ts_stats.index, ts_stats["rolling_mean"], label="Rolling Mean")
        ax.plot(ts_stats.index, ts_stats["rolling_vol"], label="Rolling Volatility")
        ax.legend()
        ax.set_title("Rolling Return Stats")
        st.pyplot(fig)

    ts_diag = analyze_time_series(close)
    if "error" not in ts_diag:
        t1, t2 = st.columns(2)
        t1.metric("ADF Statistic", fmt_num(ts_diag["adf_stat"], 4))
        t2.metric("ADF p-value", fmt_num(ts_diag["adf_pvalue"], 6))

        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar(ts_diag["acf_lags"], ts_diag["acf_values"])
        ax.set_title("Autocorrelation of Returns")
        st.pyplot(fig)

    mc_paths = monte_carlo_paths(close, n_sims=mc_sims, n_days=mc_days, seed=42)
    if not mc_paths.empty:
        terminal = mc_paths.iloc[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Median terminal", fmt_money(terminal.median()))
        m2.metric("5th pct", fmt_money(terminal.quantile(0.05)))
        m3.metric("95th pct", fmt_money(terminal.quantile(0.95)))

        fig, ax = plt.subplots(figsize=(10, 3.8))
        for i in range(min(50, mc_paths.shape[1])):
            ax.plot(mc_paths.index, mc_paths[i], alpha=0.2, linewidth=0.8)
        ax.set_title("Monte Carlo Paths")
        ax.set_xlabel("Day")
        st.pyplot(fig)


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
        fig, ax = plt.subplots(figsize=(10, 3.6))
        hist = close.tail(100)
        ax.plot(range(len(hist)), hist.values, label="Recent Close")
        ax.plot(range(len(hist), len(hist) + len(arima)), arima["forecast"].values, label="ARIMA Forecast")
        ax.set_title("ARIMA Price Forecast")
        ax.legend()
        st.pyplot(fig)

    ml = run_ml_models(close)
    if "error" in ml:
        st.info(ml["error"])
    else:
        ml1, ml2, ml3, ml4 = st.columns(4)
        ml1.metric("RF RMSE", fmt_num(ml["rmse"], 4))
        ml2.metric("RF MAE", fmt_num(ml["mae"], 4))
        ml3.metric("Next Price", fmt_money(ml["next_price"]))
        ml4.metric("Up Probability", fmt_pct(ml["up_probability"]))

        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(ml["y_test"].index, ml["y_test"].values, label="Actual")
        ax.plot(ml["y_test"].index, ml["reg_pred"], label="Predicted")
        ax.legend()
        ax.set_title("ML Regression Fit")
        st.pyplot(fig)

    if not bundle.quarterly_financials.empty and "Total Revenue" in bundle.quarterly_financials.index:
        rev_fc = forecast_statement_series(bundle.quarterly_financials.loc["Total Revenue"], periods=4)
        if not rev_fc.empty:
            st.markdown("**Quarterly Revenue Forecast (Linear Regression)**")
            st.dataframe(rev_fc, hide_index=True, use_container_width=True)


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
                st.bar_chart(w)

                p1, p2, p3 = st.columns(3)
                p1.metric("Expected Return", fmt_pct(perf[0]))
                p2.metric("Volatility", fmt_pct(perf[1]))
                p3.metric("Sharpe", fmt_num(perf[2], 3))

                corr = p_prices.pct_change().dropna().corr()
                fig, ax = plt.subplots(figsize=(7, 5))
                im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
                ax.set_xticks(range(len(corr.columns)))
                ax.set_xticklabels(corr.columns, rotation=45, ha="right")
                ax.set_yticks(range(len(corr.index)))
                ax.set_yticklabels(corr.index)
                ax.set_title("Correlation Heatmap")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig)

                st.metric("Selected ticker portfolio weight", fmt_pct(w.get(ticker, 0.0)))
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
    st.dataframe(compact_table(selected_df, max_rows=max_table_rows), use_container_width=True)


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
