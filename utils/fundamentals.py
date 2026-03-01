from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yfinance as yf


@dataclass
class TickerBundle:
    ticker: str
    info: Dict
    history: pd.DataFrame
    actions: pd.DataFrame
    dividends: pd.Series
    splits: pd.Series
    financials: pd.DataFrame
    balance_sheet: pd.DataFrame
    cashflow: pd.DataFrame
    quarterly_financials: pd.DataFrame
    quarterly_balance_sheet: pd.DataFrame
    quarterly_cashflow: pd.DataFrame
    major_holders: pd.DataFrame
    institutional_holders: pd.DataFrame
    mutualfund_holders: pd.DataFrame
    insider_transactions: pd.DataFrame
    insider_roster_holders: pd.DataFrame
    recommendations: pd.DataFrame
    upgrades_downgrades: pd.DataFrame
    earnings_dates: pd.DataFrame


def _empty_df(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df.copy()
    return pd.DataFrame()


def _empty_series(series: Optional[pd.Series]) -> pd.Series:
    if isinstance(series, pd.Series):
        return series.copy()
    return pd.Series(dtype=float)


def _safe_attr(obj, attr_name: str):
    try:
        return getattr(obj, attr_name)
    except Exception:
        return None


def _latest_value(df: pd.DataFrame, keys: Iterable[str]) -> float:
    if df.empty or df.shape[1] == 0:
        return np.nan
    for key in keys:
        if key in df.index:
            series = pd.to_numeric(df.loc[key], errors="coerce").dropna()
            if not series.empty:
                return float(series.iloc[0])
    return np.nan


def _series_for_growth(df: pd.DataFrame, keys: Iterable[str]) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    for key in keys:
        if key in df.index:
            s = pd.to_numeric(df.loc[key], errors="coerce").dropna()
            if not s.empty:
                s = s.sort_index()
                return s
    return pd.Series(dtype=float)


def _safe_div(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b) or b == 0:
        return np.nan
    return float(a / b)


def _cagr(series: pd.Series) -> float:
    if series.empty or len(series) < 2:
        return np.nan
    start = float(series.iloc[0])
    end = float(series.iloc[-1])
    n = len(series) - 1
    if start <= 0 or end <= 0 or n <= 0:
        return np.nan
    return float((end / start) ** (1 / n) - 1)


def fetch_ticker_bundle(ticker: str, period: str = "10y") -> TickerBundle:
    t = yf.Ticker(ticker)

    history = t.history(period=period, auto_adjust=True)
    history = history.dropna(how="all") if isinstance(history, pd.DataFrame) and not history.empty else pd.DataFrame()

    info = _safe_attr(t, "info") or {}

    return TickerBundle(
        ticker=ticker,
        info=info,
        history=history,
        actions=_empty_df(_safe_attr(t, "actions")),
        dividends=_empty_series(_safe_attr(t, "dividends")),
        splits=_empty_series(_safe_attr(t, "splits")),
        financials=_empty_df(_safe_attr(t, "financials")),
        balance_sheet=_empty_df(_safe_attr(t, "balance_sheet")),
        cashflow=_empty_df(_safe_attr(t, "cashflow")),
        quarterly_financials=_empty_df(_safe_attr(t, "quarterly_financials")),
        quarterly_balance_sheet=_empty_df(_safe_attr(t, "quarterly_balance_sheet")),
        quarterly_cashflow=_empty_df(_safe_attr(t, "quarterly_cashflow")),
        major_holders=_empty_df(_safe_attr(t, "major_holders")),
        institutional_holders=_empty_df(_safe_attr(t, "institutional_holders")),
        mutualfund_holders=_empty_df(_safe_attr(t, "mutualfund_holders")),
        insider_transactions=_empty_df(_safe_attr(t, "insider_transactions")),
        insider_roster_holders=_empty_df(_safe_attr(t, "insider_roster_holders")),
        recommendations=_empty_df(_safe_attr(t, "recommendations")),
        upgrades_downgrades=_empty_df(_safe_attr(t, "upgrades_downgrades")),
        earnings_dates=_empty_df(_safe_attr(t, "earnings_dates")),
    )


def compute_financial_metrics(bundle: TickerBundle) -> Dict[str, float]:
    fin = bundle.financials
    bal = bundle.balance_sheet
    cf = bundle.cashflow
    info = bundle.info

    revenue = _latest_value(fin, ["Total Revenue", "Revenue"])
    gross_profit = _latest_value(fin, ["Gross Profit"])
    operating_income = _latest_value(fin, ["Operating Income", "EBIT"])
    net_income = _latest_value(fin, ["Net Income", "Net Income Common Stockholders"])
    ebitda = _latest_value(fin, ["EBITDA"])

    total_assets = _latest_value(bal, ["Total Assets"])
    total_liabilities = _latest_value(bal, ["Total Liabilities Net Minority Interest", "Total Liab"])
    equity = _latest_value(bal, ["Stockholders Equity", "Total Stockholder Equity"])
    cash = _latest_value(bal, ["Cash And Cash Equivalents", "Cash"])
    debt = _latest_value(
        bal,
        [
            "Total Debt",
            "Long Term Debt",
            "Long Term Debt And Capital Lease Obligation",
        ],
    )

    current_assets = _latest_value(bal, ["Current Assets", "Total Current Assets"])
    current_liabilities = _latest_value(
        bal, ["Current Liabilities", "Total Current Liabilities"]
    )
    inventory = _latest_value(bal, ["Inventory"])
    receivables = _latest_value(bal, ["Accounts Receivable"])

    cfo = _latest_value(
        cf,
        ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    )
    capex = _latest_value(cf, ["Capital Expenditure", "Capital Expenditures"])
    fcf = _latest_value(cf, ["Free Cash Flow"])
    if pd.isna(fcf) and not pd.isna(cfo) and not pd.isna(capex):
        fcf = cfo + capex

    eps = info.get("trailingEps", np.nan)
    shares_outstanding = info.get("sharesOutstanding", np.nan)
    market_cap = info.get("marketCap", np.nan)

    metrics = {
        "revenue": revenue,
        "gross_profit": gross_profit,
        "operating_income": operating_income,
        "net_income": net_income,
        "ebitda": ebitda,
        "total_assets": total_assets,
        "total_liabilities": total_liabilities,
        "equity": equity,
        "cash": cash,
        "debt": debt,
        "operating_cash_flow": cfo,
        "free_cash_flow": fcf,
        "eps": eps,
        "market_cap": market_cap,
        "shares_outstanding": shares_outstanding,
    }

    ratios = {
        "gross_margin": _safe_div(gross_profit, revenue),
        "operating_margin": _safe_div(operating_income, revenue),
        "net_margin": _safe_div(net_income, revenue),
        "roe": _safe_div(net_income, equity),
        "roa": _safe_div(net_income, total_assets),
        "roic": _safe_div(operating_income, equity + debt - cash),
        "current_ratio": _safe_div(current_assets, current_liabilities),
        "quick_ratio": _safe_div(current_assets - inventory, current_liabilities)
        if not pd.isna(current_assets)
        else np.nan,
        "cash_ratio": _safe_div(cash, current_liabilities),
        "debt_to_equity": _safe_div(debt, equity),
        "debt_to_assets": _safe_div(debt, total_assets),
        "asset_turnover": _safe_div(revenue, total_assets),
        "inventory_turnover": _safe_div(revenue, inventory),
        "receivable_turnover": _safe_div(revenue, receivables),
    }

    pe = info.get("trailingPE", np.nan)
    forward_pe = info.get("forwardPE", np.nan)
    peg = info.get("pegRatio", np.nan)
    price_to_book = info.get("priceToBook", np.nan)
    enterprise_value = info.get("enterpriseValue", np.nan)
    ev_to_ebitda = _safe_div(enterprise_value, ebitda)

    valuation = {
        "pe": pe,
        "forward_pe": forward_pe,
        "peg": peg,
        "price_to_book": price_to_book,
        "ev_to_ebitda": ev_to_ebitda,
    }

    revenue_series = _series_for_growth(fin, ["Total Revenue", "Revenue"])
    net_income_series = _series_for_growth(fin, ["Net Income", "Net Income Common Stockholders"])
    fcf_series = _series_for_growth(cf, ["Free Cash Flow"])
    if fcf_series.empty:
        cfo_series = _series_for_growth(
            cf, ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"]
        )
        capex_series = _series_for_growth(cf, ["Capital Expenditure", "Capital Expenditures"])
        if not cfo_series.empty and not capex_series.empty:
            aligned = cfo_series.align(capex_series, join="inner")
            fcf_series = aligned[0] + aligned[1]

    growth = {
        "revenue_cagr": _cagr(revenue_series),
        "net_income_cagr": _cagr(net_income_series),
        "fcf_cagr": _cagr(fcf_series),
    }

    eps_growth = info.get("earningsQuarterlyGrowth", np.nan)
    growth["eps_growth_proxy"] = eps_growth

    return {**metrics, **ratios, **valuation, **growth}


def compute_health_score(metrics: Dict[str, float]) -> float:
    """Weighted financial strength score in range [0, 100]."""

    def bounded(value: float, low: float, high: float) -> float:
        if pd.isna(value):
            return 0.5
        x = (value - low) / (high - low)
        return float(np.clip(x, 0, 1))

    roe_s = bounded(metrics.get("roe", np.nan), 0.0, 0.25)
    rev_s = bounded(metrics.get("revenue_cagr", np.nan), -0.05, 0.2)
    debt_inv_s = 1 - bounded(metrics.get("debt_to_equity", np.nan), 0.0, 2.5)
    fcf_s = bounded(metrics.get("fcf_cagr", np.nan), -0.1, 0.2)
    margin_s = bounded(metrics.get("operating_margin", np.nan), 0.0, 0.35)

    score = (
        0.20 * roe_s
        + 0.20 * rev_s
        + 0.15 * debt_inv_s
        + 0.15 * fcf_s
        + 0.15 * margin_s
        + 0.15 * (1 - bounded(metrics.get("debt_to_assets", np.nan), 0.0, 0.8))
    )
    return float(100 * score)
