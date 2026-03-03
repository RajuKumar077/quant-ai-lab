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


def _safe_call(obj, fn_name: str, *args, **kwargs):
    fn = _safe_attr(obj, fn_name)
    if callable(fn):
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None
    return None


def _dict_like_to_dict(value) -> Dict:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    try:
        return dict(value)
    except Exception:
        return {}


def _merge_non_empty(base: Dict, extra: Dict) -> Dict:
    out = dict(base or {})
    for k, v in (extra or {}).items():
        if (k not in out) or pd.isna(out.get(k)):
            out[k] = v
    return out


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

    info = _dict_like_to_dict(_safe_attr(t, "info"))
    if not info:
        info = _dict_like_to_dict(_safe_call(t, "get_info"))
    fast_info = _dict_like_to_dict(_safe_attr(t, "fast_info"))
    if not fast_info:
        fast_info = _dict_like_to_dict(_safe_call(t, "get_fast_info"))
    info = _merge_non_empty(info, fast_info)

    actions = _empty_df(_safe_attr(t, "actions"))
    dividends = _empty_series(_safe_attr(t, "dividends"))
    splits = _empty_series(_safe_attr(t, "splits"))
    financials = _empty_df(_safe_attr(t, "financials"))
    balance_sheet = _empty_df(_safe_attr(t, "balance_sheet"))
    cashflow = _empty_df(_safe_attr(t, "cashflow"))
    quarterly_financials = _empty_df(_safe_attr(t, "quarterly_financials"))
    quarterly_balance_sheet = _empty_df(_safe_attr(t, "quarterly_balance_sheet"))
    quarterly_cashflow = _empty_df(_safe_attr(t, "quarterly_cashflow"))
    earnings_dates = _empty_df(_safe_attr(t, "earnings_dates"))

    # Method fallbacks for environments where property endpoints fail.
    if actions.empty:
        actions = _empty_df(_safe_call(t, "get_actions"))
    if dividends.empty:
        dividends = _empty_series(_safe_call(t, "get_dividends"))
    if splits.empty:
        splits = _empty_series(_safe_call(t, "get_splits"))
    if financials.empty:
        financials = _empty_df(_safe_call(t, "get_income_stmt"))
    if balance_sheet.empty:
        balance_sheet = _empty_df(_safe_call(t, "get_balance_sheet"))
    if cashflow.empty:
        cashflow = _empty_df(_safe_call(t, "get_cashflow"))
    if quarterly_financials.empty:
        quarterly_financials = _empty_df(_safe_call(t, "get_income_stmt", freq="quarterly"))
    if quarterly_balance_sheet.empty:
        quarterly_balance_sheet = _empty_df(_safe_call(t, "get_balance_sheet", freq="quarterly"))
    if quarterly_cashflow.empty:
        quarterly_cashflow = _empty_df(_safe_call(t, "get_cashflow", freq="quarterly"))
    if earnings_dates.empty:
        earnings_dates = _empty_df(_safe_call(t, "get_earnings_dates", limit=24))

    return TickerBundle(
        ticker=ticker,
        info=info,
        history=history,
        actions=actions,
        dividends=dividends,
        splits=splits,
        financials=financials,
        balance_sheet=balance_sheet,
        cashflow=cashflow,
        quarterly_financials=quarterly_financials,
        quarterly_balance_sheet=quarterly_balance_sheet,
        quarterly_cashflow=quarterly_cashflow,
        major_holders=_empty_df(_safe_attr(t, "major_holders")),
        institutional_holders=_empty_df(_safe_attr(t, "institutional_holders")),
        mutualfund_holders=_empty_df(_safe_attr(t, "mutualfund_holders")),
        insider_transactions=_empty_df(_safe_attr(t, "insider_transactions")),
        insider_roster_holders=_empty_df(_safe_attr(t, "insider_roster_holders")),
        recommendations=_empty_df(_safe_attr(t, "recommendations")),
        upgrades_downgrades=_empty_df(_safe_attr(t, "upgrades_downgrades")),
        earnings_dates=earnings_dates,
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


def compute_health_score_details(metrics: Dict[str, float]) -> Dict[str, float]:
    """Financial health framework with category-level transparency."""

    def bounded(value: float, low: float, high: float, inverse: bool = False) -> float:
        if pd.isna(value):
            return 0.5
        x = (value - low) / (high - low)
        x = float(np.clip(x, 0, 1))
        return 1.0 - x if inverse else x

    revenue = metrics.get("revenue", np.nan)
    fcf = metrics.get("free_cash_flow", np.nan)
    fcf_margin = fcf / revenue if pd.notna(fcf) and pd.notna(revenue) and revenue not in (0, 0.0) else np.nan

    category_scores = {
        "profitability": float(
            np.mean(
                [
                    bounded(metrics.get("gross_margin", np.nan), 0.18, 0.58),
                    bounded(metrics.get("operating_margin", np.nan), 0.06, 0.34),
                    bounded(metrics.get("net_margin", np.nan), 0.04, 0.28),
                    bounded(metrics.get("roe", np.nan), 0.08, 0.45),
                    bounded(metrics.get("roic", np.nan), 0.06, 0.32),
                ]
            )
        ),
        "growth": float(
            np.mean(
                [
                    bounded(metrics.get("revenue_cagr", np.nan), -0.01, 0.16),
                    bounded(metrics.get("net_income_cagr", np.nan), -0.03, 0.20),
                    bounded(metrics.get("fcf_cagr", np.nan), -0.03, 0.20),
                    bounded(metrics.get("eps_growth_proxy", np.nan), -0.08, 0.22),
                ]
            )
        ),
        "solvency_liquidity": float(
            np.mean(
                [
                    bounded(metrics.get("debt_to_assets", np.nan), 0.12, 0.72, inverse=True),
                    bounded(metrics.get("debt_to_equity", np.nan), 0.10, 4.00, inverse=True),
                    bounded(metrics.get("current_ratio", np.nan), 0.75, 2.20),
                    bounded(metrics.get("quick_ratio", np.nan), 0.65, 2.00),
                    bounded(metrics.get("cash_ratio", np.nan), 0.12, 1.00),
                ]
            )
        ),
        "cash_efficiency": float(
            np.mean(
                [
                    bounded(fcf_margin, 0.02, 0.32),
                    bounded(metrics.get("asset_turnover", np.nan), 0.25, 1.40),
                    bounded(1.0 if pd.notna(fcf) and fcf > 0 else 0.0, 0.0, 1.0),
                ]
            )
        ),
    }

    category_weights = {
        "profitability": 0.35,
        "growth": 0.20,
        "solvency_liquidity": 0.25,
        "cash_efficiency": 0.20,
    }

    weighted = 0.0
    weight_sum = 0.0
    for k, w in category_weights.items():
        v = category_scores.get(k, np.nan)
        if pd.notna(v):
            weighted += w * float(v)
            weight_sum += w
    base_score = (weighted / weight_sum) if weight_sum > 0 else 0.5

    bonus = 0.0
    if (
        pd.notna(metrics.get("roe", np.nan))
        and metrics.get("roe", 0.0) > 0.25
        and pd.notna(fcf_margin)
        and fcf_margin > 0.14
        and pd.notna(metrics.get("debt_to_assets", np.nan))
        and metrics.get("debt_to_assets", 1.0) < 0.55
    ):
        bonus += 0.05

    if (
        pd.notna(metrics.get("debt_to_assets", np.nan))
        and metrics.get("debt_to_assets", 0.0) > 0.80
        and pd.notna(metrics.get("current_ratio", np.nan))
        and metrics.get("current_ratio", 2.0) < 0.80
    ):
        bonus -= 0.08

    final_score = float(np.clip((base_score + bonus) * 100, 0, 100))
    coverage = float(
        np.mean(
            [
                pd.notna(metrics.get("gross_margin", np.nan)),
                pd.notna(metrics.get("operating_margin", np.nan)),
                pd.notna(metrics.get("net_margin", np.nan)),
                pd.notna(metrics.get("roe", np.nan)),
                pd.notna(metrics.get("roic", np.nan)),
                pd.notna(metrics.get("revenue_cagr", np.nan)),
                pd.notna(metrics.get("net_income_cagr", np.nan)),
                pd.notna(metrics.get("fcf_cagr", np.nan)),
                pd.notna(metrics.get("debt_to_assets", np.nan)),
                pd.notna(metrics.get("debt_to_equity", np.nan)),
                pd.notna(metrics.get("current_ratio", np.nan)),
                pd.notna(metrics.get("quick_ratio", np.nan)),
                pd.notna(metrics.get("cash_ratio", np.nan)),
                pd.notna(fcf_margin),
            ]
        )
    )

    return {
        "score": final_score,
        "coverage": coverage,
        "fcf_margin": float(fcf_margin) if pd.notna(fcf_margin) else np.nan,
        "profitability_score": float(category_scores["profitability"] * 100),
        "growth_score": float(category_scores["growth"] * 100),
        "solvency_liquidity_score": float(category_scores["solvency_liquidity"] * 100),
        "cash_efficiency_score": float(category_scores["cash_efficiency"] * 100),
    }


def compute_health_score(metrics: Dict[str, float]) -> float:
    return float(compute_health_score_details(metrics)["score"])
