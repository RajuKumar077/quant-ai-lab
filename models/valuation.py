from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def _extract_fcf_series(cashflow: pd.DataFrame) -> pd.Series:
    if cashflow.empty:
        return pd.Series(dtype=float)

    for key in ["Free Cash Flow"]:
        if key in cashflow.index:
            s = pd.to_numeric(cashflow.loc[key], errors="coerce").dropna()
            return s.sort_index()

    if (
        "Operating Cash Flow" in cashflow.index
        or "Cash Flow From Continuing Operating Activities" in cashflow.index
    ) and ("Capital Expenditure" in cashflow.index or "Capital Expenditures" in cashflow.index):
        cfo_key = (
            "Operating Cash Flow"
            if "Operating Cash Flow" in cashflow.index
            else "Cash Flow From Continuing Operating Activities"
        )
        capex_key = (
            "Capital Expenditure"
            if "Capital Expenditure" in cashflow.index
            else "Capital Expenditures"
        )
        cfo = pd.to_numeric(cashflow.loc[cfo_key], errors="coerce")
        capex = pd.to_numeric(cashflow.loc[capex_key], errors="coerce")
        s = (cfo + capex).dropna()
        return s.sort_index()

    return pd.Series(dtype=float)


def run_dcf_valuation(
    cashflow: pd.DataFrame,
    shares_outstanding: float,
    beta: float,
    current_price: float,
    forecast_years: int = 5,
    terminal_growth: float = 0.025,
    risk_free_rate: float = 0.045,
    equity_risk_premium: float = 0.055,
) -> Dict[str, float]:
    fcf_series = _extract_fcf_series(cashflow)
    if len(fcf_series) < 3:
        return {"error": "Not enough cash-flow history for DCF."}

    if not shares_outstanding or pd.isna(shares_outstanding) or shares_outstanding <= 0:
        return {"error": "Invalid shares outstanding for per-share valuation."}

    y = fcf_series.values.astype(float)
    x = np.arange(len(y)).reshape(-1, 1)

    lr = LinearRegression()
    lr.fit(x, y)
    future_x = np.arange(len(y), len(y) + forecast_years).reshape(-1, 1)
    forecast = lr.predict(future_x)
    forecast = np.maximum(forecast, 0)

    discount_rate = risk_free_rate + (beta if pd.notna(beta) else 1.0) * equity_risk_premium
    discount_rate = max(discount_rate, terminal_growth + 0.01)

    discounts = np.array([(1 + discount_rate) ** (i + 1) for i in range(forecast_years)])
    pv_fcf = (forecast / discounts).sum()

    terminal_value = forecast[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** forecast_years)

    equity_value = pv_fcf + pv_terminal
    intrinsic_per_share = equity_value / shares_outstanding
    undervaluation = (intrinsic_per_share - current_price) / current_price if current_price > 0 else np.nan

    return {
        "discount_rate": float(discount_rate),
        "terminal_growth": float(terminal_growth),
        "equity_value": float(equity_value),
        "intrinsic_per_share": float(intrinsic_per_share),
        "current_price": float(current_price),
        "undervaluation_pct": float(undervaluation),
    }
