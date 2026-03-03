from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


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

    s = pd.to_numeric(fcf_series, errors="coerce").dropna().sort_index().astype(float)
    if len(s) < 3:
        return {"error": "Not enough valid cash-flow observations for DCF."}

    positive_share = float((s > 0).mean())
    last_fcf = float(s.iloc[-1])
    if last_fcf <= 0:
        return {"error": "Latest free cash flow is non-positive; DCF is unreliable."}

    # Normalize the starting FCF to reduce one-year noise.
    tail = s.tail(3).values
    if len(tail) >= 3:
        last_fcf_norm = float(np.average(tail, weights=np.array([0.2, 0.3, 0.5])))
    else:
        last_fcf_norm = float(np.mean(tail))
    last_fcf_norm = float(np.clip(last_fcf_norm, 0.60 * last_fcf, 1.40 * last_fcf))

    growth_hist = s.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    if growth_hist.empty:
        base_growth = 0.04
        growth_vol = 0.08
    else:
        trimmed = growth_hist.clip(lower=growth_hist.quantile(0.1), upper=growth_hist.quantile(0.9))
        base_growth = float(np.nanmedian(trimmed))
        growth_vol = float(np.nanstd(trimmed))

    # Guardrails prevent extreme terminal distortions on large-cap companies.
    base_growth = float(np.clip(base_growth, -0.03, 0.18))
    growth_vol = float(np.clip(growth_vol, 0.01, 0.20))

    beta_eff = float(beta) if pd.notna(beta) else 1.0
    beta_eff = float(np.clip(beta_eff, 0.55, 1.60))
    cost_of_equity = risk_free_rate + beta_eff * equity_risk_premium
    mature_anchor = risk_free_rate + 0.75 * equity_risk_premium
    # Blend CAPM with a mature-company anchor to avoid overly punitive discount rates.
    discount_rate = 0.65 * cost_of_equity + 0.35 * mature_anchor
    discount_rate = float(np.clip(discount_rate, terminal_growth + 0.0125, 0.14))
    terminal_growth = float(np.clip(terminal_growth, 0.01, min(0.04, discount_rate - 0.0125)))

    def forecast_cashflows(start_fcf: float, g_start: float, years: int):
        vals = []
        fcf_t = start_fcf
        for yr in range(1, years + 1):
            # Fade growth toward terminal assumptions through the explicit forecast horizon.
            decay = yr / max(years, 1)
            g_t = (1 - decay) * g_start + decay * terminal_growth
            g_t = float(np.clip(g_t, -0.08, 0.20))
            fcf_t = max(fcf_t * (1 + g_t), 0.0)
            vals.append(fcf_t)
        return np.array(vals, dtype=float)

    bear_growth = base_growth - 0.75 * growth_vol
    bull_growth = base_growth + 0.75 * growth_vol
    scenario_growth = {
        "bear": float(np.clip(bear_growth, -0.06, 0.12)),
        "base": base_growth,
        "bull": float(np.clip(bull_growth, -0.02, 0.22)),
    }

    scenario_values = {}
    for name, g0 in scenario_growth.items():
        fc = forecast_cashflows(last_fcf_norm, g0, forecast_years)
        discounts = np.array([(1 + discount_rate) ** i for i in range(1, forecast_years + 1)], dtype=float)
        pv_fcf = float((fc / discounts).sum())
        terminal_value = float(fc[-1] * (1 + terminal_growth) / max(discount_rate - terminal_growth, 1e-6))
        pv_terminal = float(terminal_value / ((1 + discount_rate) ** forecast_years))
        scenario_values[name] = pv_fcf + pv_terminal

    equity_value = (
        0.20 * scenario_values["bear"]
        + 0.60 * scenario_values["base"]
        + 0.20 * scenario_values["bull"]
    )
    intrinsic_per_share = equity_value / shares_outstanding
    undervaluation = (intrinsic_per_share - current_price) / current_price if current_price > 0 else np.nan

    # Compact valuation range using small discount/terminal perturbations.
    sensitivity_cases = [(discount_rate + 0.01, terminal_growth - 0.003), (discount_rate - 0.01, terminal_growth + 0.003)]
    sensitivity_values = []
    for dr, tg in sensitivity_cases:
        dr_eff = float(np.clip(dr, tg + 0.01, 0.16))
        tg_eff = float(np.clip(tg, 0.005, dr_eff - 0.01))
        fc = forecast_cashflows(last_fcf_norm, scenario_growth["base"], forecast_years)
        discounts = np.array([(1 + dr_eff) ** i for i in range(1, forecast_years + 1)], dtype=float)
        pv_fcf = float((fc / discounts).sum())
        terminal_value = float(fc[-1] * (1 + tg_eff) / max(dr_eff - tg_eff, 1e-6))
        pv_terminal = float(terminal_value / ((1 + dr_eff) ** forecast_years))
        sensitivity_values.append((pv_fcf + pv_terminal) / shares_outstanding)

    intrinsic_low = float(min(sensitivity_values[0], sensitivity_values[1]))
    intrinsic_high = float(max(sensitivity_values[0], sensitivity_values[1]))

    # Confidence reflects historical FCF quality and stability.
    stability = 1.0 - float(np.clip(growth_vol / 0.20, 0, 1))
    history_coverage = float(np.clip(len(s) / 8.0, 0, 1))
    confidence = float(np.clip(0.45 * positive_share + 0.35 * stability + 0.20 * history_coverage, 0, 1))

    return {
        "discount_rate": float(discount_rate),
        "terminal_growth": float(terminal_growth),
        "equity_value": float(equity_value),
        "equity_value_bear": float(scenario_values["bear"]),
        "equity_value_base": float(scenario_values["base"]),
        "equity_value_bull": float(scenario_values["bull"]),
        "cost_of_equity": float(cost_of_equity),
        "base_fcf_growth": float(base_growth),
        "fcf_growth_vol": float(growth_vol),
        "fcf_last": float(last_fcf),
        "fcf_start_normalized": float(last_fcf_norm),
        "forecast_years": int(forecast_years),
        "scenario_growth_bear": float(scenario_growth["bear"]),
        "scenario_growth_base": float(scenario_growth["base"]),
        "scenario_growth_bull": float(scenario_growth["bull"]),
        "dcf_confidence": confidence,
        "intrinsic_per_share": float(intrinsic_per_share),
        "intrinsic_per_share_low": intrinsic_low,
        "intrinsic_per_share_high": intrinsic_high,
        "intrinsic_per_share_bear": float(scenario_values["bear"] / shares_outstanding),
        "intrinsic_per_share_base": float(scenario_values["base"] / shares_outstanding),
        "intrinsic_per_share_bull": float(scenario_values["bull"] / shares_outstanding),
        "current_price": float(current_price),
        "undervaluation_pct": float(undervaluation),
        "fcf_observations": int(len(s)),
    }
