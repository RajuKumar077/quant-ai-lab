from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_risk_metrics(close, confidence=0.95, horizon_days=1) -> Dict[str, float]:
    """Compute VaR/CVaR plus return-risk ratios for a price series."""
    returns = close.pct_change().dropna()
    if returns.empty:
        return {
            "historical_var": 0.0,
            "historical_cvar": 0.0,
            "parametric_var": 0.0,
            "parametric_cvar": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
        }

    alpha = 1 - confidence
    horizon_scale = np.sqrt(horizon_days)

    var_cutoff = np.percentile(returns, alpha * 100)
    hist_var = -var_cutoff * horizon_scale

    tail = returns[returns <= var_cutoff]
    hist_cvar = -tail.mean() * horizon_scale if len(tail) else hist_var

    mu = returns.mean() * horizon_days
    sigma = returns.std() * horizon_scale
    z = norm.ppf(alpha)
    param_var = -(mu + sigma * z)
    param_cvar = -(mu - sigma * norm.pdf(z) / alpha)

    ann_return = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    downside = returns[returns < 0]
    downside_std = downside.std() * np.sqrt(252) if not downside.empty else np.nan
    sharpe = ann_return / ann_vol if ann_vol and ann_vol > 0 else np.nan
    sortino = ann_return / downside_std if downside_std and downside_std > 0 else np.nan

    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = cumulative / rolling_max - 1
    max_dd = drawdown.min()

    return {
        "historical_var": float(max(hist_var, 0.0)),
        "historical_cvar": float(max(hist_cvar, 0.0)),
        "parametric_var": float(max(param_var, 0.0)),
        "parametric_cvar": float(max(param_cvar, 0.0)),
        "annualized_volatility": float(max(ann_vol, 0.0)),
        "sharpe_ratio": float(sharpe) if pd.notna(sharpe) else 0.0,
        "sortino_ratio": float(sortino) if pd.notna(sortino) else 0.0,
        "max_drawdown": float(abs(max_dd)) if pd.notna(max_dd) else 0.0,
    }


def return_series_stats(close, window=30) -> pd.DataFrame:
    returns = close.pct_change().dropna()
    if returns.empty:
        return pd.DataFrame()

    df = pd.DataFrame(index=returns.index)
    df["returns"] = returns
    df["log_returns"] = np.log1p(returns)
    df["rolling_mean"] = returns.rolling(window).mean()
    df["rolling_vol"] = returns.rolling(window).std() * np.sqrt(252)
    return df
