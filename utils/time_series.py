from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller


def analyze_time_series(close: pd.Series) -> Dict:
    returns = close.pct_change().dropna()
    if len(returns) < 60:
        return {"error": "Not enough observations for time-series diagnostics."}

    adf_stat, adf_pvalue, _, _, _, _ = adfuller(returns)
    acf_vals = acf(returns, nlags=10)

    return {
        "returns": returns,
        "adf_stat": float(adf_stat),
        "adf_pvalue": float(adf_pvalue),
        "acf_lags": list(range(len(acf_vals))),
        "acf_values": acf_vals.tolist(),
    }


def arima_forecast(close: pd.Series, steps=30, order=(1, 1, 1)) -> pd.DataFrame:
    s = pd.to_numeric(close, errors="coerce").dropna()
    if len(s) < 120:
        return pd.DataFrame()

    # Use a dense integer index to avoid unsupported datetime frequency warnings.
    s = s.reset_index(drop=True)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            model = ARIMA(s, order=order)
            fit = model.fit()
        fc = fit.forecast(steps=steps)
    except Exception:
        return pd.DataFrame()

    future_idx = pd.RangeIndex(start=1, stop=steps + 1)
    out = pd.DataFrame({"step": future_idx, "forecast": fc.values})
    return out
