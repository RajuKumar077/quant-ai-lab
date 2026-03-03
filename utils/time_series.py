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

    returns = np.log(s).diff().dropna()
    if len(returns) < 100:
        return pd.DataFrame()

    # Use integer index to avoid unsupported date frequency warnings.
    r = returns.reset_index(drop=True)

    candidate_orders = [(1, 0, 1), (1, 0, 0), (2, 0, 1), (2, 0, 2), (3, 0, 1)]
    if order is not None:
        candidate_orders = [order] + [o for o in candidate_orders if o != order]

    best_fit = None
    best_order = None
    best_aic = np.inf
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for ord_i in candidate_orders:
                try:
                    model = ARIMA(r, order=ord_i)
                    fit = model.fit()
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_fit = fit
                        best_order = ord_i
                except Exception:
                    continue
    except Exception:
        return pd.DataFrame()

    if best_fit is None:
        return pd.DataFrame()

    pred = best_fit.get_forecast(steps=steps)
    ret_fc = pd.Series(pred.predicted_mean)
    ci = pred.conf_int(alpha=0.20)  # 80% confidence band for readability

    if isinstance(ci, pd.DataFrame) and ci.shape[1] >= 2:
        lower_ret = pd.to_numeric(ci.iloc[:, 0], errors="coerce")
        upper_ret = pd.to_numeric(ci.iloc[:, 1], errors="coerce")
    else:
        std = float(np.nanstd(r.tail(60)))
        lower_ret = ret_fc - 1.28 * std
        upper_ret = ret_fc + 1.28 * std

    last_price = float(s.iloc[-1])
    cum_mean = ret_fc.cumsum()
    cum_low = lower_ret.cumsum()
    cum_up = upper_ret.cumsum()

    fc_price = last_price * np.exp(cum_mean)
    low_price = last_price * np.exp(cum_low)
    up_price = last_price * np.exp(cum_up)

    out = pd.DataFrame(
        {
            "step": pd.RangeIndex(start=1, stop=steps + 1),
            "forecast": fc_price.values,
            "lower_ci": low_price.values,
            "upper_ci": up_price.values,
            "mean_return": ret_fc.values,
            "model_order": [str(best_order)] * steps,
        }
    )
    return out
