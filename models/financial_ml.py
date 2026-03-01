from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from utils.indicators import add_indicators


def build_market_feature_frame(close: pd.Series) -> pd.DataFrame:
    df = add_indicators(close)
    df["momentum_30"] = close.pct_change(30)
    df["momentum_90"] = close.pct_change(90)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["ma_cross"] = (close.rolling(20).mean() - close.rolling(50).mean())

    df["target_reg"] = close.shift(-1)
    df["target_cls"] = (close.shift(-1) > close).astype(int)
    return df.dropna().copy()


def run_ml_models(close: pd.Series) -> Dict:
    df = build_market_feature_frame(close)
    if len(df) < 150:
        return {
            "error": "Not enough data for stable ML training. Increase history window.",
        }

    feature_cols = [c for c in df.columns if c not in ["target_reg", "target_cls"]]
    X = df[feature_cols]
    y_reg = df["target_reg"]
    y_cls = df["target_cls"]

    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    yr_train, yr_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]
    yc_train, yc_test = y_cls.iloc[:split_idx], y_cls.iloc[split_idx:]

    reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, yr_train)
    reg_pred = reg.predict(X_test)

    if yc_train.nunique() < 2:
        return {
            "error": "Classification target has only one class in training window.",
        }

    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=3000, solver="lbfgs")),
        ]
    )
    clf.fit(X_train, yc_train)
    cls_pred = clf.predict(X_test)

    next_features = X.tail(1)
    next_price = float(reg.predict(next_features)[0])
    up_prob = float(clf.predict_proba(next_features)[0, 1])

    return {
        "y_test": yr_test,
        "reg_pred": reg_pred,
        "rmse": float(np.sqrt(mean_squared_error(yr_test, reg_pred))),
        "mae": float(mean_absolute_error(yr_test, reg_pred)),
        "next_price": next_price,
        "cls_accuracy": float(accuracy_score(yc_test, cls_pred)),
        "up_probability": up_prob,
    }


def forecast_statement_series(series: pd.Series, periods=4) -> pd.DataFrame:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) < 4:
        return pd.DataFrame()

    s = s.sort_index()
    x = np.arange(len(s)).reshape(-1, 1)
    y = s.values

    model = LinearRegression()
    model.fit(x, y)

    x_future = np.arange(len(s), len(s) + periods).reshape(-1, 1)
    preds = model.predict(x_future)

    out = pd.DataFrame(
        {
            "period": [f"t+{i+1}" for i in range(periods)],
            "forecast": preds,
        }
    )
    return out
