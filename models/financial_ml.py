from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

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

    df["target_reg"] = close.pct_change().shift(-1)
    df["target_cls"] = (close.shift(-1) > close).astype(int)
    return df.dropna().copy()


def _compute_max_drawdown(return_series: np.ndarray) -> float:
    if return_series.size == 0:
        return np.nan
    eq = np.cumprod(1.0 + return_series)
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(dd))


def _strategy_stats(next_returns: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, float]:
    if next_returns.size == 0 or probs.size == 0:
        return {
            "threshold": float(threshold),
            "annual_return": np.nan,
            "annual_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
            "turnover": np.nan,
            "participation": np.nan,
            "n_trades": 0,
        }

    positions = (probs >= threshold).astype(float)  # long/flat regime
    strat_ret = positions * next_returns
    participation = float(np.mean(positions))
    trade_events = int(np.abs(np.diff(positions)).sum()) if len(positions) > 1 else int(positions[0] > 0)
    turnover = float(trade_events / max(len(positions), 1))

    mean_r = float(np.mean(strat_ret))
    std_r = float(np.std(strat_ret))
    annual_return = float((1 + mean_r) ** 252 - 1) if np.isfinite(mean_r) else np.nan
    annual_vol = float(std_r * np.sqrt(252)) if np.isfinite(std_r) else np.nan
    sharpe = float((mean_r / max(std_r, 1e-9)) * np.sqrt(252)) if np.isfinite(mean_r) else np.nan
    max_dd = _compute_max_drawdown(strat_ret)

    active = positions > 0
    hit_rate = float(np.mean(next_returns[active] > 0)) if active.any() else np.nan

    return {
        "threshold": float(threshold),
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "turnover": turnover,
        "participation": participation,
        "n_trades": trade_events,
    }


def _optimize_threshold(next_returns: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    grid = np.array([0.52, 0.55, 0.58, 0.60, 0.62, 0.65], dtype=float)
    best = None
    best_score = -np.inf
    for thr in grid:
        st = _strategy_stats(next_returns, probs, float(thr))
        sharpe = float(np.nan_to_num(st["sharpe"], nan=-1.0))
        ann_ret = float(np.nan_to_num(st["annual_return"], nan=-0.30))
        max_dd = float(np.nan_to_num(st["max_drawdown"], nan=-0.30))
        turnover = float(np.nan_to_num(st["turnover"], nan=1.0))
        hit = float(np.nan_to_num(st["hit_rate"], nan=0.50))
        # Reward risk-adjusted return; penalize churn and deep drawdowns.
        score = (
            0.45 * np.clip(sharpe, -1.0, 3.0)
            + 0.30 * np.clip(ann_ret, -0.40, 0.60)
            + 0.15 * np.clip(hit - 0.50, -0.20, 0.20)
            - 0.20 * np.clip(turnover, 0.0, 1.0)
            - 0.15 * np.clip(abs(max_dd), 0.0, 0.50)
        )
        if score > best_score:
            best_score = score
            best = st
    best = best or _strategy_stats(next_returns, probs, 0.60)
    best["optimization_score"] = float(best_score)
    return best


def run_ml_models(close: pd.Series) -> Dict:
    df = build_market_feature_frame(close)
    if len(df) < 110:
        # Fallback signal so the section remains useful even with shorter histories.
        r = close.pct_change().dropna()
        if len(r) < 30:
            return {"error": "Not enough data for ML or fallback signal. Increase history window."}
        mu = float(r.tail(min(120, len(r))).mean())
        sigma = float(r.tail(min(120, len(r))).std())
        sigma = sigma if sigma > 1e-9 else 1e-9
        z = mu / sigma
        up_prob = float(1 / (1 + np.exp(-z)))
        next_price = float(close.iloc[-1] * (1 + mu))
        return {
            "model_mode": "fallback_drift",
            "fallback_signal": True,
            "rmse": np.nan,
            "mae": np.nan,
            "next_price": next_price,
            "cls_accuracy": np.nan,
            "up_probability": up_prob,
            "up_probability_raw": up_prob,
            "confidence": float(np.clip(len(r) / 120.0, 0.15, 0.60)),
            "validation_method": "fallback_drift_signal",
            "warning": "Data window is short; using drift/volatility fallback instead of full ML.",
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
            ("model", LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced")),
        ]
    )
    clf.fit(X_train, yc_train)
    logit_prob_test = clf.predict_proba(X_test)[:, 1]

    rf_clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=7,
        min_samples_leaf=3,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    rf_clf.fit(X_train, yc_train)
    rf_prob_test = rf_clf.predict_proba(X_test)[:, 1]
    cls_prob_test = 0.55 * logit_prob_test + 0.45 * rf_prob_test
    cls_pred = (cls_prob_test >= 0.5).astype(int)

    # Walk-forward validation on the full history for more trustworthy diagnostics.
    tscv = TimeSeriesSplit(n_splits=5)
    cv_reg_true = []
    cv_reg_pred = []
    cv_cls_true = []
    cv_cls_pred = []
    cv_cls_prob = []
    cv_price_anchor = []
    cv_next_returns = []
    for train_idx, test_idx in tscv.split(X):
        Xtr, Xte = X.iloc[train_idx], X.iloc[test_idx]
        yr_tr, yr_te = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
        yc_tr, yc_te = y_cls.iloc[train_idx], y_cls.iloc[test_idx]

        if len(Xtr) < 80 or yc_tr.nunique() < 2:
            continue

        reg_cv = RandomForestRegressor(
            n_estimators=250,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )
        reg_cv.fit(Xtr, yr_tr)
        reg_hat = reg_cv.predict(Xte)
        cv_reg_true.extend(yr_te.tolist())
        cv_reg_pred.extend(reg_hat.tolist())
        cv_price_anchor.extend(close.loc[yr_te.index].tolist())

        clf_cv = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, solver="lbfgs", class_weight="balanced")),
            ]
        )
        clf_cv.fit(Xtr, yc_tr)
        logit_prob_hat = clf_cv.predict_proba(Xte)[:, 1]
        rf_cv = RandomForestClassifier(
            n_estimators=300,
            max_depth=7,
            min_samples_leaf=3,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        rf_cv.fit(Xtr, yc_tr)
        rf_prob_hat = rf_cv.predict_proba(Xte)[:, 1]
        prob_hat = 0.55 * logit_prob_hat + 0.45 * rf_prob_hat
        cls_hat = (prob_hat >= 0.5).astype(int)
        cv_cls_true.extend(yc_te.tolist())
        cv_cls_pred.extend(cls_hat.tolist())
        cv_cls_prob.extend(prob_hat.tolist())
        cv_next_returns.extend(yr_te.tolist())

    next_features = X.tail(1)
    next_ret = float(reg.predict(next_features)[0])
    last_price = float(close.iloc[-1])
    next_price = float(last_price * (1 + next_ret))
    raw_up_prob = float(
        0.55 * clf.predict_proba(next_features)[0, 1]
        + 0.45 * rf_clf.predict_proba(next_features)[0, 1]
    )

    reg_pred_price = close.loc[yr_test.index] * (1 + reg_pred)
    y_test_price = close.loc[yr_test.index] * (1 + yr_test)
    naive_pred = close.shift(1).loc[y_test_price.index]
    valid_naive = naive_pred.notna()
    naive_rmse = float(
        np.sqrt(mean_squared_error(y_test_price[valid_naive], naive_pred[valid_naive]))
    ) if valid_naive.any() else np.nan
    model_rmse = float(np.sqrt(mean_squared_error(y_test_price, reg_pred_price)))
    rmse_improvement = (
        float((naive_rmse - model_rmse) / naive_rmse) if pd.notna(naive_rmse) and naive_rmse > 0 else np.nan
    )

    cls_acc = float(accuracy_score(yc_test, cls_pred))

    if cv_reg_true and cv_price_anchor:
        cv_reg_true_arr = np.array(cv_reg_true, dtype=float)
        cv_reg_pred_arr = np.array(cv_reg_pred, dtype=float)
        cv_anchor_arr = np.array(cv_price_anchor, dtype=float)
        cv_true_price = cv_anchor_arr * (1 + cv_reg_true_arr)
        cv_pred_price = cv_anchor_arr * (1 + cv_reg_pred_arr)
        cv_rmse = float(np.sqrt(mean_squared_error(cv_true_price, cv_pred_price)))
        cv_mae = float(mean_absolute_error(cv_true_price, cv_pred_price))
    else:
        cv_rmse = np.nan
        cv_mae = np.nan

    if cv_cls_true:
        cv_accuracy = float(accuracy_score(cv_cls_true, cv_cls_pred))
        cv_brier = float(brier_score_loss(cv_cls_true, cv_cls_prob))
        if len(set(cv_cls_true)) > 1:
            cv_auc = float(roc_auc_score(cv_cls_true, cv_cls_prob))
        else:
            cv_auc = np.nan
        cv_next_returns_arr = np.array(cv_next_returns, dtype=float)
        cv_prob_arr = np.array(cv_cls_prob, dtype=float)
        bt_best = _optimize_threshold(cv_next_returns_arr, cv_prob_arr)
    else:
        cv_accuracy = np.nan
        cv_brier = np.nan
        cv_auc = np.nan
        bt_best = _strategy_stats(np.array([], dtype=float), np.array([], dtype=float), 0.60)
        bt_best["optimization_score"] = np.nan

    # Probability calibration by reliability shrinkage toward 50%.
    class_base = float(y_cls.mean())
    brier_ref = max(class_base * (1 - class_base), 1e-6)
    brier_skill = float(np.clip(1 - (cv_brier / brier_ref), -1, 1)) if pd.notna(cv_brier) else 0.0
    sample_factor = float(np.clip(len(df) / 900.0, 0.25, 1.0))
    reliability = float(np.clip(0.5 * max(brier_skill, 0.0) + 0.3 * np.nan_to_num(cv_accuracy, nan=0.5) + 0.2 * sample_factor, 0, 1))

    bt_sharpe = float(np.nan_to_num(bt_best.get("sharpe", np.nan), nan=0.0))
    bt_hit = float(np.nan_to_num(bt_best.get("hit_rate", np.nan), nan=0.5))
    edge_factor = float(
        np.clip(
            0.65 * np.clip((bt_sharpe - 0.15) / 1.30, 0, 1)
            + 0.35 * np.clip((bt_hit - 0.50) / 0.12, 0, 1),
            0,
            1,
        )
    )
    shrink = float(np.clip(0.10 + 0.55 * reliability + 0.35 * edge_factor, 0.15, 0.95))
    up_prob = float(0.5 + (raw_up_prob - 0.5) * shrink)

    train_cls_acc = float(accuracy_score(yc_train, clf.predict(X_train)))
    overfit_gap = float(max(train_cls_acc - cls_acc, 0.0))
    confidence = float(
        np.clip(
            0.30 * (1 - np.clip(model_rmse / max(last_price, 1e-9), 0, 1))
            + 0.25 * np.clip(np.nan_to_num(cv_accuracy, nan=0.5), 0, 1)
            + 0.20 * np.clip(np.nan_to_num(cv_auc, nan=0.5), 0, 1)
            + 0.15 * np.clip(reliability, 0, 1)
            + 0.10 * np.clip(edge_factor, 0, 1)
            - 0.15 * np.clip(overfit_gap, 0, 0.5),
            0,
            1,
        )
    )

    threshold_selected = float(bt_best.get("threshold", 0.60))
    signal = "LONG" if up_prob >= threshold_selected and edge_factor >= 0.25 else "HOLD"

    warning = None
    if pd.notna(overfit_gap) and overfit_gap > 0.12:
        warning = "Potential overfit: training accuracy materially exceeds holdout accuracy."
    elif pd.notna(cv_brier) and cv_brier > 0.24:
        warning = "Probability calibration is weak (high Brier loss); treat up-probability as directional only."
    elif bt_sharpe < 0.20:
        warning = "Backtest edge is weak; treat ML output as low-conviction."

    return {
        "model_mode": "full_ml",
        "fallback_signal": False,
        "y_test": yr_test,
        "reg_pred": reg_pred,
        "rmse": model_rmse,
        "mae": float(mean_absolute_error(y_test_price, reg_pred_price)),
        "naive_rmse": naive_rmse,
        "rmse_improvement_vs_naive": rmse_improvement,
        "next_price": next_price,
        "predicted_next_return": next_ret,
        "cls_accuracy": cls_acc,
        "up_probability": up_prob,
        "up_probability_raw": raw_up_prob,
        "probability_reliability": reliability,
        "signal": signal,
        "signal_threshold": threshold_selected,
        "edge_factor": edge_factor,
        "confidence": confidence,
        "cv_rmse": cv_rmse,
        "cv_mae": cv_mae,
        "cv_accuracy": cv_accuracy,
        "cv_auc": cv_auc,
        "cv_brier": cv_brier,
        "overfit_gap": overfit_gap,
        "train_cls_accuracy": train_cls_acc,
        "test_cls_accuracy": cls_acc,
        "n_samples": int(len(df)),
        "n_features": int(len(feature_cols)),
        "validation_method": "walk_forward_timeseries_cv_5fold + threshold_backtest",
        "bt_annual_return": bt_best.get("annual_return", np.nan),
        "bt_annual_volatility": bt_best.get("annual_volatility", np.nan),
        "bt_sharpe": bt_best.get("sharpe", np.nan),
        "bt_max_drawdown": bt_best.get("max_drawdown", np.nan),
        "bt_hit_rate": bt_best.get("hit_rate", np.nan),
        "bt_turnover": bt_best.get("turnover", np.nan),
        "bt_participation": bt_best.get("participation", np.nan),
        "bt_n_trades": int(bt_best.get("n_trades", 0) or 0),
        **({"warning": warning} if warning else {}),
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
