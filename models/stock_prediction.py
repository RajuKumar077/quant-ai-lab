import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

from utils.indicators import add_indicators


def build_training_frame(close):
    """Create model-ready features and next-day target."""
    df = add_indicators(close)
    df["lag_1"] = close.shift(1)
    df["lag_2"] = close.shift(2)
    df["lag_5"] = close.shift(5)
    df["target"] = close.shift(-1)
    return df.dropna().copy()


def train_predict_model(df, test_ratio=0.2):
    """Train a random-forest regressor and return prediction diagnostics."""
    feature_cols = [c for c in df.columns if c != "target"]
    X = df[feature_cols]
    y = df["target"]

    split_idx = int(len(df) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))

    next_prediction = float(model.predict(X.tail(1))[0])
    return {
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
        "rmse": rmse,
        "next_prediction": next_prediction,
    }
