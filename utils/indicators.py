import pandas as pd


def add_indicators(close):
    """Build simple technical indicator features from close prices."""
    df = pd.DataFrame(index=close.index)
    df["close"] = close
    df["return_1d"] = close.pct_change()
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["vol_20"] = df["return_1d"].rolling(20).std() * (252**0.5)

    delta = close.diff()
    gains = delta.clip(lower=0).rolling(14).mean()
    losses = -delta.clip(upper=0).rolling(14).mean()
    rs = gains / losses.replace(0, pd.NA)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    return df
