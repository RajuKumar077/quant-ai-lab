import numpy as np
import pandas as pd


def monte_carlo_paths(close, n_sims=1000, n_days=252, seed=42):
    """Simulate future price paths using a geometric Brownian motion model."""
    returns = close.pct_change().dropna()
    if returns.empty:
        return pd.DataFrame()

    mu = returns.mean()
    sigma = returns.std()
    last_price = float(close.iloc[-1])

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal((n_days, n_sims))
    daily_growth = np.exp((mu - 0.5 * sigma**2) + sigma * shocks)

    paths = np.zeros((n_days + 1, n_sims))
    paths[0] = last_price
    for t in range(1, n_days + 1):
        paths[t] = paths[t - 1] * daily_growth[t - 1]

    return pd.DataFrame(paths)
