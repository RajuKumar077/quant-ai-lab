from pypfopt import EfficientFrontier, expected_returns, risk_models


def optimize_portfolio(price_df, objective="max_sharpe"):
    """Optimize a long-only portfolio and return cleaned weights and performance."""
    clean_df = price_df.dropna(axis=1, how="any")
    if clean_df.shape[1] < 2:
        raise ValueError("Need at least two assets with valid data for optimization.")

    mu = expected_returns.mean_historical_return(clean_df, frequency=252)
    cov = risk_models.sample_cov(clean_df, frequency=252)

    ef = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    if objective == "min_volatility":
        ef.min_volatility()
    else:
        ef.max_sharpe()

    weights = ef.clean_weights()
    performance = ef.portfolio_performance(verbose=False)
    return weights, performance
