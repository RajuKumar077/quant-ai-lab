from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.financial_ml import run_ml_models
from models.portfolio_optimization import optimize_portfolio
from models.valuation import run_dcf_valuation
from utils.data_loader import load_price_data
from utils.fundamentals import compute_financial_metrics, fetch_ticker_bundle
from utils.risk import calculate_risk_metrics
from utils.time_series import analyze_time_series, arima_forecast


def fail(msg: str) -> tuple[bool, str]:
    return False, msg


def ok(msg: str) -> tuple[bool, str]:
    return True, msg


def check_single_ticker(ticker: str, start: date, end: date) -> tuple[bool, str]:
    prices = load_price_data([ticker], start, end)
    if prices.empty or ticker not in prices.columns:
        return fail(f"{ticker}: price data missing")

    close = prices[ticker].dropna()
    if len(close) < 250:
        return fail(f"{ticker}: insufficient history ({len(close)} rows)")

    bundle = fetch_ticker_bundle(ticker)
    metrics = compute_financial_metrics(bundle)

    risk = calculate_risk_metrics(close, confidence=0.95, horizon_days=5)
    for key in [
        "historical_var",
        "historical_cvar",
        "parametric_var",
        "parametric_cvar",
        "annualized_volatility",
        "max_drawdown",
    ]:
        if risk[key] < 0:
            return fail(f"{ticker}: negative risk metric {key}={risk[key]}")

    ts = analyze_time_series(close)
    if "error" in ts:
        return fail(f"{ticker}: time series diagnostics failed ({ts['error']})")

    ar = arima_forecast(close, steps=10)
    if ar.empty:
        return fail(f"{ticker}: ARIMA forecast empty")

    ml = run_ml_models(close)
    if "error" in ml:
        return fail(f"{ticker}: ML check failed ({ml['error']})")
    if not (0 <= ml["up_probability"] <= 1):
        return fail(f"{ticker}: invalid ML probability ({ml['up_probability']})")

    dcf = run_dcf_valuation(
        cashflow=bundle.cashflow,
        shares_outstanding=metrics.get("shares_outstanding", np.nan),
        beta=bundle.info.get("beta", 1.0),
        current_price=float(close.iloc[-1]),
    )
    if "error" in dcf:
        return fail(f"{ticker}: DCF unavailable ({dcf['error']})")
    if not np.isfinite(dcf["intrinsic_per_share"]):
        return fail(f"{ticker}: DCF intrinsic value is not finite")

    return ok(f"{ticker}: all checks passed")


def check_portfolio(start: date, end: date) -> tuple[bool, str]:
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL"]
    prices = load_price_data(tickers, start, end)
    if prices.empty or prices.shape[1] < 2:
        return fail("portfolio: insufficient price data")

    weights, perf = optimize_portfolio(prices, "max_sharpe")
    wsum = float(sum(weights.values()))
    if not (0.95 <= wsum <= 1.05):
        return fail(f"portfolio: weights sum out of tolerance ({wsum:.6f})")
    if any((w < -1e-8 or w > 1 + 1e-8) for w in weights.values()):
        return fail("portfolio: weights out of [0, 1] range")

    if any(pd.isna(x) for x in perf):
        return fail("portfolio: invalid performance metrics")

    return ok("portfolio: optimization checks passed")


def main() -> int:
    start = date.today() - timedelta(days=365 * 5)
    end = date.today()

    checks: list[tuple[bool, str]] = []
    for ticker in ["AAPL", "MSFT", "NVDA"]:
        checks.append(check_single_ticker(ticker, start, end))
    checks.append(check_portfolio(start, end))

    failures = [msg for passed, msg in checks if not passed]
    for passed, msg in checks:
        prefix = "PASS" if passed else "FAIL"
        print(f"[{prefix}] {msg}")

    if failures:
        print(f"\nHealth check failed with {len(failures)} issue(s).", file=sys.stderr)
        return 1

    print("\nHealth check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
