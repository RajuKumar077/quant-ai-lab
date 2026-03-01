import pandas as pd
import yfinance as yf


def load_price_data(tickers, start, end):
    """Load adjusted close prices as a DataFrame with ticker columns."""
    if isinstance(tickers, str):
        tickers = [tickers]

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        close = data.get("Close")
        if close is None:
            return pd.DataFrame()
        prices = close.copy()
    else:
        if "Close" not in data.columns:
            return pd.DataFrame()
        prices = data[["Close"]].copy()
        prices.columns = [tickers[0]]

    prices = prices.dropna(how="all")
    prices = prices.loc[:, ~prices.columns.duplicated()]
    return prices
