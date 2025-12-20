#!/usr/bin/env python
"""
Fetch SPY monthly closing prices for 2024 and calculate returns.
"""
import yfinance as yf
import pandas as pd

# Fetch SPY data for 2024
ticker = "SPY"
start_date = "2024-01-01"
end_date = "2024-12-31"

print(f"Fetching {ticker} data from {start_date} to {end_date}...")

try:
    # Download data with monthly interval
    spy = yf.download(ticker, start=start_date, end=end_date, interval="1mo", progress=False, proxy=None)

    # Get closing prices
    if isinstance(spy, pd.DataFrame) and not spy.empty:
        # If multi-column DataFrame
        if 'Close' in spy.columns:
            prices = spy['Close']
        else:
            prices = spy[ticker] if ticker in spy.columns else spy.iloc[:, 0]
    else:
        # If download failed, use sample data for demonstration
        print("\nNote: Unable to fetch real data. Using sample data for demonstration.")
        dates = pd.date_range(start='2024-01-31', end='2024-12-31', freq='ME')
        sample_prices = [477.68, 496.83, 516.21, 512.02, 527.37, 543.52,
                        552.35, 556.08, 572.59, 579.68, 594.13, 600.00]
        prices = pd.Series(sample_prices[:len(dates)], index=dates, name=ticker)

    print(f"\nMonthly Closing Prices for {ticker} in 2024:")
    print("=" * 50)
    print(prices)

    # Calculate returns as percent changes
    returns = prices.pct_change() * 100  # Convert to percentage

    print(f"\n\nMonthly Returns (%) for {ticker} in 2024:")
    print("=" * 50)
    print(returns)

    # Summary statistics
    print(f"\n\nSummary Statistics:")
    print("=" * 50)
    mean_ret = returns.mean()
    std_ret = returns.std()
    min_ret = returns.min()
    max_ret = returns.max()
    cum_ret = ((prices.iloc[-1] / prices.iloc[0]) - 1) * 100

    print(f"Mean Monthly Return: {mean_ret:.2f}%")
    print(f"Std Dev: {std_ret:.2f}%")
    print(f"Min Return: {min_ret:.2f}%")
    print(f"Max Return: {max_ret:.2f}%")
    print(f"Cumulative Return: {cum_ret:.2f}%")

except Exception as e:
    print(f"Error: {e}")
    print("\nUsing sample data for demonstration:")
    dates = pd.date_range(start='2024-01-31', end='2024-12-31', freq='ME')
    sample_prices = [477.68, 496.83, 516.21, 512.02, 527.37, 543.52,
                    552.35, 556.08, 572.59, 579.68, 594.13, 600.00]
    prices = pd.Series(sample_prices[:len(dates)], index=dates, name=ticker)

    print(f"\nMonthly Closing Prices for {ticker} in 2024:")
    print("=" * 50)
    print(prices)

    returns = prices.pct_change() * 100
    print(f"\n\nMonthly Returns (%) for {ticker} in 2024:")
    print("=" * 50)
    print(returns)

    print(f"\n\nSummary Statistics:")
    print("=" * 50)
    print(f"Mean Monthly Return: {returns.mean():.2f}%")
    print(f"Std Dev: {returns.std():.2f}%")
    print(f"Min Return: {returns.min():.2f}%")
    print(f"Max Return: {returns.max():.2f}%")
    print(f"Cumulative Return: {((prices.iloc[-1] / prices.iloc[0]) - 1) * 100:.2f}%")
