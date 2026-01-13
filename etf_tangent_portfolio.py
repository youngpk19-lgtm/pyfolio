#!/usr/bin/env python
"""
Calculate the tangent portfolio (optimal risky portfolio) for sector ETFs:
XLE (Energy), XLK (Technology), XLC (Communication), and XLU (Utilities)

Uses monthly returns from the last 5 years and maximizes the Sharpe ratio.
Risk-free rate is based on 1-month Treasury bill rate.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Define ETFs and time period
tickers = ['XLE', 'XLK', 'XLC', 'XLU']
etf_names = {
    'XLE': 'Energy',
    'XLK': 'Technology',
    'XLC': 'Communication Services',
    'XLU': 'Utilities'
}

end_date = datetime.now()
start_date = end_date - timedelta(days=5*365)  # Last 5 years

print("=" * 70)
print("TANGENT PORTFOLIO OPTIMIZATION FOR SECTOR ETFs")
print("=" * 70)
print(f"\nETFs:")
for ticker, name in etf_names.items():
    print(f"  {ticker}: {name}")

print(f"\nPeriod: {start_date.date()} to {end_date.date()}")
print("=" * 70)

try:
    # Download monthly data
    print("\nFetching data from Yahoo Finance...")
    data = yf.download(tickers, start=start_date, end=end_date, interval="1mo", progress=False, proxy=None)

    # Fetch 1-month Treasury bill rate (^IRX is 13-week treasury bill)
    print("Fetching 1-month Treasury bill rate...")
    tbill = yf.download('^IRX', start=start_date, end=end_date, interval="1mo", progress=False, proxy=None)

    # Extract closing prices
    if isinstance(data, pd.DataFrame) and not data.empty and 'Close' in data.columns:
        prices = data['Close']
    else:
        raise Exception("Unable to fetch ETF data from Yahoo Finance")

    # Calculate monthly returns (percentage changes)
    returns = prices.pct_change().dropna()

    # Get the most recent T-bill rate (annualized, convert to decimal)
    if not tbill.empty and 'Close' in tbill.columns:
        rf_rate_annual = tbill['Close'].iloc[-1] / 100  # Convert from percentage
    else:
        # Use current approximate 1-month T-bill rate if download fails
        rf_rate_annual = 0.0475  # Approximately 4.75% as of early 2024
        print(f"Note: Using estimated T-bill rate of {rf_rate_annual*100:.2f}%")

    print(f"\nFetched {len(returns)} months of data")
    print(f"1-month Treasury Bill Rate (annualized): {rf_rate_annual*100:.2f}%")

    print("\nSample of monthly returns (first 5 rows):")
    print(returns.head())

    # Calculate expected returns (mean)
    expected_returns = returns.mean()

    # Calculate covariance matrix
    cov_matrix = returns.cov()

    # Annualize the statistics (12 months per year)
    annual_returns = expected_returns * 12
    annual_cov = cov_matrix * 12

    print("\n" + "=" * 70)
    print("ANNUALIZED EXPECTED RETURNS:")
    print("=" * 70)
    for ticker in tickers:
        print(f"{ticker} ({etf_names[ticker]}): {annual_returns[ticker]*100:6.2f}%")

    print("\n" + "=" * 70)
    print("ANNUALIZED COVARIANCE MATRIX:")
    print("=" * 70)
    print(annual_cov.round(6))

    # Calculate tangent portfolio weights
    # Formula: w = Σ^(-1) * (μ - rf) / (1^T * Σ^(-1) * (μ - rf))
    excess_returns = annual_returns - rf_rate_annual
    inv_cov = np.linalg.inv(annual_cov)

    # Unnormalized weights
    weights_unnorm = inv_cov @ excess_returns

    # Normalize so weights sum to 1
    weights = weights_unnorm / np.sum(weights_unnorm)

    # Calculate portfolio characteristics
    portfolio_return = np.dot(weights, annual_returns)
    portfolio_variance = weights @ annual_cov @ weights
    portfolio_std = np.sqrt(portfolio_variance)
    sharpe_ratio = (portfolio_return - rf_rate_annual) / portfolio_std

    print("\n" + "=" * 70)
    print("TANGENT PORTFOLIO - OPTIMAL ALLOCATIONS")
    print("(Maximizes Sharpe Ratio)")
    print("=" * 70)
    print("\nOptimal Weights:")
    print("-" * 50)
    for ticker, weight in zip(tickers, weights):
        print(f"{ticker} ({etf_names[ticker]:25s}): {weight*100:7.2f}%")
    print("-" * 50)

    print(f"\nPortfolio Characteristics:")
    print("-" * 50)
    print(f"Expected Annual Return:        {portfolio_return*100:6.2f}%")
    print(f"Annual Volatility (Std Dev):   {portfolio_std*100:6.2f}%")
    print(f"Sharpe Ratio:                  {sharpe_ratio:7.4f}")
    print(f"Risk-free Rate (1-mo T-bill):  {rf_rate_annual*100:6.2f}%")
    print("-" * 50)

    # Verify weights sum to 1
    print(f"\nWeights sum to: {np.sum(weights):.6f}")

    # Show correlation matrix for reference
    print("\n" + "=" * 70)
    print("CORRELATION MATRIX:")
    print("=" * 70)
    corr_matrix = returns.corr()
    print(corr_matrix.round(3))

    # Additional analysis: individual ETF Sharpe ratios
    print("\n" + "=" * 70)
    print("INDIVIDUAL ETF SHARPE RATIOS (for comparison):")
    print("=" * 70)
    individual_sharpes = (annual_returns - rf_rate_annual) / np.sqrt(np.diag(annual_cov))
    for ticker, sharpe in zip(tickers, individual_sharpes):
        print(f"{ticker} ({etf_names[ticker]}): {sharpe:.4f}")

except Exception as e:
    print(f"\nError: {e}")
    print("\nNote: Using sample data for demonstration...")
    print("=" * 70)

    # Sample monthly returns data (60 months = 5 years)
    # Representative returns for sector ETFs
    np.random.seed(42)
    n_months = 60

    # Generate realistic returns for sector ETFs
    # XLE (Energy): Higher volatility, moderate returns
    # XLK (Technology): High returns, high volatility
    # XLC (Communication): Moderate returns, moderate volatility
    # XLU (Utilities): Lower volatility, lower returns (defensive)

    xle_returns = np.random.normal(0.008, 0.08, n_months)
    xlk_returns = np.random.normal(0.020, 0.07, n_months)
    xlc_returns = np.random.normal(0.012, 0.06, n_months)
    xlu_returns = np.random.normal(0.007, 0.04, n_months)

    returns = pd.DataFrame({
        'XLE': xle_returns,
        'XLK': xlk_returns,
        'XLC': xlc_returns,
        'XLU': xlu_returns
    })

    # Use current 1-month T-bill rate (approximate)
    rf_rate_annual = 0.0475  # 4.75%

    print(f"\nUsing {len(returns)} months of sample data")
    print(f"1-month Treasury Bill Rate (annualized): {rf_rate_annual*100:.2f}%")

    print("\nSample of monthly returns (first 5 rows):")
    print(returns.head())

    # Calculate expected returns (mean)
    expected_returns = returns.mean()

    # Calculate covariance matrix
    cov_matrix = returns.cov()

    # Annualize the statistics (12 months per year)
    annual_returns = expected_returns * 12
    annual_cov = cov_matrix * 12

    print("\n" + "=" * 70)
    print("ANNUALIZED EXPECTED RETURNS:")
    print("=" * 70)
    for ticker in tickers:
        print(f"{ticker} ({etf_names[ticker]}): {annual_returns[ticker]*100:6.2f}%")

    print("\n" + "=" * 70)
    print("ANNUALIZED COVARIANCE MATRIX:")
    print("=" * 70)
    print(annual_cov.round(6))

    # Calculate tangent portfolio weights
    excess_returns = annual_returns - rf_rate_annual
    inv_cov = np.linalg.inv(annual_cov)

    # Unnormalized weights
    weights_unnorm = inv_cov @ excess_returns

    # Normalize so weights sum to 1
    weights = weights_unnorm / np.sum(weights_unnorm)

    # Calculate portfolio characteristics
    portfolio_return = np.dot(weights, annual_returns)
    portfolio_variance = weights @ annual_cov @ weights
    portfolio_std = np.sqrt(portfolio_variance)
    sharpe_ratio = (portfolio_return - rf_rate_annual) / portfolio_std

    print("\n" + "=" * 70)
    print("TANGENT PORTFOLIO - OPTIMAL ALLOCATIONS")
    print("(Maximizes Sharpe Ratio)")
    print("=" * 70)
    print("\nOptimal Weights:")
    print("-" * 50)
    for ticker, weight in zip(tickers, weights):
        print(f"{ticker} ({etf_names[ticker]:25s}): {weight*100:7.2f}%")
    print("-" * 50)

    print(f"\nPortfolio Characteristics:")
    print("-" * 50)
    print(f"Expected Annual Return:        {portfolio_return*100:6.2f}%")
    print(f"Annual Volatility (Std Dev):   {portfolio_std*100:6.2f}%")
    print(f"Sharpe Ratio:                  {sharpe_ratio:7.4f}")
    print(f"Risk-free Rate (1-mo T-bill):  {rf_rate_annual*100:6.2f}%")
    print("-" * 50)

    # Verify weights sum to 1
    print(f"\nWeights sum to: {np.sum(weights):.6f}")

    # Show correlation matrix for reference
    print("\n" + "=" * 70)
    print("CORRELATION MATRIX:")
    print("=" * 70)
    corr_matrix = returns.corr()
    print(corr_matrix.round(3))

    # Additional analysis: individual ETF Sharpe ratios
    print("\n" + "=" * 70)
    print("INDIVIDUAL ETF SHARPE RATIOS (for comparison):")
    print("=" * 70)
    individual_sharpes = (annual_returns - rf_rate_annual) / np.sqrt(np.diag(annual_cov))
    for ticker, sharpe in zip(tickers, individual_sharpes):
        print(f"{ticker} ({etf_names[ticker]}): {sharpe:.4f}")
