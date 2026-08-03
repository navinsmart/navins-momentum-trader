"""
Momentum + Volatility Targeting Live Trader for Alpaca
Now using the full Nasdaq-100
"""

import os
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

# ============================================================
# CREDENTIALS (from GitHub Secrets)
# ============================================================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY")

PAPER = True   # Keep True for paper trading

# ============================================================
# STRATEGY SETTINGS
# ============================================================
LOOKBACK_MONTHS = 12
SKIP_MONTHS = 1
TOP_N = 10                    # Hold top 10 Nasdaq-100 stocks
VOL_LOOKBACK = 63
TARGET_PORTFOLIO_VOL = 0.12
MAX_POSITION_WEIGHT = 0.15
MIN_PRICE = 5.0
USE_ABSOLUTE_MOMENTUM = True
CASH_BUFFER = 0.02

# ============================================================
# GET CURRENT NASDAQ-100 LIST
# ============================================================
def get_nasdaq100_tickers():
    """Download the current Nasdaq-100 constituents from Wikipedia"""
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)

    # The main constituents table is usually the second or third table
    for table in tables:
        if "Ticker" in table.columns or "Symbol" in table.columns:
            col = "Ticker" if "Ticker" in table.columns else "Symbol"
            tickers = table[col].tolist()
            # Clean the list
            tickers = [str(t).strip().upper() for t in tickers if isinstance(t, str)]
            # Remove any non-ticker values
            tickers = [t for t in tickers if t.isalpha() or "." in t]
            if len(tickers) > 80:   # Sanity check
                print(f"Found {len(tickers)} Nasdaq-100 tickers")
                return tickers

    # Fallback if Wikipedia structure changes
    print("Could not read Wikipedia table – using backup list")
    return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA", "COST"]

# ============================================================
# CONNECT TO ALPACA
# ============================================================
trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)

def get_account_equity():
    account = trading_client.get_account()
    return float(account.equity)

def get_current_positions():
    positions = trading_client.get_all_positions()
    return {p.symbol: float(p.qty) for p in positions}

def get_latest_prices(symbols):
    data = yf.download(symbols, period="5d", auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.Series):
        return {data.name: float(data.iloc[-1])}
    return data.iloc[-1].to_dict()

# ============================================================
# CALCULATE TARGET WEIGHTS
# ============================================================
def calculate_target_weights():
    print("Getting Nasdaq-100 list...")
    universe = get_nasdaq100_tickers()

    end = datetime.now()
    start = end - timedelta(days=450)

    print("Downloading price data for Nasdaq-100 (this may take 30-60 seconds)...")
    raw = yf.download(universe, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(axis=1, how="all")

    print(f"Successfully downloaded data for {len(raw.columns)} stocks")

    monthly = raw.resample("ME").last()
    momentum = (monthly.shift(SKIP_MONTHS) / monthly.shift(LOOKBACK_MONTHS)) - 1

    latest = momentum.dropna(how="all").index[-1]
    mom = momentum.loc[latest].dropna()

    if USE_ABSOLUTE_MOMENTUM:
        sma200 = raw.rolling(200).mean().iloc[-1]
        mom = mom[raw.iloc[-1] > sma200.reindex(mom.index)]

    last_prices = raw.iloc[-1]
    mom = mom[last_prices.reindex(mom.index) > MIN_PRICE]

    if len(mom) < TOP_N:
        print("Not enough stocks passed the filters.")
        return {}

    # Show the full ranking
    ranked = mom.sort_values(ascending=False)
    print("\n===== TOP 20 NASDAQ-100 MOMENTUM STOCKS =====")
    for i, (ticker, score) in enumerate(ranked.head(20).items(), 1):
        print(f"{i:2d}. {ticker:6s}  {score:7.1%}")
    print("=============================================\n")

    top = ranked.head(TOP_N).index.tolist()
    print(f"Selected portfolio: {top}")

    # Volatility weighting
    daily_rets = raw.pct_change()
    vols = daily_rets[top].iloc[-VOL_LOOKBACK:].std() * np.sqrt(252)
    vols = vols.replace(0, np.nan).dropna()

    if len(vols) < 3:
        weight = (1.0 - CASH_BUFFER) / len(top)
        return {t: weight for t in top}

    inv_vol = 1.0 / vols
    raw_w = inv_vol / inv_vol.sum()
    raw_w = raw_w.clip(upper=MAX_POSITION_WEIGHT)
    raw_w = raw_w / raw_w.sum()

    port_vol = np.sqrt((raw_w ** 2 * vols ** 2).sum())
    scale = min(TARGET_PORTFOLIO_VOL / port_vol, 1.5) if port_vol > 0 else 1.0

    final = (raw_w * scale).clip(upper=MAX_POSITION_WEIGHT)
    final = final / final.sum() * (1.0 - CASH_BUFFER)

    return final.to_dict()

# ============================================================
# REBALANCE
# ============================================================
def rebalance():
    print(f"\n[{datetime.now()}] Starting Nasdaq-100 Momentum rebalance...")
    equity = get_account_equity()
    print(f"Account equity: ${equity:,.2f}")

    targets = calculate_target_weights()
    if not targets:
        print("No targets generated.")
        return

    print("\nFinal target weights:")
    for symbol, weight in sorted(targets.items(), key=lambda x: -x[1]):
        print(f"  {symbol}: {weight:.1%}")

    current_positions = get_current_positions()
    all_symbols = list(set(list(targets.keys()) + list(current_positions.keys())))
    prices = get_latest_prices(all_symbols)

    target_shares = {}
    for symbol, weight in targets.items():
        price = prices.get(symbol, 0)
        if price and price > 0:
            dollars = equity * weight
            target_shares[symbol] = int(dollars / price)

    # Close unwanted positions
    for symbol, qty in current_positions.items():
        if symbol not in target_shares and abs(qty) > 0:
            side = OrderSide.SELL if qty > 0 else OrderSide.BUY
            order = MarketOrderRequest(
                symbol=symbol,
                qty=abs(int(qty)),
                side=side,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order)
            print(f"Closing {symbol}")

    time.sleep(3)

    # Adjust positions
    current_positions = get_current_positions()
    for symbol, target_qty in target_shares.items():
        current_qty = current_positions.get(symbol, 0)
        difference = target_qty - current_qty

        if abs(difference) < 1:
            continue

        side = OrderSide.BUY if difference > 0 else OrderSide.SELL
        order = MarketOrderRequest(
            symbol=symbol,
            qty=abs(int(difference)),
            side=side,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        action = "BUY" if difference > 0 else "SELL"
        print(f"{action} {abs(int(difference))} {symbol}")

    print("\nRebalance completed.")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Nasdaq-100 Momentum Trader started")
    print(f"Paper trading: {PAPER}")
    rebalance()
    print("Script finished.")
