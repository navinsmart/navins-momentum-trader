"""

Momentum + Volatility Targeting Live Trader for Alpaca

Designed to run automatically via GitHub Actions

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

# CREDENTIALS (read from GitHub Secrets)

# ============================================================

API_KEY = os.getenv("ALPACA_API_KEY")

API_SECRET = os.getenv("ALPACA_SECRET_KEY")



if not API_KEY or not API_SECRET:

    raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY. Add them as GitHub Secrets.")



# Force paper trading unless you change this

PAPER = True



# ============================================================

# STRATEGY SETTINGS (you can change these later)

# ============================================================

UNIVERSE = [

    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA", "AVGO",

    "JPM", "V", "UNH", "XOM", "LLY", "MA", "COST", "HD", "PG", "JNJ",

    "ABBV", "CRM", "NFLX", "AMD", "KO", "PEP", "WMT", "BAC", "ORCL",

    "XLK", "XLF", "XLE", "XLV", "XLY", "XLI", "XLP", "XLU"

]



LOOKBACK_MONTHS = 12          # How far back we look for momentum

SKIP_MONTHS = 1               # Skip the most recent month

TOP_N = 8                     # How many stocks to hold

VOL_LOOKBACK = 63             # Days used to measure volatility

TARGET_PORTFOLIO_VOL = 0.12   # Target yearly volatility (12%)

MAX_POSITION_WEIGHT = 0.18    # Maximum weight for any single stock

MIN_PRICE = 5.0               # Ignore very cheap stocks

USE_ABSOLUTE_MOMENTUM = True  # Only buy stocks above their 200-day average

CASH_BUFFER = 0.02            # Keep 2% in cash



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

        return {data.name: data.iloc[-1]}

    return data.iloc[-1].to_dict()



# ============================================================

# CALCULATE TARGET WEIGHTS

# ============================================================

def calculate_target_weights():

    end = datetime.now()

    start = end - timedelta(days=400)



    print("Downloading price data...")

    raw = yf.download(UNIVERSE, start=start, end=end, auto_adjust=True, progress=False)["Close"]

    raw = raw.dropna(axis=1, how="all")



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



    top = mom.nlargest(TOP_N).index.tolist()

    print(f"Top momentum stocks: {top}")



    # Volatility calculation

    daily_rets = raw.pct_change()

    vols = daily_rets[top].iloc[-VOL_LOOKBACK:].std() * np.sqrt(252)

    vols = vols.replace(0, np.nan).dropna()



    if len(vols) < 3:

        # Fallback to equal weight

        weight = (1.0 - CASH_BUFFER) / len(top)

        return {t: weight for t in top}



    # Inverse-volatility weights

    inv_vol = 1.0 / vols

    raw_w = inv_vol / inv_vol.sum()

    raw_w = raw_w.clip(upper=MAX_POSITION_WEIGHT)

    raw_w = raw_w / raw_w.sum()



    # Scale to target portfolio volatility

    port_vol = np.sqrt((raw_w ** 2 * vols ** 2).sum())

    scale = min(TARGET_PORTFOLIO_VOL / port_vol, 1.5) if port_vol > 0 else 1.0



    final = (raw_w * scale).clip(upper=MAX_POSITION_WEIGHT)

    final = final / final.sum() * (1.0 - CASH_BUFFER)



    return final.to_dict()



# ============================================================

# REBALANCE FUNCTION

# ============================================================

def rebalance():

    print(f"\n[{datetime.now()}] Starting rebalance...")

    equity = get_account_equity()

    print(f"Account equity: ${equity:,.2f}")



    targets = calculate_target_weights()

    if not targets:

        print("No targets generated. Exiting.")

        return



    print("\nTarget portfolio:")

    for symbol, weight in sorted(targets.items(), key=lambda x: -x[1]):

        print(f"  {symbol}: {weight:.1%}")



    current_positions = get_current_positions()

    all_symbols = list(set(list(targets.keys()) + list(current_positions.keys())))

    prices = get_latest_prices(all_symbols)



    # Calculate how many shares we want

    target_shares = {}

    for symbol, weight in targets.items():

        price = prices.get(symbol, 0)

        if price > 0:

            dollars = equity * weight

            target_shares[symbol] = int(dollars / price)



    # 1. Close positions that are no longer wanted

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

            print(f"Closing entire position in {symbol}")



    time.sleep(3)



    # 2. Adjust the remaining positions

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

        print(f"{action} {abs(int(difference))} shares of {symbol}")



    print("\nRebalance finished successfully.")



# ============================================================

# MAIN

# ============================================================

if __name__ == "__main__":

    print("Momentum Trader started")

    print(f"Paper trading mode: {PAPER}")



    try:

        clock = trading_client.get_clock()

        print(f"Market is open: {clock.is_open}")

    except Exception as e:

        print(f"Could not check market clock: {e}")



    rebalance()

    print("Script completed.")

