"""
Nasdaq-100 Momentum Trading Robot
- 6-month momentum
- Top 10 stocks
- Weekly rebalancing
- Volatility targeting
- Supports fractional shares
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
# CREDENTIALS
# ============================================================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY")

PAPER = False   # False = Live trading

# ============================================================
# STRATEGY SETTINGS
# ============================================================
TOP_N = 10
MOMENTUM_DAYS = 126
VOL_LOOKBACK = 63
TARGET_PORTFOLIO_VOL = 0.12
MAX_POSITION_WEIGHT = 0.15
MIN_PRICE = 5.0
CASH_BUFFER = 0.02

# ============================================================
# GET NASDAQ-100 LIST
# ============================================================
def get_nasdaq100_tickers():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        import requests
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(response.text)
        for table in tables:
            if "Ticker" in table.columns or "Symbol" in table.columns:
                col = "Ticker" if "Ticker" in table.columns else "Symbol"
                tickers = table[col].astype(str).str.strip().str.upper().tolist()
                tickers = [t for t in tickers if t.isalpha() or "." in t]
                if len(tickers) > 80:
                    print(f"Loaded {len(tickers)} Nasdaq-100 tickers")
                    return tickers
    except Exception as e:
        print(f"Wikipedia error: {e}")

    print("Using fallback list")
    return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","AVGO","TSLA","COST","NFLX",
            "AMD","ADBE","PEP","CSCO","INTC","CMCSA","INTU","QCOM","AMGN","HON",
            "TXN","AMAT","SBUX","ISRG","BKNG","ADP","GILD","VRTX","REGN","LRCX",
            "PANW","KLAC","SNPS","CDNS","CRWD","FTNT","DDOG","TEAM","MDB","ZS"]

# ============================================================
# ALPACA HELPERS
# ============================================================
trading_client = TradingClient(API_KEY, API_SECRET, paper=PAPER)

def get_account_equity():
    return float(trading_client.get_account().equity)

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
    start = end - timedelta(days=300)

    print("Downloading price data...")
    raw = yf.download(universe, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(axis=1, how="all")
    print(f"Data ready for {len(raw.columns)} stocks")

    momentum = raw.pct_change(periods=MOMENTUM_DAYS).iloc[-1].dropna()
    last_prices = raw.iloc[-1]
    momentum = momentum[last_prices.reindex(momentum.index) > MIN_PRICE]

    if len(momentum) < TOP_N:
        print("Not enough stocks passed filters")
        return {}

    ranked = momentum.sort_values(ascending=False)
    print("\n===== TOP 15 (6-Month Momentum) =====")
    for i, (ticker, score) in enumerate(ranked.head(15).items(), 1):
        print(f"{i:2d}. {ticker:6s}  {score*100:6.1f}%")
    print("=====================================\n")

    top = ranked.head(TOP_N).index.tolist()
    print(f"Selected Top {TOP_N} portfolio: {top}")

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

    port_vol = np.sqrt((raw_w**2 * vols**2).sum())
    scale = min(TARGET_PORTFOLIO_VOL / port_vol, 1.5) if port_vol > 0 else 1.0

    final = (raw_w * scale).clip(upper=MAX_POSITION_WEIGHT)
    final = final / final.sum() * (1.0 - CASH_BUFFER)

    return final.to_dict()

# ============================================================
# REBALANCE (with fractional shares)
# ============================================================
def rebalance():
    print(f"\n[{datetime.now()}] Starting weekly rebalance...")
    equity = get_account_equity()
    print(f"Account equity: ${equity:,.2f}")

    targets = calculate_target_weights()
    if not targets:
        print("No targets generated.")
        return

    print("\nTarget weights:")
    for symbol, weight in sorted(targets.items(), key=lambda x: -x[1]):
        print(f"  {symbol}: {weight:.1%}")

    current_positions = get_current_positions()
    all_symbols = list(set(list(targets.keys()) + list(current_positions.keys())))
    prices = get_latest_prices(all_symbols)

    target_dollars = {symbol: equity * weight for symbol, weight in targets.items()}

    # Close unwanted positions
    for symbol, qty in current_positions.items():
        if symbol not in target_dollars and abs(qty) > 0:
            side = OrderSide.SELL if qty > 0 else OrderSide.BUY
            order = MarketOrderRequest(
                symbol=symbol,
                qty=abs(qty),
                side=side,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order)
            print(f"Closing {symbol}")

    time.sleep(2)

    # Adjust positions with fractional support
    current_positions = get_current_positions()
    for symbol, target_value in target_dollars.items():
        current_qty = current_positions.get(symbol, 0)
        current_price = prices.get(symbol, 0)

        if current_price <= 0:
            continue

        current_value = current_qty * current_price
        diff_value = target_value - current_value

        if abs(diff_value) < 5:
            continue

        if diff_value > 0:
            order = MarketOrderRequest(
                symbol=symbol,
                notional=round(diff_value, 2),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order)
            print(f"BUY ${diff_value:.2f} of {symbol}")
        else:
            sell_qty = abs(diff_value) / current_price
            order = MarketOrderRequest(
                symbol=symbol,
                qty=round(sell_qty, 4),
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
            trading_client.submit_order(order)
            print(f"SELL {sell_qty:.4f} shares of {symbol}")

    print("\nWeekly rebalance completed successfully.")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Nasdaq-100 Momentum Robot started")
    print(f"Paper trading mode: {PAPER}")
    rebalance()
    print("Script finished.")
