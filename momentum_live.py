"""
Nasdaq-100 Momentum Trading Robot
- 6-month momentum (including most recent month)
- Holds Top 10 stocks
- Weekly rebalancing
- Volatility targeting
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
# CREDENTIALS (from GitHub Secrets or environment variables)
# ============================================================
API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not API_SECRET:
    raise ValueError("Missing ALPACA_API_KEY or ALPACA_SECRET_KEY")

PAPER = True   # Keep True for paper trading. Change to False only when ready for real money.

# ============================================================
# STRATEGY SETTINGS
# ============================================================
TOP_N = 10                      # Hold the top 10 stocks
MOMENTUM_DAYS = 126             # ~6 months of trading days
VOL_LOOKBACK = 63               # Days used to measure volatility
TARGET_PORTFOLIO_VOL = 0.12     # Target 12% annual volatility
MAX_POSITION_WEIGHT = 0.15      # Maximum 15% in any single stock
MIN_PRICE = 5.0                 # Ignore very cheap stocks
CASH_BUFFER = 0.02              # Keep 2% in cash

# ============================================================
# GET CURRENT NASDAQ-100 LIST
# ============================================================
def get_nasdaq100_tickers():
    """Get current Nasdaq-100 tickers from Wikipedia with proper headers"""
    import requests
    
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
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
        print(f"Could not load from Wikipedia: {e}")
    
    # Fallback list (major Nasdaq-100 stocks) if Wikipedia fails
    print("Using fallback Nasdaq-100 list")
    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "AVGO", "TSLA", "COST",
        "NFLX", "AMD", "ADBE", "PEP", "CSCO", "INTC", "CMCSA", "INTU", "QCOM", "AMGN",
        "HON", "TXN", "AMAT", "SBUX", "ISRG", "BKNG", "ADP", "GILD", "VRTX", "REGN",
        "LRCX", "MDLZ", "ADI", "PANW", "KLAC", "SNPS", "CDNS", "MAR", "ORLY", "CTAS",
        "FTNT", "NXPI", "AEP", "KDP", "CSX", "PCAR", "ROST", "FAST", "ODFL", "IDXX",
        "BKR", "GEHC", "XEL", "WBD", "DDOG", "ZS", "TEAM", "CRWD", "TTD", "MDB"
    ]


# ============================================================
# ALPACA CONNECTION
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

    print("Downloading price data (this may take 30-60 seconds)...")
    raw = yf.download(universe, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    raw = raw.dropna(axis=1, how="all")
    print(f"Data ready for {len(raw.columns)} stocks")

    # 6-month momentum (includes the most recent month)
    momentum = raw.pct_change(periods=MOMENTUM_DAYS).iloc[-1].dropna()

    # Filter by minimum price
    last_prices = raw.iloc[-1]
    momentum = momentum[last_prices.reindex(momentum.index) > MIN_PRICE]

    if len(momentum) < TOP_N:
        print("Not enough stocks passed the filters.")
        return {}

    # Show ranking
    ranked = momentum.sort_values(ascending=False)
    print("\n===== TOP 15 NASDAQ-100 (6-Month Momentum) =====")
    for i, (ticker, score) in enumerate(ranked.head(15).items(), 1):
        print(f"{i:2d}. {ticker:6s}  {score*100:6.1f}%")
    print("================================================\n")

    top = ranked.head(TOP_N).index.tolist()
    print(f"Selected Top {TOP_N} portfolio: {top}")

    # Volatility targeting
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
# REBALANCE FUNCTION
# ============================================================
def rebalance():
    print(f"\n[{datetime.now()}] Starting weekly rebalance...")
    equity = get_account_equity()
    print(f"Account equity: ${equity:,.2f}")

    targets = calculate_target_weights()
    if not targets:
        print("No targets generated. Exiting.")
        return

    print("\nTarget weights:")
    for symbol, weight in sorted(targets.items(), key=lambda x: -x[1]):
        print(f"  {symbol}: {weight:.1%}")

    current_positions = get_current_positions()
    all_symbols = list(set(list(targets.keys()) + list(current_positions.keys())))
    prices = get_latest_prices(all_symbols)

    # Calculate target number of shares
    target_shares = {}
    for symbol, weight in targets.items():
        price = prices.get(symbol, 0)
        if price > 0:
            target_shares[symbol] = int((equity * weight) / price)

    # 1. Close positions that are no longer in the Top 10
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

    # 2. Adjust the remaining positions
    current_positions = get_current_positions()
    for symbol, target_qty in target_shares.items():
        current_qty = current_positions.get(symbol, 0)
        diff = target_qty - current_qty

        if abs(diff) < 1:
            continue

        side = OrderSide.BUY if diff > 0 else OrderSide.SELL
        order = MarketOrderRequest(
            symbol=symbol,
            qty=abs(int(diff)),
            side=side,
            time_in_force=TimeInForce.DAY
        )
        trading_client.submit_order(order)
        action = "BUY" if diff > 0 else "SELL"
        print(f"{action} {abs(int(diff))} shares of {symbol}")

    print("\nWeekly rebalance completed successfully.")

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    print("Nasdaq-100 Momentum Robot started")
    print(f"Paper trading mode: {PAPER}")
    rebalance()
    print("Script finished.")
