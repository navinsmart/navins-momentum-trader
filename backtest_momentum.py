"""
Backtest: Nasdaq-100 Top 10 Momentum Strategy
- 6-month momentum
- Weekly rebalancing
- Volatility targeting
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# SETTINGS
# ============================================================
START_DATE = "2018-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
TOP_N = 10
MOMENTUM_DAYS = 126          # ~6 months
VOL_LOOKBACK = 63
TARGET_VOL = 0.12
MAX_WEIGHT = 0.15
MIN_PRICE = 5.0
INITIAL_CAPITAL = 100000

# ============================================================
# GET NASDAQ-100 TICKERS
# ============================================================
def get_nasdaq100_tickers():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        tables = pd.read_html(url)
        for table in tables:
            if "Ticker" in table.columns or "Symbol" in table.columns:
                col = "Ticker" if "Ticker" in table.columns else "Symbol"
                tickers = table[col].astype(str).str.strip().str.upper().tolist()
                tickers = [t for t in tickers if t.isalpha() or "." in t]
                if len(tickers) > 80:
                    return tickers
    except:
        pass
    # Fallback list
    return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","AVGO","TSLA","COST",
            "NFLX","AMD","ADBE","PEP","CSCO","INTC","CMCSA","INTU","QCOM","AMGN",
            "HON","TXN","AMAT","SBUX","ISRG","BKNG","ADP","GILD","VRTX","REGN",
            "LRCX","MDLZ","ADI","PANW","KLAC","SNPS","CDNS","MAR","ORLY","CTAS"]

# ============================================================
# DOWNLOAD DATA
# ============================================================
print("Downloading data... this may take 1-2 minutes")
tickers = get_nasdaq100_tickers()
tickers = list(set(tickers + ["QQQ"]))  # Add benchmark

data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)["Close"]
data = data.dropna(axis=1, how="all")
print(f"Data loaded for {len(data.columns)} symbols")

# ============================================================
# BACKTEST ENGINE
# ============================================================
def run_backtest(prices):
    # Weekly dates
    weekly_prices = prices.resample("W-FRI").last()
    
    # 6-month momentum
    momentum = weekly_prices.pct_change(periods=26)  # ~26 weeks = 6 months
    
    portfolio_value = []
    dates = []
    current_weights = {}
    capital = INITIAL_CAPITAL
    
    valid_dates = momentum.dropna(how="all").index[26:]  # Need enough history
    
    for i, date in enumerate(valid_dates[:-1]):
        next_date = valid_dates[i+1]
        
        # Current momentum ranking
        mom = momentum.loc[date].dropna()
        last_prices = weekly_prices.loc[date]
        mom = mom[last_prices.reindex(mom.index) > MIN_PRICE]
        
        if len(mom) < TOP_N:
            continue
            
        top = mom.nlargest(TOP_N).index.tolist()
        
        # Volatility targeting
        daily_rets = prices.pct_change()
        vol_window = daily_rets.loc[:date].iloc[-VOL_LOOKBACK:]
        vols = vol_window[top].std() * np.sqrt(252)
        vols = vols.replace(0, np.nan).dropna()
        
        if len(vols) < 3:
            weights = {t: (1/len(top)) for t in top}
        else:
            inv_vol = 1 / vols
            raw_w = inv_vol / inv_vol.sum()
            raw_w = raw_w.clip(upper=MAX_WEIGHT)
            raw_w = raw_w / raw_w.sum()
            
            port_vol = np.sqrt((raw_w**2 * vols**2).sum())
            scale = min(TARGET_VOL / port_vol, 1.5) if port_vol > 0 else 1.0
            final_w = (raw_w * scale).clip(upper=MAX_WEIGHT)
            final_w = final_w / final_w.sum()
            weights = final_w.to_dict()
        
        # Calculate return for next week
        period_prices = weekly_prices.loc[date:next_date]
        if len(period_prices) < 2:
            continue
            
        week_ret = 0
        for ticker, w in weights.items():
            if ticker in period_prices.columns:
                r = period_prices[ticker].iloc[-1] / period_prices[ticker].iloc[0] - 1
                week_ret += w * r
        
        capital *= (1 + week_ret)
        portfolio_value.append(capital)
        dates.append(next_date)
        current_weights = weights
    
    results = pd.Series(portfolio_value, index=dates)
    return results

# ============================================================
# RUN BACKTEST
# ============================================================
print("\nRunning backtest...")
strategy = run_backtest(data)

# Benchmark (QQQ)
qqq = data["QQQ"].resample("W-FRI").last().dropna()
qqq = qqq.loc[strategy.index[0]:strategy.index[-1]]
qqq_value = INITIAL_CAPITAL * (qqq / qqq.iloc[0])

# ============================================================
# PERFORMANCE METRICS
# ============================================================
def performance_stats(series, name):
    total_return = series.iloc[-1] / series.iloc[0] - 1
    years = (series.index[-1] - series.index[0]).days / 365.25
    cagr = (series.iloc[-1] / series.iloc[0]) ** (1/years) - 1
    daily_rets = series.pct_change().dropna()
    vol = daily_rets.std() * np.sqrt(52)  # weekly to annual
    sharpe = cagr / vol if vol > 0 else 0
    max_dd = (series / series.cummax() - 1).min()
    
    print(f"\n===== {name} =====")
    print(f"Final Value:     ${series.iloc[-1]:,.0f}")
    print(f"Total Return:    {total_return*100:.1f}%")
    print(f"CAGR:            {cagr*100:.1f}%")
    print(f"Volatility:      {vol*100:.1f}%")
    print(f"Sharpe Ratio:    {sharpe:.2f}")
    print(f"Max Drawdown:    {max_dd*100:.1f}%")

print("\n" + "="*50)
print("BACKTEST RESULTS")
print("="*50)
performance_stats(strategy, "Momentum Strategy (Top 10)")
performance_stats(qqq_value, "QQQ Buy & Hold")

print("\nBacktest completed.")
