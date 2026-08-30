import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import plotly.express as px

st.set_page_config(
    page_title="Momentum Rankings",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { max-width: 520px; margin: auto; }
    .status-healthy {
        background-color: #d4edda;
        color: #155724;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
    .status-cash {
        background-color: #f8d7da;
        color: #721c24;
        padding: 12px;
        border-radius: 8px;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Nasdaq-100 Momentum")
st.caption("6-month momentum ranking + Crash Protection")

# ============================================================
# MARKET FILTER STATUS
# ============================================================
st.subheader("Market Filter Status")

@st.cache_data(ttl=300)
def get_market_status():
    data = yf.download("QQQ", period="1y", auto_adjust=True, progress=False)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    current = float(close.iloc[-1])
    ma100 = float(close.rolling(100).mean().iloc[-1])
    ma55 = float(close.rolling(55).mean().iloc[-1])

    return current, ma100, ma55

try:
    qqq_price, ma100, ma55 = get_market_status()

    col1, col2, col3 = st.columns(3)
    col1.metric("QQQ Price", f"${qqq_price:.2f}")
    col2.metric("100-day MA (Exit)", f"${ma100:.2f}")
    col3.metric("55-day MA (Re-entry)", f"${ma55:.2f}")

    # Simple status logic for display
    if qqq_price > ma100:
        st.markdown('<div class="status-healthy">✅ HEALTHY – Invested Mode</div>', unsafe_allow_html=True)
        st.caption("QQQ is above 100-day MA → Robot is allowed to hold stocks")
    elif qqq_price > ma55:
        st.markdown('<div class="status-healthy">🟡 WATCH – Between 55 & 100 MA</div>', unsafe_allow_html=True)
        st.caption("QQQ is between 55-day and 100-day MA")
    else:
        st.markdown('<div class="status-cash">🛡️ CASH MODE – Protection Active</div>', unsafe_allow_html=True)
        st.caption("QQQ is below 55-day MA → Robot stays in cash")

except Exception as e:
    st.warning("Could not load market filter data.")

st.markdown("---")

# ============================================================
# PORTFOLIO SECTION
# ============================================================
st.subheader("Your Portfolio")

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

if API_KEY and API_SECRET:
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(API_KEY, API_SECRET, paper=False)
        account = client.get_account()
        equity = float(account.equity)
        cash = float(account.cash)
        col1, col2 = st.columns(2)
        col1.metric("Account Equity", f"${equity:,.2f}")
        col2.metric("Cash", f"${cash:,.2f}")
    except:
        st.info("Could not load Alpaca account.")
else:
    st.info("Add ALPACA_API_KEY and ALPACA_SECRET_KEY in Streamlit Secrets to see portfolio.")

st.markdown("---")

# ============================================================
# CURRENT RANKING
# ============================================================
top_n = st.selectbox("Show Top", [5, 10, 15, 20], index=1)

@st.cache_data(ttl=1800)
def get_nasdaq100():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    try:
        import requests
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        tables = pd.read_html(response.text)
        for table in tables:
            if "Ticker" in table.columns or "Symbol" in table.columns:
                col = "Ticker" if "Ticker" in table.columns else "Symbol"
                tickers = table[col].astype(str).str.strip().str.upper().tolist()
                tickers = [t for t in tickers if t.isalpha() or "." in t]
                if len(tickers) > 80:
                    return tickers
    except:
        pass
    return ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","AVGO","TSLA","COST","NFLX",
            "AMD","ADBE","CSCO","INTC","AMAT","PANW","CRWD","FTNT","DDOG","TEAM",
            "MU","STX","WDC","MRVL","LRCX","KLAC","SNPS","CDNS"]

@st.cache_data(ttl=1800)
def get_price_data(tickers):
    end = datetime.now()
    start = end - timedelta(days=400)
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    return data.dropna(axis=1, how="all")

@st.cache_data(ttl=1800)
def calculate_current_ranking(data, top_n):
    momentum = data.pct_change(periods=126).iloc[-1].dropna()
    last_price = data.iloc[-1]
    momentum = momentum[last_price.reindex(momentum.index) > 5.0]
    ranked = momentum.sort_values(ascending=False)

    df = pd.DataFrame({
        "Rank": range(1, len(ranked)+1),
        "Symbol": ranked.index,
        "Momentum": (ranked.values * 100).round(1),
        "Price": last_price.reindex(ranked.index).round(2).values
    })
    df["Momentum"] = df["Momentum"].astype(str) + "%"
    return df, ranked.head(top_n).index.tolist()

@st.cache_data(ttl=1800)
def calculate_historical_ranks(data, current_top_symbols, weeks=20):
    weekly = data.resample("W-FRI").last()
    records = []

    for i in range(weeks, 0, -1):
        if i >= len(weekly):
            continue
        date = weekly.index[-i]
        window = weekly.loc[:date]
        if len(window) < 27:
            continue
        mom = window.pct_change(periods=26).iloc[-1].dropna()
        mom = mom.sort_values(ascending=False)
        rank_map = {sym: rank+1 for rank, sym in enumerate(mom.index)}

        for sym in current_top_symbols:
            if sym in rank_map:
                records.append({
                    "Date": date,
                    "Symbol": sym,
                    "Rank": rank_map[sym]
                })

    return pd.DataFrame(records)

# Main
with st.spinner("Loading data..."):
    tickers = get_nasdaq100()
    price_data = get_price_data(tickers)
    ranking, top_symbols = calculate_current_ranking(price_data, top_n)

st.subheader(f"Top {top_n} Stocks")
st.dataframe(
    ranking.head(top_n),
    use_container_width=True,
    hide_index=True
)

st.markdown("---")
st.subheader("Historical Rank Performance")
st.caption("How the current top stocks ranked over the past 20 weeks (lower is better)")

with st.spinner("Calculating historical ranks..."):
    hist_df = calculate_historical_ranks(price_data, top_symbols, weeks=20)

if not hist_df.empty:
    fig = px.line(
        hist_df,
        x="Date",
        y="Rank",
        color="Symbol",
        markers=True,
        title="Historical Rank Performance"
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=-0.4),
        margin=dict(l=10, r=10, t=40, b=10)
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Not enough historical data to build the chart.")

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
