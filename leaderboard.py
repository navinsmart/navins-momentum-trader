import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Momentum Rankings",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Mobile-friendly styling
st.markdown("""
<style>
    .stApp {
        max-width: 480px;
        margin: auto;
        padding-top: 1rem;
    }
    .big-number {
        font-size: 28px;
        font-weight: 700;
        color: #00C853;
    }
    .card {
        background: #1E1E1E;
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📈 Nasdaq-100 Momentum")
st.caption("6-month momentum ranking")

# ========== ALPACA PORTFOLIO SECTION ==========
st.subheader("Your Portfolio")

API_KEY = os.getenv("ALPACA_API_KEY")
API_SECRET = os.getenv("ALPACA_SECRET_KEY")

if API_KEY and API_SECRET:
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(API_KEY, API_SECRET, paper=False)  # Change to True if using paper
        account = client.get_account()
        equity = float(account.equity)
        cash = float(account.cash)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Account Equity", f"${equity:,.2f}")
        with col2:
            st.metric("Cash", f"${cash:,.2f}")
    except Exception as e:
        st.info("Could not load Alpaca account. Check your API keys.")
else:
    st.info("Add ALPACA_API_KEY and ALPACA_SECRET_KEY to see your portfolio.")

st.markdown("---")

# ========== MOMENTUM LEADERBOARD ==========
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
            "AMD","ADBE","CSCO","INTC","AMAT","PANW","CRWD","FTNT","DDOG","TEAM"]

@st.cache_data(ttl=1800)
def calculate_momentum(tickers):
    end = datetime.now()
    start = end - timedelta(days=250)
    
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    data = data.dropna(axis=1, how="all")
    
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
    return df

with st.spinner("Updating rankings..."):
    tickers = get_nasdaq100()
    ranking = calculate_momentum(tickers)

st.subheader(f"Top {top_n} Stocks")
st.dataframe(
    ranking.head(top_n),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank": st.column_config.NumberColumn(width="small"),
        "Symbol": st.column_config.TextColumn(width="medium"),
        "Momentum": st.column_config.TextColumn(width="medium"),
        "Price": st.column_config.NumberColumn(format="$%.2f"),
    }
)

st.markdown("---")
st.caption(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
