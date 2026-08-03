import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Nasdaq-100 Momentum Leaderboard", layout="wide")

st.title("Nasdaq-100 Momentum Rankings")
st.caption("Stocks ranked by **6-month momentum** score")

# Sidebar
st.sidebar.header("Settings")
top_n = st.sidebar.slider("Show Top N stocks", 5, 30, 10)
min_price = st.sidebar.number_input("Minimum price ($)", 1.0, 50.0, 5.0)

@st.cache_data(ttl=3600)
def get_nasdaq100():
    url = "https://en.wikipedia.org/wiki/Nasdaq-100"
    tables = pd.read_html(url)
    for table in tables:
        if "Ticker" in table.columns or "Symbol" in table.columns:
            col = "Ticker" if "Ticker" in table.columns else "Symbol"
            tickers = table[col].astype(str).str.strip().str.upper().tolist()
            tickers = [t for t in tickers if t.isalpha() or "." in t]
            if len(tickers) > 80:
                return tickers
    return ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "TSLA", "COST", "NFLX"]

@st.cache_data(ttl=3600)
def calculate_momentum(tickers):
    end = datetime.now()
    start = end - timedelta(days=250)   # enough history for 6 months

    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    data = data.dropna(axis=1, how="all")

    # === 6-MONTH MOMENTUM ===
    # Return over the last 6 months (approximately 126 trading days)
    momentum = data.pct_change(periods=126).iloc[-1].dropna()

    # Price filter
    last_price = data.iloc[-1]
    momentum = momentum[last_price.reindex(momentum.index) > min_price]

    # Create ranking table
    df = pd.DataFrame({
        "Symbol": momentum.index,
        "Momentum Score": momentum.values
    })

    df = df.sort_values("Momentum Score", ascending=False).reset_index(drop=True)
    df["Rank"] = df.index + 1
    df["Momentum Score"] = (df["Momentum Score"] * 100).round(1).astype(str) + "%"
    df["Current Price"] = last_price.reindex(df["Symbol"]).round(2).values

    return df[["Rank", "Symbol", "Momentum Score", "Current Price"]]

# Main app
with st.spinner("Calculating 6-month momentum for Nasdaq-100..."):
    tickers = get_nasdaq100()
    ranking = calculate_momentum(tickers)

st.subheader(f"Top {top_n} Stocks – 6-Month Momentum")
st.dataframe(
    ranking.head(top_n),
    use_container_width=True,
    hide

