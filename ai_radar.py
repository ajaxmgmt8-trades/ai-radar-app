import streamlit as st
import requests

st.title("🔑 Polygon API Test")

# Load your key from secrets
try:
    POLYGON_KEY = st.secrets["POLYGON_API_KEY"]
except KeyError:
    st.error("⚠️ Missing POLYGON_API_KEY in your secrets.toml")
    st.stop()

def test_trade():
    url = f"https://api.polygon.io/v2/last/trade/AAPL?apiKey={POLYGON_KEY}"
    return requests.get(url, timeout=10).json()

def test_prev_close():
    url = f"https://api.polygon.io/v2/aggs/ticker/AAPL/prev?apiKey={POLYGON_KEY}"
    return requests.get(url, timeout=10).json()

def test_snapshot():
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/gainers?apiKey={POLYGON_KEY}"
    return requests.get(url, timeout=10).json()

st.subheader("🔎 Testing Polygon Endpoints")

if st.button("Run Tests"):
    st.markdown("### Last Trade (AAPL)")
    st.json(test_trade())

    st.markdown("### Previous Close (AAPL)")
    st.json(test_prev_close())

    st.markdown("### Snapshot Gainers")
    st.json(test_snapshot())
