
import streamlit as st
import requests
import pandas as pd
import os

# === PAGE CONFIG ===
st.set_page_config(page_title="AI Radar - UW Edition", layout="wide")

# === UW API SETUP ===
UNUSUAL_WHALES_KEY = st.secrets.get("UNUSUAL_WHALES_KEY", "YOUR_BACKUP_KEY")

HEADERS = {
    "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
    "accept": "application/json, text/plain"
}

# === SIDEBAR ===
st.sidebar.title("🔌 Unusual Whales Connection")

# === API TEST ===
def test_uw_api():
    try:
        res = requests.get("https://phx.unusualwhales.com/api/historic_chains/flow/SPY", headers=HEADERS)
        if res.status_code == 200:
            st.sidebar.success("✅ Connected to Unusual Whales")
            st.sidebar.write(res.json()[:2])
        else:
            st.sidebar.error(f"UW API failed: {res.status_code}")
    except Exception as e:
        st.sidebar.error(f"Exception: {e}")

test_uw_api()

# === MAIN TABS ===
tabs = st.tabs(["🎯 Lottos", "📈 Flow", "📅 Earnings", "📰 News"])

# === TAB: LOTTOS ===
with tabs[0]:
    st.header("🎯 Lotto Contracts (UW API)")
    try:
        r = requests.get("https://phx.unusualwhales.com/api/historic_chains/lottos/all", headers=HEADERS)
        data = r.json()
        df = pd.DataFrame(data)
        if not df.empty:
            st.dataframe(df)
    except Exception as e:
        st.error(f"Failed to load Lottos: {e}")

# === TAB: FLOW ===
with tabs[1]:
    st.header("📈 Recent Flow (SPY)")
    try:
        r = requests.get("https://phx.unusualwhales.com/api/historic_chains/flow/SPY", headers=HEADERS)
        data = r.json()
        df = pd.DataFrame(data)
        if not df.empty:
            st.dataframe(df)
    except Exception as e:
        st.error(f"Failed to load flow: {e}")

# === TAB: EARNINGS ===
with tabs[2]:
    st.header("📅 Upcoming Earnings")
    try:
        r = requests.get("https://phx.unusualwhales.com/api/equities/earnings/SPY", headers=HEADERS)
        data = r.json()
        df = pd.DataFrame(data)
        if not df.empty:
            st.dataframe(df)
    except Exception as e:
        st.error(f"Failed to load earnings: {e}")

# === TAB: NEWS ===
with tabs[3]:
    st.header("📰 Important News")
    try:
        r = requests.get("https://phx.unusualwhales.com/api/news", headers=HEADERS)
        data = r.json()
        df = pd.DataFrame(data)
        if not df.empty:
            st.dataframe(df)
    except Exception as e:
        st.error(f"Failed to load news: {e}")

