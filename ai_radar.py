
import streamlit as st
from unusual_whales_api_module import *

st.title("🧪 Unusual Whales API Test Harness")

ticker = st.text_input("Enter Ticker", "AAPL")

if st.button("🔄 Get Stock State"):
    result = get_stock_state(ticker)
    st.subheader("Stock State")
    st.json(result)

if st.button("📊 Get Option Chains"):
    result = get_option_chains(ticker)
    st.subheader("Option Chains")
    st.json(result)

if st.button("🐋 Get Recent Option Flow"):
    result = get_recent_flow(ticker)
    st.subheader("Recent Flow")
    st.json(result)

if st.button("🎯 Get Flow by Strike"):
    result = get_flow_by_strike(ticker)
    st.subheader("Flow by Strike")
    st.json(result)

if st.button("📆 Get Flow by Expiry"):
    result = get_flow_by_expiry(ticker)
    st.subheader("Flow by Expiry")
    st.json(result)

if st.button("⏪ Get Historic Trades"):
    result = get_historic_trades(ticker)
    st.subheader("Historic Trades")
    st.json(result)

if st.button("🧠 Get Earnings (This Ticker)"):
    result = get_earnings_for_ticker(ticker)
    st.subheader("Earnings for Ticker")
    st.json(result)

if st.button("☀️ Get Earnings (Premarket)"):
    result = get_earnings_premarket()
    st.subheader("Premarket Earnings")
    st.json(result)

if st.button("🌙 Get Earnings (Afterhours)"):
    result = get_earnings_afterhours()
    st.subheader("Afterhours Earnings")
    st.json(result)

if st.button("🚨 Get Flow Alerts"):
    result = get_flow_alerts(ticker)
    st.subheader("Flow Alerts")
    st.json(result)

