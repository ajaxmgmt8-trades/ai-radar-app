# test_unusual_whales.py
import streamlit as st
from ai_radar_fixed import (  # or copy-paste the functions directly
    get_stock_state,
    get_option_chains,
    get_recent_flow,
    get_flow_by_strike,
    get_flow_by_expiry,
    get_historic_trades,
    get_earnings_premarket,
    get_earnings_afterhours
)

st.title("🧪 Unusual Whales API Test")

ticker = st.text_input("Enter Ticker", "AAPL")

if st.button("Test Stock State"):
    st.json(get_stock_state(ticker))

if st.button("Test Option Chains"):
    st.json(get_option_chains(ticker))

if st.button("Test Recent Flow"):
    st.json(get_recent_flow(ticker))

if st.button("Test Flow by Strike"):
    st.json(get_flow_by_strike(ticker))

if st.button("Test Flow by Expiry"):
    st.json(get_flow_by_expiry(ticker))

if st.button("Test Historic Trades"):
    st.json(get_historic_trades(ticker))

st.divider()

if st.button("Test Premarket Earnings"):
    st.json(get_earnings_premarket())

if st.button("Test Afterhours Earnings"):
    st.json(get_earnings_afterhours())
