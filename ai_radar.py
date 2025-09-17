import requests
import streamlit as st
from datetime import date

UNUSUAL_WHALES_KEY = st.secrets["UNUSUAL_WHALES_KEY"]
headers = {
    "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
    "accept": "application/json"
}

st.title("🧪 Test UW - Chains by Date")

ticker = st.text_input("Ticker", value="AAPL")
chosen_date = st.date_input("Choose Date", value=date.today())

if st.button("Test UW Chains by Date"):
    url = f"https://api.unusualwhales.com/api/option_chains/{ticker}/chains/date/{chosen_date}"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        st.success("✅ Connected")
        st.json(r.json())
    except Exception as e:
        st.error(f"❌ Error: {e}")
