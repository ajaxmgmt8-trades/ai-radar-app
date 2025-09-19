
import streamlit as st
import requests
from datetime import date

# Load API key securely
UNUSUAL_WHALES_KEY = st.secrets["UNUSUAL_WHALES_KEY"]
HEADERS = {
    "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
    "accept": "application/json, text/plain"
}

# Comprehensive endpoint dictionary
endpoints = {
    "Institutions - Details": "/api/institutions?name={name}",
    "Institutions - Latest Filings": "/api/institutions/latest_filings?name={name}&date={date}",
    "Market - Correlations": "/api/market/correlations",
    "Market - Economic Calendar": "/api/market/economic-calendar",
    "Market - FDA Calendar": "/api/market/fda-calendar?ticker={ticker}",
    "Market - Insider Buy/Sells": "/api/market/insider-buy-sells",
    "Market - Market Tide": "/api/market/market-tide?date={date}",
    "Market - OI Change": "/api/market/oi-change?date={date}",
    "Market - Sector ETFs": "/api/market/sector-etfs",
    "Market - Spike": "/api/market/spike?date={date}",
    "Market - Top Net Impact": "/api/market/top-net-impact?date={date}",
    "Market - Total Options Volume": "/api/market/total-options-volume",
    "Sector Tide (Basic Materials)": "/api/market/Basic%20Materials/sector-tide?date={date}",
    "ETF Tide for Ticker": "/api/market/{ticker}/etf-tide?date={date}",
    "Net Flow by Expiry": "/api/net-flow/expiry?date={date}",
    "News Headlines": "/api/news/headlines",
    "Option Contract - Flow": "/api/option-contract/{contract}/flow?date={date}",
    "Option Contract - Historic": "/api/option-contract/{contract}/historic",
    "Option Contract - Intraday": "/api/option-contract/{contract}/intraday?date={date}",
    "Option Contract - Volume Profile": "/api/option-contract/{contract}/volume-profile?date={date}",
    "Stock - Expiry Breakdown": "/api/stock/{ticker}/expiry-breakdown?date={date}",
    "Stock - Option Contracts": "/api/stock/{ticker}/option-contracts",
    "Flow Alerts": "/api/option-trades/flow-alerts?all_opening=true&is_floor=true&is_sweep=true&is_call=true&is_put=true&is_ask_side=true&is_bid_side=true&is_otm=true",
    "Full Tape": "/api/option-trades/full-tape/{date}",
    "Analyst Ratings": "/api/screener/analysts?ticker={ticker}",
    "Screener OTM Contracts": "/api/screener/option-contracts?is_otm=true&date={date}",
    "Screener Stocks": "/api/screener/stocks?ticker={ticker}&date={date}",
    "Shorts - Data": "/api/shorts/{ticker}/data",
    "Shorts - FTDs": "/api/shorts/{ticker}/ftds",
    "Shorts - Interest Float": "/api/shorts/{ticker}/interest-float",
    "Shorts - Volume and Ratio": "/api/shorts/{ticker}/volume-and-ratio",
    "Shorts - Volumes by Exchange": "/api/shorts/{ticker}/volumes-by-exchange",
    "Stock - Sector Tickers": "/api/stock/{sector}/tickers",
    "Stock - ATM Chains": "/api/stock/{ticker}/atm-chains",
    "Stock - Flow Alerts": "/api/stock/{ticker}/flow-alerts?is_ask_side=true&is_bid_side=true",
    "Stock - Flow Per Expiry": "/api/stock/{ticker}/flow-per-expiry",
    "Stock - Flow Per Strike": "/api/stock/{ticker}/flow-per-strike?date={date}",
    "Stock - Flow Per Strike Intraday": "/api/stock/{ticker}/flow-per-strike-intraday?date={date}",
    "Stock - Flow Recent": "/api/stock/{ticker}/flow-recent",
    "Stock - Greek Exposure": "/api/stock/{ticker}/greek-exposure?date={date}",
    "Stock - Greek Exposure by Expiry": "/api/stock/{ticker}/greek-exposure/expiry?date={date}",
    "Stock - Greek Exposure by Strike": "/api/stock/{ticker}/greek-exposure/strike?date={date}",
    "Stock - Greek Exposure by Strike & Expiry": "/api/stock/{ticker}/greek-exposure/strike-expiry?date={date}",
    "Stock - Greek Flow": "/api/stock/{ticker}/greek-flow?date={date}",
    "Stock - Greeks": "/api/stock/{ticker}/greeks?date={date}",
}

st.title("🧪 Unusual Whales API Tester")

endpoint_choice = st.selectbox("Choose an Endpoint", list(endpoints.keys()))
ticker = st.text_input("Ticker", value="AAPL")
contract = st.text_input("Contract ID", value="TSLA230526P00167500")
sector = st.text_input("Sector", value="Basic Materials")
name = st.text_input("Institution Name", value="VANGUARD GROUP INC")
selected_date = st.date_input("Date", value=date.today())

# Format endpoint path
raw_path = endpoints[endpoint_choice]
path = raw_path.replace("{ticker}", ticker).replace("{contract}", contract).replace("{sector}", sector).replace("{name}", name).replace("{date}", selected_date.isoformat())

url = f"https://api.unusualwhales.com{path}"

if st.button("Run API Request"):
    try:
        response = requests.get(url, headers=HEADERS, timeout=15)
        response.raise_for_status()
        st.success("✅ API Response:")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"HTTP Error: {e}")
    except Exception as e:
        st.error(f"Other Error: {e}")
