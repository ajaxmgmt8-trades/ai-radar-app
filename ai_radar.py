import streamlit as st
import requests
from datetime import date

# Load API key securely
UNUSUAL_WHALES_KEY = st.secrets["UNUSUAL_WHALES_KEY"]
HEADERS = {
    "Authorization": f"Bearer {UNUSUAL_WHALES_KEY}",
    "accept": "application/json, text/plain"
}

# Supported endpoints from documentation
endpoints = {
    "📊 Stock State": "/api/stock/{ticker}/stock-state",
    "📈 Options Volume": "/api/stock/{ticker}/options-volume",
    "📉 Volatility (Realized)": "/api/stock/{ticker}/volatility/realized",
    "📊 Volatility Stats": "/api/stock/{ticker}/volatility/stats",
    "📉 Volatility Term Structure": "/api/stock/{ticker}/volatility/term-structure",
    "📊 Spot GEX (1min)": "/api/stock/{ticker}/spot-exposures",
    "🎯 Spot GEX (By Strike)": "/api/stock/{ticker}/spot-exposures/strike",
    "📆 Spot GEX (By Expiry & Strike)": "/api/stock/{ticker}/spot-exposures/expiry-strike",
    "📈 Price Levels (Lit/Off)": "/api/stock/{ticker}/stock-volume-price-levels",
    "🕰️ Intraday Stats": "/api/stock/{ticker}/intraday/stats",
    "📅 Option Chains by Date": "/api/stock/{ticker}/option-chains?date={date}",
    "📜 Historic Chains by Date": "/api/historic_chains/{ticker}?date={date}"
}

# ──────────────────────────────────────────────────────────────
st.title("🧪 Test Unusual Whales API Endpoints")

endpoint_key = st.selectbox("Choose an endpoint to test:", list(endpoints.keys()))
ticker = st.text_input("Ticker Symbol", value="AAPL")
selected_date = st.date_input("Select Date (if needed)", value=date.today())

if st.button("Run Test"):
    path_template = endpoints[endpoint_key]
    path = path_template.replace("{ticker}", ticker.upper())
    if "{date}" in path:
        path = path.replace("{date}", selected_date.isoformat())

    # Handle query parameters if needed
    if "?" in path:
        base_path, query = path.split("?")
        query_params = dict(param.split("=") for param in query.split("&"))
        query_params["date"] = selected_date.isoformat()
        url = f"https://api.unusualwhales.com{base_path}"
        params = query_params
    else:
        url = f"https://api.unusualwhales.com{path}"
        params = {}

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        st.success("✅ Connected!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ──────────────────────────────────────────────────────────────
st.divider()
st.header("📜 Option Contract History Test")

contract_id = st.text_input("Enter Option Contract ID (e.g. TSLA230526P00167500)")
limit = st.number_input("Limit (optional)", min_value=1, value=1, step=1)

if st.button("Run Option Contract History"):
    url = f"https://api.unusualwhales.com/api/option-contract/{contract_id}/historic"
    params = {"limit": int(limit)} if limit else {}

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        st.success("✅ Data retrieved!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ──────────────────────────────────────────────────────────────
st.divider()
st.header("🕰️ Option Contract Intraday Data Test")

intraday_contract_id = st.text_input("Intraday Contract ID (e.g. TSLA230526P00167500)")
intraday_date = st.date_input("Intraday Date", value=date.today(), key="intraday_date")

if st.button("Run Intraday Option Contract Test"):
    url = f"https://api.unusualwhales.com/api/option-contract/{intraday_contract_id}/intraday"
    params = {"date": intraday_date.isoformat()}

    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        st.success("✅ Intraday Data Retrieved!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ──────────────────────────────────────────────────────────────
st.divider()
st.header("📊 Historical Earnings (Ticker)")

earnings_ticker = st.text_input("Earnings Ticker Symbol", value="AAPL", key="earnings_ticker")

if st.button("Run Earnings Test"):
    url = f"https://api.unusualwhales.com/api/earnings/{earnings_ticker.upper()}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        st.success("✅ Earnings Data Retrieved!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")



# ──────────────────────────────────────────────────────────────
st.divider()
st.header("📅 Afterhours Earnings (By Date)")

afterhours_date = st.date_input("Afterhours Date", value=date.today(), key="afterhours_date")
if st.button("Run Afterhours Earnings Test"):
    url = "https://api.unusualwhales.com/api/earnings/afterhours"
    params = {"date": afterhours_date.isoformat()}
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        st.success("✅ Afterhours Earnings Retrieved!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ──────────────────────────────────────────────────────────────
st.divider()
st.header("🚨 Alerts Triggered")

if st.button("Run Alerts Triggered Test"):
    url = "https://api.unusualwhales.com/api/alerts"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        st.success("✅ Alerts Retrieved!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ──────────────────────────────────────────────────────────────
st.divider()
st.header("🛠️ Alert Configurations")

if st.button("Run Alert Configs Test"):
    url = "https://api.unusualwhales.com/api/alerts/configuration"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        st.success("✅ Alert Configs Retrieved!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ──────────────────────────────────────────────────────────────
st.divider()
st.header("🏛️ Recent Reports by Trader")

congress_date = st.date_input("Congress Date", value=date.today(), key="congress_date")
congress_ticker = st.text_input("Congress Ticker", value="AAPL", key="congress_ticker")
if st.button("Run Reports by Trader"):
    url = "https://api.unusualwhales.com/api/congress/congress-trader"
    params = {
        "date": congress_date.isoformat(),
        "ticker": congress_ticker.upper()
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        st.success("✅ Congress Reports Retrieved!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ──────────────────────────────────────────────────────────────
st.divider()
st.header("🏛️ Recent Congress Trades")

recent_congress_date = st.date_input("Recent Congress Date", value=date.today(), key="recent_congress_date")
recent_congress_ticker = st.text_input("Recent Congress Ticker", value="AAPL", key="recent_congress_ticker")
if st.button("Run Recent Congress Trades"):
    url = "https://api.unusualwhales.com/api/congress/recent-trades"
    params = {
        "date": recent_congress_date.isoformat(),
        "ticker": recent_congress_ticker.upper()
    }
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        st.success("✅ Recent Congress Trades Retrieved!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ──────────────────────────────────────────────────────────────
st.divider()
st.header("🌑 Recent Darkpool Trades")

darkpool_date = st.date_input("Darkpool Date", value=date.today(), key="darkpool_date")
if st.button("Run Recent Darkpool Trades"):
    url = "https://api.unusualwhales.com/api/darkpool/recent"
    params = {"date": darkpool_date.isoformat()}
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        st.success("✅ Darkpool Data Retrieved!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ──────────────────────────────────────────────────────────────
st.divider()
st.header("🌑 Ticker Darkpool Trades")

ticker_darkpool = st.text_input("Darkpool Ticker", value="AAPL", key="darkpool_ticker")
ticker_darkpool_date = st.date_input("Ticker Darkpool Date", value=date.today(), key="darkpool_ticker_date")
if st.button("Run Ticker Darkpool Trades"):
    url = f"https://api.unusualwhales.com/api/darkpool/{ticker_darkpool.upper()}"
    params = {"date": ticker_darkpool_date.isoformat()}
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=10)
        response.raise_for_status()
        st.success("✅ Ticker Darkpool Data Retrieved!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

# ──────────────────────────────────────────────────────────────
st.divider()
st.header("📦 ETF Exposure for Ticker")

etf_ticker = st.text_input("ETF Ticker", value="AAPL", key="etf_ticker")
if st.button("Run ETF Exposure"):
    url = f"https://api.unusualwhales.com/api/etfs/{etf_ticker.upper()}/exposure"
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        st.success("✅ ETF Exposure Retrieved!")
        st.json(response.json())
    except requests.exceptions.HTTPError as e:
        st.error(f"❌ HTTP Error: {e}")
    except Exception as e:
        st.error(f"❌ Error: {e}")

