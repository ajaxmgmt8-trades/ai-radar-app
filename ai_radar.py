import streamlit as st
import pandas as pd
import requests
import datetime
from polygon import RESTClient
import finnhub

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🔥 AI Radar Pro — Market Scanner", layout="wide")

# 🔑 API KEYS
POLYGON_KEY = st.secrets["POLYGON_API_KEY"]
NEWS_API_KEY = st.secrets.get("NEWS_API_KEY", None)

polygon_client = RESTClient(POLYGON_KEY)
finnhub_client = finnhub.Client(api_key=NEWS_API_KEY) if NEWS_API_KEY else None

# Default watchlist (can be expanded)
WATCHLIST = ["AAPL", "NVDA", "TSLA", "SPY", "AMD", "MSFT", "META", "ORCL", "MDB", "GOOG", "NFLX","SPX","APP","NDX","SMCI","QUBT","IONQ","QBTS","SOFI","IBM","COST","MSTR","COIN","OSCR","LYFT","JOBY","ACHR","LLY","UNH","OPEN","UPST","NOW","ISRG","RR","FIG","HOOD","IBIT","WULF,"WOLF",OKLO,"APLD","HUT","SNPS","SE","ETHU","TSM","AVGO","BITF","HIMS","BULL","SPOT","LULU","CRCL","SOUN","QMMM","BMNR","SBET","GEMI","CRWV","KLAR","BABA","INTC","CMG","UAMY","IREN","BBAI","BRKB","TEM","GLD","IWM","LMND"]

# ---------------- HELPERS ----------------
def get_session_times():
    return {
        "premarket": (datetime.time(4, 0), datetime.time(9, 30)),
        "intraday": (datetime.time(9, 30), datetime.time(16, 0)),
        "postmarket": (datetime.time(16, 0), datetime.time(20, 0)),
    }

def get_session_data(ticker, session):
    start_t, end_t = get_session_times()[session]
    today = datetime.date.today()

    try:
        bars = polygon_client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="minute",
            from_=today.strftime("%Y-%m-%d"),
            to=today.strftime("%Y-%m-%d")
        )
    except Exception:
        return None, None

    if not bars:
        return None, None

    df = pd.DataFrame(bars)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
    df.set_index("timestamp", inplace=True)

    session_df = df.between_time(start_t, end_t)
    if session_df.empty:
        return None, None

    open_price = session_df.iloc[0]["open"]
    close_price = session_df.iloc[-1]["close"]
    change_pct = ((close_price - open_price) / open_price) * 100

    relvol = session_df["volume"].sum() / (df["volume"].sum() / 3)  # crude session relvol
    return round(change_pct, 2), round(relvol, 2)

def scan_session(session):
    movers = []
    for t in WATCHLIST:
        change, relvol = get_session_data(t, session)
        if change is not None:
            movers.append({"Ticker": t, "Change %": change, "RelVol": relvol})
    df = pd.DataFrame(movers).sort_values("Change %", ascending=False).head(10)
    df.index = range(1, len(df) + 1)
    return df

def get_stocktwits_feed(ticker):
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        data = r.json()
        msgs = [m["body"] for m in data.get("messages", [])[:3]]
        return msgs if msgs else ["No chatter found"]
    except Exception as e:
        return [f"StockTwits error: {e}"]

def get_catalysts():
    if not finnhub_client:
        return pd.DataFrame()
    news_items = []
    for t in WATCHLIST:
        try:
            res = finnhub_client.company_news(
                t,
                _from=str(datetime.date.today()),
                to=str(datetime.date.today())
            )
            for n in res[:2]:
                news_items.append({
                    "Ticker": t,
                    "Headline": n["headline"],
                    "Source": n["source"],
                    "Time": datetime.datetime.fromtimestamp(n["datetime"]).strftime("%H:%M")
                })
        except:
            pass
    return pd.DataFrame(news_items)

# ---------------- UI ----------------
st.title("🔥 AI Radar Pro — Market Scanner")
st.caption("Live Premarket, Intraday, Postmarket Movers with RelVol, StockTwits chatter & Catalysts")

query = st.text_input("🔍 Search a ticker (e.g., TSLA, NVDA, SPY)", "")

tabs = st.tabs(["📈 Premarket", "🌞 Intraday", "🌙 Postmarket", "💬 StockTwits Feed", "📰 Catalysts"])

with tabs[0]:
    st.subheader("Premarket Movers (04:00–09:30 ET)")
    df = scan_session("premarket")
    st.dataframe(df, use_container_width=True)

with tabs[1]:
    st.subheader("Intraday Movers (09:30–16:00 ET)")
    df = scan_session("intraday")
    st.dataframe(df, use_container_width=True)

with tabs[2]:
    st.subheader("Postmarket Movers (16:00–20:00 ET)")
    df = scan_session("postmarket")
    st.dataframe(df, use_container_width=True)

with tabs[3]:
    st.subheader("💬 StockTwits Feed")
    for t in WATCHLIST:
        st.markdown(f"### {t}")
        msgs = get_stocktwits_feed(t)
        for m in msgs:
            st.write(f"👉 {m}")

with tabs[4]:
    st.subheader("📰 Catalysts")
    df_cat = get_catalysts()
    if not df_cat.empty:
        st.dataframe(df_cat, use_container_width=True, height=400)
    else:
        st.info("No major catalysts today.")
