import streamlit as st
import pandas as pd
import yfinance as yf
from openai import OpenAI

# =====================
# CONFIG
# =====================
st.set_page_config(page_title="🔥 AI Radar Pro — Live Trading Assistant", layout="wide")
REFRESH_INTERVAL = 5000  # ms (5 seconds)

# Safe auto-refresh (handles old/new Streamlit)
try:
    st_autorefresh = st.autorefresh(interval=REFRESH_INTERVAL, key="refresh")
except AttributeError:
    try:
        st_autorefresh = st.experimental_autorefresh(interval=REFRESH_INTERVAL, key="refresh")
    except Exception:
        st.warning("⚠️ Auto-refresh not available in this Streamlit version. Please upgrade.")

# =====================
# HELPERS
# =====================
CORE_TICKERS = ["AAPL","MSFT","NVDA","TSLA","AMZN"]

def get_quote(ticker):
    try:
        data = yf.download(ticker, period="2d", interval="1m", prepost=True, progress=False)
        if data.empty:
            return None
        last = float(data["Close"].iloc[-1])
        prev = float(data["Close"].iloc[0])
        change = (last - prev) / prev * 100 if prev > 0 else 0
        vol = data["Volume"].sum()
        avg_vol = data["Volume"].mean()
        relvol = vol / avg_vol if avg_vol > 0 else 1
        return {"last": last, "prev": prev, "change": change, "relvol": relvol}
    except Exception:
        return None

def get_top_movers():
    movers = []
    for t in CORE_TICKERS:  # can expand universe
        q = get_quote(t)
        if not q:
            continue
        movers.append({
            "Ticker": t,
            "Price": q["last"],
            "Change %": q["change"],
            "RelVol": q["relvol"]
        })
    if not movers:
        return pd.DataFrame(columns=["Ticker","Price","Change %","RelVol"])
    df = pd.DataFrame(movers)
    df = df.sort_values("Change %", ascending=False).head(10).reset_index(drop=True)
    # Format
    df["Price"] = df["Price"].map(lambda x: f"${x:.2f}")
    df["Change %"] = df["Change %"].map(lambda x: f"{x:+.2f}%")
    df["RelVol"] = df["RelVol"].map(lambda x: f"{x:.2f}x")
    return df

def ai_playbook(ticker, change, catalyst=""):
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        prompt = f"""
        Ticker: {ticker}
        Change: {change:+.2f}%
        Catalyst: {catalyst}

        Generate setups:
        1) Sentiment
        2) Scalp plan (1-5m)
        3) Day trade plan (15-30m)
        4) Swing plan (4H-Daily)
        """
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3,
            max_tokens=300
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"⚠️ AI Playbook unavailable: {e}"

# =====================
# MAIN LAYOUT
# =====================
st.title("🔥 AI Radar Pro — Live Trading Assistant")

tabs = st.tabs(["📊 Premarket", "📈 Intraday", "🌙 Postmarket", "🤖 AI Playbooks"])

# -------- Session Tabs --------
for i, session in enumerate(["Premarket","Intraday","Postmarket"]):
    with tabs[i]:
        st.subheader(f"{session} Movers")
        st.markdown("### 🔥 Top 10 Market Movers")
        st.dataframe(get_top_movers(), use_container_width=True)

        st.markdown("### 📌 Your Watchlist")
        wl_data = []
        for t in CORE_TICKERS:
            q = get_quote(t)
            if not q: continue
            wl_data.append({
                "Ticker": t,
                "Price": f"${q['last']:.2f}",
                "Change %": f"{q['change']:+.2f}%",
                "RelVol": f"{q['relvol']:.2f}x"
            })
        if wl_data:
            st.dataframe(pd.DataFrame(wl_data), use_container_width=True)

# -------- Playbook Tab --------
with tabs[3]:
    st.subheader("🤖 AI Playbooks")
    ticker = st.text_input("Enter Ticker", value="AAPL").upper()
    catalyst = st.text_input("Catalyst (optional)")
    if st.button("Generate AI Playbook"):
        q = get_quote(ticker)
        change = q["change"] if q else 0
        pb = ai_playbook(ticker, change, catalyst)
        st.markdown(pb)

# Footer
st.markdown("---")
st.caption("🔥 Live data via yfinance | AI via OpenAI | Built with Streamlit")
