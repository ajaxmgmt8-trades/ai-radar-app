import streamlit as st
import pandas as pd
import requests
import datetime
import json
import plotly.graph_objects as go
from polygon import RESTClient
from zoneinfo import ZoneInfo
import openai
import os
from typing import Dict, List, Optional

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🔥 AI Radar Pro", layout="wide")
TZ_CT = ZoneInfo("America/Chicago")
WATCHLIST_FILE = "watchlists.json"

# 🔑 API KEYS - Fixed OpenAI client initialization
try:
    POLYGON_KEY = st.secrets["POLYGON_API_KEY"]
    OPENAI_KEY = st.secrets["OPENAI_API_KEY"]
    polygon_client = RESTClient(POLYGON_KEY)
    openai_client = openai.OpenAI(api_key=OPENAI_KEY)
except Exception as e:
    st.error(f"API Key Error: {e}")
    st.stop()

# ---------------- WATCHLIST PERSISTENCE ----------------
def load_watchlists() -> Dict:
    try:
        with open(WATCHLIST_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"Default": ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"]}
    except json.JSONDecodeError:
        st.error("Corrupted watchlist file. Creating new one.")
        return {"Default": ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"]}

def save_watchlists(watchlists: Dict):
    try:
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(watchlists, f, indent=2)
    except Exception as e:
        st.error(f"Error saving watchlists: {e}")

# Initialize session state
if "watchlists" not in st.session_state:
    st.session_state.watchlists = load_watchlists()
if "active_watchlist" not in st.session_state:
    st.session_state.active_watchlist = list(st.session_state.watchlists.keys())[0]
if "show_sparklines" not in st.session_state:
    st.session_state.show_sparklines = True

# ---------------- HELPERS ----------------
@st.cache_data(ttl=30)  # Cache for 30 seconds
def get_quote(ticker: str) -> Dict:
    """Get real-time quote data from Polygon."""
    url = f"https://api.polygon.io/v2/last/nbbo/{ticker}?apiKey={POLYGON_KEY}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        
        if "results" not in data:
            return {"last": 0, "bid": 0, "ask": 0, "error": "No data"}
            
        q = data["results"]
        return {
            "last": q.get("p", 0),
            "bid": q.get("bP", 0),
            "ask": q.get("aP", 0),
            "error": None
        }
    except requests.RequestException as e:
        st.warning(f"Quote error for {ticker}: {e}")
        return {"last": 0, "bid": 0, "ask": 0, "error": str(e)}
    except Exception as e:
        return {"last": 0, "bid": 0, "ask": 0, "error": str(e)}

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_previous_close(ticker: str) -> float:
    """Get previous day's close price."""
    yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        bars = list(polygon_client.get_aggs(
            ticker, multiplier=1, timespan="day",
            from_=yesterday, to=yesterday, limit=1
        ))
        if bars:
            return bars[0].close
        return 0
    except Exception as e:
        st.warning(f"Previous close error for {ticker}: {e}")
        return 0

@st.cache_data(ttl=300)  # Cache for 5 minutes  
def get_intraday_sparkline(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch today's intraday candles for sparkline."""
    today = datetime.date.today().strftime("%Y-%m-%d")
    try:
        bars = list(polygon_client.get_aggs(
            ticker, multiplier=5, timespan="minute",
            from_=today, to=today, limit=78  # ~6.5 hours of 5-min bars
        ))
        
        if not bars:
            return None
            
        data = []
        for bar in bars:
            data.append({
                "timestamp": bar.timestamp,
                "open": bar.open,
                "high": bar.high, 
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume
            })
            
        df = pd.DataFrame(data)
        df["t"] = pd.to_datetime(df["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(TZ_CT)
        return df
        
    except Exception as e:
        st.warning(f"Sparkline error for {ticker}: {e}")
        return None

def render_sparkline(df: pd.DataFrame, change: float) -> go.Figure:
    """Create a mini sparkline chart."""
    color = "#00ff88" if change >= 0 else "#ff4444"
    fill_color = "rgba(0,255,136,0.1)" if change >= 0 else "rgba(255,68,68,0.1)"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["t"], 
        y=df["close"],
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy", 
        fillcolor=fill_color,
        hoverinfo="skip",
        showlegend=False
    ))
    
    fig.update_layout(
        xaxis=dict(visible=False), 
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        height=40, 
        width=120, 
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig

def calculate_rel_volume(ticker: str, current_vol: float) -> float:
    """Calculate relative volume vs 20-day average."""
    try:
        end_date = datetime.date.today()
        start_date = end_date - datetime.timedelta(days=30)
        
        bars = list(polygon_client.get_aggs(
            ticker, multiplier=1, timespan="day",
            from_=start_date.strftime("%Y-%m-%d"), 
            to=end_date.strftime("%Y-%m-%d"),
            limit=20
        ))
        
        if len(bars) < 5:
            return 1.0
            
        avg_vol = sum(bar.volume for bar in bars[-20:]) / len(bars[-20:])
        return current_vol / avg_vol if avg_vol > 0 else 1.0
        
    except Exception:
        return 1.0

def ai_playbook(ticker: str, change: float, relvol: float, catalyst: str = "") -> str:
    """Generate AI trading playbook using OpenAI."""
    prompt = f"""
    You are an expert day trader and swing trader. Analyze this stock:
    
    Ticker: {ticker}
    Price Change: {change:.2f}%
    Relative Volume: {relvol:.2f}x
    Catalyst: {catalyst if catalyst else "No specific catalyst"}
    
    Provide a concise trading analysis with:
    
    1. **Sentiment**: Bullish/Bearish/Neutral with confidence %
    2. **Scalp Setup (1-5m)**: Quick entry/exit with tight stops
    3. **Day Trade Setup (15-30m)**: Intraday momentum play  
    4. **Swing Setup (4H-Daily)**: Multi-day position
    
    For each setup include:
    - Entry strategy
    - Price targets
    - Stop loss levels
    - Key levels to watch
    
    Keep response under 200 words, bullet points preferred.
    """
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"❌ AI Error: {str(e)}\n\nPlease check your OpenAI API key and connection."

# ---------------- SIDEBAR (WATCHLIST) ----------------
with st.sidebar:
    st.header("📌 Watchlist Manager")

    # Watchlist selector
    list_name = st.selectbox("Active Watchlist", list(st.session_state.watchlists.keys()))
    st.session_state.active_watchlist = list_name
    tickers = st.session_state.watchlists[list_name].copy()  # Work with copy to avoid reference issues

    # Add new ticker
    col1, col2 = st.columns([3, 1])
    new_ticker = col1.text_input("Add Symbol", placeholder="TSLA").upper().strip()
    if col2.button("➕"):
        if new_ticker and len(new_ticker) <= 5 and new_ticker not in tickers:
            tickers.append(new_ticker)
            st.session_state.watchlists[list_name] = tickers
            save_watchlists(st.session_state.watchlists)
            st.rerun()
        elif new_ticker in tickers:
            st.warning(f"{new_ticker} already in watchlist")

    # Display tickers with remove buttons
    if tickers:
        st.subheader("Current Symbols")
        for t in tickers:
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{t}**")
            if col2.button("🗑️", key=f"remove_{t}", help=f"Remove {t}"):
                tickers.remove(t)
                st.session_state.watchlists[list_name] = tickers
                save_watchlists(st.session_state.watchlists)
                st.rerun()

    # Settings
    st.markdown("---")
    st.subheader("⚙️ Settings")
    st.session_state.show_sparklines = st.checkbox("⚡ Show Sparklines", value=st.session_state.show_sparklines)

    # Quick watchlist overview
    st.markdown("---")
    st.subheader("📊 Quick Overview")
    
    if tickers:
        with st.spinner("Loading quotes..."):
            overview_data = []
            for ticker in tickers:
                quote = get_quote(ticker)
                if quote["error"]:
                    continue
                    
                prev_close = get_previous_close(ticker)
                change_pct = ((quote["last"] - prev_close) / prev_close * 100) if prev_close > 0 else 0
                
                overview_data.append({
                    "ticker": ticker,
                    "price": quote["last"],
                    "change": change_pct
                })
            
            # Display mini cards
            for data in overview_data:
                color = "🟢" if data["change"] >= 0 else "🔴"
                st.markdown(f"""
                **{data['ticker']}** {color}  
                ${data['price']:.2f} ({data['change']:+.2f}%)
                """)
    else:
        st.info("Add symbols to see overview")

# ---------------- MAIN AREA ----------------
st.title("🔥 AI Radar Pro — Trading Assistant")

if not tickers:
    st.warning("⚠️ No symbols in watchlist. Add some symbols in the sidebar to get started!")
    st.stop()

tabs = st.tabs(["📊 Live Quotes", "📈 Market Analysis", "🤖 AI Playbooks"])

with tabs[0]:
    st.subheader("📊 Real-Time Watchlist")
    
    if st.button("🔄 Refresh Quotes", type="primary"):
        st.cache_data.clear()
        st.rerun()
    
    # Create quotes table
    quotes_data = []
    progress_bar = st.progress(0)
    
    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) / len(tickers))
        
        # Use demo data if enabled or API fails
        if st.session_state.demo_mode:
            quote = get_demo_quote(ticker)
            prev_close = get_demo_previous_close(ticker)
        else:
            quote = get_quote(ticker)
            if quote["error"]:
                continue
            prev_close = get_previous_close(ticker)
        
        change_pct = ((quote["last"] - prev_close) / prev_close * 100) if prev_close > 0 else 0
        
        # Get volume data for relative volume calculation
        sparkline_df = get_intraday_sparkline(ticker) if st.session_state.show_sparklines and not st.session_state.demo_mode else None
        current_vol = sparkline_df["volume"].sum() if sparkline_df is not None and not sparkline_df.empty else random.randint(1000000, 5000000)
        rel_vol = calculate_rel_volume(ticker, current_vol)
        
        quotes_data.append({
            "Ticker": ticker,
            "Price": f"${quote['last']:.2f}",
            "Change": f"{change_pct:+.2f}%",
            "Bid": f"${quote['bid']:.2f}",
            "Ask": f"${quote['ask']:.2f}",
            "RelVol": f"{rel_vol:.2f}x",
            "change_num": change_pct,
            "sparkline_df": sparkline_df,
            "error": quote.get("error")
        })
    
    progress_bar.empty()
    
    # Display quotes in a nice format
    if quotes_data:
        for i, data in enumerate(quotes_data):
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
                
                # Ticker and price
                col1.metric(
                    label=data["Ticker"],
                    value=data["Price"],
                    delta=data["Change"]
                )
                
                # Bid/Ask
                col2.write("**Bid/Ask**")
                col2.write(f"{data['Bid']} / {data['Ask']}")
                
                # Volume
                col3.write("**Rel Volume**")
                vol_color = "🔥" if float(data['RelVol'][:-1]) >= 2.0 else "📊"
                col3.write(f"{vol_color} {data['RelVol']}")
                
                # Sparkline
                if st.session_state.show_sparklines and data['sparkline_df'] is not None:
                    col4.plotly_chart(
                        render_sparkline(data['sparkline_df'], data['change_num']), 
                        use_container_width=False
                    )
                
                st.divider()

with tabs[1]:
    st.subheader("📈 Market Analysis")
    st.info("🚧 Coming Soon: Market-wide movers, sector analysis, and news catalysts")
    
    # Placeholder for future features
    with st.expander("Planned Features"):
        st.markdown("""
        - **Market Movers**: Top gainers/losers across all markets
        - **Sector Rotation**: Heat map of sector performance  
        - **News Catalysts**: Real-time news affecting your watchlist
        - **Economic Calendar**: Key events and earnings
        - **Options Flow**: Unusual options activity
        """)

with tabs[3]:
    st.subheader("🤖 AI Trading Playbooks")
    
    # Ticker selection for AI analysis
    selected_ticker = st.selectbox("Select Symbol for AI Analysis", tickers, key="ai_ticker")
    
    col1, col2 = st.columns([3, 1])
    catalyst_input = col1.text_input(
        "Market Catalyst (optional)", 
        placeholder="Earnings beat, FDA approval, etc.",
        key="catalyst"
    )
    
    if col2.button("🤖 Generate Playbook", type="primary"):
        if selected_ticker:
            with st.spinner(f"AI analyzing {selected_ticker}..."):
                # Get current data
                quote = get_quote(selected_ticker)
                prev_close = get_previous_close(selected_ticker)
                change_pct = ((quote["last"] - prev_close) / prev_close * 100) if prev_close > 0 else 0
                
                # Get volume data
                sparkline_df = get_intraday_sparkline(selected_ticker)
                current_vol = sparkline_df["volume"].sum() if sparkline_df is not None and not sparkline_df.empty else 0
                rel_vol = calculate_rel_volume(selected_ticker, current_vol) if current_vol > 0 else 1.0
                
                # Generate AI analysis
                playbook = ai_playbook(selected_ticker, change_pct, rel_vol, catalyst_input)
                
                # Display results
                st.success(f"✅ AI Analysis Complete for **{selected_ticker}**")
                
                # Current stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Current Price", f"${quote['last']:.2f}", f"{change_pct:+.2f}%")
                col2.metric("Relative Volume", f"{rel_vol:.2f}x")
                col3.metric("Bid/Ask Spread", f"${quote['ask'] - quote['bid']:.2f}")
                
                # AI Playbook
                st.markdown("### 🎯 Trading Playbook")
                st.markdown(playbook)
                
                # Add to favorites option
                if st.button("⭐ Save This Analysis"):
                    st.success("Analysis saved! (Feature coming soon)")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🔥 AI Radar Pro | Real-time data from Polygon.io | AI powered by OpenAI"
    "</div>", 
    unsafe_allow_html=True
)
