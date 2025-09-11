import streamlit as st
import pandas as pd
import requests
import datetime
import json
import plotly.graph_objects as go
import random
from typing import Dict, List, Optional

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Radar Pro", layout="wide")

# Initialize ALL session state variables first
if "watchlists" not in st.session_state:
    st.session_state.watchlists = {"Default": ["AAPL", "NVDA", "TSLA", "MSFT", "AMZN"]}
if "active_watchlist" not in st.session_state:
    st.session_state.active_watchlist = "Default"
if "show_sparklines" not in st.session_state:
    st.session_state.show_sparklines = True
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = True  # Start in demo mode to avoid API issues

# API Keys (optional - app works in demo mode without them)
try:
    POLYGON_KEY = st.secrets.get("POLYGON_API_KEY", "")
    OPENAI_KEY = st.secrets.get("OPENAI_API_KEY", "")
    if OPENAI_KEY:
        import openai
        openai_client = openai.OpenAI(api_key=OPENAI_KEY)
    else:
        openai_client = None
except:
    POLYGON_KEY = ""
    OPENAI_KEY = ""
    openai_client = None

# ---------------- HELPER FUNCTIONS ----------------
def get_demo_quote(ticker: str) -> Dict:
    """Generate realistic demo quote data."""
    base_prices = {
        "AAPL": 150.00, "NVDA": 450.00, "TSLA": 250.00,
        "MSFT": 350.00, "AMZN": 140.00, "GOOGL": 120.00,
        "META": 300.00, "AMD": 100.00, "PLTR": 25.00
    }
    
    base_price = base_prices.get(ticker, random.uniform(50, 200))
    price_change = random.uniform(-3, 3)
    current_price = round(base_price + price_change, 2)
    spread = current_price * 0.001
    
    return {
        "last": current_price,
        "bid": round(current_price - spread/2, 2),
        "ask": round(current_price + spread/2, 2),
        "error": None
    }

def get_demo_previous_close(ticker: str) -> float:
    """Get demo previous close price."""
    current = get_demo_quote(ticker)["last"]
    return round(current * random.uniform(0.97, 1.03), 2)

def render_sparkline(ticker: str, change: float):
    """Create a simple sparkline chart."""
    # Generate sample intraday data
    base_price = get_demo_quote(ticker)["last"]
    prices = []
    for i in range(20):
        price_change = random.uniform(-0.02, 0.02)
        base_price = base_price * (1 + price_change)
        prices.append(base_price)
    
    color = "#00ff88" if change >= 0 else "#ff4444"
    fill_color = "rgba(0,255,136,0.1)" if change >= 0 else "rgba(255,68,68,0.1)"
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=prices,
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

def get_market_news():
    """Get demo market news data."""
    return [
        {
            "title": "Tesla Reports Record Q4 Deliveries, Stock Surges in Pre-Market",
            "author": "MarketWatch",
            "published_utc": "2024-01-02T14:30:00Z",
            "tickers": ["TSLA"],
            "keywords": ["earnings", "deliveries", "record"],
            "description": "Tesla exceeded delivery expectations with 484,507 vehicles delivered in Q4"
        },
        {
            "title": "NVIDIA Announces New AI Chip Partnership with Microsoft", 
            "author": "Reuters",
            "published_utc": "2024-01-02T13:15:00Z",
            "tickers": ["NVDA", "MSFT"],
            "keywords": ["ai", "partnership", "chips"],
            "description": "Strategic partnership to develop next-generation AI infrastructure"
        },
        {
            "title": "Apple Faces Production Delays in China Amid Supply Chain Issues",
            "author": "Bloomberg", 
            "published_utc": "2024-01-02T12:00:00Z",
            "tickers": ["AAPL"],
            "keywords": ["supply chain", "production", "delays"],
            "description": "iPhone production may be impacted by ongoing supply chain disruptions"
        },
        {
            "title": "AMD Stock Jumps on AI Server Chip Demand Surge",
            "author": "CNBC",
            "published_utc": "2024-01-02T11:45:00Z", 
            "tickers": ["AMD"],
            "keywords": ["ai", "server", "demand", "surge"],
            "description": "Strong demand for AI server chips boosts AMD's data center revenue"
        }
    ]

def analyze_news_sentiment(title, description):
    """Analyze news for explosive potential."""
    explosive_keywords = ["surge", "soars", "jumps", "rocket", "announces", "beats", "record", "partnership"]
    bearish_keywords = ["delays", "issues", "problems", "cuts", "falls", "plunges"]
    
    text = (title + " " + description).lower()
    
    explosive_score = sum(1 for word in explosive_keywords if word in text)
    bearish_score = sum(1 for word in bearish_keywords if word in text)
    
    if explosive_score >= 2:
        return "🚀 EXPLOSIVE", "bullish", explosive_score * 25
    elif explosive_score >= 1:
        return "📈 Bullish", "bullish", explosive_score * 20
    elif bearish_score >= 2:
        return "💥 CRASH RISK", "bearish", bearish_score * 25
    elif bearish_score >= 1:
        return "📉 Bearish", "bearish", bearish_score * 20
    else:
        return "⚪ Neutral", "neutral", 15

def get_catalyst_alerts():
    """Generate catalyst alerts."""
    news_items = get_market_news()
    alerts = []
    
    for item in news_items:
        sentiment, direction, confidence = analyze_news_sentiment(
            item.get("title", ""),
            item.get("description", "")
        )
        
        hours_ago = random.randint(1, 6)
        
        alerts.append({
            "title": item["title"],
            "tickers": item.get("tickers", []),
            "sentiment": sentiment,
            "direction": direction, 
            "confidence": confidence,
            "hours_ago": hours_ago,
            "author": item.get("author", "Unknown"),
            "description": item.get("description", "")
        })
    
    return sorted(alerts, key=lambda x: x["confidence"], reverse=True)

def ai_playbook(ticker: str, change: float, catalyst: str = "") -> str:
    """Generate AI trading playbook."""
    if not openai_client:
        return f"""
**Demo AI Analysis for {ticker}**

**Sentiment:** Neutral (Demo Mode)
**Current Change:** {change:+.2f}%

**Scalp Setup (1-5m):**
- Entry: Look for breakout above recent highs with volume
- Target: +0.5-1% quick scalp
- Stop: -0.3% below entry

**Day Trade Setup (15-30m):**  
- Entry: Pullback to key support level
- Target: Previous resistance levels
- Stop: Below swing low

**Swing Setup (4H-Daily):**
- Entry: Break and hold above daily resistance
- Target: Next major resistance zone
- Stop: Below daily support

*Note: Enable OpenAI API key for full AI analysis*
        """
    
    try:
        prompt = f"""
        Analyze {ticker} with {change:+.2f}% change.
        Catalyst: {catalyst}
        
        Provide brief trading setups:
        1. Sentiment (Bullish/Bearish/Neutral)
        2. Scalp setup (1-5m timeframe)
        3. Day trade setup (15-30m)
        4. Swing setup (4H-Daily)
        
        Keep it concise and actionable.
        """
        
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content
        
    except Exception as e:
        return f"AI Error: {str(e)}. Using demo mode analysis instead."

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.header("📌 Watchlist Manager")

    # Watchlist selector
    list_name = st.selectbox("Active Watchlist", list(st.session_state.watchlists.keys()))
    st.session_state.active_watchlist = list_name
    tickers = st.session_state.watchlists[list_name].copy()

    # Add new ticker
    col1, col2 = st.columns([3, 1])
    new_ticker = col1.text_input("Add Symbol", placeholder="TSLA").upper().strip()
    if col2.button("➕"):
        if new_ticker and len(new_ticker) <= 5 and new_ticker not in tickers:
            tickers.append(new_ticker)
            st.session_state.watchlists[list_name] = tickers
            st.rerun()

    # Display current tickers
    if tickers:
        st.subheader("Current Symbols")
        for t in tickers:
            col1, col2 = st.columns([4, 1])
            col1.write(f"**{t}**")
            if col2.button("🗑️", key=f"remove_{t}"):
                tickers.remove(t)
                st.session_state.watchlists[list_name] = tickers
                st.rerun()

    # Settings
    st.markdown("---")
    st.subheader("⚙️ Settings")
    st.session_state.show_sparklines = st.checkbox("⚡ Show Sparklines", value=st.session_state.show_sparklines)
    st.session_state.demo_mode = st.checkbox("🎮 Demo Mode", value=st.session_state.demo_mode)
    
    if st.session_state.demo_mode:
        st.info("Demo mode active - using sample data")
    
    # Quick overview
    if tickers:
        st.markdown("---")
        st.subheader("📊 Quick Overview")
        for ticker in tickers[:5]:  # Show first 5
            quote = get_demo_quote(ticker)
            prev_close = get_demo_previous_close(ticker)
            change_pct = ((quote["last"] - prev_close) / prev_close * 100)
            
            color = "🟢" if change_pct >= 0 else "🔴"
            st.markdown(f"**{ticker}** {color} ${quote['last']:.2f} ({change_pct:+.2f}%)")

# ---------------- MAIN AREA ----------------
st.title("🔥 AI Radar Pro — Trading Assistant")

if not tickers:
    st.warning("⚠️ Add some symbols to your watchlist to get started!")
    st.stop()

tabs = st.tabs(["📊 Live Quotes", "🔥 Catalyst Scanner", "📈 Market Analysis", "🤖 AI Playbooks"])

# TAB 1: Live Quotes
with tabs[0]:
    st.subheader("📊 Real-Time Watchlist")
    
    if st.button("🔄 Refresh Quotes", type="primary"):
        st.rerun()
    
    # Create quotes display
    for ticker in tickers:
        quote = get_demo_quote(ticker)
        prev_close = get_demo_previous_close(ticker) 
        change_pct = ((quote["last"] - prev_close) / prev_close * 100)
        rel_vol = round(random.uniform(0.5, 3.5), 2)
        
        with st.container():
            col1, col2, col3, col4 = st.columns([2, 2, 2, 4])
            
            # Price and change
            col1.metric(
                label=ticker,
                value=f"${quote['last']:.2f}",
                delta=f"{change_pct:+.2f}%"
            )
            
            # Bid/Ask
            col2.write("**Bid/Ask**")
            col2.write(f"${quote['bid']:.2f} / ${quote['ask']:.2f}")
            
            # Volume
            col3.write("**Rel Volume**")
            vol_color = "🔥" if rel_vol >= 2.0 else "📊"
            col3.write(f"{vol_color} {rel_vol:.2f}x")
            
            # Sparkline
            if st.session_state.show_sparklines:
                col4.plotly_chart(
                    render_sparkline(ticker, change_pct),
                    use_container_width=False
                )
            
            st.divider()

# TAB 2: Catalyst Scanner  
with tabs[1]:
    st.subheader("🔥 Real-Time Catalyst Scanner")
    st.caption("24/7 monitoring for explosive stock movements")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    auto_refresh = col1.checkbox("🔄 Auto Refresh", value=False)
    col2.button("🔍 Scan Now", type="primary")
    col3.metric("Scanner", "🟢 ACTIVE" if auto_refresh else "⏸️ PAUSED")
    
    # Filters
    st.markdown("### 🎯 Catalyst Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        show_explosive = st.checkbox("🚀 Explosive", value=True)
        show_bullish = st.checkbox("📈 Bullish", value=True)
    
    with filter_col2:
        show_bearish = st.checkbox("📉 Bearish", value=True)
        min_confidence = st.slider("Min Confidence", 0, 100, 15)
    
    with filter_col3:
        max_hours = st.slider("Max Hours Old", 1, 24, 8)
    
    # Get and display alerts
    alerts = get_catalyst_alerts()
    
    # Filter alerts
    filtered_alerts = []
    for alert in alerts:
        if alert["confidence"] < min_confidence or alert["hours_ago"] > max_hours:
            continue
        if "EXPLOSIVE" in alert["sentiment"] and not show_explosive:
            continue
        if "Bullish" in alert["sentiment"] and not show_bullish:
            continue
        if "Bearish" in alert["sentiment"] and not show_bearish:
            continue
        filtered_alerts.append(alert)
    
    if filtered_alerts:
        st.markdown(f"### 📊 Found {len(filtered_alerts)} Active Catalysts")
        
        for i, alert in enumerate(filtered_alerts):
            with st.container():
                # Header
                col1, col2, col3 = st.columns([3, 3, 2])
                col1.markdown(f"**{alert['sentiment']}** ({alert['confidence']}%)")
                col2.markdown(f"**Tickers:** {', '.join(alert['tickers'])}")
                col3.markdown(f"**{alert['hours_ago']}h ago**")
                
                # Content
                st.markdown(f"#### {alert['title']}")
                st.markdown(alert['description'])
                
                # Actions
                action_col1, action_col2 = st.columns(2)
                
                if action_col1.button("📊 AI Analysis", key=f"analyze_{i}"):
                    if alert['tickers']:
                        ticker = alert['tickers'][0]
                        quote = get_demo_quote(ticker)
                        prev_close = get_demo_previous_close(ticker)
                        change = ((quote["last"] - prev_close) / prev_close * 100)
                        
                        with st.spinner(f"Analyzing {ticker}..."):
                            analysis = ai_playbook(ticker, change, alert['title'])
                            st.success(f"🤖 Analysis for {ticker}")
                            st.markdown(analysis)
                
                if action_col2.button("⚡ Add to Watchlist", key=f"add_{i}"):
                    added = []
                    for ticker in alert['tickers']:
                        if ticker not in tickers:
                            tickers.append(ticker)
                            added.append(ticker)
                    if added:
                        st.session_state.watchlists[list_name] = tickers
                        st.success(f"Added: {', '.join(added)}")
                
                st.divider()
    else:
        st.info("🔍 No catalysts match your filters. Try adjusting settings.")

# TAB 3: Market Analysis
with tabs[2]:
    st.subheader("📈 Market Analysis")
    st.info("🚧 Coming Soon: Market-wide analysis and sector rotation")
    
    with st.expander("Planned Features"):
        st.markdown("""
        - **Market Movers:** Top gainers/losers
        - **Sector Heat Map:** Performance by sector
        - **Economic Calendar:** Key events
        - **Options Flow:** Unusual activity
        """)

# TAB 4: AI Playbooks
with tabs[3]:
    st.subheader("🤖 AI Trading Playbooks")
    
    selected_ticker = st.selectbox("Select Symbol", tickers)
    catalyst_input = st.text_input("Market Catalyst (optional)", placeholder="Earnings, news, etc.")
    
    if st.button("🤖 Generate Playbook", type="primary"):
        if selected_ticker:
            quote = get_demo_quote(selected_ticker)
            prev_close = get_demo_previous_close(selected_ticker)
            change_pct = ((quote["last"] - prev_close) / prev_close * 100)
            
            with st.spinner(f"AI analyzing {selected_ticker}..."):
                playbook = ai_playbook(selected_ticker, change_pct, catalyst_input)
                
                st.success(f"✅ Analysis Complete for **{selected_ticker}**")
                
                # Current stats
                col1, col2, col3 = st.columns(3)
                col1.metric("Price", f"${quote['last']:.2f}", f"{change_pct:+.2f}%")
                col2.metric("Bid/Ask", f"${quote['bid']:.2f} / ${quote['ask']:.2f}")
                col3.metric("Spread", f"${quote['ask'] - quote['bid']:.2f}")
                
                # AI Analysis
                st.markdown("### 🎯 Trading Playbook")
                st.markdown(playbook)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "🔥 AI Radar Pro | Demo Mode Active | Enable API keys for live data"
    "</div>",
    unsafe_allow_html=True
)
