# --- keep your full code above unchanged ---

# New: Auto-generate plays
def ai_auto_generate_plays(limit: int = 5):
    plays = []
    tickers = st.session_state.watchlists.get(st.session_state.active_watchlist, CORE_TICKERS[:20])

    for ticker in tickers:
        quote = get_live_quote(ticker)
        if quote["error"]:
            continue

        # Only pick movers
        if abs(quote["change_percent"]) < 1.5:
            continue

        # Get catalyst
        news = get_finnhub_news(ticker)
        catalyst = news[0].get("headline", "") if news else "No major catalyst"

        # Generate AI playbook
        analysis = ai_playbook(ticker, quote["change_percent"], catalyst)

        plays.append({
            "ticker": ticker,
            "current_price": quote["last"],
            "change_percent": quote["change_percent"],
            "volume": quote["volume"],
            "catalyst": catalyst,
            "play_analysis": analysis,
            "session_data": {
                "premarket": quote["premarket_change"],
                "intraday": quote["intraday_change"],
                "afterhours": quote["postmarket_change"],
            }
        })

    # Sort by absolute % move
    plays.sort(key=lambda x: abs(x["change_percent"]), reverse=True)
    return plays[:limit]

# --- in TAB 5 (AI Playbooks), replace the Auto-Generated Plays section with this ---

# Auto-generated plays section
st.markdown("### 🎯 Auto-Generated Trading Plays")
col1, col2 = st.columns([3, 1])
with col1:
    st.caption("AI scans your watchlist + movers for trading setups")
with col2:
    if st.button("🚀 Generate Auto Plays", type="primary"):
        with st.spinner("AI generating trading plays from market scan..."):
            auto_plays = ai_auto_generate_plays()

            if auto_plays:
                st.success(f"🤖 Generated {len(auto_plays)} Trading Plays")
                for i, play in enumerate(auto_plays):
                    with st.expander(f"🎯 {play['ticker']} - ${play['current_price']:.2f} ({play['change_percent']:+.2f}%)"):
                        sess_col1, sess_col2, sess_col3 = st.columns(3)
                        sess_col1.metric("Premarket", f"{play['session_data']['premarket']:+.2f}%")
                        sess_col2.metric("Intraday", f"{play['session_data']['intraday']:+.2f}%")
                        sess_col3.metric("After Hours", f"{play['session_data']['afterhours']:+.2f}%")

                        st.write(f"**Catalyst:** {play['catalyst']}")
                        st.markdown("**AI Trading Play:**")
                        st.markdown(play['play_analysis'])

                        if st.button(f"Add {play['ticker']} to Watchlist", key=f"add_auto_{i}"):
                            wl = st.session_state.watchlists[st.session_state.active_watchlist]
                            if play['ticker'] not in wl:
                                wl.append(play['ticker'])
                                st.session_state.watchlists[st.session_state.active_watchlist] = wl
                                st.success(f"Added {play['ticker']}")
                                st.rerun()
            else:
                st.info("No significant setups found now.")

