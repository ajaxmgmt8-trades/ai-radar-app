def ai_auto_generate_plays() -> List[Dict]:
    """
    Auto-generates trading plays by scanning watchlist and market movers
    """
    plays = []
    
    try:
        # Get current watchlist
        current_watchlist = st.session_state.watchlists[st.session_state.active_watchlist]
        
        # Combine watchlist with core tickers for broader scan
        scan_tickers = list(set(current_watchlist + CORE_TICKERS[:30]))
        
        # Scan for significant movers
        candidates = []
        
        for ticker in scan_tickers:
            quote = get_live_quote(ticker)
            if not quote["error"]:
                # Look for significant moves (>1.5% change)
                if abs(quote["change_percent"]) >= 1.5:
                    candidates.append({
                        "ticker": ticker,
                        "quote": quote,
                        "significance": abs(quote["change_percent"])
                    })
        
        # Sort by significance and take top candidates
        candidates.sort(key=lambda x: x["significance"], reverse=True)
        top_candidates = candidates[:5]  # Limit to top 5 to avoid API limits
        
        # Generate plays for top candidates
        for candidate in top_candidates:
            ticker = candidate["ticker"]
            quote = candidate["quote"]
            
            # Get recent news for context
            news = get_finnhub_news(ticker)
            catalyst = ""
            if news:
                catalyst = news[0].get('headline', '')[:100] + "..."
            
            # Generate AI analysis if OpenAI is available
            if openai_client:
                try:
                    play_prompt = f"""
                    Generate a concise trading play for {ticker}:
                    
                    Current Price: ${quote['last']:.2f}
                    Change: {quote['change_percent']:+.2f}%
                    Premarket: {quote['premarket_change']:+.2f}%
                    Intraday: {quote['intraday_change']:+.2f}%
                    After Hours: {quote['postmarket_change']:+.2f}%
                    Volume: {quote['volume']:,}
                    Catalyst: {catalyst if catalyst else "Market movement"}
                    
                    Provide:
                    1. Play type (Scalp/Day/Swing)
                    2. Entry strategy and levels
                    3. Target and stop levels
                    4. Risk/reward ratio
                    5. Confidence (1-10)
                    
                    Keep under 200 words, be specific and actionable.
                    """
                    
                    response = openai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[{"role": "user", "content": play_prompt}],
                        temperature=0.3,
                        max_tokens=300
                    )
                    
                    play_analysis = response.choices[0].message.content
                    
                except Exception as ai_error:
                    play_analysis = f"""
                    **{ticker} Trading Opportunity**
                    
                    **Movement:** {quote['change_percent']:+.2f}% change with {quote['volume']:,} volume
                    
                    **Session Breakdown:**
                    • Premarket: {quote['premarket_change']:+.2f}%
                    • Intraday: {quote['intraday_change']:+.2f}%
                    • After Hours: {quote['postmarket_change']:+.2f}%
                    
                    **Quick Setup:** Watch for continuation or reversal around current levels
                    
                    *AI analysis unavailable: {str(ai_error)[:50]}...*
                    """
            else:
                # Fallback analysis without AI
                direction = "bullish" if quote['change_percent'] > 0 else "bearish"
                play_analysis = f"""
                **{ticker} Trading Setup**
                
                **Movement:** {quote['change_percent']:+.2f}% change showing {direction} momentum
                
                **Session Analysis:**
                • Premarket: {quote['premarket_change']:+.2f}%
                • Intraday: {quote['intraday_change']:+.2f}%
                • After Hours: {quote['postmarket_change']:+.2f}%
                
                **Volume:** {quote['volume']:,} shares
                
                **Setup:** Monitor for continuation or reversal. Consider risk management around current price levels.
                
                *Configure OpenAI API for detailed AI analysis*
                """
            
            # Create play dictionary
            play = {
                "ticker": ticker,
                "current_price": quote['last'],
                "change_percent": quote['change_percent'],
                "session_data": {
                    "premarket": quote['premarket_change'],
                    "intraday": quote['intraday_change'],
                    "afterhours": quote['postmarket_change']
                },
                "catalyst": catalyst if catalyst else f"Market movement: {quote['change_percent']:+.2f}%",
                "play_analysis": play_analysis,
                "volume": quote['volume'],
                "timestamp": quote['last_updated']
            }
            
            plays.append(play)
        
        return plays
        
    except Exception as e:
        st.error(f"Error generating auto plays: {str(e)}")
        return []
