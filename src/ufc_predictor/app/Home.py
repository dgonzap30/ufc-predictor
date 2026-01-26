"""
UFC Sniper Dashboard - Main Page
Modern, clean UI for fight predictions and betting analysis
"""
import streamlit as st
from pathlib import Path
from ufc_predictor.app.components.styling import get_custom_css
from ufc_predictor.app.components.cards import (
    render_fight_card,
    render_quick_stats,
    render_sniper_summary
)
from ufc_predictor.inference.predict import predict_matchup
from ufc_predictor.data.upcoming import get_upcoming_fight_data, get_recent_events, get_fight_card


# Page Configuration
st.set_page_config(
    page_title="UFC Sniper",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


# Initialize session state
if "event" not in st.session_state:
    st.session_state["event"] = None
if "predictions" not in st.session_state:
    st.session_state["predictions"] = []


# Sidebar: Event Loader
with st.sidebar:
    st.markdown("### 📅 Event Selector")

    # Event source tabs in sidebar
    event_source = st.radio(
        "Event Source",
        ["Next Upcoming", "Recent Events"],
        label_visibility="collapsed",
        horizontal=True
    )

    if event_source == "Next Upcoming":
        if st.button("🔄 Load Next Event", type="primary", use_container_width=True):
            with st.spinner("Fetching fight card and odds..."):
                try:
                    card = get_upcoming_fight_data()
                    if card and card.get("fights"):
                        st.session_state["event"] = card
                        st.session_state["predictions"] = []  # Reset predictions
                        st.success("✓ Event loaded")
                        st.rerun()
                    else:
                        st.error("Could not load fight card")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        # Load recent events
        if "recent_events" not in st.session_state:
            with st.spinner("Loading recent events..."):
                st.session_state["recent_events"] = get_recent_events(10)

        recent_events = st.session_state.get("recent_events", [])

        if recent_events:
            event_names = [e["name"] for e in recent_events]
            selected_event = st.selectbox(
                "Select Event",
                event_names,
                label_visibility="collapsed"
            )

            if st.button("🔄 Load Event", type="primary", use_container_width=True):
                with st.spinner(f"Loading {selected_event}..."):
                    try:
                        event_url = next(e["url"] for e in recent_events if e["name"] == selected_event)
                        card = get_fight_card(event_url)
                        if card and card.get("fights"):
                            st.session_state["event"] = card
                            st.session_state["predictions"] = []
                            st.success("✓ Event loaded")
                            st.rerun()
                        else:
                            st.error("Could not load fight card")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        else:
            st.error("Could not fetch events")

    # Display loaded event info
    if st.session_state["event"]:
        event = st.session_state["event"]
        st.divider()
        st.markdown("**Loaded Event**")
        st.markdown(f"**{event.get('event_name', 'Unknown')}**")
        st.caption(f"📅 {event.get('date', 'TBA')}")
        st.caption(f"📍 {event.get('location', 'TBA')}")
        st.caption(f"🥊 {len(event.get('fights', []))} fights")

    # Strategy info
    st.markdown("### 🎯 Sniper Strategy")
    st.caption("**Filters:**")
    st.caption("✓ Confidence > 65%")
    st.caption("✓ Favorites only")
    st.caption("✓ Safe weight classes")


# Main Header
st.markdown("# UFC Sniper 🎯")
st.markdown("#### AI-Powered Fight Predictions & Profitable Betting Strategy")


# Main Content
if not st.session_state["event"]:
    # Welcome screen
    st.info(
        "👈 **Load an event from the sidebar to begin analyzing fights**",
        icon="📊"
    )

    st.markdown("### What is UFC Sniper?")
    st.markdown(
        """
        UFC Sniper is an AI-powered prediction system that identifies profitable betting opportunities
        using a calibrated XGBoost model trained on 98 engineered features.

        **Key Features:**
        - 61.3% prediction accuracy on test set
        - +2.2% ROI on real Vegas odds (Sniper Strategy)
        - Filters out negative-EV weight classes
        - Only recommends high-confidence favorites
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("##### 🎯 Sniper Strategy")
        st.markdown("Highly selective betting approach that filters for maximum profitability")

    with col2:
        st.markdown("##### 📊 Data-Driven")
        st.markdown("Trained on comprehensive UFC fight history with ELO ratings")

    with col3:
        st.markdown("##### ⚡ Real-Time Odds")
        st.markdown("Live odds integration from BestFightOdds.com")

else:
    event = st.session_state["event"]

    # Event Header
    st.markdown(f"## 🏟️ {event['event_name']}")
    st.caption(f"📅 {event['date']} • 📍 {event['location']}")

    # Generate predictions if not already done
    if not st.session_state["predictions"]:
        with st.spinner("Analyzing fights..."):
            predictions = []

            for fight in event["fights"]:
                try:
                    odds_f1 = fight.get("odds_f1")
                    odds_f2 = fight.get("odds_f2")

                    result = predict_matchup(
                        fight["fighter1"],
                        fight["fighter2"],
                        fight["weight_class"],
                        odds_f1=odds_f1,
                        odds_f2=odds_f2
                    )

                    has_odds = odds_f1 is not None and odds_f2 is not None
                    is_sniper_bet = has_odds and result.get("sniper", {}).get("is_sniper_bet", False)

                    pred_data = {
                        "fighter1": fight["fighter1"],
                        "fighter2": fight["fighter2"],
                        "winner": result["winner"],
                        "loser": fight["fighter2"] if result["winner"] == fight["fighter1"] else fight["fighter1"],
                        "confidence": result["confidence"],
                        "weight_class": fight["weight_class"],
                        "has_odds": has_odds,
                        "is_sniper_bet": is_sniper_bet,
                        "fight": fight,
                        "result": result
                    }

                    if has_odds:
                        pred_data["odds_f1"] = odds_f1
                        pred_data["odds_f2"] = odds_f2
                        pred_data["winner_odds"] = odds_f1 if result["winner"] == fight["fighter1"] else odds_f2

                        if "ev_f1" in result and "ev_f2" in result:
                            pred_data["ev"] = result["ev_f1"] if result["winner"] == fight["fighter1"] else result["ev_f2"]

                        if not is_sniper_bet and "sniper" in result:
                            pred_data["reasons"] = result["sniper"].get("reasons", [])

                    predictions.append(pred_data)

                except Exception as e:
                    st.error(f"Error predicting {fight['fighter1']} vs {fight['fighter2']}: {str(e)}")

            st.session_state["predictions"] = predictions

    # Quick Stats
    render_quick_stats(st.session_state["predictions"])

    st.divider()

    # Sniper Summary
    render_sniper_summary(st.session_state["predictions"])

    # Tabs for organization
    tab1, tab2 = st.tabs(["🥊 All Fights", "🎯 Sniper Bets Only"])

    with tab1:
        st.markdown("### All Fight Predictions")

        for pred in st.session_state["predictions"]:
            render_fight_card(
                pred["fight"],
                pred["result"],
                pred["has_odds"],
                pred["is_sniper_bet"]
            )

    with tab2:
        st.markdown("### Sniper Bet Recommendations")

        sniper_bets = [p for p in st.session_state["predictions"] if p.get("is_sniper_bet", False)]

        if not sniper_bets:
            st.info("No Sniper bets on this card. All fights filtered by strategy criteria.")
        else:
            for pred in sniper_bets:
                render_fight_card(
                    pred["fight"],
                    pred["result"],
                    pred["has_odds"],
                    pred["is_sniper_bet"]
                )
