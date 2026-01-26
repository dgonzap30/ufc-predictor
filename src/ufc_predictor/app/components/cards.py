"""
Reusable card components for fight displays
"""
import streamlit as st
from ufc_predictor.inference.predict import EXCLUDED_WEIGHT_CLASSES


def render_fight_card(fight, result, has_odds, is_sniper_bet):
    """
    Render an individual fight prediction card

    Args:
        fight: Fight data dict with fighter names, weight class, odds
        result: Prediction result dict
        has_odds: Boolean indicating if odds are available
        is_sniper_bet: Boolean indicating if this meets Sniper criteria
    """
    # Determine status badge
    if has_odds and is_sniper_bet:
        badge = '<span class="sniper-bet-badge">🎯 SNIPER BET</span>'
        expanded = True
    else:
        badge = '<span class="pass-badge">PASS</span>' if has_odds else ''
        expanded = fight.get("is_main_event", False)

    # Fighter names with badge
    label = f"{fight['fighter1']} vs {fight['fighter2']}"

    with st.expander(label, expanded=expanded):
        # Header with badge
        if badge:
            st.markdown(badge, unsafe_allow_html=True)

        # Metrics row
        col1, col2, col3, col4 = st.columns(4)

        # Winner with trophy icon
        winner_display = f"🏆 {result['winner']}"
        col1.metric("Predicted Winner", winner_display)

        # Confidence with color coding
        confidence = result['confidence']
        conf_color = "🟢" if confidence > 65 else "🟡" if confidence > 55 else "🔴"
        col2.metric("Confidence", f"{conf_color} {confidence:.1f}%")

        # Weight class with risk indicator
        weight_class = fight["weight_class"].replace(" Bout", "")
        is_risky = fight["weight_class"] in EXCLUDED_WEIGHT_CLASSES
        wc_display = f"⚠️ {weight_class}" if is_risky else weight_class
        col3.metric("Weight Class", wc_display)

        # Odds display
        if has_odds:
            odds_f1 = fight.get("odds_f1")
            odds_f2 = fight.get("odds_f2")
            odds_display = f"{odds_f1:+d} / {odds_f2:+d}"
            col4.metric("Odds", odds_display)
        else:
            col4.metric("Odds", "N/A")

        # Status banner
        if has_odds and "sniper" in result:
            if result["sniper"]["is_sniper_bet"]:
                # Calculate potential return
                if result["winner"] == fight["fighter1"]:
                    winner_odds = fight.get("odds_f1")
                else:
                    winner_odds = fight.get("odds_f2")

                # Show EV if available
                ev_text = ""
                if "ev_f1" in result and "ev_f2" in result:
                    ev = result["ev_f1"] if result["winner"] == fight["fighter1"] else result["ev_f2"]
                    ev_text = f" | Expected Value: **{ev*100:+.1f}%**"

                st.success(
                    f"🎯 **SNIPER BET:** Back {result['winner']} @ {winner_odds:+d}{ev_text}"
                )
            else:
                reasons = ", ".join(result["sniper"]["reasons"])
                st.warning(f"⛔ **PASS:** {reasons}")
        else:
            # No odds - show candidate status
            is_high_conf = result["confidence"] > 65
            is_safe_class = fight["weight_class"] not in EXCLUDED_WEIGHT_CLASSES

            if is_high_conf and is_safe_class:
                st.info(
                    "📊 High confidence + Safe weight class "
                    "(Would qualify as Sniper bet with odds)"
                )
            elif not is_high_conf:
                st.warning(f"⚠️ Low confidence ({result['confidence']:.1f}% < 65%)")
            else:
                st.error("⛔ High-risk weight class (Excluded from Sniper strategy)")


def render_quick_stats(predictions):
    """
    Render quick statistics summary for loaded event

    Args:
        predictions: List of prediction dicts
    """
    if not predictions:
        return

    # Calculate stats
    total_fights = len(predictions)
    sniper_bets = sum(1 for p in predictions if p.get("is_sniper_bet", False))
    high_confidence = sum(1 for p in predictions if p["confidence"] > 65)
    avg_confidence = sum(p["confidence"] for p in predictions) / total_fights

    # Display in columns
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Fights",
        total_fights,
        help="Number of fights on this card"
    )

    col2.metric(
        "Sniper Bets",
        sniper_bets,
        delta=f"{sniper_bets/total_fights*100:.0f}%" if total_fights > 0 else None,
        help="Fights meeting all Sniper criteria"
    )

    col3.metric(
        "High Confidence",
        high_confidence,
        delta=f"{high_confidence/total_fights*100:.0f}%" if total_fights > 0 else None,
        help="Predictions with >65% confidence"
    )

    col4.metric(
        "Avg Confidence",
        f"{avg_confidence:.1f}%",
        help="Average model confidence across all fights"
    )


def render_sniper_summary(predictions):
    """
    Render summary table of Sniper bets vs Passes

    Args:
        predictions: List of prediction dicts
    """
    sniper_bets = [p for p in predictions if p.get("is_sniper_bet", False)]

    if not sniper_bets:
        st.info("🎯 No Sniper bets identified on this card. All fights filtered by strategy.")
        return

    st.success(f"🎯 **{len(sniper_bets)} Sniper Bet{'s' if len(sniper_bets) != 1 else ''} Found**")

    # Create summary table
    import pandas as pd

    rows = []
    for p in sniper_bets:
        row = {
            "Fight": f"{p['winner']} vs {p['loser']}",
            "Pick": p['winner'],
            "Confidence": f"{p['confidence']:.1f}%",
            "Odds": f"{p.get('winner_odds', 'N/A'):+d}" if p.get('winner_odds') else "N/A",
            "EV": f"{p.get('ev', 0)*100:+.1f}%" if 'ev' in p else "—"
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)
