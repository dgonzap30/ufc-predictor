"""
Manual Matchup Prediction Page
Create custom fighter vs fighter predictions
"""
import streamlit as st
from ufc_predictor.app.components.styling import get_custom_css
from ufc_predictor.inference.predict import predict_matchup, EXCLUDED_WEIGHT_CLASSES


# Page Configuration
st.set_page_config(
    page_title="Manual Matchup - UFC Sniper",
    page_icon="🔧",
    layout="wide"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


# Weight Class Mapping
WEIGHT_CLASSES = {
    "Heavyweight (High Risk)": "Heavyweight Bout",
    "Light Heavyweight": "Light Heavyweight Bout",
    "Middleweight": "Middleweight Bout",
    "Welterweight": "Welterweight Bout",
    "Lightweight": "Lightweight Bout",
    "Featherweight": "Featherweight Bout",
    "Bantamweight": "Bantamweight Bout",
    "Flyweight (High Risk)": "Flyweight Bout",
    "Women's Bantamweight": "Women's Bantamweight Bout",
    "Women's Flyweight": "Women's Flyweight Bout",
    "Women's Strawweight (High Risk)": "Women's Strawweight Bout",
}


# Header
st.markdown("# 🔧 Manual Matchup")
st.markdown("#### Create custom fighter vs fighter predictions")


# Instructions
with st.expander("ℹ️ How to use this tool", expanded=False):
    st.markdown(
        """
        **Create hypothetical matchups to test the model:**

        1. Enter both fighter names (must exist in database)
        2. Select the weight class for the bout
        3. Optionally include betting odds for EV calculation
        4. Click 'Analyze Matchup' to see prediction

        **Note:** The model is trained on historical UFC data. Fighters not in the database
        will return an error. Try using full names (e.g., "Islam Makhachev" not "Islam").
        """
    )

# Input Form
st.markdown("### Fighter Selection")

col1, col2 = st.columns(2)

with col1:
    fighter_a = st.text_input(
        "Fighter A",
        placeholder="e.g., Islam Makhachev",
        help="Enter the first fighter's full name"
    )

with col2:
    fighter_b = st.text_input(
        "Fighter B",
        placeholder="e.g., Dustin Poirier",
        help="Enter the second fighter's full name"
    )

# Weight class selection
weight_class_display = st.selectbox(
    "Weight Class",
    options=list(WEIGHT_CLASSES.keys()),
    index=4,  # Default to Lightweight
    help="Select the weight class for this matchup"
)
weight_class_internal = WEIGHT_CLASSES[weight_class_display]

# Show risk warning if applicable
if weight_class_internal in EXCLUDED_WEIGHT_CLASSES:
    st.warning(
        "⚠️ **High Risk Weight Class:** This weight class is excluded from Sniper strategy "
        "due to negative historical ROI"
    )

# Odds Section
st.markdown("### Betting Odds (Optional)")

include_odds = st.checkbox(
    "Include betting odds for Expected Value calculation",
    help="Add American odds to see if this matchup qualifies as a Sniper bet"
)

odds_a = None
odds_b = None

if include_odds:
    col1, col2 = st.columns(2)

    with col1:
        odds_a = st.number_input(
            f"Odds for {fighter_a or 'Fighter A'} (American)",
            step=10,
            format="%d",
            value=-150,
            help="Negative for favorites (e.g., -150), positive for underdogs (e.g., +200)"
        )

    with col2:
        odds_b = st.number_input(
            f"Odds for {fighter_b or 'Fighter B'} (American)",
            step=10,
            format="%d",
            value=150,
            help="Negative for favorites (e.g., -150), positive for underdogs (e.g., +200)"
        )

# Action Button
if st.button("🔍 Analyze Matchup", type="primary", use_container_width=True):
    # Validation
    if not fighter_a or not fighter_b:
        st.warning("⚠️ Please enter both fighter names")
    else:
        with st.spinner("Analyzing matchup..."):
            try:
                # Call prediction
                result = predict_matchup(
                    fighter_a,
                    fighter_b,
                    weight_class_internal,
                    odds_f1=odds_a if include_odds else None,
                    odds_f2=odds_b if include_odds else None
                )

                # Results Section
                st.divider()
                st.markdown("### 📊 Prediction Results")

                # Status Banner
                if include_odds and "sniper" in result:
                    if result["sniper"]["is_sniper_bet"]:
                        st.success(
                            f"🎯 **SNIPER BET FOUND:** Back {result['winner']} to win",
                            icon="✅"
                        )
                    else:
                        reasons = ", ".join(result["sniper"]["reasons"])
                        st.warning(f"⛔ **PASS:** {reasons}", icon="⚠️")
                else:
                    st.info(
                        f"📊 **PREDICTION:** {result['winner']} wins with "
                        f"{result['confidence']:.0f}% confidence",
                        icon="📈"
                    )

                # Metrics Display
                col1, col2, col3, col4 = st.columns(4)

                # Winner
                col1.metric(
                    "Predicted Winner",
                    f"🏆 {result['winner']}",
                    help="Fighter predicted to win"
                )

                # Confidence
                confidence = result['confidence']
                conf_indicator = "🟢" if confidence > 65 else "🟡" if confidence > 55 else "🔴"
                col2.metric(
                    "Confidence",
                    f"{conf_indicator} {confidence:.1f}%",
                    help="Model confidence in this prediction"
                )

                # Win Probability
                prob_f1 = result.get('prob_f1', 0)
                prob_f2 = result.get('prob_f2', 0)
                col3.metric(
                    "Win Probabilities",
                    f"{prob_f1*100:.1f}% / {prob_f2*100:.1f}%",
                    help=f"{fighter_a} / {fighter_b}"
                )

                # Expected Value (if odds provided)
                if include_odds and "ev_f1" in result and "ev_f2" in result:
                    winner_ev = result['ev_f1'] if result['winner'] == fighter_a else result['ev_f2']
                    ev_indicator = "🟢" if winner_ev > 0.05 else "🟡" if winner_ev > 0 else "🔴"
                    col4.metric(
                        "Expected Value",
                        f"{ev_indicator} {winner_ev * 100:+.1f}%",
                        help="Expected value of betting on the predicted winner"
                    )
                else:
                    col4.metric(
                        "Expected Value",
                        "—",
                        help="Add odds to calculate EV"
                    )

                # Additional Details
                if include_odds:
                    st.markdown("### 💰 Betting Analysis")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**Fighter A Analysis**")
                        st.markdown(f"**{fighter_a}**")
                        st.markdown(f"- Win Probability: {prob_f1*100:.1f}%")
                        st.markdown(f"- Odds: {odds_a:+d}")
                        if "ev_f1" in result:
                            ev_f1_pct = result['ev_f1'] * 100
                            st.markdown(f"- Expected Value: **{ev_f1_pct:+.2f}%**")

                    with col2:
                        st.markdown("**Fighter B Analysis**")
                        st.markdown(f"**{fighter_b}**")
                        st.markdown(f"- Win Probability: {prob_f2*100:.1f}%")
                        st.markdown(f"- Odds: {odds_b:+d}")
                        if "ev_f2" in result:
                            ev_f2_pct = result['ev_f2'] * 100
                            st.markdown(f"- Expected Value: **{ev_f2_pct:+.2f}%**")

            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.caption(
                    "**Possible causes:**\n"
                    "- Fighter names not found in database\n"
                    "- Check spelling and use full names\n"
                    "- Ensure fighters have fought in this weight class"
                )


# Sidebar Info
with st.sidebar:
    st.markdown("### 🎯 Sniper Criteria")
    st.caption("**For a bet to qualify:**")
    st.caption("✅ Model confidence > 65%")
    st.caption("✅ Betting on the favorite")
    st.caption("✅ Safe weight class")
    st.caption("✅ Positive expected value")

    st.markdown("### 📚 Examples")
    st.caption("**Try these matchups:**")
    st.caption("- Islam Makhachev vs Charles Oliveira")
    st.caption("- Alex Pereira vs Jamahal Hill")
    st.caption("- Leon Edwards vs Colby Covington")
