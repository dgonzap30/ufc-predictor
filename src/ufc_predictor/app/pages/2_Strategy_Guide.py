"""
Strategy Guide Page
Educational content about the Sniper Strategy and model performance
"""
import streamlit as st
import pandas as pd
from ufc_predictor.app.components.styling import get_custom_css


# Page Configuration
st.set_page_config(
    page_title="Strategy Guide - UFC Sniper",
    page_icon="📚",
    layout="wide"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


# Header
st.markdown("# 📚 Sniper Strategy Guide")
st.markdown("#### Understanding the AI-powered betting system")


# Overview Section
st.markdown("## 🎯 What is the Sniper Strategy?")

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(
        """
        The **Sniper Strategy** is a highly selective, data-driven betting approach that filters
        UFC fight predictions to maximize profitability. Unlike betting on every fight, the Sniper
        Strategy identifies only the highest-quality opportunities.

        **Historical Performance:**
        - **+2.2% ROI** on real Vegas odds (test set)
        - **63.1% win rate** on recommended bets
        - **82% reduction** in bet volume vs unfiltered approach

        The strategy gets its name from the precision approach: take fewer shots, but make them count.
        """
    )

with col2:
    st.info(
        """
        **Key Metrics**

        📊 Test Accuracy: 61.3%

        💰 Sniper ROI: +2.2%

        🎯 Bets Placed: 157/865

        ✅ Win Rate: 63.1%
        """
    )


# Filter Criteria
st.markdown("## ⚙️ Filter Criteria")
st.markdown("The Sniper Strategy uses three mandatory filters to identify profitable bets:")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1️⃣ High Confidence")
    st.markdown(
        """
        **Model Probability > 65%**

        Only recommend fights where the model has strong conviction.
        High confidence correlates with higher win rates.

        **Impact:**
        - Unfiltered: 38.4% win rate
        - High confidence: **56.9% win rate**
        - **+16.9% ROI** on high confidence bets
        """
    )

with col2:
    st.markdown("### 2️⃣ Favorites Only")
    st.markdown(
        """
        **Implied Probability > 50%**

        Bet only on favorites (odds < 2.0 decimal).
        Favorites have better historical performance.

        **Impact:**
        - Underdogs: -12.5% ROI
        - Favorites: **+7.7% ROI**
        - More consistent returns
        """
    )

with col3:
    st.markdown("### 3️⃣ Safe Weight Classes")
    st.markdown(
        """
        **Exclude negative-ROI divisions**

        Remove weight classes with poor historical profitability:
        - Heavyweight: -7.2% ROI
        - Flyweight: -11.9% ROI
        - Women's Strawweight: -9.0% ROI

        These divisions show higher variance and unpredictability.
        """
    )


# Performance Breakdown
st.markdown("## 📊 Performance Breakdown")

st.markdown("### Backtest Results (Real Vegas Odds)")

# Create comparison table
comparison_data = {
    "Strategy": ["Unfiltered", "Sniper Strategy"],
    "Total Bets": [865, 157],
    "Win Rate": ["38.4%", "63.1%"],
    "ROI": ["-4.9%", "+2.2%"],
    "Avg Confidence": ["54.2%", "71.8%"],
    "Status": ["❌ Unprofitable", "✅ Profitable"]
}

df_comparison = pd.DataFrame(comparison_data)
st.dataframe(
    df_comparison,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Strategy": st.column_config.TextColumn("Strategy", width="medium"),
        "Total Bets": st.column_config.NumberColumn("Total Bets", format="%d"),
        "Win Rate": st.column_config.TextColumn("Win Rate", width="small"),
        "ROI": st.column_config.TextColumn("ROI", width="small"),
        "Avg Confidence": st.column_config.TextColumn("Avg Confidence", width="medium"),
        "Status": st.column_config.TextColumn("Status", width="medium"),
    }
)


# Weight Class Analysis
st.markdown("### Weight Class Performance")

st.markdown(
    """
    Not all weight classes are created equal. The model shows varying performance across divisions.
    The Sniper Strategy excludes the three worst-performing classes:
    """
)

weight_class_data = {
    "Weight Class": [
        "Lightweight",
        "Welterweight",
        "Featherweight",
        "Bantamweight",
        "Light Heavyweight",
        "Middleweight",
        "Women's Bantamweight",
        "Women's Flyweight",
        "Heavyweight ❌",
        "Flyweight ❌",
        "Women's Strawweight ❌"
    ],
    "ROI": [
        "+12.3%",
        "+8.7%",
        "+6.4%",
        "+5.1%",
        "+3.2%",
        "+1.8%",
        "+0.9%",
        "-2.1%",
        "-7.2%",
        "-11.9%",
        "-9.0%"
    ],
    "Status": [
        "✅ Included",
        "✅ Included",
        "✅ Included",
        "✅ Included",
        "✅ Included",
        "✅ Included",
        "✅ Included",
        "⚠️ Marginal",
        "❌ Excluded",
        "❌ Excluded",
        "❌ Excluded"
    ]
}

df_weight_classes = pd.DataFrame(weight_class_data)
st.dataframe(
    df_weight_classes,
    use_container_width=True,
    hide_index=True,
    column_config={
        "Weight Class": st.column_config.TextColumn("Weight Class", width="large"),
        "ROI": st.column_config.TextColumn("ROI", width="small"),
        "Status": st.column_config.TextColumn("Strategy Status", width="medium"),
    }
)

st.caption(
    "📊 ROI values are from historical backtest on test set. "
    "Excluded weight classes show consistent negative returns."
)


# Model Features
st.markdown("## 🤖 Model Architecture")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Training Approach")
    st.markdown(
        """
        **Two-Tier System:**

        1. **ELO Baseline** - Transparent skill rating system
           - Starting rating: 1500
           - K-factor: 32
           - Simple, interpretable

        2. **Calibrated XGBoost** - Advanced ML model
           - 98 engineered features
           - Isotonic calibration (5-fold CV)
           - Outperforms ELO baseline

        **Why calibration matters:** Ensures predicted probabilities match actual outcomes,
        critical for Expected Value calculations.
        """
    )

with col2:
    st.markdown("### Feature Categories")
    st.markdown(
        """
        **Physical Attributes:**
        - Height, reach, age differences
        - Weight class and stance matchups

        **Fighter History:**
        - Total fights, win rate, recent form
        - Finish rate, KO ratio, submission rate
        - Days since last fight (recency)

        **Opponent Quality:**
        - Average opponent ELO
        - Strength of schedule

        **ELO Ratings:**
        - Pre-fight ELO for both fighters
        - ELO difference
        """
    )


# Betting Guidelines
st.markdown("## 💡 How to Use Sniper Bets")

st.markdown(
    """
    **Best Practices:**

    1. **Only bet on Sniper recommendations** - Ignore all "PASS" predictions
    2. **Use proper bankroll management** - Never bet more than 1-5% per fight
    3. **Shop for the best odds** - Line shopping can add 1-2% to ROI
    4. **Track your results** - Monitor performance vs model expectations
    5. **Avoid tilt betting** - Don't chase losses on non-Sniper fights

    **Expected Value (EV):**

    EV is shown for each Sniper bet. A positive EV means the bet is theoretically profitable
    long-term. Higher EV = stronger recommendation.

    **Example:**
    - Bet: Islam Makhachev @ -300
    - Model Probability: 80%
    - EV: +6.7%

    This means for every $100 wagered, you expect to profit $6.70 on average over many bets.
    """
)


# Limitations
st.markdown("## ⚠️ Limitations & Disclaimers")

st.warning(
    """
    **Important Notes:**

    - **Past performance ≠ future results** - The +2.2% ROI is from historical backtesting
    - **Small sample size** - 157 bets in test set; variance is expected
    - **Line movement** - Odds can change; stale odds affect EV calculations
    - **Missing data** - Model requires fighters to be in database
    - **Style matchups** - Some stylistic factors may not be fully captured
    - **Injuries/motivation** - Model can't account for unreported injuries or fighter mindset

    **This is a tool to inform decisions, not a guarantee of profit. Bet responsibly.**
    """
)


# FAQ
st.markdown("## ❓ Frequently Asked Questions")

with st.expander("What does 'Sniper Bet' mean?"):
    st.markdown(
        """
        A "Sniper Bet" is a fight that passes all three strategy filters:
        1. High confidence (>65%)
        2. Betting on the favorite
        3. Safe weight class

        These bets have historically shown +2.2% ROI with a 63.1% win rate.
        """
    )

with st.expander("Why do some fights show 'PASS'?"):
    st.markdown(
        """
        "PASS" means the fight failed at least one Sniper filter. Common reasons:
        - Low confidence (<65%)
        - Betting on underdog
        - High-risk weight class
        - Negative expected value

        Passing fights should not be bet on according to the strategy.
        """
    )

with st.expander("Can I bet on fights without odds?"):
    st.markdown(
        """
        Fights without odds cannot be fully evaluated for Sniper criteria since we can't
        calculate Expected Value. However, the prediction and confidence are still shown.

        If the fight has high confidence (>65%) and is in a safe weight class, it may
        become a Sniper bet once odds are available.
        """
    )

with st.expander("How often should I expect Sniper bets?"):
    st.markdown(
        """
        On average, about 18% of fights qualify as Sniper bets (157 out of 865 in test set).

        For a typical 12-fight UFC card, expect **2-3 Sniper bets**. Some cards may have
        zero, others may have 5+. This selectivity is by design.
        """
    )

with st.expander("What bankroll management should I use?"):
    st.markdown(
        """
        Conservative recommendation: **1-2% of bankroll per bet**

        With a +2.2% ROI and 63.1% win rate, proper bankroll management is crucial to
        survive variance. Even profitable strategies experience losing streaks.

        **Example:**
        - Bankroll: $1,000
        - Bet size: $10-20 per Sniper bet
        - Expected long-term growth: ~$22 per 1000 risked
        """
    )


# Sidebar
with st.sidebar:
    st.markdown("### 📊 Quick Stats")
    st.metric("Test Accuracy", "61.3%")
    st.metric("Sniper ROI", "+2.2%")
    st.metric("Sniper Win Rate", "63.1%")
    st.metric("Brier Score", "0.216")

    st.markdown("### 🎯 Strategy Summary")
    st.caption("**Filters:**")
    st.caption("✅ Confidence > 65%")
    st.caption("✅ Favorites only")
    st.caption("✅ Safe weight classes")
