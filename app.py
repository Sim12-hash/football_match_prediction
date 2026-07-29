import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Tactical Predictor Dashboard",
    layout="wide"
)

st.title("⚽ Tactical Predictor Dashboard for Coaches")
st.markdown("---")

# ---------------------------------------------------------
# Step 1: Team Selection & Historical Auto-fill (Mocked)
# ---------------------------------------------------------
st.header("1. Match Setup & Historical Baseline")

# Mock options for teams and formations
teams = ["Argentina", "France", "Croatia", "Morocco", "Brazil", "England"]
formations = ["4-3-3", "4-2-3-1", "3-5-2", "4-4-2", "5-3-2"]

col_team1, col_team2 = st.columns(2)

with col_team1:
    home_team = st.selectbox("Select Home Team", teams, index=0)

with col_team2:
    away_team = st.selectbox("Select Away Team", teams, index=1)

# Mock historical averages retrieved from dataset
# (Replace with real pandas groupby logic later)
mock_hist_data = {
    "Argentina": {"xg": 1.85, "possession": 58, "pass_acc": 85.0},
    "France": {"xg": 1.65, "possession": 52, "pass_acc": 83.5},
    "Croatia": {"xg": 1.20, "possession": 54, "pass_acc": 84.0},
    "Morocco": {"xg": 0.95, "possession": 38, "pass_acc": 78.0},
    "Brazil": {"xg": 2.10, "possession": 60, "pass_acc": 87.0},
    "England": {"xg": 1.70, "possession": 56, "pass_acc": 85.5}
}

h_defaults = mock_hist_data.get(home_team, {"xg": 1.5, "possession": 50, "pass_acc": 80.0})
a_defaults = mock_hist_data.get(away_team, {"xg": 1.5, "possession": 50, "pass_acc": 80.0})

st.info(f"💡 System auto-loaded historical averages: {home_team} (Avg xG: {h_defaults['xg']}) vs {away_team} (Avg xG: {a_defaults['xg']})")

# ---------------------------------------------------------
# Step 2: Tactical Adjustments (Interactive Sliders)
# ---------------------------------------------------------
st.header("2. Tactical Adjustments")

col_h_tactics, col_a_tactics = st.columns(2)

with col_h_tactics:
    st.subheader(f"🏠 {home_team} (Home)")
    home_fmt = st.selectbox("Formation", formations, index=0, key="h_fmt")
    home_xg = st.slider("Expected xG", 0.0, 4.0, float(h_defaults['xg']), 0.1, key="h_xg")
    home_pos = st.slider("Expected Possession (%)", 20, 80, int(h_defaults['possession']), key="h_pos")

with col_a_tactics:
    st.subheader(f"✈️ {away_team} (Away)")
    away_fmt = st.selectbox("Formation", formations, index=1, key="a_fmt")
    away_xg = st.slider("Expected xG", 0.0, 4.0, float(a_defaults['xg']), 0.1, key="a_xg")
    
    # Auto-link possession so the total equals 100%
    away_pos = 100 - home_pos
    st.write(f"Expected Possession (%): **{away_pos}%** *(Auto-aligned)*")

# ---------------------------------------------------------
# Step 3: Prediction & Interpretability (XAI)
# ---------------------------------------------------------
st.markdown("---")
st.header("3. Match Prediction & Decision Support")

if st.button("🚀 Run Tactical Simulation", type="primary"):
    # ---------------------------------------------------------
    # Mock Prediction Results (Replace with model.predict_proba later)
    # ---------------------------------------------------------
    win_prob = 62.5
    draw_prob = 21.0
    lose_prob = 16.5

    st.success(f"### Prediction Result: {home_team} Win: {win_prob}% | Draw: {draw_prob}% | {away_team} Win: {lose_prob}%")

    # Metrics Layout
    m1, m2, m3 = st.columns(3)
    m1.metric(label=f"{home_team} Win Chance", value=f"{win_prob}%", delta="+5.2% vs Baseline")
    m2.metric(label="Draw Chance", value=f"{draw_prob}%")
    m3.metric(label=f"{away_team} Win Chance", value=f"{lose_prob}%", delta="-5.2% vs Baseline")

    # ---------------------------------------------------------
    # Mock Feature Importance (Explainable AI - XAI)
    # ---------------------------------------------------------
    st.subheader("💡 Model Feature Importance (Explainable AI)")
    st.caption("Shows which tactical variables contributed most to this prediction.")

    feature_names = ["xG Difference", "Possession Diff", "Formation Matchup", "Shot Precision Diff"]
    importance_scores = [0.42, 0.28, 0.18, 0.12]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(feature_names, importance_scores, color="#1f77b4")
    ax.set_xlabel("Relative Importance Weight")
    ax.set_title("Tactical Drivers Influencing Prediction")
    st.pyplot(fig)