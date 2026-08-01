import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(page_title="World Cup Tactical Predictor", layout="wide")
st.title("⚽ World Cup Tactical Outcome Predictor")

# 加载模型
@st.cache_resource
def load_model():
    return joblib.load('world_cup_rf_model.pkl')

try:
    model = load_model()
    st.sidebar.success("✅ ML Model Loaded Successfully!")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {e}")

# 侧边栏输入
st.sidebar.header("🎯 Coach Tactical Inputs")
xg = st.sidebar.slider("Expected Goals (xG)", 0.0, 5.0, 1.5, 0.1)
possession = st.sidebar.slider("Possession (%)", 10, 90, 50, 1)
shots_on_target = st.sidebar.slider("Shots on Target", 0, 20, 5, 1)
shots_total = st.sidebar.slider("Total Shots", 0, 35, 12, 1)
passes_completed = st.sidebar.slider("Passes Completed", 50, 1000, 450, 10)
pass_accuracy = st.sidebar.slider("Pass Accuracy (%)", 30, 100, 82, 1)
ppda = st.sidebar.slider("PPDA (Pressing Intensity)", 1.0, 30.0, 10.5, 0.5)
tackles_successful = st.sidebar.slider("Successful Tackles", 0, 40, 15, 1)
interceptions = st.sidebar.slider("Interceptions", 0, 30, 10, 1)
clearances = st.sidebar.slider("Clearances", 0, 50, 18, 1)
fouls_committed = st.sidebar.slider("Fouls Committed", 0, 30, 12, 1)
yellow_cards = st.sidebar.number_input("Yellow Cards", 0, 10, 2)
corners = st.sidebar.slider("Corners", 0, 20, 5, 1)
crosses_completed = st.sidebar.slider("Completed Crosses", 0, 25, 6, 1)
aerial_duels_won_pct = st.sidebar.slider("Aerial Duels Won (%)", 0, 100, 50, 1)
errors_leading_to_shot = st.sidebar.number_input("Errors Leading to Shot", 0, 5, 0)

# 预测展示
if st.button("🚀 Run Real-time Match Simulation", type="primary"):
    # 构造 DataFrame 保持与训练时相同的特征名称，避免版本 Warning
    input_df = pd.DataFrame([[
        xg, possession, shots_on_target, shots_total, passes_completed, pass_accuracy,
        ppda, tackles_successful, interceptions, clearances, fouls_committed,
        yellow_cards, corners, crosses_completed, aerial_duels_won_pct, errors_leading_to_shot
    ]], columns=[
        'xg', 'possession', 'shots_on_target', 'shots_total',
        'passes_completed', 'pass_accuracy', 'ppda', 'tackles_successful',
        'interceptions', 'clearances', 'fouls_committed', 'yellow_cards',
        'corners', 'crosses_completed', 'aerial_duels_won_pct', 'errors_leading_to_shot'
    ])
    
    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]
    prob_dict = dict(zip(model.classes_, probabilities))
    
    st.subheader(f"Predicted Outcome: **{prediction.upper()}**")
    m1, m2, m3 = st.columns(3)
    m1.metric("Win Probability", f"{prob_dict.get('Win', 0.0)*100:.1f}%")
    m2.metric("Draw Probability", f"{prob_dict.get('Draw', 0.0)*100:.1f}%")
    m3.metric("Loss Probability", f"{prob_dict.get('Loss', 0.0)*100:.1f}%")

    # Feature Importance 可视化
    st.markdown("---")
    st.header("Model Explainability (XAI)")
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Feature': input_df.columns, 'Importance': importances}).sort_values('Importance').tail(8)
    
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(feat_df['Feature'], feat_df['Importance'], color='#2ca02c')
    st.pyplot(fig)
