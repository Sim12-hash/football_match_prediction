import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pro Football Tactical Decision Engine",
    page_icon="⚽",
    layout="wide"
)

# Custom CSS for Professional Dashboard Aesthetics
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 15px; border-radius: 10px; border-left: 5px solid #3b82f6; }
    .tactical-card { background-color: #111827; padding: 20px; border-radius: 10px; border: 1px solid #374151; margin-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ Pro Match Tactical Simulator & Decision Engine")
st.caption("正式足球联赛主教练赛前推演与战术决策支持系统")

# ---------------------------------------------------------
# 1. Load Model & Historical Data
# ---------------------------------------------------------
@st.cache_resource
def load_resources():
    model = joblib.load('world_cup_rf_model.pkl')
    try:
        df = pd.read_csv('clean_world_cup_2022.csv')
    except:
        df = None
    return model, df

model, match_df = load_resources()

# Feature List Mapping
tactical_features = [
    'xg', 'possession', 'shots_on_target', 'shots_total',
    'passes_completed', 'pass_accuracy', 'ppda', 'tackles_successful',
    'interceptions', 'clearances', 'fouls_committed', 'yellow_cards',
    'corners', 'crosses_completed', 'aerial_duels_won_pct', 'errors_leading_to_shot'
]

# ---------------------------------------------------------
# 2. Top Header: Match Setup & Tactical Presets
# ---------------------------------------------------------
st.sidebar.header("⚙️ 1. 比赛与对手设置")

# Preset Selector for Coaches
preset = st.sidebar.selectbox(
    "🎯 战术风格预设 (Tactical Style Presets)",
    ["自定义 (Custom)", "高位逼抢 (High Press)", "传控主导 (Tiki-Taka)", "低位防守反击 (Low Block Counter)", "高空轰炸/长传 (Direct & Aerial)"]
)

# Tactical Preset Defaults
preset_values = {
    "自定义 (Custom)": {"xg": 1.5, "possession": 50, "ppda": 10.5, "shots_on_target": 5, "passes_completed": 450, "pass_accuracy": 82, "crosses_completed": 6, "aerial_duels_won_pct": 50},
    "高位逼抢 (High Press)": {"xg": 2.1, "possession": 58, "ppda": 6.5, "shots_on_target": 7, "passes_completed": 520, "pass_accuracy": 84, "crosses_completed": 8, "aerial_duels_won_pct": 55},
    "传控主导 (Tiki-Taka)": {"xg": 1.8, "possession": 68, "ppda": 9.0, "shots_on_target": 6, "passes_completed": 680, "pass_accuracy": 89, "crosses_completed": 4, "aerial_duels_won_pct": 45},
    "低位防守反击 (Low Block Counter)": {"xg": 1.2, "possession": 38, "ppda": 18.0, "shots_on_target": 4, "passes_completed": 310, "pass_accuracy": 75, "crosses_completed": 3, "aerial_duels_won_pct": 52},
    "高空轰炸/长传 (Direct & Aerial)": {"xg": 1.6, "possession": 45, "ppda": 13.0, "shots_on_target": 5, "passes_completed": 350, "pass_accuracy": 74, "crosses_completed": 12, "aerial_duels_won_pct": 68}
}

current_preset = preset_values.get(preset, preset_values["自定义 (Custom)"])

# ---------------------------------------------------------
# 3. Main Screen: Tactical Directives (Grouped Controls)
# ---------------------------------------------------------
st.subheader("📋 2. 战术方案推演 (Tactical Directives)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("#### 🎯 进攻终结 (Attacking)")
    xg = st.slider("预期进球 (xG Target)", 0.2, 4.5, float(current_preset["xg"]), 0.1)
    shots_on_target = st.slider("射正数 Target", 1, 15, int(current_preset["shots_on_target"]))
    shots_total = st.slider("总射门数 Target", 3, 30, max(shots_on_target + 4, 12))
    corners = st.slider("角球次数", 0, 15, 5)

with col2:
    st.markdown("#### 🔄 组织与控球 (Build-up)")
    possession = st.slider("目标控球率 (%)", 20, 80, int(current_preset["possession"]))
    passes_completed = st.slider("成功传球数", 150, 900, int(current_preset["passes_completed"]), 10)
    pass_accuracy = st.slider("传球成功率 (%)", 50, 95, int(current_preset["pass_accuracy"]))
    crosses_completed = st.slider("成功传中数", 0, 20, int(current_preset["crosses_completed"]))

with col3:
    st.markdown("#### 🛡️ 防守与逼抢 (Defense & Pressing)")
    ppda = st.slider("PPDA (逼抢强度, 越低越激进)", 3.0, 25.0, float(current_preset["ppda"]), 0.5, help="Opponent Passes Per Defensive Action: 越低代表前场逼抢越激进")
    tackles_successful = st.slider("成功抢断", 5, 35, 15)
    interceptions = st.slider("拦截次数", 3, 25, 10)
    clearances = st.slider("解围次数", 5, 45, 18)

with col4:
    st.markdown("#### ⚔️ 对抗与纪律 (Duels & Discipline)")
    aerial_duels_won_pct = st.slider("争顶胜率 (%)", 20, 80, int(current_preset["aerial_duels_won_pct"]))
    fouls_committed = st.slider("犯规次数", 2, 25, 11)
    yellow_cards = st.number_input("黄牌数", 0, 8, 2)
    errors_leading_to_shot = st.number_input("致命失误致射门", 0, 3, 0)

# Construct Input Vector
input_vector = np.array([[
    xg, possession, shots_on_target, shots_total,
    passes_completed, pass_accuracy, ppda, tackles_successful,
    interceptions, clearances, fouls_committed, yellow_cards,
    corners, crosses_completed, aerial_duels_won_pct, errors_leading_to_shot
]])

# ---------------------------------------------------------
# 4. Real-time Prediction Engine
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 3. 模拟比赛结果与胜率诊断 (Match Simulation Output)")

probabilities = model.predict_proba(input_vector)[0]
classes = model.classes_
prob_dict = dict(zip(classes, probabilities))

win_p = prob_dict.get('Win', 0.0) * 100
draw_p = prob_dict.get('Draw', 0.0) * 100
loss_p = prob_dict.get('Loss', 0.0) * 100

res_col1, res_col2 = st.columns([1, 2])

with res_col1:
    st.markdown("### 🏆 预测胜率模型评估")
    st.metric("胜率 (Win Probability)", f"{win_p:.1f}%", delta=f"{win_p-33.3:.1f}% vs 基准")
    st.metric("平局概率 (Draw)", f"{draw_p:.1f}%")
    st.metric("败率 (Loss Probability)", f"{loss_p:.1f}%")

with res_col2:
    # Plotly Probabilities Doughnut Chart
    fig_pie = go.Figure(data=[go.Pie(
        labels=['胜 (Win)', '平 (Draw)', '负 (Loss)'],
        values=[win_p, draw_p, loss_p],
        hole=.5,
        marker_colors=['#10b981', '#f59e0b', '#ef4444']
    )])
    fig_pie.update_layout(
        title="比赛结果概率分布 (Match Outcome Distribution)",
        margin=dict(l=20, r=20, t=40, b=20),
        height=260,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white")
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ---------------------------------------------------------
# 5. Coach Actionable Decision Support System (战术敏感度与推荐引擎)
# ---------------------------------------------------------
st.markdown("---")
st.subheader("💡 4. 主教练决策与战术微调建议 (Coach Executive Brief)")

st.caption("基于随机森林特征贡献度与敏感度分析，为教练组提供的行动指南：")

# Calculate sensitivity: What happens if PPDA is pushed more aggressive (decreased by 3)?
sim_vector_high_press = input_vector.copy()
sim_vector_high_press[0][tactical_features.index('ppda')] = max(3.0, ppda - 3.0)
sim_win_press = dict(zip(classes, model.predict_proba(sim_vector_high_press)[0])).get('Win', 0.0) * 100

# Calculate sensitivity: What happens if xG increases by 0.4?
sim_vector_xg = input_vector.copy()
sim_vector_xg[0][tactical_features.index('xg')] += 0.4
sim_win_xg = dict(zip(classes, model.predict_proba(sim_vector_xg)[0])).get('Win', 0.0) * 100

advice_col1, advice_col2 = st.columns(2)

with advice_col1:
    st.markdown("#### 🎯 战术敏感度推演 (What-If Insights)")
    
    delta_press = sim_win_press - win_p
    press_direction = "提升" if delta_press >= 0 else "降低"
    st.info(f"👉 **加强前场逼抢**：若将 PPDA 从 **{ppda:.1f}** 压低至 **{max(3.0, ppda - 3.0):.1f}**，胜率预计将 **{press_direction} {abs(delta_press):.1f}%**。")

    delta_xg = sim_win_xg - win_p
    st.success(f"👉 **提高进攻终结质量**：若通过定位球或进攻战术提升 +0.4 xG，胜率预计将 **提升 +{delta_xg:.1f}%**。")

with advice_col2:
    st.markdown("#### 📈 关键战术权重 (Top Tactical Drivers)")
    # Feature Importance Plot
    importances = model.feature_importances_
    feat_df = pd.DataFrame({'Tactical Variable': tactical_features, 'Importance': importances})
    feat_df = feat_df.sort_values(by='Importance', ascending=True).tail(6)

    fig_bar = px.bar(
        feat_df, x='Importance', y='Tactical Variable', orientation='h',
        color='Importance', color_continuous_scale='Blues'
    )
    fig_bar.update_layout(
        height=220, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"), coloraxis_showscale=False
    )
    st.plotly_chart(fig_bar, use_container_width=True)
