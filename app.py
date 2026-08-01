import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. 页面基本配置与样式
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pro AI Football Tactical Engine",
    page_icon="⚽",
    layout="wide"
)

# 职业软件暗黑风主题 CSS
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 12px; border-radius: 8px; border-left: 4px solid #3b82f6; }
    .coach-box { background-color: #111827; padding: 18px; border-radius: 10px; border: 1px solid #374151; margin-bottom: 15px; }
    .status-ok { color: #10b981; font-weight: bold; }
    .status-warn { color: #f59e0b; font-weight: bold; }
    .status-danger { color: #ef4444; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ Pro AI Tactical Decision Support System")
st.caption("职业足球赛前博弈推演、蒙特卡洛模拟与 AI 战术决策引擎 (Professional Coach Edition)")

# ---------------------------------------------------------
# 2. 资源加载与基准预设定义
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load('world_cup_rf_model.pkl')

try:
    model = load_model()
    st.sidebar.success("✅ ML 模型加载成功 (Random Forest)")
except Exception as e:
    st.sidebar.error(f"❌ 模型加载失败: {e}")
    st.stop()

# 16 个底层特征定义（必须与模型训练顺序完全一致）
tactical_features = [
    'xg', 'possession', 'shots_on_target', 'shots_total',
    'passes_completed', 'pass_accuracy', 'ppda', 'tackles_successful',
    'interceptions', 'clearances', 'fouls_committed', 'yellow_cards',
    'corners', 'crosses_completed', 'aerial_duels_won_pct', 'errors_leading_to_shot'
]

# 矩阵基准参考值（用于单场归因 XAI 计算）
FEATURE_BASELINES = {
    'xg': 1.3, 'possession': 50.0, 'shots_on_target': 4.5, 'shots_total': 12.0,
    'passes_completed': 420.0, 'pass_accuracy': 80.0, 'ppda': 11.0, 'tackles_successful': 15.0,
    'interceptions': 10.0, 'clearances': 18.0, 'fouls_committed': 12.0, 'yellow_cards': 1.8,
    'corners': 4.5, 'crosses_completed': 5.0, 'aerial_duels_won_pct': 50.0, 'errors_leading_to_shot': 0.3
}

# ---------------------------------------------------------
# 3. 侧边栏：球队设置、阵型与战术情景 preset (需求模块 1, 2, 4, 5)
# ---------------------------------------------------------
st.sidebar.header("⚙️ 1. 比赛与对阵设置")

# 对阵设置
home_team = st.sidebar.text_input("我方球队 (Home Team)", "Argentina")
away_team = st.sidebar.text_input("对手球队 (Away Team)", "France")

st.sidebar.markdown("---")
st.sidebar.header("📐 2. 阵型与对手风格映射")

# 阵型选择器
formation = st.sidebar.selectbox(
    "我方阵型 (Formation)",
    ["4-3-3", "4-2-3-1", "3-5-2", "4-4-2", "5-4-1", "3-4-3"]
)

# 对手战术风格
opp_style = st.sidebar.selectbox(
    "对手战术风格 (Opponent Style)",
    ["常规平衡 (Balanced)", "高位逼抢 (High Press)", "传控主导 (Tiki-Taka)", "低位摆大巴 (Low Block)", "快速反击 (Counter Attack)"]
)

# 一键战术情景 (Scenario Simulation)
st.sidebar.markdown("---")
st.sidebar.header("🎯 3. 一键战术情景 (Scenario)")
scenario = st.sidebar.radio(
    "战术倾向选择",
    ["常规推演 (Balanced)", "全员激进压迫 (Press All Out)", "极端防守摆大巴 (Parking Bus)", "全线压上攻坚 (Ultra Attack)"],
    index=0
)

# ---------------------------------------------------------
# 4. 战术映射引擎：根据阵型、对手风格、情景自动计算参数
# ---------------------------------------------------------
# 阵型基础参数映射
formation_rules = {
    "4-3-3": {"possession": 56, "ppda": 8.0, "passes": 520, "crosses": 7, "aerial": 52},
    "4-2-3-1": {"possession": 52, "ppda": 10.0, "passes": 470, "crosses": 6, "aerial": 50},
    "3-5-2": {"possession": 54, "ppda": 9.5, "passes": 490, "crosses": 10, "aerial": 58},
    "4-4-2": {"possession": 48, "ppda": 12.0, "passes": 410, "crosses": 8, "aerial": 54},
    "5-4-1": {"possession": 38, "ppda": 17.0, "passes": 320, "crosses": 3, "aerial": 55},
    "3-4-3": {"possession": 55, "ppda": 7.5, "passes": 510, "crosses": 8, "aerial": 50}
}
f_base = formation_rules[formation]

# 计算初始值
init_possession = f_base["possession"]
init_ppda = f_base["ppda"]
init_passes = f_base["passes"]
init_crosses = f_base["crosses"]
init_aerial = f_base["aerial"]
init_xg = 1.5
init_sot = 5
init_tackles = 15
init_clearances = 18

# 对手风格对数据的修正
if opp_style == "高位逼抢 (High Press)":
    init_ppda = max(4.0, init_ppda - 1.5)
    init_passes = int(init_passes * 0.9)
elif opp_style == "低位摆大巴 (Low Block)":
    init_possession = min(75, init_possession + 10)
    init_ppda += 4.0
    init_clearances += 8
elif opp_style == "传控主导 (Tiki-Taka)":
    init_possession = max(30, init_possession - 10)
    init_tackles += 4

# 情景（Scenario）覆盖修正
if scenario == "全员激进压迫 (Press All Out)":
    init_ppda = 5.5
    init_tackles = 22
    init_xg = 2.0
elif scenario == "极端防守摆大巴 (Parking Bus)":
    init_possession = 33
    init_ppda = 20.0
    init_clearances = 32
    init_xg = 0.8
elif scenario == "全线压上攻坚 (Ultra Attack)":
    init_xg = 2.4
    init_sot = 8
    init_possession = max(60, init_possession)

# ---------------------------------------------------------
# 5. 主界面排版：三列布局 (左: 战术指令 | 中: 模拟结果 | 右: Coach Report)
# ---------------------------------------------------------
col_left, col_mid, col_right = st.columns([1.1, 1.2, 1.3])

# --- 左列：战术指令微调面板 ---
with col_left:
    st.subheader("📋 战术指令微调 (Tactical Panel)")
    st.caption("系统已根据阵型与对阵映射初始值，教练组可微调：")

    with st.expander("🎯 进攻终结 (Attacking)", expanded=True):
        xg = st.slider("预期进球 (xG Target)", 0.2, 4.0, float(init_xg), 0.1)
        shots_on_target = st.slider("射正数 Target", 1, 15, int(init_sot))
        shots_total = st.slider("总射门数 Target", 3, 30, max(shots_on_target + 5, 12))
        corners = st.slider("角球次数", 0, 15, 5)

    with st.expander("🔄 组织控球 (Build-up)", expanded=False):
        possession = st.slider("控球率 (%)", 20, 80, int(init_possession))
        passes_completed = st.slider("成功传球数", 150, 900, int(init_passes), 10)
        pass_accuracy = st.slider("传球成功率 (%)", 50, 95, 82)
        crosses_completed = st.slider("成功传中数", 0, 20, int(init_crosses))

    with st.expander("🛡️ 防守压迫 (Defensive)", expanded=False):
        ppda = st.slider("PPDA (逼抢强度, 越低越高压)", 3.0, 25.0, float(init_ppda), 0.5)
        tackles_successful = st.slider("成功抢断", 5, 35, int(init_tackles))
        interceptions = st.slider("拦截次数", 3, 25, 10)
        clearances = st.slider("解围次数", 5, 45, int(init_clearances))

    with st.expander("⚔️ 对抗纪律 (Duels & Errors)", expanded=False):
        aerial_duels_won_pct = st.slider("争顶胜率 (%)", 20, 80, int(init_aerial))
        fouls_committed = st.slider("犯规次数", 2, 25, 11)
        yellow_cards = st.number_input("黄牌数", 0, 8, 2)
        errors_leading_to_shot = st.number_input("致命失误致射门", 0, 3, 0)

# 构造输入特征向量
input_vector = np.array([[
    xg, possession, shots_on_target, shots_total,
    passes_completed, pass_accuracy, ppda, tackles_successful,
    interceptions, clearances, fouls_committed, yellow_cards,
    corners, crosses_completed, aerial_duels_won_pct, errors_leading_to_shot
]])

# --- 中列：蒙特卡洛模拟与结果诊断 (需求模块 7) ---
with col_mid:
    st.subheader("🎲 蒙特卡洛模拟 (Monte Carlo Engine)")
    st.caption("模拟 1,000 场比赛微观碰撞，评估概率分布与置信区间：")

    # 蒙特卡洛 1000 次算法实现
    N_SIM = 1000
    np.random.seed(42)

    # 注入符合足球比赛规律的高斯噪声微扰
    noise = np.random.normal(0, 1, (N_SIM, 16))
    scale = np.array([0.15, 2.5, 0.8, 1.5, 25, 1.5, 0.8, 1.2, 1.0, 1.5, 1.0, 0.3, 0.8, 0.8, 2.0, 0.2])
    sim_inputs = np.clip(input_vector + noise * scale, a_min=0, a_max=None)

    # 批量预测
    sim_preds = model.predict(sim_inputs)
    sim_probs = model.predict_proba(sim_inputs)

    classes = model.classes_ # ['Draw', 'Loss', 'Win']
    win_idx = list(classes).index('Win') if 'Win' in classes else 2
    draw_idx = list(classes).index('Draw') if 'Draw' in classes else 0
    loss_idx = list(classes).index('Loss') if 'Loss' in classes else 1

    win_count = np.sum(sim_preds == 'Win')
    draw_count = np.sum(sim_preds == 'Draw')
    loss_count = np.sum(sim_preds == 'Loss')

    mc_win_pct = (win_count / N_SIM) * 100
    mc_draw_pct = (draw_count / N_SIM) * 100
    mc_loss_pct = (loss_count / N_SIM) * 100

    # 95% 置信区间计算
    win_probs_series = sim_probs[:, win_idx]
    ci_lower = np.percentile(win_probs_series, 2.5) * 100
    ci_upper = np.percentile(win_probs_series, 97.5) * 100

    # 显示核心 Metric 评估
    m1, m2, m3 = st.columns(3)
    m1.metric("模拟胜率 (Win)", f"{mc_win_pct:.1f}%")
    m2.metric("平局率 (Draw)", f"{mc_draw_pct:.1f}%")
    m3.metric("不败率 (Win/Draw)", f"{(mc_win_pct + mc_draw_pct):.1f}%")

    st.info(f"📊 **95% 胜率置信区间 (Confidence Interval)**: [{ci_lower:.1f}%, {ci_upper:.1f}%]")

    # 蒙特卡洛模拟分布直方图
    fig_mc, ax_mc = plt.subplots(figsize=(6, 3.2))
    ax_mc.hist(win_probs_series * 100, bins=25, color='#10b981', alpha=0.7, edgecolor='black')
    ax_mc.axvline(mc_win_pct, color='red', linestyle='--', linewidth=2, label=f'Mean Win Prob ({mc_win_pct:.1f}%)')
    ax_mc.set_title("1,000 Runs Win Probability Distribution", color='white', fontsize=10)
    ax_mc.set_xlabel("Predicted Win Probability (%)", color='white')
    ax_mc.set_ylabel("Frequency", color='white')
    ax_mc.tick_params(colors='white')
    ax_mc.legend(loc='upper right', facecolor='#111827', labelcolor='white')
    fig_mc.patch.set_facecolor('#111827')
    ax_mc.set_facecolor('#1f2937')
    st.pyplot(fig_mc)

# --- 右列：单场 XAI 战术归因与自动教练报告 (需求模块 6, 8, 9) ---
with col_right:
    st.subheader("📑 赛前战术分析报告 (Executive Coach Report)")

    # 计算当前比赛的战术贡献归因（Match Attribution）
    rf_importances = model.feature_importances_
    current_vals = input_vector[0]

    contributions = []
    for i, feat in enumerate(tactical_features):
        base_val = FEATURE_BASELINES[feat]
        curr_val = current_vals[i]
        imp = rf_importances[i]

        # 计算偏离方向：例如 xG 越高越利好 Win；PPDA 越低越利好 Win
        if feat in ['ppda', 'fouls_committed', 'yellow_cards', 'errors_leading_to_shot']:
            diff = base_val - curr_val  # 数值越小代表表现越好
        else:
            diff = curr_val - base_val  # 数值越大代表表现越好

        score = diff * imp
        contributions.append((feat, curr_val, score))

    # 按贡献得分排序
    contributions.sort(key=lambda x: x[2], reverse=True)
    top_positives = [c for c in contributions if c[2] > 0][:3]
    top_negatives = [c for c in contributions if c[2] < 0][-3:]

    # 动态构建结构化报告
    st.markdown(f"""
    <div class="coach-box">
        <h4>🏟️ {home_team} vs {away_team}</h4>
        <p><b>我方阵型：</b>{formation} | <b>对手风格：</b>{opp_style}</p>
        <p><b>战术倾向：</b>{scenario}</p>
        <hr style="border:0.5px solid #374151;">
        
        <h5>✅ 本场战术优势红利 (Key Strengths)</h5>
        <ul>
            {"".join([f"<li><b>{c[0].upper()}</b> ({c[1]:.1f}): 偏离基准带来正向胜率加成</li>" for c in top_positives])}
        </ul>

        <h5>⚠️ 关键战术风险点 (Key Vulnerabilities)</h5>
        <ul>
            {"".join([f"<li><b>{c[0].upper()}</b> ({c[1]:.1f}): 当前设定可能导致掌控力下降</li>" for c in top_negatives])}
        </ul>

        <hr style="border:0.5px solid #374151;">
        <h5>💡 临场指挥与换人建议 (Actionable Directives)</h5>
        <p>1. <b>逼抢节奏控制：</b>当前 PPDA 为 {ppda:.1f}。若下半场需破局，建议在 60' 降至 {max(3.0, ppda-2.0):.1f} 实施全场压迫。</p>
        <p>2. <b>高空与边路：</b>针对 {opp_style} 风格，建议增加边路传中（当前 {crosses_completed} 次），并在 70' 换上高大中锋冲击禁区。</p>
    </div>
    """, unsafe_allow_html=True)

    # 单场 XAI 归因柱状图
    st.markdown("#### 🔍 单场战术归因 (Match Attribution XAI)")
    top_xai = sorted(contributions, key=lambda x: abs(x[2]), reverse=True)[:6]
    xai_df = pd.DataFrame({
        'Feature': [x[0] for x in top_xai],
        'Contribution': [x[2] for x in top_xai]
    }).sort_values(by='Contribution')

    fig_xai, ax_xai = plt.subplots(figsize=(6, 2.8))
    colors = ['#10b981' if v >= 0 else '#ef4444' for v in xai_df['Contribution']]
    ax_xai.barh(xai_df['Feature'], xai_df['Contribution'], color=colors)
    ax_xai.set_title("Tactical Drivers Impacting This Match", color='white', fontsize=10)
    ax_xai.tick_params(colors='white')
    fig_xai.patch.set_facecolor('#111827')
    ax_xai.set_facecolor('#1f2937')
    st.pyplot(fig_xai)
