import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------
# 1. 页面配置与暗黑风 CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pro AI Football Tactical Engine",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #1f2937; padding: 12px; border-radius: 8px; border-left: 4px solid #10b981; }
    .card-box { background-color: #111827; padding: 16px; border-radius: 8px; border: 1px solid #374151; margin-bottom: 12px; }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ Pro AI Tactical Decision Support System")
st.caption("职业足球赛前博弈推演、数据驱动战术映射与蒙特卡洛引擎 (Data-Driven Coach Edition)")

# ---------------------------------------------------------
# 2. 真实历史数据集与 ML 模型动态加载
# ---------------------------------------------------------
@st.cache_data
def load_datasets():
    df_clean = pd.read_csv('clean_world_cup_2022.csv')
    df_raw = pd.read_csv('data.csv')
    return df_clean, df_raw

@st.cache_resource
def load_model():
    return joblib.load('world_cup_rf_model.pkl')

try:
    df_clean, df_raw = load_datasets()
    model = load_model()
    st.sidebar.success("✅ 2022 世界杯真实数据集 & ML 模型加载成功")
except Exception as e:
    st.sidebar.error(f"❌ 数据或模型加载失败: {e}")
    st.stop()

# 16 个底层特征定义
tactical_features = [
    'xg', 'possession', 'shots_on_target', 'shots_total',
    'passes_completed', 'pass_accuracy', 'ppda', 'tackles_successful',
    'interceptions', 'clearances', 'fouls_committed', 'yellow_cards',
    'corners', 'crosses_completed', 'aerial_duels_won_pct', 'errors_leading_to_shot'
]

FEATURE_BASELINES = df_clean[tactical_features].mean().to_dict()

# ---------------------------------------------------------
# 3. 2D 足球场与阵型绘制函数 (Matplotlib Engine)
# ---------------------------------------------------------
def draw_2d_pitch(formation_name, team_name):
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='#0e1117')
    ax.set_facecolor('#1e293b') # 深绿青草色暗黑风

    # 画外场线与中线
    ax.plot([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], color="white", alpha=0.3, linewidth=1.5)
    ax.plot([50, 50], [0, 100], color="white", alpha=0.3, linewidth=1.5)
    
    # 画中圈与禁区
    center_circle = patches.Circle((50, 50), 12, color="white", fill=False, alpha=0.3, linewidth=1.5)
    left_penalty = patches.Rectangle((0, 20), 18, 60, color="white", fill=False, alpha=0.3, linewidth=1.5)
    right_penalty = patches.Rectangle((82, 20), 18, 60, color="white", fill=False, alpha=0.3, linewidth=1.5)
    ax.add_patch(center_circle)
    ax.add_patch(left_penalty)
    ax.add_patch(right_penalty)

    # 11 门员及队员坐标映射
    formations_coords = {
        "4-3-3": [(8,50), (28,18), (25,38), (25,62), (28,82), (50,28), (45,50), (50,72), (80,20), (85,50), (80,80)],
        "4-2-3-1": [(8,50), (28,18), (25,38), (25,62), (28,82), (42,35), (42,65), (65,20), (68,50), (65,80), (85,50)],
        "3-5-2": [(8,50), (25,28), (23,50), (25,72), (45,15), (48,35), (45,50), (48,65), (45,85), (82,38), (82,62)],
        "4-4-2": [(8,50), (28,18), (25,38), (25,62), (28,82), (52,18), (50,38), (50,62), (52,82), (82,38), (82,62)],
        "5-4-1": [(8,50), (28,12), (25,31), (23,50), (25,69), (28,88), (50,20), (48,40), (48,60), (50,80), (82,50)],
        "3-4-3": [(8,50), (25,28), (23,50), (25,72), (50,18), (48,38), (48,62), (50,82), (80,20), (85,50), (80,80)]
    }

    coords = formations_coords.get(formation_name, formations_coords["4-3-3"])

    # 绘制球员节点
    for idx, (x, y) in enumerate(coords):
        node_color = '#10b981' if idx > 0 else '#f59e0b'
        ax.scatter(x, y, s=280, color=node_color, edgecolors='white', linewidth=2, zorder=5)
        label = "GK" if idx == 0 else str(idx+1)
        ax.text(x, y, label, color='white', fontsize=8, fontweight='bold', ha='center', va='center', zorder=6)

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.axis('off')
    ax.set_title(f"{team_name} Formation Pitch View ({formation_name})", color='white', fontsize=11, pad=10)
    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# 4. 侧边栏设置 (对阵、阵型、战术指令与情景)
# ---------------------------------------------------------
st.sidebar.header("⚙️ 1. 比赛对阵设置 (Data-Driven)")
all_teams = sorted(df_clean['team'].unique().tolist())

col_h, col_a = st.sidebar.columns(2)
home_team = col_h.selectbox("我方球队", all_teams, index=all_teams.index("Argentina") if "Argentina" in all_teams else 0)
away_team = col_a.selectbox("对手球队", all_teams, index=all_teams.index("France") if "France" in all_teams else 1)

# 动态计算我方球队在 2022 世界杯的真实场均数据作为 Baseline
home_historical_data = df_clean[df_clean['team'] == home_team]
if not home_historical_data.empty:
    team_baseline = home_historical_data[tactical_features].mean().to_dict()
else:
    team_baseline = FEATURE_BASELINES.copy()

st.sidebar.markdown("---")
st.sidebar.header("📐 2. 阵型与战术意图 (Stage 1 Mapper)")

formation = st.sidebar.selectbox(
    "我方部署阵型",
    ["4-3-3", "4-2-3-1", "3-5-2", "4-4-2", "5-4-1", "3-4-3"]
)

tactical_style = st.sidebar.selectbox(
    "我方主导战术 (Our Instruction)",
    ["常规平衡 (Balanced)", "高位逼抢 (High Pressing)", "低位反击 (Low Block Counter)", "控球主导 (Possession Focus)"]
)

opp_style = st.sidebar.selectbox(
    "对手战术风格 (Opponent Style)",
    ["常规平衡 (Balanced)", "高位逼抢 (High Press)", "传控主导 (Tiki-Taka)", "低位摆大巴 (Low Block)", "快速反击 (Counter Attack)"]
)

st.sidebar.markdown("---")
st.sidebar.header("🎯 3. 一键战术情景 (Scenario)")
scenario = st.sidebar.radio(
    "战术倾向选择",
    ["常规推演 (Balanced)", "全员激进压迫 (Press All Out)", "极端防守摆大巴 (Parking Bus)", "全线压上攻坚 (Ultra Attack)"],
    index=0
)

# ---------------------------------------------------------
# 5. Stage 1: 战术映射器逻辑 (Instruction -> Mapped Stats)
# ---------------------------------------------------------
def apply_tactical_mapping(baseline, style, opp_style, scenario, formation):
    mapped = baseline.copy()
    
    # A. 我方战术指令 (Our Instruction)
    if style == "高位逼抢 (High Pressing)":
        mapped['ppda'] = max(4.0, mapped['ppda'] * 0.7)
        mapped['tackles_successful'] = mapped['tackles_successful'] * 1.2
        mapped['xg'] = mapped['xg'] * 1.15
        mapped['errors_leading_to_shot'] = mapped['errors_leading_to_shot'] + 0.2
    elif style == "低位反击 (Low Block Counter)":
        mapped['possession'] = min(40.0, mapped['possession'] * 0.75)
        mapped['ppda'] = mapped['ppda'] * 1.4
        mapped['clearances'] = mapped['clearances'] * 1.3
        mapped['passes_completed'] = mapped['passes_completed'] * 0.8
    elif style == "控球主导 (Possession Focus)":
        mapped['possession'] = max(60.0, mapped['possession'] * 1.15)
        mapped['passes_completed'] = mapped['passes_completed'] * 1.2
        mapped['pass_accuracy'] = min(92.0, mapped['pass_accuracy'] * 1.05)

    # B. 对手风格制约 (Opponent Influence)
    if opp_style == "高位逼抢 (High Press)":
        mapped['ppda'] = max(4.0, mapped['ppda'] - 1.5)
        mapped['pass_accuracy'] = max(60.0, mapped['pass_accuracy'] - 4.0)
    elif opp_style == "低位摆大巴 (Low Block)":
        mapped['possession'] = min(75.0, mapped['possession'] + 8.0)
        mapped['clearances'] = mapped['clearances'] + 6.0
    elif opp_style == "传控主导 (Tiki-Taka)":
        mapped['possession'] = max(25.0, mapped['possession'] - 8.0)
        mapped['tackles_successful'] = mapped['tackles_successful'] + 3.0

    # C. 战术情景微调 (Scenario)
    if scenario == "全员激进压迫 (Press All Out)":
        mapped['ppda'] = 5.5
        mapped['tackles_successful'] = mapped['tackles_successful'] + 5.0
        mapped['xg'] = mapped['xg'] * 1.2
    elif scenario == "极端防守摆大巴 (Parking Bus)":
        mapped['possession'] = 33.0
        mapped['ppda'] = 20.0
        mapped['clearances'] = mapped['clearances'] + 10.0
        mapped['xg'] = mapped['xg'] * 0.6
    elif scenario == "全线压上攻坚 (Ultra Attack)":
        mapped['xg'] = mapped['xg'] * 1.3
        mapped['shots_on_target'] = mapped['shots_on_target'] + 2.0
        mapped['possession'] = max(60.0, mapped['possession'])

    return mapped

mapped_stats = apply_tactical_mapping(team_baseline, tactical_style, opp_style, scenario, formation)

st.sidebar.caption(f"💡 指标已基于 **{home_team}** 真实数据及 **{tactical_style}** 指令自动推算。")

# ---------------------------------------------------------
# 6. Tab 选项卡主界面排版
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🏟️ 1. 赛前阵型与战术部署 (Tactical Pitch)",
    "🎲 2. 蒙特卡洛模拟与胜率诊断 (Monte Carlo Engine)",
    "📑 3. AI 临场决策与战术报告 (Executive Brief)"
])

# =========================================================
# TAB 1: 2D 足球场 + 战术指令微调面板
# =========================================================
with tab1:
    col_pitch, col_panel = st.columns([1.2, 1.0])

    with col_pitch:
        st.subheader("🏟️ 2D 战术阵型部署")
        fig_pitch = draw_2d_pitch(formation, home_team)
        st.pyplot(fig_pitch)

    with col_panel:
        st.subheader("📋 战术指令微调 (Tactical Panel)")
        st.caption("系统已基于真实数据集与 Stage 1 战术映射自动生成推算值，教练组可手动微调：")

        with st.expander("🎯 进攻终结 (Attacking)", expanded=True):
            xg = st.slider("预期进球 (xG Target)", 0.1, 4.0, float(round(mapped_stats['xg'], 2)), 0.1)
            shots_on_target = st.slider("射正数 Target", 0, 15, int(round(mapped_stats['shots_on_target'])))
            shots_total = st.slider("总射门数 Target", 1, 30, max(int(round(mapped_stats['shots_total'])), shots_on_target + 2))
            corners = st.slider("角球次数", 0, 15, int(round(mapped_stats['corners'])))

        with st.expander("🔄 组织控球 (Build-up)", expanded=False):
            possession = st.slider("控球率 (%)", 20, 80, int(round(mapped_stats['possession'])))
            passes_completed = st.slider("成功传球数", 100, 900, int(round(mapped_stats['passes_completed'])), 10)
            pass_accuracy = st.slider("传球成功率 (%)", 50, 95, int(round(mapped_stats['pass_accuracy'])))
            crosses_completed = st.slider("成功传中数", 0, 25, int(round(mapped_stats['crosses_completed'])))

        with st.expander("🛡️ 防守压迫 (Defensive)", expanded=False):
            ppda = st.slider("PPDA (逼抢强度, 越低越高压)", 3.0, 30.0, float(round(mapped_stats['ppda'], 1)), 0.5)
            tackles_successful = st.slider("成功抢断", 3, 40, int(round(mapped_stats['tackles_successful'])))
            interceptions = st.slider("拦截次数", 1, 30, int(round(mapped_stats['interceptions'])))
            clearances = st.slider("解围次数", 3, 50, int(round(mapped_stats['clearances'])))

        with st.expander("⚔️ 对抗纪律 (Duels & Errors)", expanded=False):
            aerial_duels_won_pct = st.slider("争顶胜率 (%)", 20, 80, int(round(mapped_stats['aerial_duels_won_pct'])))
            fouls_committed = st.slider("犯规次数", 1, 30, int(round(mapped_stats['fouls_committed'])))
            yellow_cards = st.number_input("黄牌数", 0, 8, int(round(mapped_stats['yellow_cards'])))
            errors_leading_to_shot = st.number_input("致命失误致射门", 0, 3, int(round(mapped_stats['errors_leading_to_shot'])))

# 构造标准输入向量
input_vector = np.array([[
    xg, possession, shots_on_target, shots_total,
    passes_completed, pass_accuracy, ppda, tackles_successful,
    interceptions, clearances, fouls_committed, yellow_cards,
    corners, crosses_completed, aerial_duels_won_pct, errors_leading_to_shot
]])

# =========================================================
# TAB 2: 蒙特卡洛 1000 次模拟引擎
# =========================================================
with tab2:
    st.subheader("🎲 蒙特卡洛 1,000 场平行宇宙模拟")
    st.caption("在赛前战术框架下，注入符合比赛规律的临场随机扰动，评估风险与置信区间：")

    N_SIM = 1000
    np.random.seed(42)
    noise = np.random.normal(0, 1, (N_SIM, 16))
    scale = np.array([0.15, 2.5, 0.8, 1.5, 25, 1.5, 0.8, 1.2, 1.0, 1.5, 1.0, 0.3, 0.8, 0.8, 2.0, 0.2])
    sim_inputs = np.clip(input_vector + noise * scale, a_min=0, a_max=None)

    sim_preds = model.predict(sim_inputs)
    sim_probs = model.predict_proba(sim_inputs)

    classes = model.classes_
    win_idx = list(classes).index('Win') if 'Win' in classes else 2

    win_count = np.sum(sim_preds == 'Win')
    draw_count = np.sum(sim_preds == 'Draw')
    loss_count = np.sum(sim_preds == 'Loss')

    mc_win_pct = (win_count / N_SIM) * 100
    mc_draw_pct = (draw_count / N_SIM) * 100
    mc_loss_pct = (loss_count / N_SIM) * 100

    win_probs_series = sim_probs[:, win_idx]
    ci_lower = np.percentile(win_probs_series, 2.5) * 100
    ci_upper = np.percentile(win_probs_series, 97.5) * 100

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("模拟胜率 (Win)", f"{mc_win_pct:.1f}%")
    m2.metric("平局率 (Draw)", f"{mc_draw_pct:.1f}%")
    m3.metric("败率 (Loss)", f"{mc_loss_pct:.1f}%")
    m4.metric("不败率 (Win/Draw)", f"{(mc_win_pct + mc_draw_pct):.1f}%")

    st.markdown(f"##### 📊 **95% 胜率置信区间 (Confidence Interval)**: `{ci_lower:.1f}%` ~ `{ci_upper:.1f}%`")

    # 直方图
    fig_mc, ax_mc = plt.subplots(figsize=(8, 3.2))
    ax_mc.hist(win_probs_series * 100, bins=30, color='#10b981', alpha=0.75, edgecolor='black')
    ax_mc.axvline(mc_win_pct, color='#ef4444', linestyle='--', linewidth=2, label=f'Mean Win Prob ({mc_win_pct:.1f}%)')
    ax_mc.set_title("1,000 Runs Win Probability Distribution", color='white', fontsize=11)
    ax_mc.set_xlabel("Predicted Win Probability (%)", color='white')
    ax_mc.set_ylabel("Frequency", color='white')
    ax_mc.tick_params(colors='white')
    ax_mc.legend(loc='upper right', facecolor='#111827', labelcolor='white')
    fig_mc.patch.set_facecolor('#111827')
    ax_mc.set_facecolor('#1f2937')
    st.pyplot(fig_mc)

# =========================================================
# TAB 3: XAI 归因与动态教练报告
# =========================================================
with tab3:
    st.subheader("📑 赛前战术决策报告 (Executive Brief)")

    rf_importances = model.feature_importances_
    current_vals = input_vector[0]

    contributions = []
    for i, feat in enumerate(tactical_features):
        base_val = FEATURE_BASELINES[feat]
        curr_val = current_vals[i]
        imp = rf_importances[i]

        if feat in ['ppda', 'fouls_committed', 'yellow_cards', 'errors_leading_to_shot']:
            diff = base_val - curr_val
        else:
            diff = curr_val - base_val

        score = diff * imp
        contributions.append((feat, curr_val, score))

    contributions.sort(key=lambda x: x[2], reverse=True)
    top_positives = [c for c in contributions if c[2] > 0][:3]
    top_negatives = [c for c in contributions if c[2] < 0][-3:]

    # 动态建议生成逻辑
    dynamic_directives = []
    for feat, val, score in top_negatives:
        if feat == 'ppda':
            if val > 12.0:
                dynamic_directives.append(f"🔥 **加强前场压迫**：当前 PPDA 为 `{val:.1f}`（偏被动）。建议下半场压低 PPDA 至 `{max(4.0, val - 3.5):.1f}`，提高前场抢断效率。")
            else:
                dynamic_directives.append(f"⚠️ **防守体能风险**：当前 PPDA 高达 `{val:.1f}`（极高压），需警惕下半场体能快速消耗与后场留白。")
        elif feat == 'crosses_completed':
            if val < 5:
                dynamic_directives.append(f"⚽ **利用边路宽度**：当前成功传中仅 `{val}` 次。针对 **{opp_style}** 防线，建议增加边路套上并提高传中频率。")
            else:
                dynamic_directives.append(f"🎯 **中路渗透切入**：传中次数已达 `{val}` 次但收益递减，建议减少盲目吊传，增加肋部直塞。")
        elif feat == 'xg':
            dynamic_directives.append(f"🎯 **提升终结质量**：当前预期进球 xG 仅为 `{val:.2f}`。建议通过定位球战术或提升远射质量来制造绝对得分机会。")
        elif feat == 'possession':
            if val < 45:
                dynamic_directives.append(f"🔄 **争夺中场控球**：控球率仅 `{val:.0f}%`。建议增加安全短传，稳住比赛节奏。")
            else:
                dynamic_directives.append(f"⚡ **加快进攻节奏**：控球率高达 `{val:.0f}%` 但转换效率不足，建议减少无效倒脚，提速向前直塞。")
        elif feat in ['fouls_committed', 'yellow_cards']:
            dynamic_directives.append(f"🟨 **控制动作纪律**：当前犯规/黄牌偏高（犯规 `{fouls_committed}` 次，黄牌 `{yellow_cards}` 张），需提醒中场控制动作，避免红牌减员。")
        elif feat == 'aerial_duels_won_pct':
            dynamic_directives.append(f"🛡️ **高空球保护**：争顶胜率仅 `{val:.0f}%`。建议中后卫收缩禁区，减少毁灭性头球争顶。")
        elif feat == 'errors_leading_to_shot':
            dynamic_directives.append(f"🚨 **严禁出球失误**：后场已出现 `{val}` 次致命失误。后场出球时应减少高风险横传，简化解围路线。")

    if not dynamic_directives:
        dynamic_directives.append("✨ **战术体系高度平衡**：当前各指标运行良好，建议保持现有阵型与比赛节奏。")

    col_report, col_xai = st.columns([1.1, 0.9])

    with col_report:
        with st.container(border=True):
            st.markdown(f"### 🏟️ {home_team} vs {away_team}")
            st.caption(f"**我方阵型**：{formation} | **主导战术**：{tactical_style} | **对手风格**：{opp_style}")
            st.divider()

            st.markdown("##### ✅ 本场战术优势红利 (Key Strengths)")
            if top_positives:
                for feat, val, score in top_positives:
                    st.success(f"**{feat.upper()}** (`{val:.1f}`): 正向偏离基准，带来胜率加成")
            else:
                st.caption("暂无显著优势指标")

            st.markdown("##### ⚠️ 关键战术风险点 (Key Vulnerabilities)")
            if top_negatives:
                for feat, val, score in top_negatives:
                    st.warning(f"**{feat.upper()}** (`{val:.1f}`): 拖累当前战术掌控力")
            else:
                st.caption("当前战术风险较低")

            st.divider()
            st.markdown("##### 💡 动态临场指挥建议 (Dynamic Directives)")
            for idx, directive in enumerate(dynamic_directives, 1):
                st.info(f"{idx}. {directive}")

    with col_xai:
        st.markdown("#### 🔍 单场战术归因 (Match Attribution XAI)")
        top_xai = sorted(contributions, key=lambda x: abs(x[2]), reverse=True)[:6]
        xai_df = pd.DataFrame({
            'Feature': [x[0] for x in top_xai],
            'Contribution': [x[2] for x in top_xai]
        }).sort_values(by='Contribution')

        fig_xai, ax_xai = plt.subplots(figsize=(6, 3.8))
        colors = ['#10b981' if v >= 0 else '#ef4444' for v in xai_df['Contribution']]
        ax_xai.barh(xai_df['Feature'], xai_df['Contribution'], color=colors)
        ax_xai.set_title("Tactical Drivers Impacting This Match", color='white', fontsize=10)
        ax_xai.tick_params(colors='white')
        fig_xai.patch.set_facecolor('#111827')
        ax_xai.set_facecolor('#1f2937')
        st.pyplot(fig_xai)
