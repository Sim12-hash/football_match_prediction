import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------
# 1. 页面配置与超高对比度暗黑风 CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pro AI Football Tactical Engine",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #0b0f19; }
    
    /* 高对比度视觉勋章卡片 */
    .metric-card-win {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border: 2px solid #00FF87;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 0 12px rgba(0, 255, 135, 0.2);
    }
    .metric-card-draw {
        background: linear-gradient(135deg, #713f12 0%, #451a03 100%);
        border: 2px solid #FACC15;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 0 12px rgba(250, 204, 21, 0.2);
    }
    .metric-card-loss {
        background: linear-gradient(135deg, #881337 0%, #4c0519 100%);
        border: 2px solid #FF0055;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 0 12px rgba(255, 0, 85, 0.2);
    }
    .metric-card-undefeated {
        background: linear-gradient(135deg, #1e1b4b 0%, #0f172a 100%);
        border: 2px solid #38BDF8;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        box-shadow: 0 0 12px rgba(56, 189, 248, 0.2);
    }
    
    .metric-title { color: #94a3b8; font-size: 13px; font-weight: 600; text-transform: uppercase; }
    .metric-value-win { color: #00FF87; font-size: 28px; font-weight: 800; }
    .metric-value-draw { color: #FACC15; font-size: 28px; font-weight: 800; }
    .metric-value-loss { color: #FF0055; font-size: 28px; font-weight: 800; }
    .metric-value-undefeated { color: #38BDF8; font-size: 28px; font-weight: 800; }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ Pro AI Tactical Decision Support System")
st.caption("职业足球赛前博弈推演、审计级战术映射与高对比度蒙特卡洛引擎 (Audited Coach Edition)")

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
# 3. 2D 足球场绘制 (含战术落位图层)
# ---------------------------------------------------------
def draw_2d_pitch_enhanced(formation_name, team_name, tactical_style):
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='#0b0f19')
    ax.set_facecolor('#1e293b')

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

    # 动态战术图层
    if "高位逼抢" in tactical_style:
        ax.axvline(x=68, color='#FF0055', linestyle='--', linewidth=2, alpha=0.8)
        ax.text(69, 92, '🔥 HIGH PRESS LINE', color='#FF0055', fontsize=8, fontweight='bold')
    elif "低位反击" in tactical_style:
        ax.axvline(x=32, color='#38BDF8', linestyle='--', linewidth=2, alpha=0.8)
        ax.text(33, 92, '🛡️ LOW BLOCK LINE', color='#38BDF8', fontsize=8, fontweight='bold')
    elif "控球主导" in tactical_style:
        ax.axvline(x=50, color='#00FF87', linestyle='--', linewidth=2, alpha=0.8)
        ax.text(51, 92, '🔄 MIDFIELD CONTROL', color='#00FF87', fontsize=8, fontweight='bold')

    formations_coords = {
        "4-3-3": [(8,50), (28,18), (25,38), (25,62), (28,82), (50,28), (45,50), (50,72), (80,20), (85,50), (80,80)],
        "4-2-3-1": [(8,50), (28,18), (25,38), (25,62), (28,82), (42,35), (42,65), (65,20), (68,50), (65,80), (85,50)],
        "3-5-2": [(8,50), (25,28), (23,50), (25,72), (45,15), (48,35), (45,50), (48,65), (45,85), (82,38), (82,62)],
        "4-4-2": [(8,50), (28,18), (25,38), (25,62), (28,82), (52,18), (50,38), (50,62), (52,82), (82,38), (82,62)],
        "5-4-1": [(8,50), (28,12), (25,31), (23,50), (25,69), (28,88), (50,20), (48,40), (48,60), (50,80), (82,50)],
        "3-4-3": [(8,50), (25,28), (23,50), (25,72), (50,18), (48,38), (48,62), (50,82), (80,20), (85,50), (80,80)]
    }

    coords = formations_coords.get(formation_name, formations_coords["4-3-3"])

    for idx, (x, y) in enumerate(coords):
        node_color = '#00FF87' if idx > 0 else '#FACC15'
        ax.scatter(x, y, s=280, color=node_color, edgecolors='white', linewidth=2, zorder=5)
        label = "GK" if idx == 0 else str(idx+1)
        ax.text(x, y, label, color='black' if idx == 0 else 'white', fontsize=8, fontweight='bold', ha='center', va='center', zorder=6)

    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    ax.axis('off')
    ax.set_title(f"{team_name} ({formation_name}) - {tactical_style}", color='white', fontsize=11, pad=10)
    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# 4. 侧边栏设置 (数据驱动对阵与战术选择)
# ---------------------------------------------------------
st.sidebar.header("⚙️ 1. 比赛对阵设置 (Data-Driven)")
all_teams = sorted(df_clean['team'].unique().tolist())

col_h, col_a = st.sidebar.columns(2)
home_team = col_h.selectbox("我方球队", all_teams, index=all_teams.index("Argentina") if "Argentina" in all_teams else 0)
away_team = col_a.selectbox("对手球队", all_teams, index=all_teams.index("France") if "France" in all_teams else 1)

# 动态计算我方与对手球队在 2022 世界杯的真实场均数据作为 Baseline
home_data = df_clean[df_clean['team'] == home_team]
away_data = df_clean[df_clean['team'] == away_team]

team_baseline = home_data[tactical_features].mean().to_dict() if not home_data.empty else FEATURE_BASELINES.copy()
opp_baseline = away_data[tactical_features].mean().to_dict() if not away_data.empty else FEATURE_BASELINES.copy()

st.sidebar.markdown("---")
st.sidebar.header("📐 2. 阵型与战术意图 (Stage 1 Mapper)")

formation = st.sidebar.selectbox("我方部署阵型", ["4-3-3", "4-2-3-1", "3-5-2", "4-4-2", "5-4-1", "3-4-3"])
tactical_style = st.sidebar.selectbox("我方主导战术", ["常规平衡 (Balanced)", "高位逼抢 (High Pressing)", "低位反击 (Low Block Counter)", "控球主导 (Possession Focus)"])
opp_style = st.sidebar.selectbox("对手战术风格", ["常规平衡 (Balanced)", "高位逼抢 (High Press)", "传控主导 (Tiki-Taka)", "低位摆大巴 (Low Block)", "快速反击 (Counter Attack)"])

st.sidebar.markdown("---")
st.sidebar.header("🎯 3. 一键战术情景 (Scenario)")
scenario = st.sidebar.radio("战术倾向选择", ["常规推演 (Balanced)", "全员激进压迫 (Press All Out)", "极端防守摆大巴 (Parking Bus)", "全线压上攻坚 (Ultra Attack)"], index=0)

# ---------------------------------------------------------
# 5. 审计级 Stage 1 战术映射器逻辑 (含漏洞修复)
# ---------------------------------------------------------
def scale_attack_metrics(mapped_stats, xg_multiplier):
    """漏洞 3 修复：进攻指标物理比例同步缩放"""
    mapped_stats['xg'] *= xg_multiplier
    mapped_stats['shots_on_target'] = max(1.0, mapped_stats['shots_on_target'] * xg_multiplier)
    mapped_stats['shots_total'] = max(mapped_stats['shots_on_target'] + 2.0, mapped_stats['shots_total'] * (1 + (xg_multiplier - 1) * 0.8))
    return mapped_stats

def apply_error_and_card_penalty(mapped_stats):
    """漏洞 2 修复：模型死特征代理惩罚代换"""
    if mapped_stats['errors_leading_to_shot'] > 0:
        penalty = mapped_stats['errors_leading_to_shot'] * 0.15
        mapped_stats['xg'] = max(0.1, mapped_stats['xg'] - penalty)
        mapped_stats['pass_accuracy'] = max(50.0, mapped_stats['pass_accuracy'] - 3.0)
    if mapped_stats['yellow_cards'] >= 2:
        mapped_stats['tackles_successful'] *= 0.85
        mapped_stats['ppda'] += 1.5
    return mapped_stats

def apply_tactical_mapping_audited(home_base, opp_base, style, opp_style, scenario):
    mapped = home_base.copy()
    
    # 漏洞 4 修复：对位强弱修正因子 (Matchup Adjustment Factor)
    xg_diff_factor = (home_base['xg'] - opp_base['xg']) * 0.1
    poss_diff_factor = (home_base['possession'] - opp_base['possession']) * 0.15

    mapped['xg'] = max(0.2, mapped['xg'] + xg_diff_factor)
    mapped['possession'] = np.clip(mapped['possession'] + poss_diff_factor, 25.0, 75.0)

    # A. 我方战术指令 (物理联动更新)
    if style == "高位逼抢 (High Pressing)":
        mapped['ppda'] = max(4.0, mapped['ppda'] * 0.7)
        mapped['tackles_successful'] *= 1.2
        mapped = scale_attack_metrics(mapped, xg_multiplier=1.15)
        mapped['errors_leading_to_shot'] += 0.2
    elif style == "低位反击 (Low Block Counter)":
        mapped['possession'] = min(42.0, mapped['possession'] * 0.75)
        mapped['ppda'] *= 1.4
        mapped['clearances'] *= 1.3
        mapped['passes_completed'] *= 0.8
        mapped = scale_attack_metrics(mapped, xg_multiplier=0.85)
    elif style == "控球主导 (Possession Focus)":
        mapped['possession'] = np.clip(mapped['possession'] * 1.15, 58.0, 78.0)
        mapped['passes_completed'] *= 1.2
        mapped['pass_accuracy'] = min(94.0, mapped['pass_accuracy'] * 1.05)

    # B. 对手风格制约
    if opp_style == "高位逼抢 (High Press)":
        mapped['ppda'] = max(4.0, mapped['ppda'] - 1.5)
        mapped['pass_accuracy'] = max(60.0, mapped['pass_accuracy'] - 4.0)
    elif opp_style == "低位摆大巴 (Low Block)":
        mapped['possession'] = min(78.0, mapped['possession'] + 8.0)
        mapped['clearances'] += 6.0

    # C. 情景微调
    if scenario == "全员激进压迫 (Press All Out)":
        mapped['ppda'] = 5.5
        mapped['tackles_successful'] += 5.0
        mapped = scale_attack_metrics(mapped, xg_multiplier=1.2)
    elif scenario == "极端防守摆大巴 (Parking Bus)":
        mapped['possession'] = 33.0
        mapped['ppda'] = 20.0
        mapped['clearances'] += 10.0
        mapped = scale_attack_metrics(mapped, xg_multiplier=0.6)

    # D. 代理惩罚代换
    mapped = apply_error_and_card_penalty(mapped)

    return mapped

mapped_stats = apply_tactical_mapping_audited(team_baseline, opp_baseline, tactical_style, opp_style, scenario)

# ---------------------------------------------------------
# 6. Tab 选项卡主界面排版
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🏟️ 1. 赛前阵型与战术部署 (Tactical Pitch)",
    "🎲 2. 蒙特卡洛模拟与胜率诊断 (Monte Carlo Engine)",
    "📑 3. AI 临场决策与战术报告 (Executive Brief)"
])

# =========================================================
# TAB 1: 2D 足球场 + 物理指标联动微调面板
# =========================================================
with tab1:
    col_pitch, col_panel = st.columns([1.2, 1.0])

    with col_pitch:
        st.subheader("🏟️ 2D 战术阵型部署与意图图层")
        fig_pitch = draw_2d_pitch_enhanced(formation, home_team, tactical_style)
        st.pyplot(fig_pitch)

        st.markdown("##### 📈 战术部署对 KPI 的预期影响 (vs 历史基准)")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("预期进球 xG", f"{mapped_stats['xg']:.2f}", f"{mapped_stats['xg'] - team_baseline['xg']:+.2f}")
        k2.metric("控球率 Possession", f"{mapped_stats['possession']:.1f}%", f"{mapped_stats['possession'] - team_baseline['possession']:+.1f}%")
        k3.metric("逼抢强度 PPDA", f"{mapped_stats['ppda']:.1f}", f"{mapped_stats['ppda'] - team_baseline['ppda']:+.1f}")
        k4.metric("成功抢断 Tackles", f"{mapped_stats['tackles_successful']:.1f}", f"{mapped_stats['tackles_successful'] - team_baseline['tackles_successful']:+.1f}")

    with col_panel:
        st.subheader("📋 战术指令微调 (Tactical Panel)")
        st.caption("系统已生成推算值，教练组可手动微调：")

        with st.expander("🎯 进攻终结 (Attacking)", expanded=True):
            xg = st.slider("预期进球 (xG Target)", 0.1, 4.0, float(round(mapped_stats['xg'], 2)), 0.1)
            shots_on_target = st.slider("射正数 Target", 0, 15, int(round(mapped_stats['shots_on_target'])))
            shots_total = st.slider("总射门数 Target", 1, 30, max(int(round(mapped_stats['shots_total'])), shots_on_target + 2))
            corners = st.slider("角球次数", 0, 15, int(round(mapped_stats['corners'])))

        with st.expander("🔄 组织控球 (Build-up)", expanded=False):
            possession = st.slider("控球率 (%)", 20, 80, int(round(mapped_stats['possession'])))
            passes_completed = st.slider("成功传球数", 100, 900, int(round(mapped_stats['passes_completed'])), 10)
            pass_accuracy = st.slider("传球成功率 (%)", 50, 98, int(round(mapped_stats['pass_accuracy'])))
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

input_vector = np.array([[
    xg, possession, shots_on_target, shots_total,
    passes_completed, pass_accuracy, ppda, tackles_successful,
    interceptions, clearances, fouls_committed, yellow_cards,
    corners, crosses_completed, aerial_duels_won_pct, errors_leading_to_shot
]])

# =========================================================
# TAB 2: 高对比度蒙特卡洛引擎 (含漏洞 1 物理上下界硬约束)
# =========================================================
with tab2:
    st.subheader("🎲 蒙特卡洛 1,000 场平行宇宙模拟 & 高对比度诊判")

    # 漏洞 1 修复：16 维特征的物理上下界硬约束
    a_min_bounds = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    a_max_bounds = np.array([10.0, 100.0, 30.0, 50.0, 1500.0, 100.0, 50.0, 60.0, 50.0, 80.0, 40.0, 11.0, 25.0, 40.0, 100.0, 10.0])

    N_SIM = 1000
    np.random.seed(42)
    noise = np.random.normal(0, 1, (N_SIM, 16))
    scale = np.array([0.15, 2.5, 0.8, 1.5, 25, 1.5, 0.8, 1.2, 1.0, 1.5, 1.0, 0.3, 0.8, 0.8, 2.0, 0.2])
    
    sim_inputs = np.clip(input_vector + noise * scale, a_min=a_min_bounds, a_max=a_max_bounds)

    sim_preds = model.predict(sim_inputs)
    sim_probs = model.predict_proba(sim_inputs)

    classes = model.classes_
    win_idx = list(classes).index('Win') if 'Win' in classes else 2

    mc_win_pct = (np.sum(sim_preds == 'Win') / N_SIM) * 100
    mc_draw_pct = (np.sum(sim_preds == 'Draw') / N_SIM) * 100
    mc_loss_pct = (np.sum(sim_preds == 'Loss') / N_SIM) * 100

    win_probs_series = sim_probs[:, win_idx]
    ci_lower = np.percentile(win_probs_series, 2.5) * 100
    ci_upper = np.percentile(win_probs_series, 97.5) * 100

    # 重构高对比度视觉 Metric 勋章
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f'<div class="metric-card-win"><div class="metric-title">模拟胜率 (WIN)</div><div class="metric-value-win">{mc_win_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card-draw"><div class="metric-title">平局率 (DRAW)</div><div class="metric-value-draw">{mc_draw_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card-loss"><div class="metric-title">败率 (LOSS)</div><div class="metric-value-loss">{mc_loss_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with m4:
        st.markdown(f'<div class="metric-card-undefeated"><div class="metric-title">不败率 (WIN/DRAW)</div><div class="metric-value-undefeated">{(mc_win_pct + mc_draw_pct):.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"##### 📊 **95% 胜率置信区间 (Confidence Interval)**: `{ci_lower:.1f}%` ~ `{ci_upper:.1f}%`")

    col_sim_chart, col_ab = st.columns([1.1, 0.9])

    with col_sim_chart:
        fig_mc, ax_mc = plt.subplots(figsize=(6, 3.4), facecolor='#0b0f19')
        ax_mc.set_facecolor('#1e293b')
        
        n_counts, bins, patches_hist = ax_mc.hist(win_probs_series * 100, bins=25, color='#00FF87', alpha=0.85, edgecolor='black', linewidth=1.2)
        ax_mc.axvline(mc_win_pct, color='#FF0055', linestyle='--', linewidth=2.5, label=f'Mean Win ({mc_win_pct:.1f}%)')
        
        ax_mc.set_title("1,000 Runs Win Probability Distribution", color='white', fontsize=11, fontweight='bold')
        ax_mc.set_xlabel("Predicted Win Probability (%)", color='#94a3b8', fontsize=9)
        ax_mc.set_ylabel("Frequency", color='#94a3b8', fontsize=9)
        ax_mc.tick_params(colors='white')
        ax_mc.legend(loc='upper right', facecolor='#0b0f19', edgecolor='#00FF87', labelcolor='white')
        plt.tight_layout()
        st.pyplot(fig_mc)

    with col_ab:
        with st.container(border=True):
            st.markdown("#### ⚖️ 战术方案 A/B 对比矩阵")
            alt_style = st.selectbox(
                "选择备选方案 B 战术",
                ["低位反击 (Low Block Counter)", "高位逼抢 (High Pressing)", "控球主导 (Possession Focus)", "常规平衡 (Balanced)"]
            )

            alt_mapped = apply_tactical_mapping_audited(team_baseline, opp_baseline, alt_style, opp_style, scenario)
            alt_vector = np.array([[alt_mapped[f] for f in tactical_features]])
            alt_sim_inputs = np.clip(alt_vector + noise * scale, a_min=a_min_bounds, a_max=a_max_bounds)
            alt_sim_preds = model.predict(alt_sim_inputs)
            
            alt_win_pct = (np.sum(alt_sim_preds == 'Win') / N_SIM) * 100
            diff = alt_win_pct - mc_win_pct

            st.divider()
            st.metric("方案 A 预测胜率", f"{mc_win_pct:.1f}%")
            st.metric("方案 B 预测胜率", f"{alt_win_pct:.1f}%", f"{diff:+.1f}%")

            if diff > 3.0:
                st.success(f"💡 **推荐方案 B**：改用 **{alt_style}** 胜率更优！")
            else:
                st.info("⚖️ 方案 A 保持优势，建议沿用当前部署。")

# =========================================================
# TAB 3: XAI 归因与动态教练报告 + Markdown 下载
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

    dynamic_directives = []
    for feat, val, score in top_negatives:
        if feat == 'ppda':
            dynamic_directives.append(f"🔥 **优化压迫节奏**：PPDA (`{val:.1f}`)。建议提升中前场抢断效率。")
        elif feat == 'xg':
            dynamic_directives.append(f"🎯 **提升终结质量**：当前 xG 仅为 `{val:.2f}`。增加远射或定位球威胁。")
        elif feat == 'possession':
            dynamic_directives.append(f"⚡ **加快转化速度**：控球率 `{val:.0f}%`，应提速减少无效传导。")

    if not dynamic_directives:
        dynamic_directives.append("✨ **战术体系高度平衡**：当前部署运转良好。")

    col_report, col_xai = st.columns([1.1, 0.9])

    with col_report:
        with st.container(border=True):
            st.markdown(f"### 🏟️ {home_team} vs {away_team}")
            st.caption(f"**我方阵型**：{formation} | **主导战术**：{tactical_style} | **对手风格**：{opp_style}")
            st.divider()

            st.markdown("##### ✅ 本场战术优势红利")
            for feat, val, score in top_positives:
                st.success(f"**{feat.upper()}** (`{val:.1f}`): 正向加成胜率")

            st.markdown("##### ⚠️ 关键战术风险点")
            for feat, val, score in top_negatives:
                st.warning(f"**{feat.upper()}** (`{val:.1f}`): 拖累当前战术掌控力")

            st.divider()
            report_text = f"# ⚽ 赛前战术简报: {home_team} vs {away_team}\n- 预测胜率: {mc_win_pct:.1f}%\n- 95% 置信区间: {ci_lower:.1f}% ~ {ci_upper:.1f}%\n\n---\n## 💡 临场指挥指令:\n" + "\n".join([f"{i}. {d}" for i, d in enumerate(dynamic_directives, 1)])
            
            st.download_button(
                label="📥 一键下载赛前战术简报 (.md)",
                data=report_text,
                file_name=f"Tactical_Brief_{home_team}_vs_{away_team}.md",
                mime="text/markdown"
            )

    with col_xai:
        st.markdown("#### 🔍 单场战术归因 (XAI High-Contrast)")
        top_xai = sorted(contributions, key=lambda x: abs(x[2]), reverse=True)[:6]
        xai_df = pd.DataFrame({
            'Feature': [x[0] for x in top_xai],
            'Contribution': [x[2] for x in top_xai]
        }).sort_values(by='Contribution')

        fig_xai, ax_xai = plt.subplots(figsize=(6, 3.8), facecolor='#0b0f19')
        ax_xai.set_facecolor('#1e293b')
        colors = ['#00FF87' if v >= 0 else '#FF0055' for v in xai_df['Contribution']]
        
        bars = ax_xai.barh(xai_df['Feature'], xai_df['Contribution'], color=colors, edgecolor='black', linewidth=1)
        ax_xai.set_title("Tactical Drivers Impacting This Match", color='white', fontsize=10, fontweight='bold')
        ax_xai.tick_params(colors='white')
        plt.tight_layout()
        st.pyplot(fig_xai)
