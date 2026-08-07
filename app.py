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
    .metric-card-win { background: linear-gradient(135deg, #064e3b 0%, #022c22 100%); border: 2px solid #00FF87; border-radius: 10px; padding: 14px; text-align: center; box-shadow: 0 0 12px rgba(0, 255, 135, 0.2); }
    .metric-card-draw { background: linear-gradient(135deg, #713f12 0%, #451a03 100%); border: 2px solid #FACC15; border-radius: 10px; padding: 14px; text-align: center; box-shadow: 0 0 12px rgba(250, 204, 21, 0.2); }
    .metric-card-loss { background: linear-gradient(135deg, #881337 0%, #4c0519 100%); border: 2px solid #FF0055; border-radius: 10px; padding: 14px; text-align: center; box-shadow: 0 0 12px rgba(255, 0, 85, 0.2); }
    .metric-title { color: #94a3b8; font-size: 13px; font-weight: 600; text-transform: uppercase; }
    .metric-value-win { color: #00FF87; font-size: 28px; font-weight: 800; }
    .metric-value-draw { color: #FACC15; font-size: 28px; font-weight: 800; }
    .metric-value-loss { color: #FF0055; font-size: 28px; font-weight: 800; }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ Pro AI Tactical Decision Support System")
st.caption("职业足球赛前博弈推演、阵型对位引擎与 7 黄金特征蒙特卡洛引擎 (Head Coach Efficiency Edition)")

# ---------------------------------------------------------
# 2. 真实历史数据集与 v2 版 ML 模型动态加载
# ---------------------------------------------------------
@st.cache_data
def load_datasets():
    df_clean = pd.read_csv('clean_world_cup_2022.csv')
    return df_clean

@st.cache_resource
def load_model():
    return joblib.load('world_cup_rf_model_v2.pkl')

try:
    df_clean = load_datasets()
    model = load_model()
except Exception as e:
    st.sidebar.error(f"❌ 数据或模型加载失败: {e}")
    st.stop()

# 核心 7 黄金特征定义 (与 v2 模型完全对齐)
tactical_features = [
    'xg', 'possession', 'shots_on_target', 
    'ppda', 'tackles_successful', 'interceptions', 'aerial_duels_won_pct'
]

FEATURE_BASELINES = df_clean[tactical_features].mean().to_dict()

FORMATION_TACTICS = {
    "4-3-3": {"style": "控球高压", "color": "#FF0055", "line_x": 68, "label": "🔥 HIGH PRESS LINE"},
    "4-2-3-1": {"style": "常规平衡", "color": "#FACC15", "line_x": 55, "label": "⚖️ BALANCED LINE"},
    "3-5-2": {"style": "中路控制", "color": "#00FF87", "line_x": 50, "label": "🔄 MIDFIELD CONTROL"},
    "4-4-2": {"style": "传统反击", "color": "#38BDF8", "line_x": 40, "label": "⚡ COUNTER LINE"},
    "5-4-1": {"style": "低位防反", "color": "#3b82f6", "line_x": 25, "label": "🛡️ LOW BLOCK LINE"},
    "3-4-3": {"style": "边路强攻", "color": "#a855f7", "line_x": 65, "label": "⚔️ WIDE OVERLOAD"},
    "4-1-4-1": {"style": "中场延误", "color": "#6366f1", "line_x": 45, "label": "🛑 DELAY & BLOCK"}
}

def draw_2d_pitch_enhanced(formation_name, team_name):
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='#0b0f19')
    ax.set_facecolor('#1e293b')

    ax.plot([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], color="white", alpha=0.3, linewidth=1.5)
    ax.plot([50, 50], [0, 100], color="white", alpha=0.3, linewidth=1.5)
    ax.add_patch(patches.Circle((50, 50), 12, color="white", fill=False, alpha=0.3, linewidth=1.5))
    ax.add_patch(patches.Rectangle((0, 20), 18, 60, color="white", fill=False, alpha=0.3, linewidth=1.5))
    ax.add_patch(patches.Rectangle((82, 20), 18, 60, color="white", fill=False, alpha=0.3, linewidth=1.5))

    tactic_info = FORMATION_TACTICS[formation_name]
    ax.axvline(x=tactic_info["line_x"], color=tactic_info["color"], linestyle='--', linewidth=2, alpha=0.8)
    ax.text(tactic_info["line_x"] + 1, 92, tactic_info["label"], color=tactic_info["color"], fontsize=8, fontweight='bold')

    formations_coords = {
        "4-3-3": [(8,50), (28,18), (25,38), (25,62), (28,82), (50,28), (45,50), (50,72), (80,20), (85,50), (80,80)],
        "4-2-3-1": [(8,50), (28,18), (25,38), (25,62), (28,82), (42,35), (42,65), (65,20), (68,50), (65,80), (85,50)],
        "3-5-2": [(8,50), (25,28), (23,50), (25,72), (45,15), (48,35), (45,50), (48,65), (45,85), (82,38), (82,62)],
        "4-4-2": [(8,50), (28,18), (25,38), (25,62), (28,82), (52,18), (50,38), (50,62), (52,82), (82,38), (82,62)],
        "5-4-1": [(8,50), (28,12), (25,31), (23,50), (25,69), (28,88), (50,20), (48,40), (48,60), (50,80), (82,50)],
        "3-4-3": [(8,50), (25,28), (23,50), (25,72), (50,18), (48,38), (48,62), (50,82), (80,20), (85,50), (80,80)],
        "4-1-4-1": [(8,50), (28,18), (25,38), (25,62), (28,82), (40,50), (60,18), (60,38), (60,62), (60,82), (82,50)]
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
    ax.set_title(f"{team_name} [{formation_name} | {tactic_info['style']}]", color='white', fontsize=11, pad=10)
    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# 4. 侧边栏设置
# ---------------------------------------------------------
st.sidebar.header("⚙️ 1. 比赛对位设置 (Matchup)")
all_teams = sorted(df_clean['team'].unique().tolist())

col_h, col_a = st.sidebar.columns(2)
home_team = col_h.selectbox("我方球队", all_teams, index=all_teams.index("Argentina") if "Argentina" in all_teams else 0)
away_team = col_a.selectbox("对手球队", all_teams, index=all_teams.index("France") if "France" in all_teams else 1)

home_data = df_clean[df_clean['team'] == home_team]
away_data = df_clean[df_clean['team'] == away_team]
team_baseline = home_data[tactical_features].mean().to_dict() if not home_data.empty else FEATURE_BASELINES.copy()
opp_baseline = away_data[tactical_features].mean().to_dict() if not away_data.empty else FEATURE_BASELINES.copy()

st.sidebar.markdown("---")
st.sidebar.header("📐 2. 阵型沙盘博弈 (Formations)")
formation_list = list(FORMATION_TACTICS.keys())
home_formation = st.sidebar.selectbox("我方部署阵型 (Our Formation)", formation_list, index=0)
opp_formation = st.sidebar.selectbox("敌方部署阵型 (Opp Formation)", formation_list, index=1)

st.sidebar.markdown("---")
st.sidebar.header("🎯 3. 比赛情景干预 (Scenarios)")
scenario = st.sidebar.radio("比赛所处情景", ["常规开局 (0-0 Balanced)", "落后狂攻 (Press All Out)", "领先后缩 (Parking Bus)"], index=0)

# ---------------------------------------------------------
# 5. 7 黄金维度阵型对弈引擎 & 战术风格修饰器
# ---------------------------------------------------------
def apply_formation_clash_engine(home_base, opp_base, h_form, a_form, scenario):
    mapped = home_base.copy()
    
    xg_diff_factor = (home_base['xg'] - opp_base['xg']) * 0.1
    poss_diff_factor = (home_base['possession'] - opp_base['possession']) * 0.15
    mapped['xg'] = max(0.2, mapped['xg'] + xg_diff_factor)
    mapped['possession'] = np.clip(mapped['possession'] + poss_diff_factor, 25.0, 75.0)

    if h_form == "4-3-3":
        mapped['ppda'] *= 0.75
        mapped['possession'] *= 1.1
        mapped['xg'] *= 1.15
        mapped['shots_on_target'] *= 1.15
    elif h_form == "5-4-1":
        mapped['possession'] *= 0.75
        mapped['ppda'] *= 1.3
        mapped['xg'] *= 0.8
        mapped['shots_on_target'] *= 0.8
    elif h_form == "4-1-4-1":
        mapped['interceptions'] *= 1.25
        mapped['tackles_successful'] *= 1.15
        mapped['possession'] *= 0.95
    elif h_form == "3-4-3":
        mapped['xg'] *= 1.2
        mapped['shots_on_target'] *= 1.2
    elif h_form == "3-5-2":
        mapped['possession'] *= 1.05
        mapped['tackles_successful'] *= 1.1
    elif h_form == "4-4-2":
        mapped['possession'] *= 0.9

    if a_form in ["4-3-3", "3-4-3"]:
        mapped['possession'] -= 5.0
        mapped['ppda'] -= 1.0
    elif a_form in ["5-4-1", "4-1-4-1"]:
        mapped['possession'] += 6.0
    elif a_form in ["3-5-2", "4-2-3-1"]:
        mapped['tackles_successful'] += 2.0

    if scenario == "落后狂攻 (Press All Out)":
        mapped['ppda'] = 5.5
        mapped['tackles_successful'] += 5.0
        mapped['xg'] *= 1.2
        mapped['shots_on_target'] *= 1.2
    elif scenario == "领先后缩 (Parking Bus)":
        mapped['possession'] = 33.0
        mapped['ppda'] = 20.0
        mapped['xg'] *= 0.6
        mapped['shots_on_target'] *= 0.6

    return mapped

# 核心解耦：独立的战术风格修饰函数
def apply_tactical_style(stats_dict, style):
    adj = stats_dict.copy()
    if "疯狗式" in style:
        adj['ppda'] *= 0.65  
        adj['tackles_successful'] *= 1.25
        adj['possession'] *= 1.1
    elif "铁桶阵" in style:
        adj['possession'] *= 0.65
        adj['ppda'] *= 1.5
        adj['interceptions'] *= 1.3
        adj['xg'] *= 0.8
    elif "两翼齐飞" in style:
        adj['aerial_duels_won_pct'] = min(80.0, adj['aerial_duels_won_pct'] * 1.25)
        adj['shots_on_target'] *= 1.1
    return adj

# 先算出底层的阵型对冲数据
mapped_stats_base = apply_formation_clash_engine(team_baseline, opp_baseline, home_formation, opp_formation, scenario)

# ---------------------------------------------------------
# 6. Tab 选项卡主界面排版
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🏟️ 1. 战术沙盘微调 (Tactical Board)",
    "⚖️ 2. A/B 变阵决策矩阵 (Manager Board)",
    "📑 3. 教练战术简报 (Executive Brief)"
])

# =========================================================
# TAB 1: 2D 足球场 + 战术指令下达
# =========================================================
with tab1:
    col_pitch, col_panel = st.columns([1.2, 1.0])

    with col_pitch:
        st.subheader("🏟️ 阵型对弈与防线落位图层")
        fig_pitch = draw_2d_pitch_enhanced(home_formation, home_team)
        st.pyplot(fig_pitch)

        st.markdown("##### 📈 阵型博弈对球队 KPI 的预期影响")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("预期进球 xG", f"{mapped_stats_base['xg']:.2f}", f"{mapped_stats_base['xg'] - team_baseline['xg']:+.2f}")
        k2.metric("控球率 Possession", f"{mapped_stats_base['possession']:.1f}%", f"{mapped_stats_base['possession'] - team_baseline['possession']:+.1f}%")
        k3.metric("逼抢强度 PPDA", f"{mapped_stats_base['ppda']:.1f}", f"{mapped_stats_base['ppda'] - team_baseline['ppda']:+.1f}", delta_color="inverse")
        k4.metric("成功抢断 Tackles", f"{mapped_stats_base['tackles_successful']:.1f}", f"{mapped_stats_base['tackles_successful'] - team_baseline['tackles_successful']:+.1f}")

    with col_panel:
        st.subheader("📋 主帅赛前战术定调 (Manager Directives)")
        st.caption("选择本场核心战略，AI将自动解算为球员的量化执行 KPI。")

        tactical_style = st.radio(
            "本场比赛核心战术倾向",
            [
                "⚖️ 常规平衡 (按原定阵型运转)", 
                "🔥 疯狗式压迫 (Gegenpressing - 极致体能消耗)", 
                "🛡️ 铁桶阵防反 (Low Block & Counter - 放弃球权)", 
                "⚔️ 两翼齐飞轰炸 (Wide & Aerial - 主打边路传中)"
            ],
            index=0
        )

        # 应用战术修饰得到最终 A 计划数据
        adj_stats = apply_tactical_style(mapped_stats_base, tactical_style)

        st.markdown("#### 🎯 球员执行 KPI 目标 (可直接下达更衣室)")
        with st.container(border=True):
            st.success(f"**中前场任务**: 必须将对手的逼抢压迫度 (PPDA) 限制在 **{adj_stats['ppda']:.1f}** 以内。")
            st.info(f"**后防线任务**: 全场需保持高度专注，完成至少 **{int(adj_stats['interceptions'])}** 次拦截。")
            st.warning(f"**整体节奏**: 预期控球率将维持在 **{adj_stats['possession']:.1f}%** 左右，射正需达到 **{int(adj_stats['shots_on_target'])}** 次。")
            st.error(f"**对抗要求**: 争顶胜率必须咬住 **{adj_stats['aerial_duels_won_pct']:.1f}%** 的底线。")

# 生成最终的 7 维向量喂给引擎
input_vector = np.array([[
    adj_stats['xg'], adj_stats['possession'], adj_stats['shots_on_target'], 
    adj_stats['ppda'], adj_stats['tackles_successful'], adj_stats['interceptions'], adj_stats['aerial_duels_won_pct']
]])

# =========================================================
# TAB 2: 赛前 A/B 变阵决策矩阵
# =========================================================
with tab2:
    a_min_bounds = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    a_max_bounds = np.array([10.0, 100.0, 30.0, 50.0, 60.0, 50.0, 100.0])

    N_SIM = 1000
    np.random.seed(42)
    noise = np.random.normal(0, 1, (N_SIM, 7))
    scale = np.array([0.15, 2.5, 0.8, 0.8, 1.2, 1.0, 2.0])
    
    sim_inputs = np.clip(input_vector + noise * scale, a_min=a_min_bounds, a_max=a_max_bounds)
    sim_probs = model.predict_proba(sim_inputs)

    classes = list(model.classes_)
    win_idx = classes.index('Win') if 'Win' in classes else 2
    draw_idx = classes.index('Draw') if 'Draw' in classes else 0
    loss_idx = classes.index('Loss') if 'Loss' in classes else 1

    mc_win_pct = np.mean(sim_probs[:, win_idx]) * 100
    mc_draw_pct = np.mean(sim_probs[:, draw_idx]) * 100
    mc_loss_pct = np.mean(sim_probs[:, loss_idx]) * 100

    win_probs_series = sim_probs[:, win_idx]
    ci_lower = np.percentile(win_probs_series, 2.5) * 100
    ci_upper = np.percentile(win_probs_series, 97.5) * 100

    st.subheader("⚖️ 战术底盘：A/B 变阵红利与 KPI 代价测算")
    st.caption("放弃花哨图表，直击核心利益。对比备选阵型对胜率及球队底层运转指标的真实影响。")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card-win"><div class="metric-title">方案A ({home_formation}) 期望胜率</div><div class="metric-value-win">{mc_win_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card-draw"><div class="metric-title">期望平局</div><div class="metric-value-draw">{mc_draw_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card-loss"><div class="metric-title">期望负率</div><div class="metric-value-loss">{mc_loss_pct:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    with st.container(border=True):
        st.markdown("#### 🔄 启动备选方案 (Plan B)")
        
        col_select, col_advice = st.columns([1, 1.5])
        with col_select:
            alt_formation = st.selectbox(
                "若需变阵，请选择 B 计划落位：",
                [f for f in formation_list if f != home_formation]
            )

        # 完美修补逻辑断层：B计划也必须继承战术修饰器！
        alt_mapped_base = apply_formation_clash_engine(team_baseline, opp_baseline, alt_formation, opp_formation, scenario)
        alt_mapped_styled = apply_tactical_style(alt_mapped_base, tactical_style)
        alt_vector = np.array([[alt_mapped_styled[f] for f in tactical_features]])
        alt_sim_inputs = np.clip(alt_vector + noise * scale, a_min=a_min_bounds, a_max=a_max_bounds)
        
        alt_sim_probs = model.predict_proba(alt_sim_inputs)
        alt_win_pct = np.mean(alt_sim_probs[:, win_idx]) * 100
        diff_win = alt_win_pct - mc_win_pct

        with col_advice:
            if diff_win > 3.0:
                st.success(f"💡 **教练组建议**：改打 **{alt_formation}** 预期胜率将暴涨 **+{diff_win:.1f}%**！存在极佳战术克制红利，强烈推荐。")
            elif diff_win < -2.0:
                st.error(f"⚠️ **高危警告**：改打 **{alt_formation}** 预期胜率将跌至 **{alt_win_pct:.1f}%**。被对手严重限制，严禁盲目尝试。")
            else:
                st.info(f"⚖️ **战术评估**：改打 **{alt_formation}** 胜率变化为 **{diff_win:+.1f}%**。收益不明显，建议优先临场微调。")

        st.markdown("##### 📊 变阵付出的战术代价与收益 (KPI Delta)")
        
        k1, k2, k3, k4 = st.columns(4)
        diff_xg = alt_mapped_styled['xg'] - adj_stats['xg']
        diff_poss = alt_mapped_styled['possession'] - adj_stats['possession']
        diff_ppda = alt_mapped_styled['ppda'] - adj_stats['ppda']
        diff_tackles = alt_mapped_styled['tackles_successful'] - adj_stats['tackles_successful']
        
        k1.metric("预期进球 xG", f"{alt_mapped_styled['xg']:.2f}", f"{diff_xg:+.2f}")
        k2.metric("控球率 %", f"{alt_mapped_styled['possession']:.1f}%", f"{diff_poss:+.1f}%")
        k3.metric("前场压迫 PPDA", f"{alt_mapped_styled['ppda']:.1f}", f"{diff_ppda:+.1f}", delta_color="inverse")
        k4.metric("成功抢断", f"{alt_mapped_styled['tackles_successful']:.1f}", f"{diff_tackles:+.1f}")

# =========================================================
# TAB 3: 带有实战指令翻译的教练简报
# =========================================================
with tab3:
    st.subheader("📑 赛前主帅执行单 (Executive Brief)")
    st.caption("AI提炼核心优劣势，并翻译为可直接向球员下达的战术指令。")

    # 战术指令字典：将冷冰冰的数据指标翻译成球员听得懂的“人话”
    TACTICAL_ADVICE = {
        'xg': "创造空当打透防线，提高禁区内射门转化率。",
        'possession': "稳住球权节奏，通过快速传导消耗对手体能。",
        'shots_on_target': "加强禁区前沿的远射尝试与门前二次抢点。",
        'ppda': "全军压上！丢球后必须在3秒内就地合围反抢。",
        'tackles_successful': "提升中场绞杀硬度，遇险果断下脚破坏。",
        'interceptions': "保持防线紧凑，提前预判并切断对手核心传球路线。",
        'aerial_duels_won_pct': "控制第一落点！边路起球果断找高中锋头顶。"
    }

    rf_importances = model.feature_importances_
    current_vals = input_vector[0]

    contributions = []
    for i, feat in enumerate(tactical_features):
        base_val = FEATURE_BASELINES[feat]
        curr_val = current_vals[i]
        imp = rf_importances[i]

        if feat == 'ppda':
            diff = base_val - curr_val
        else:
            diff = curr_val - base_val

        score = diff * imp
        contributions.append((feat, curr_val, score))

    contributions.sort(key=lambda x: x[2], reverse=True)
    top_positives = [c for c in contributions if c[2] > 0][:3]
    top_negatives = [c for c in contributions if c[2] < 0][-3:]

    with st.container(border=True):
        st.markdown(f"### 🏟️ 赛前定调: {home_team} vs {away_team}")
        st.caption(f"**部署阵型**：{home_formation} | **战术打法**：{tactical_style.split(' ')[1]} | **基础胜率**: {mc_win_pct:.1f}%")
        
        col_pos, col_neg = st.columns(2)
        
        with col_pos:
            st.markdown("##### ✅ 战术红利点 (主攻方向)")
            if top_positives:
                for feat, val, score in top_positives:
                    advice = TACTICAL_ADVICE.get(feat, "")
                    st.success(f"**{feat.upper()}** (预估: `{val:.1f}`)\n\n*指令：{advice}*")
            else:
                st.caption("暂无明显数据优势。")

        with col_neg:
            st.markdown("##### ⚠️ 致命阿喀琉斯之踵 (严防死守)")
            if top_negatives:
                for feat, val, score in top_negatives:
                    advice = TACTICAL_ADVICE.get(feat, "")
                    st.error(f"**{feat.upper()}** (预估: `{val:.1f}`)\n\n*警告：此环节易崩盘，需针对性保护协防！*")
            else:
                st.caption("风险受控。")

        st.divider()
        
        report_text = f"""# ⚽ 赛前战术执行单: {home_team} vs {away_team}
- **首发阵型**: {home_formation} ({tactical_style.split(' ')[1]})
- **预期胜率**: {mc_win_pct:.1f}% (置信区间 {ci_lower:.1f}% ~ {ci_upper:.1f}%)

---
### ⚔️ 进攻端主攻指令:
"""
        report_text += "\n".join([f"- 发挥 **{f.upper()}** 优势。要求：{TACTICAL_ADVICE.get(f, '')}" for f, v, s in top_positives]) + "\n\n"
        report_text += "### 🛡️ 防守端避险指令:\n"
        report_text += "\n".join([f"- 警惕 **{f.upper()}** 被打穿！要求：加强协防，掩盖此项短板。" for f, v, s in top_negatives])

        st.download_button(
            label="📥 导出 Markdown 战术执行单 (交由队长传达)",
            data=report_text,
            file_name=f"Tactical_Sheet_{home_team}_vs_{away_team}.md",
            mime="text/markdown"
        )
            data=report_text,
            file_name=f"Tactical_Sheet_{home_team}_vs_{away_team}.md",
            mime="text/markdown"
        )
