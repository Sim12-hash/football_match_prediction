import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------
# 1. Page Configuration & High-Contrast Dark Theme CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pro AI Football Tactical Engine",
    page_icon="⚽",
    layout="wide"
)

# Customizing the UI and applying the "Ultimate Sticky Tabs" CSS hack
st.markdown("""
    <style>
    .main { background-color: #0b0f19; }
    
    /* Ultimate Sticky Tabs Fix for Streamlit BaseWeb container */
    div[data-testid="stTabs"] [data-baseweb="tab-list"] {
        position: -webkit-sticky !important;
        position: sticky !important;
        top: 2.875rem !important; 
        z-index: 9999 !important;
        background-color: #0b0f19 !important; 
        padding-top: 15px !important;
        padding-bottom: 10px !important;
        border-bottom: 2px solid #1e293b !important;
    }
    
    .metric-card-win { background: linear-gradient(135deg, #064e3b 0%, #022c22 100%); border: 2px solid #00FF87; border-radius: 10px; padding: 14px; text-align: center; box-shadow: 0 0 12px rgba(0, 255, 135, 0.2); }
    .metric-card-draw { background: linear-gradient(135deg, #713f12 0%, #451a03 100%); border: 2px solid #FACC15; border-radius: 10px; padding: 14px; text-align: center; box data-shadow: 0 0 12px rgba(250, 204, 21, 0.2); }
    .metric-card-loss { background: linear-gradient(135deg, #881337 0%, #4c0519 100%); border: 2px solid #FF0055; border-radius: 10px; padding: 14px; text-align: center; box-shadow: 0 0 12px rgba(255, 0, 85, 0.2); }
    .metric-title { color: #94a3b8; font-size: 13px; font-weight: 600; text-transform: uppercase; }
    .metric-value-win { color: #00FF87; font-size: 28px; font-weight: 800; }
    .metric-value-draw { color: #FACC15; font-size: 28px; font-weight: 800; }
    .metric-value-loss { color: #FF0055; font-size: 28px; font-weight: 800; }
    </style>
""", unsafe_allow_html=True)

st.title("⚽ Pro AI Tactical Decision Support System")
st.caption("Pre-match Tactical Simulator, Formation Matchup & Monte Carlo Engine (Head Coach Edition)")

# ---------------------------------------------------------
# 2. Data Ingestion & Model Loading
# ---------------------------------------------------------
@st.cache_data
def load_datasets():
    return pd.read_csv('data.csv')

@st.cache_resource
def load_model():
    return joblib.load('world_cup_rf_model_v2.pkl')

try:
    df_raw = load_datasets()
    model = load_model()
except Exception as e:
    st.sidebar.error(f"❌ Initialization Failed: {e}")
    st.stop()

tactical_features = [
    'xg', 'possession', 'shots_on_target', 
    'ppda', 'tackles_successful', 'interceptions', 'aerial_duels_won_pct'
]

def calculate_global_baselines(df):
    baselines = {}
    home_pass_acc = df['home_completed_passes'].sum() / df['home_attempted_pases'].sum() * 100
    away_pass_acc = df['away_completed_passes'].sum() / df['away_attempted_pases'].sum() * 100
    baselines['pass_accuracy'] = (home_pass_acc + away_pass_acc) / 2
    
    h_ppda = df['away_completed_passes'].sum() / max((df['home_tackles'] + df['home_interceptions']).sum(), 1)
    a_ppda = df['home_completed_passes'].sum() / max((df['away_tackles'] + df['away_interceptions']).sum(), 1)
    baselines['ppda'] = (h_ppda + a_ppda) / 2
    
    baselines['xg'] = (df['home_xg'].mean() + df['away_xg'].mean()) / 2
    baselines['possession'] = 50.0 
    baselines['shots_on_target'] = (df['home_sot'].mean() + df['away_sot'].mean()) / 2
    baselines['tackles_successful'] = (df['home_tackles'].mean() + df['away_tackles'].mean()) / 2
    baselines['interceptions'] = (df['home_interceptions'].mean() + df['away_interceptions'].mean()) / 2
    baselines['aerial_duels_won_pct'] = 50.0 
    return baselines

FEATURE_BASELINES = calculate_global_baselines(df_raw)

# ---------------------------------------------------------
# 3. Tactical Dictionaries
# ---------------------------------------------------------
FORMATION_TACTICS = {
    "4-3-3": {"style": "High Press", "color": "#FF0055", "line_x": 68, "label": "🔥 HIGH PRESS LINE"},
    "4-2-3-1": {"style": "Balanced", "color": "#FACC15", "line_x": 55, "label": "⚖️ BALANCED LINE"},
    "3-5-2": {"style": "Midfield Control", "color": "#00FF87", "line_x": 50, "label": "🔄 MIDFIELD CONTROL"},
    "4-4-2": {"style": "Counter Attack", "color": "#38BDF8", "line_x": 40, "label": "⚡ COUNTER LINE"},
    "5-4-1": {"style": "Low Block", "color": "#3b82f6", "line_x": 25, "label": "🛡️ LOW BLOCK LINE"},
    "3-4-3": {"style": "Wide Overload", "color": "#a855f7", "line_x": 65, "label": "⚔️ WIDE OVERLOAD"},
    "4-1-4-1": {"style": "Delay & Block", "color": "#6366f1", "line_x": 45, "label": "🛑 DELAY & BLOCK"}
}

FORMATION_PHILOSOPHIES = {
    "4-3-3": ["High Block Possession (Default)", "Gegenpressing (High Intensity)", "Wide Overload & Crossing"],
    "4-1-4-1": ["Midfield Chokehold (Default)", "Mid-Block Press"],
    "4-2-3-1": ["Balanced Double Pivot (Default)", "Fast Counter-Attack", "Playmaker Central Penetration"],
    "3-4-3": ["All-Out Attack (Default)", "High Press Man-to-Man"],
    "3-5-2": ["Balanced Attack/Defense (Default)", "Twin Striker Aerial Target"],
    "4-4-2": ["Fast Counter-Attack (Default)", "Full-Pitch High Press"],
    "5-4-1": ["Park the Bus (Default)", "Long Ball to Target Man"]
}

# ---------------------------------------------------------
# 4. Sidebar Configuration (User Inputs)
# ---------------------------------------------------------
st.sidebar.header("⚙️ 1. Matchup Configuration")

all_teams = sorted(set(df_raw['home_team'].unique().tolist() + df_raw['away_team'].unique().tolist()))

col_h, col_a = st.sidebar.columns(2)
home_team = col_h.selectbox("Our Team", all_teams, index=all_teams.index("Argentina") if "Argentina" in all_teams else 0)
away_team = col_a.selectbox("Opponent Team", all_teams, index=all_teams.index("France") if "France" in all_teams else 1)

home_data = df_raw[(df_raw['home_team'] == home_team) | (df_raw['away_team'] == home_team)]
away_data = df_raw[(df_raw['home_team'] == away_team) | (df_raw['away_team'] == away_team)]

def get_team_baseline(team_df, team_name, global_base):
    if team_df.empty: return global_base.copy()
    
    total_matches = len(team_df)
    t_xg = 0; t_poss = 0; t_sot = 0; t_tackles = 0; t_inter = 0
    t_ppda_num = 0; t_ppda_den = 0
    
    for _, row in team_df.iterrows():
        if row['home_team'] == team_name:
            t_xg += row['home_xg']; t_poss += row['home_possession']; t_sot += row['home_sot']
            t_tackles += row['home_tackles']; t_inter += row['home_interceptions']
            t_ppda_num += row['away_completed_passes']; t_ppda_den += (row['home_tackles'] + row['home_interceptions'])
        else:
            t_xg += row['away_xg']; t_poss += row['away_possession']; t_sot += row['away_sot']
            t_tackles += row['away_tackles']; t_inter += row['away_interceptions']
            t_ppda_num += row['home_completed_passes']; t_ppda_den += (row['away_tackles'] + row['away_interceptions'])
            
    base = global_base.copy()
    base['xg'] = t_xg / total_matches
    base['possession'] = t_poss / total_matches
    base['shots_on_target'] = t_sot / total_matches
    base['tackles_successful'] = t_tackles / total_matches
    base['interceptions'] = t_inter / total_matches
    base['ppda'] = t_ppda_num / max(t_ppda_den, 1)
    return base

team_baseline = get_team_baseline(home_data, home_team, FEATURE_BASELINES)
opp_baseline = get_team_baseline(away_data, away_team, FEATURE_BASELINES)

st.sidebar.markdown("---")
st.sidebar.header("📐 2. Formation")
formation_list = list(FORMATION_TACTICS.keys())
home_formation = st.sidebar.selectbox("Our Formation", formation_list, index=0)
opp_formation = st.sidebar.selectbox("Opponent Formation", formation_list, index=1)

st.sidebar.markdown("---")
st.sidebar.header("🎯 3. Match Scenarios")
scenario = st.sidebar.radio("Current Game State", ["Balanced Start (0-0)", "Trailing - Press All Out", "Leading - Park the Bus"], index=0)

# ---------------------------------------------------------
# 5. Tactical Engine & Style Modifiers
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

    if scenario == "Trailing - Press All Out":
        mapped['ppda'] = 5.5
        mapped['tackles_successful'] += 5.0
        mapped['xg'] *= 1.2
        mapped['shots_on_target'] *= 1.2
    elif scenario == "Leading - Park the Bus":
        mapped['possession'] = 33.0
        mapped['ppda'] = 20.0
        mapped['xg'] *= 0.6
        mapped['shots_on_target'] *= 0.6

    return mapped

def apply_tactical_style(stats_dict, style):
    adj = stats_dict.copy()
    
    if any(k in style for k in ["Gegenpressing", "High Press", "All-Out Attack"]):
        adj['ppda'] *= 0.65  
        adj['tackles_successful'] *= 1.25
        adj['possession'] *= 1.1
        if "All-Out Attack" in style:
            adj['xg'] *= 1.3
            adj['shots_on_target'] *= 1.3
            
    elif any(k in style for k in ["Park the Bus", "Counter-Attack", "Long Ball"]):
        adj['possession'] *= 0.65 
        adj['xg'] *= 1.1 
        adj['shots_on_target'] *= 1.15
        if "Park the Bus" in style:
            adj['ppda'] *= 1.5
            adj['interceptions'] *= 1.3
            adj['xg'] *= 0.8 
        if "Long Ball" in style:
            adj['aerial_duels_won_pct'] = min(85.0, adj['aerial_duels_won_pct'] * 1.3)
            
    elif any(k in style for k in ["Chokehold", "Mid-Block", "High Block Possession", "Playmaker"]):
        adj['possession'] *= 1.15
        adj['interceptions'] *= 1.2
        if "Chokehold" in style or "Mid-Block" in style:
            adj['tackles_successful'] *= 1.15
            adj['possession'] *= 0.9 
            
    elif any(k in style for k in ["Wide Overload", "Twin Striker"]):
        adj['aerial_duels_won_pct'] = min(80.0, adj['aerial_duels_won_pct'] * 1.25)
        adj['shots_on_target'] *= 1.15
        adj['xg'] *= 1.1
        
    return adj

mapped_stats_base = apply_formation_clash_engine(team_baseline, opp_baseline, home_formation, opp_formation, scenario)

# ---------------------------------------------------------
# 6. 2D Pitch Rendering Engine
# ---------------------------------------------------------
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
# 7. Main Dashboard layout (Tabs)
# ---------------------------------------------------------
tab1, tab2, tab3 = st.tabs([
    "🏟️ 1. Tactical Board",
    "⚖️ 2. Manager's A/B Matrix",
    "📑 3. Executive Brief"
])

# =========================================================
# TAB 1: 2D Pitch & Directives
# =========================================================
with tab1:
    st.subheader("📋 Matchup & Tactical Execution Board")
    
    # Philosophy selector restored to Tab 1
    available_philosophies = FORMATION_PHILOSOPHIES.get(home_formation, ["Standard (Balanced setup)"])
    tactical_style = st.radio(
        f"Core Philosophy for {home_formation}",
        available_philosophies,
        index=0,
        horizontal=True
    )
    
    # Calculate final stats using the selected tactical style
    adj_stats = apply_tactical_style(mapped_stats_base, tactical_style)
    
    # Prepare final feature vector for the ML model
    input_vector = np.array([[
        adj_stats['xg'], adj_stats['possession'], adj_stats['shots_on_target'], 
        adj_stats['ppda'], adj_stats['tackles_successful'], adj_stats['interceptions'], adj_stats['aerial_duels_won_pct']
    ]])

    st.divider()

    col_pitch, col_panel = st.columns([1.2, 1.0])

    with col_pitch:
        st.markdown("##### 🏟️ Formation Clash & Defensive Line")
        fig_pitch = draw_2d_pitch_enhanced(home_formation, home_team)
        st.pyplot(fig_pitch)

        st.markdown("##### 📈 Dynamic Impact on Team KPIs")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Expected Goals (xG)", f"{adj_stats['xg']:.2f}", f"{adj_stats['xg'] - team_baseline['xg']:+.2f}")
        k2.metric("Possession", f"{adj_stats['possession']:.1f}%", f"{adj_stats['possession'] - team_baseline['possession']:+.1f}%")
        k3.metric("Pressing (PPDA)", f"{adj_stats['ppda']:.1f}", f"{adj_stats['ppda'] - team_baseline['ppda']:+.1f}", delta_color="inverse")
        k4.metric("Tackles Won", f"{adj_stats['tackles_successful']:.1f}", f"{adj_stats['tackles_successful'] - team_baseline['tackles_successful']:+.1f}")

    with col_panel:
        st.markdown("#### 🎯 Player Execution KPIs (Locker Room Directives)")
        with st.container(border=True):
            st.success(f"**Midfield Task**: Restrict the opponent's build-up. Keep our PPDA strictly under **{adj_stats['ppda']:.1f}**.")
            st.info(f"**Defensive Task**: Maintain absolute positional discipline. We need at least **{int(adj_stats['interceptions'])}** clean interceptions.")
            st.warning(f"**Tempo Control**: Expect to hold approximately **{adj_stats['possession']:.1f}%** possession. Offensive units must secure **{int(adj_stats['shots_on_target'])}** shots on target.")
            st.error(f"**Physicality**: Aerial duels are non-negotiable today. Win rate must stay above **{adj_stats['aerial_duels_won_pct']:.1f}%**.")

# =========================================================
# TAB 2: A/B Formation Decision Matrix
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

    st.subheader("⚖️ Tactical Core: A/B Formation ROI & KPI Cost")
    st.caption("Evaluate the true probability impacts of your tactical setup. Pure, data-driven insights.")

    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown(f'<div class="metric-card-win"><div class="metric-title">Plan A ({home_formation}) Win Prob</div><div class="metric-value-win">{mc_win_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with m2:
        st.markdown(f'<div class="metric-card-draw"><div class="metric-title">Draw Prob</div><div class="metric-value-draw">{mc_draw_pct:.1f}%</div></div>', unsafe_allow_html=True)
    with m3:
        st.markdown(f'<div class="metric-card-loss"><div class="metric-title">Loss Prob</div><div class="metric-value-loss">{mc_loss_pct:.1f}%</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.divider()

    with st.container(border=True):
        st.markdown("#### 🔄 Initiate Plan B")
        
        col_select, col_advice = st.columns([1, 1.5])
        with col_select:
            alt_formation = st.selectbox(
                "Select alternative formation to evaluate:",
                [f for f in formation_list if f != home_formation]
            )

        alt_default_style = FORMATION_PHILOSOPHIES.get(alt_formation, ["Standard"])[0]
        alt_mapped_base = apply_formation_clash_engine(team_baseline, opp_baseline, alt_formation, opp_formation, scenario)
        alt_mapped_styled = apply_tactical_style(alt_mapped_base, alt_default_style)
        
        alt_vector = np.array([[alt_mapped_styled[f] for f in tactical_features]])
        alt_sim_inputs = np.clip(alt_vector + noise * scale, a_min=a_min_bounds, a_max=a_max_bounds)
        
        alt_sim_probs = model.predict_proba(alt_sim_inputs)
        alt_win_pct = np.mean(alt_sim_probs[:, win_idx]) * 100
        diff_win = alt_win_pct - mc_win_pct

        with col_advice:
            if diff_win > 3.0:
                st.success(f"💡 **Coaching Staff Advice**: Switching to **{alt_formation}** boosts expected win rate by **+{diff_win:.1f}%**! Highly recommended due to a strong tactical mismatch.")
            elif diff_win < -2.0:
                st.error(f"⚠️ **High-Risk Warning**: Switching to **{alt_formation}** drops win rate to **{alt_win_pct:.1f}%**. The opponent counters this setup heavily. Avoid.")
            else:
                st.info(f"⚖️ **Tactical Assessment**: Changing to **{alt_formation}** yields a **{diff_win:+.1f}%** shift. Marginal impact; rely on player execution and in-game tweaks.")

        st.markdown("##### 📊 Tactical Cost & Benefit of Formation Change (KPI Delta)")
        
        k1, k2, k3, k4 = st.columns(4)
        diff_xg = alt_mapped_styled['xg'] - adj_stats['xg']
        diff_poss = alt_mapped_styled['possession'] - adj_stats['possession']
        diff_ppda = alt_mapped_styled['ppda'] - adj_stats['ppda']
        diff_tackles = alt_mapped_styled['tackles_successful'] - adj_stats['tackles_successful']
        
        k1.metric("Expected Goals (xG)", f"{alt_mapped_styled['xg']:.2f}", f"{diff_xg:+.2f}")
        k2.metric("Possession %", f"{alt_mapped_styled['possession']:.1f}%", f"{diff_poss:+.1f}%")
        k3.metric("Pressing (PPDA)", f"{alt_mapped_styled['ppda']:.1f}", f"{diff_ppda:+.1f}", delta_color="inverse")
        k4.metric("Tackles Won", f"{alt_mapped_styled['tackles_successful']:.1f}", f"{diff_tackles:+.1f}")

# =========================================================
# TAB 3: Data-Driven Narrative Engine (Executive Brief)
# =========================================================
with tab3:
    st.subheader("📑 Pre-Match Executive Brief")
    st.caption("Dynamically generated tactical report based on Random Forest feature weights and real-time numerical deviations.")

    def get_dynamic_advice(feat, curr_val, base_val, is_advantage, style):
        """Translates raw mathematical deviations into human-readable coaching intelligence."""
        diff = curr_val - base_val
        abs_diff = abs(diff)
        direction = "higher" if diff > 0 else "lower"
        
        if feat == 'ppda':
            press_intensity = "Highly Aggressive (High Press)" if curr_val < 11 else "Conservative (Low Block)"
            return f"Current PPDA is tracking at {curr_val:.1f} (which is {abs_diff:.1f} {direction} than baseline). Reflecting our '{style.split(' ')[0]}' approach, the defensive unit is operating in a **{press_intensity}** state. Watch out for spaces left behind the backline."
            
        elif feat == 'possession':
            control_type = "Absolute ball control" if curr_val > 50 else "Conceding possession to hit on the counter"
            return f"Expected possession is {curr_val:.1f}% (fluctuating {diff:+.1f}% from norm). This indicates **{control_type}**. Midfielders must prioritize pass completion in the middle third."
            
        elif feat == 'xg':
            threat_level = "Lethal, capable of consistent scoring" if curr_val > 1.3 else "Sub-optimal, requiring efficient chance creation"
            return f"Expected Goals (xG) evaluated at {curr_val:.2f} (shifting {diff:+.2f} from historical average). Offensive threat is currently **{threat_level}**. Conversion in the final third will dictate the outcome."
            
        elif feat == 'shots_on_target':
            status = "plentiful" if curr_val >= base_val else "lacking"
            return f"Projected shots on target: {int(curr_val)}. Offensive output is **{status}**. Attackers are instructed to increase shots from outside the box and hunt for rebounds."
            
        elif feat == 'tackles_successful':
            status = "sufficient" if curr_val >= base_val else "vulnerable"
            return f"Successful tackles estimated at {int(curr_val)}. Midfield grit and defensive solidity is **{status}**. Crucial for disrupting the opponent's primary playmakers."
            
        elif feat == 'interceptions':
            status = "Excellent" if curr_val >= base_val else "Sub-par"
            return f"Interceptions projected at {int(curr_val)}. Defensive anticipation is **{status}**. The backline must actively cut off passing lanes and through-balls."
            
        elif feat == 'aerial_duels_won_pct':
            air_status = "Dominant in the air" if curr_val >= 50 else "Struggling with aerial battles"
            return f"Aerial win rate anticipated at {curr_val:.1f}%. Assessment: **{air_status}**. This directly impacts our vulnerability to set-pieces and wide crosses."
            
        return f"Metric value is {curr_val:.1f} (deviation {diff:+.1f} from norm). Requires specific in-game monitoring."

    rf_importances = model.feature_importances_
    current_vals = input_vector[0]

    contributions = []
    for i, feat in enumerate(tactical_features):
        base_val = team_baseline[feat] 
        curr_val = current_vals[i]
        imp = rf_importances[i]

        diff = (base_val - curr_val) if feat == 'ppda' else (curr_val - base_val)
        score = diff * imp
        contributions.append((feat, curr_val, score))

    contributions.sort(key=lambda x: x[2], reverse=True)
    top_positives = [c for c in contributions if c[2] > 0][:3]
    top_negatives = [c for c in contributions if c[2] < 0][-3:]

    with st.container(border=True):
        st.markdown(f"### 🏟️ Match Preview: {home_team} vs {away_team}")
        st.caption(f"**Formation**: {home_formation} | **Style**: {tactical_style.split(' ')[0]} | **Base Win Prob**: {mc_win_pct:.1f}%")
        
        col_pos, col_neg = st.columns(2)
        
        with col_pos:
            st.markdown("##### ✅ Tactical Advantages (Primary Attack Vectors)")
            if top_positives:
                for feat, val, score in top_positives:
                    base_val = team_baseline[feat] 
                    advice = get_dynamic_advice(feat, val, base_val, True, tactical_style)
                    st.success(f"**{feat.upper()}** (Value: `{val:.1f}`)\n\n*Analysis: {advice}*")
            else:
                st.caption("No distinct statistical advantages found.")

        with col_neg:
            st.markdown("##### ⚠️ Achilles' Heel (Areas to Defend)")
            if top_negatives:
                for feat, val, score in top_negatives:
                    base_val = team_baseline[feat] 
                    advice = get_dynamic_advice(feat, val, base_val, False, tactical_style)
                    st.error(f"**{feat.upper()}** (Value: `{val:.1f}`)\n\n*Warning: {advice}*")
            else:
                st.caption("Risks are currently managed.")

        st.divider()
        
        report_text = f"""# ⚽ Pre-Match Tactical Sheet: {home_team} vs {away_team}
- **Starting Formation**: {home_formation} ({tactical_style.split(' ')[0]})
- **Expected Win Probability**: {mc_win_pct:.1f}% (Confidence Interval: {ci_lower:.1f}% ~ {ci_upper:.1f}%)

---
### ⚔️ Offensive Directives (Exploit Advantages):
"""
        report_text += "\n".join([f"- Leverage our **{f.upper()}** dominance (Current: {v:.1f}). Execution: {get_dynamic_advice(f, v, team_baseline[f], True, tactical_style)}" for f, v, s in top_positives]) + "\n\n"
        report_text += "### 🛡️ Defensive Directives (Mitigate Risks):\n"
        report_text += "\n".join([f"- Guard against **{f.upper()}** vulnerabilities (Current: {v:.1f}). Execution: {get_dynamic_advice(f, v, team_baseline[f], False, tactical_style)}" for f, v, s in top_negatives])

        st.download_button(
            label="📥 Export Markdown Tactical Sheet (For the Captain)",
            data=report_text,
            file_name=f"Tactical_Sheet_{home_team}_vs_{away_team}.md",
            mime="text/markdown"
        )
