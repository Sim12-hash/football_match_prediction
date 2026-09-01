import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# ---------------------------------------------------------
# 1. Page Configuration & Clean Dark Theme CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Pro AI Football Tactical Engine",
    page_icon="⚽",
    layout="wide"
)

# We removed all the complex CSS tab hacks. 
# Keeping only the clean UI styling for our metric cards.
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
st.caption("Pre-match Tactical Simulator, Formation Matchup & Monte Carlo Engine (Head Coach Edition)")

# ---------------------------------------------------------
# 2. Data Ingestion & Model Loading
# ---------------------------------------------------------
@st.cache_data
def load_datasets():
    return pd.read_csv('data.csv')

@st.cache_resource
def load_model():
    return joblib.load(
        'world_cup_rf_model_v2.pkl'
    )


@st.cache_resource
def load_comparison_models():
    return joblib.load(
        'comparison_models.pkl'
    )


try:
    # Dataset
    df_raw = load_datasets()

    # Main selected model
    model = load_model()

    # Other two models + metrics
    comparison_bundle = load_comparison_models()

    mlp_model = comparison_bundle[
        "mlp_model"
    ]

    xgb_model = comparison_bundle[
        "xgb_model"
    ]

    xgb_classes = comparison_bundle[
        "xgb_classes"
    ]

    comparison_metrics = pd.DataFrame(
        comparison_bundle[
            "cv_metrics"
        ]
    )

except Exception as e:

    st.sidebar.error(
        f"❌ Initialization Failed: {e}"
    )

    st.stop()

tactical_features = [
    'xg', 'possession', 'shots_on_target', 
    'ppda', 'tackles_successful', 'interceptions', 'aerial_duels_won_pct'
]

FEATURE_LABELS = {
    "xg": "Expected Goals (xG)",
    "possession": "Pass Share % (Possession-Control Proxy)",
    "shots_on_target": "Threatening Shot Count",
    "ppda": "PPDA-Style Pressing Proxy",
    "tackles_successful": "Defensive Duel Count",
    "interceptions": "Interception Count",
    "aerial_duels_won_pct": "50/50 Duel Share %"
}
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
# 3. Tactical Configuration Engine
# ---------------------------------------------------------
# Maps team formations and tactical philosophies to specific pitch 
# coordinates (X-axis depth: 0-100) and line markers for the 2D UI rendering.
TACTICAL_CONFIG = {
    "4-3-3": {
        "High Block Possession (Default)": {"color": "#FF0055", "line_x": 75, "label": " HIGH PRESS LINE"},
        "Gegenpressing (High Intensity)": {"color": "#FF0055", "line_x": 88, "label": " GEGENPRESS LINE"},
        "Wide Overload & Crossing": {"color": "#a855f7", "line_x": 65, "label": " OVERLOAD LINE"}
    },
    "4-2-3-1": {
        "Balanced Double Pivot (Default)": {"color": "#FACC15", "line_x": 55, "label": " BALANCED LINE"},
        "Fast Counter-Attack": {"color": "#38BDF8", "line_x": 35, "label": " RETREAT LINE"},
        "Playmaker Central Penetration": {"color": "#00FF87", "line_x": 75, "label": " PENETRATION LINE"}
    },
    "3-5-2": {
        "Balanced Attack/Defense (Default)": {"color": "#00FF87", "line_x": 58, "label": " MIDFIELD CONTROL LINE"},
        "Twin Striker Aerial Target": {"color": "#f97316", "line_x": 80, "label": " TARGET ZONE LINE"}
    },
    "4-4-2": {
        "Fast Counter-Attack (Default)": {"color": "#38BDF8", "line_x": 45, "label": " COUNTER LINE"},
        "Full-Pitch High Press": {"color": "#FF0055", "line_x": 82, "label": " HIGH PRESS LINE"}
    },
    "5-4-1": {
        "Park the Bus (Default)": {"color": "#3b82f6", "line_x": 28, "label": " LOW BLOCK LINE"},
        "Long Ball to Target Man": {"color": "#eab308", "line_x": 68, "label": " LONG BALL OUTLET LINE"}
    },
    "3-4-3": {
        "All-Out Attack (Default)": {"color": "#FF0055", "line_x": 78, "label": " WIDE OVERLOAD LINE"},
        "High Press Man-to-Man": {"color": "#ef4444", "line_x": 88, "label": " MAN-TO-MAN PRESS LINE"}
    },
    "4-1-4-1": {
        "Midfield Chokehold (Default)": {"color": "#6366f1", "line_x": 58, "label": " DELAY & BLOCK LINE"},
        "Mid-Block Press": {"color": "#f59e0b", "line_x": 68, "label": " MID-BLOCK PRESS LINE"}
    }
}

# Dynamically extract philosophy options to populate the sidebar UI
FORMATION_PHILOSOPHIES = {k: list(v.keys()) for k, v in TACTICAL_CONFIG.items()}

# ---------------------------------------------------------
# 4. SIDEBAR CONFIGURATION (Optimized UX Flow)
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
    t_xg = 0
    t_poss = 0
    t_sot = 0
    t_tackles = 0
    t_inter = 0
    t_5050_share = 0
    t_ppda_num = 0; t_ppda_den = 0
    
    for _, row in team_df.iterrows():
        if row['home_team'] == team_name:
            duel_total = max(
                row['home_aerials_won'] + row['away_aerials_won'],
                1
            )

            t_5050_share += (
                row['home_aerials_won'] / duel_total
            ) * 100
            t_xg += row['home_xg']; t_poss += row['home_possession']; t_sot += row['home_sot']
            t_tackles += row['home_tackles']; t_inter += row['home_interceptions']
            t_ppda_num += row['away_completed_passes']; t_ppda_den += (row['home_tackles'] + row['home_interceptions'])
        else:
            duel_total = max(
                row['home_aerials_won'] + row['away_aerials_won'],
                1
            )

            t_5050_share += (
                row['home_aerials_won'] / duel_total
            ) * 100
            t_xg += row['away_xg']; t_poss += row['away_possession']; t_sot += row['away_sot']
            t_tackles += row['away_tackles']; t_inter += row['away_interceptions']
            t_ppda_num += row['home_completed_passes']; t_ppda_den += (row['away_tackles'] + row['away_interceptions'])
            
    base = global_base.copy()
    base['xg'] = t_xg / total_matches
    base['possession'] = t_poss / total_matches
    base['shots_on_target'] = t_sot / total_matches
    base['tackles_successful'] = t_tackles / total_matches
    base['interceptions'] = t_inter / total_matches
    base['aerial_duels_won_pct'] = (
        t_5050_share / total_matches
    )
    base['ppda'] = t_ppda_num / max(t_ppda_den, 1)
    return base

team_baseline = get_team_baseline(home_data, home_team, FEATURE_BASELINES)
opp_baseline = get_team_baseline(away_data, away_team, FEATURE_BASELINES)

st.sidebar.markdown("---")
st.sidebar.header("📐 2. Formation")
formation_list = list(TACTICAL_CONFIG.keys())
home_formation = st.sidebar.selectbox("Our Formation", formation_list, index=0)
opp_formation = st.sidebar.selectbox("Opponent Formation", formation_list, index=1)

# ✨ THE UX FIX: Navigation dynamically rests at the bottom as the final action step
st.sidebar.markdown("---")
st.sidebar.header("🧭 3. Dashboard View")
app_mode = st.sidebar.radio(
    "Select output panel:",
    ["🏟️ 1. Tactical Board", "⚖️ 2. Manager's A/B Matrix", "📑 3. Executive Brief", "📊 4. Model Comparison"],
    index=0
)

# ---------------------------------------------------------
# ---------------------------------------------------------
# 5. Tactical Engine & Style Modifiers
# ---------------------------------------------------------

@st.cache_data
def load_feature_reference():
    return pd.read_csv("clean_master_dataset.csv")


feature_reference = load_feature_reference()

GLOBAL_COV_MATRIX = feature_reference[tactical_features].cov().values

STAT_LIMITS = {
    feat: (
        float(feature_reference[feat].min()),
        float(feature_reference[feat].max())
    )
    for feat in tactical_features
}

def enforce_realistic_bounds(stats_dict):
    """Clips inflated metrics back into realistic historical distributions."""
    bounded = stats_dict.copy()
    for feat, (min_val, max_val) in STAT_LIMITS.items():
        if feat in bounded:
            bounded[feat] = max(min_val, min(max_val, bounded[feat]))
    return bounded

def apply_formation_clash_engine(home_base, opp_base, h_form, a_form ):
    mapped = home_base.copy()
    
    xg_diff_factor = (home_base['xg'] - opp_base['xg']) * 0.1
    poss_diff_factor = (home_base['possession'] - opp_base['possession']) * 0.15
    mapped['xg'] = max(0.2, mapped['xg'] + xg_diff_factor)
    mapped['possession'] = np.clip(mapped['possession'] + poss_diff_factor, 25.0, 75.0)

    if h_form == "4-3-3":
        mapped['ppda'] *= 0.75; mapped['possession'] *= 1.1; mapped['xg'] *= 1.15; mapped['shots_on_target'] *= 1.15
    elif h_form == "5-4-1":
        mapped['possession'] *= 0.75; mapped['ppda'] *= 1.3; mapped['xg'] *= 0.8; mapped['shots_on_target'] *= 0.8
    elif h_form == "4-1-4-1":
        mapped['interceptions'] *= 1.25; mapped['tackles_successful'] *= 1.15; mapped['possession'] *= 0.95
    elif h_form == "3-4-3":
        mapped['xg'] *= 1.2; mapped['shots_on_target'] *= 1.2
    elif h_form == "3-5-2":
        mapped['possession'] *= 1.05; mapped['tackles_successful'] *= 1.1
    elif h_form == "4-4-2":
        mapped['possession'] *= 0.9

    if a_form in ["4-3-3", "3-4-3"]:
        mapped['possession'] -= 5.0; mapped['ppda'] -= 1.0
    elif a_form in ["5-4-1", "4-1-4-1"]:
        mapped['possession'] += 6.0
    elif a_form in ["3-5-2", "4-2-3-1"]:
        mapped['tackles_successful'] += 2.0

    # Return the bounded results
    return enforce_realistic_bounds(mapped)

def apply_tactical_style(stats_dict, style):
    adj = stats_dict.copy()
    
    if any(k in style for k in ["Gegenpressing", "High Press", "All-Out Attack"]):
        adj['ppda'] *= 0.65; adj['tackles_successful'] *= 1.25; adj['possession'] *= 1.1
        if "All-Out Attack" in style:
            adj['xg'] *= 1.3; adj['shots_on_target'] *= 1.3
            
    elif any(k in style for k in ["Park the Bus", "Counter-Attack", "Long Ball"]):
        adj['possession'] *= 0.65; adj['xg'] *= 1.1; adj['shots_on_target'] *= 1.15
        if "Park the Bus" in style:
            adj['ppda'] *= 1.5; adj['interceptions'] *= 1.3; adj['xg'] *= 0.8 
        if "Long Ball" in style:
            adj['aerial_duels_won_pct'] = min(85.0, adj['aerial_duels_won_pct'] * 1.3)
            
    elif any(k in style for k in ["Chokehold", "Mid-Block", "High Block Possession", "Playmaker"]):
        adj['possession'] *= 1.15; adj['interceptions'] *= 1.2
        if "Chokehold" in style or "Mid-Block" in style:
            adj['tackles_successful'] *= 1.15; adj['possession'] *= 0.9 
            
    elif any(k in style for k in ["Wide Overload", "Twin Striker"]):
        adj['aerial_duels_won_pct'] = min(80.0, adj['aerial_duels_won_pct'] * 1.25); adj['shots_on_target'] *= 1.15; adj['xg'] *= 1.1
        
    # Return the bounded results
    return enforce_realistic_bounds(adj)

# Calculate base collision stats
mapped_stats_base = apply_formation_clash_engine(team_baseline, opp_baseline, home_formation, opp_formation )

# Fetch available philosophies based on the chosen formation
available_philosophies = FORMATION_PHILOSOPHIES.get(home_formation, ["Standard (Balanced setup)"])


# ---------------------------------------------------------
# 6. 2D Pitch Rendering Component
# ---------------------------------------------------------
def draw_2d_pitch_enhanced(formation_name, team_name, philosophy_name):
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor='#0b0f19')
    ax.set_facecolor('#1e293b')

    # Render fundamental pitch boundaries and zones
    ax.plot([0, 0, 100, 100, 0], [0, 100, 100, 0, 0], color="white", alpha=0.3, linewidth=1.5)
    ax.plot([50, 50], [0, 100], color="white", alpha=0.3, linewidth=1.5)
    ax.add_patch(patches.Circle((50, 50), 12, color="white", fill=False, alpha=0.3, linewidth=1.5))
    ax.add_patch(patches.Rectangle((0, 20), 18, 60, color="white", fill=False, alpha=0.3, linewidth=1.5))
    ax.add_patch(patches.Rectangle((82, 20), 18, 60, color="white", fill=False, alpha=0.3, linewidth=1.5))

    # Fetch corresponding tactical depth configuration based on user selection
    tactic_info = TACTICAL_CONFIG[formation_name].get(philosophy_name, list(TACTICAL_CONFIG[formation_name].values())[0])
    
    # Overlay tactical defensive/offensive indicator line
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
    
    # Format header title by trimming default suffix
    clean_philosophy = philosophy_name.replace(" (Default)", "")
    ax.set_title(f"{team_name} [{formation_name} | {clean_philosophy}]", color='white', fontsize=11, pad=10)
    plt.tight_layout()
    return fig

# ---------------------------------------------------------
# Persistent Tactical Philosophy State
# ---------------------------------------------------------

# This value is NOT tied directly to a widget.
# Therefore it survives when the Tactical Board is not displayed.
if "selected_philosophy_value" not in st.session_state:
    st.session_state.selected_philosophy_value = available_philosophies[0]

# If the coach changes formation, make sure the previously
# selected philosophy is still valid for the new formation.
if st.session_state.selected_philosophy_value not in available_philosophies:
    st.session_state.selected_philosophy_value = available_philosophies[0]


def save_philosophy_selection():
    """
    Copy the temporary radio-widget value into a permanent
    session-state variable.
    """
    st.session_state.selected_philosophy_value = (
        st.session_state._philosophy_widget
    )


if app_mode == "🏟️ 1. Tactical Board":

    st.subheader("📋 Matchup & Tactical Execution Board")

    # Restore the last saved philosophy whenever this
    # dashboard panel is opened again.
    st.session_state._philosophy_widget = (
        st.session_state.selected_philosophy_value
    )

    tactical_style = st.radio(
        f"Core Philosophy for {home_formation}",
        available_philosophies,
        key="_philosophy_widget",
        on_change=save_philosophy_selection,
        horizontal=True
    )

    # Keep the permanent state synchronized.
    st.session_state.selected_philosophy_value = tactical_style
    
    # Calculate final stats using the actively selected tactical style
    adj_stats = apply_tactical_style(mapped_stats_base, tactical_style)
    
    st.divider()

    col_pitch, col_panel = st.columns([1.2, 1.0])

    with col_pitch:
        st.markdown("##### 🏟️ Formation Clash & Defensive Line")
        fig_pitch = draw_2d_pitch_enhanced(home_formation, home_team, tactical_style)
        st.pyplot(fig_pitch)

        st.markdown("##### 📈 Projected Match KPIs vs. Historical Baseline")

    st.caption(
        "The delta values show how the current tactical scenario changes "
        "each projected indicator relative to the team's historical baseline."
    )

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        label="🎯 Expected Goals (xG)",
        value=f"{adj_stats['xg']:.2f}",
        delta=f"{adj_stats['xg'] - team_baseline['xg']:+.2f}",
        help="Expected scoring opportunity quality. Higher values indicate greater projected chance quality."
    )

    k2.metric(
        label="⚽ Pass Share (Control Proxy)",
        value=f"{adj_stats['possession']:.1f}%",
        delta=f"{adj_stats['possession'] - team_baseline['possession']:+.1f}%",
        help=(
            "The team's share of total pass events. "
            "It is used as a possession-control proxy rather than official possession time."
        )
    )

    k3.metric(
        label="🏃 PPDA-Style Pressing Proxy",
        value=f"{adj_stats['ppda']:.1f}",
        delta=f"{adj_stats['ppda'] - team_baseline['ppda']:+.1f}",
        delta_color="inverse",
        help=(
            "Opponent completed passes divided by recorded Duel and Interception events. "
            "Lower values indicate greater defensive-action intensity."
        )
    )

    k4.metric(
        label="🛡️ Defensive Duel Count",
        value=f"{adj_stats['tackles_successful']:.1f}",
        delta=f"{adj_stats['tackles_successful'] - team_baseline['tackles_successful']:+.1f}",
        help=(
            "Recorded Duel events used as a proxy for defensive duel activity. "
            "This is not strictly a successful-tackle count."
        )
    )

    with col_panel:
        st.markdown("#### 🎯 Player Execution KPIs (Locker Room Directives)")
        with st.container(border=True):
            st.success( f"**Midfield Task**: The current scenario projects a " f"**PPDA-style pressing proxy of {adj_stats['ppda']:.1f}**. " f"Lower values represent more intensive pressing.")
            st.info( f"**Defensive Task**: The current scenario projects approximately " f"**{int(adj_stats['interceptions'])} interception events**.")
            st.warning(
                f"**Tempo Control**: The current scenario projects a "
                f"**{adj_stats['possession']:.1f}% pass share**, used here as a "
                f"possession-control proxy. Offensive units project "
                f"**{int(adj_stats['shots_on_target'])} threatening shots**."
            )
            st.error(
                f"**Physicality**: The current scenario projects a "
                f"**{adj_stats['aerial_duels_won_pct']:.1f}% share of recorded "
                f"50/50 duel activity**. This represents contested-duel involvement, "
                f"not an aerial-duel win rate."
            )


elif app_mode == "⚖️ 2. Manager's A/B Matrix":
    
    # Fetch the exact tactical style chosen by the coach in Tab 1 from memory
    tactical_style = st.session_state.selected_philosophy_value
    adj_stats = apply_tactical_style(mapped_stats_base, tactical_style)
    
    input_vector = np.array([[
        adj_stats['xg'], adj_stats['possession'], adj_stats['shots_on_target'], 
        adj_stats['ppda'], adj_stats['tackles_successful'], adj_stats['interceptions'], adj_stats['aerial_duels_won_pct']
    ]])
    
    a_min_bounds = np.array([
    STAT_LIMITS[f][0]
    for f in tactical_features
])

    a_max_bounds = np.array([
    STAT_LIMITS[f][1]
    for f in tactical_features
])
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

    st.subheader("⚖️ Tactical Core: A/B Formation Comparison & KPI Trade-offs")
    st.caption("Compare model-estimated outcome probabilities under alternative "
    "tactical scenarios.")

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
        alt_mapped_base = apply_formation_clash_engine(team_baseline, opp_baseline, alt_formation, opp_formation )
        alt_mapped_styled = apply_tactical_style(alt_mapped_base, alt_default_style)
        
        alt_vector = np.array([[alt_mapped_styled[f] for f in tactical_features]])
        alt_sim_inputs = np.clip(alt_vector + noise * scale, a_min=a_min_bounds, a_max=a_max_bounds)
        
        alt_sim_probs = model.predict_proba(alt_sim_inputs)
        alt_win_pct = np.mean(alt_sim_probs[:, win_idx]) * 100
        diff_win = alt_win_pct - mc_win_pct

        with col_advice:
            if diff_win > 3.0:
                st.success(f"💡 **Coaching Staff Advice**: Under the current scenario assumptions, "f"switching to **{alt_formation}** increases the model-estimated win "f"probability by **{diff_win:+.1f} percentage points** compared with Plan A.")
            elif diff_win < -2.0:
                st.error(f"⚠️ **Scenario Warning**: Under the current assumptions, switching to " f"**{alt_formation}** reduces the model-estimated win probability to "f"**{alt_win_pct:.1f}%**, a change of **{diff_win:.1f} percentage points** "f"relative to Plan A.")
            else:
                st.info(f"⚖️ **Tactical Assessment**: Switching to **{alt_formation}** changes the "f"model-estimated win probability by **{diff_win:+.1f} percentage points**. "f"The estimated difference between the two scenarios is relatively small.")

        st.markdown("##### 📊 Tactical Cost & Benefit of Formation Change (KPI Delta)")
        k1, k2, k3, k4 = st.columns(4)
        diff_xg = alt_mapped_styled['xg'] - adj_stats['xg']
        diff_poss = alt_mapped_styled['possession'] - adj_stats['possession']
        diff_ppda = alt_mapped_styled['ppda'] - adj_stats['ppda']
        diff_tackles = alt_mapped_styled['tackles_successful'] - adj_stats['tackles_successful']
        
        k1.metric(
            "🎯 Expected Goals (xG)",
            f"{alt_mapped_styled['xg']:.2f}",
            f"{diff_xg:+.2f}",
            help=(
                "Change in projected Expected Goals when switching "
                "from Plan A to Plan B."
            )
        )

        k2.metric(
            "⚽ Pass Share (Control Proxy)",
            f"{alt_mapped_styled['possession']:.1f}%",
            f"{diff_poss:+.1f}%",
            help=(
                "Change in the team's projected share of total pass events. "
                "This is used as a possession-control proxy."
            )
        )

        k3.metric(
            "🏃 PPDA-Style Pressing Proxy",
            f"{alt_mapped_styled['ppda']:.1f}",
            f"{diff_ppda:+.1f}",
            delta_color="inverse",
            help=(
                "Change in the simplified pressing proxy. "
                "A lower value represents greater defensive-action intensity."
            )
        )

        k4.metric(
            "🛡️ Defensive Duel Count",
            f"{alt_mapped_styled['tackles_successful']:.1f}",
            f"{diff_tackles:+.1f}",
            help=(
                "Change in recorded defensive-duel activity "
                "between Plan A and Plan B."
            )
        )


elif app_mode == "📑 3. Executive Brief":
    
    # Retrieve the exact tactical philosophy the manager selected in Tab 1 from our session memory, 
    # ensuring absolute continuity across the dashboard.
    tactical_style = st.session_state.selected_philosophy_value
    
    # Re-evaluate the baseline collision stats using the manager's chosen philosophy.
    adj_stats = apply_tactical_style(mapped_stats_base, tactical_style)
    
    # Construct the final tactical feature vector. 
    # This is the exact payload our Random Forest model will analyze.
    input_vector = np.array([[
        adj_stats['xg'], adj_stats['possession'], adj_stats['shots_on_target'], 
        adj_stats['ppda'], adj_stats['tackles_successful'], adj_stats['interceptions'], adj_stats['aerial_duels_won_pct']
    ]])
    
    # Interrogate the Random Forest model to calculate the baseline win probability for the executive summary.
    classes = list(model.classes_)
    win_idx = classes.index('Win') if 'Win' in classes else 2
    mc_win_pct = np.mean(model.predict_proba(input_vector)[:, win_idx]) * 100
    
    st.subheader("📑 Pre-Match Executive Brief")
    st.caption("Scenario summary based on global Random Forest feature importance, ""scenario-specific deviations from the historical baseline, and ""predefined tactical rules.")
    st.caption( "This module is not a local explanation of an individual model prediction.")

    def get_dynamic_advice(feat, curr_val, base_val, is_advantage, style):
        """
        Converts scenario deviations into coach-friendly tactical summaries.
        The statements describe scenario changes relative to the historical
        baseline and should not be interpreted as causal explanations.
        """

        diff = curr_val - base_val
        abs_diff = abs(diff)
        direction = "higher" if diff > 0 else "lower"

        # PPDA-style pressing proxy
        if feat == "ppda":
            press_intensity = (
                "more intensive pressing"
                if curr_val < base_val
                else "less intensive pressing"
            )
            return (
                f"PPDA-Style Pressing Proxy is projected at {curr_val:.1f}, "
                f"which is {abs_diff:.1f} {direction} than the historical baseline. "
                f"Lower values represent greater defensive-action intensity, so the "
                f"current scenario indicates **{press_intensity}**."
            )

        # Pass share
        elif feat == "possession":
            control_type = (
                "higher relative pass control"
                if curr_val >= base_val
                else "lower relative pass control"
            )
            return (
                f"Projected Pass Share is {curr_val:.1f}% "
                f"({diff:+.1f} percentage points from the historical baseline). "
                f"This indicates **{control_type}** under the current scenario. "
                f"Pass Share is used here as a possession-control proxy."
            )

        # Expected Goals
        elif feat == "xg":
            status = (
                "higher projected chance quality"
                if curr_val >= base_val
                else "lower projected chance quality"
            )
            return (
                f"Expected Goals (xG) is projected at {curr_val:.2f} "
                f"({diff:+.2f} from the historical baseline), indicating "
                f"**{status}** under the current tactical scenario."
            )

        # Threatening shot count
        elif feat == "shots_on_target":
            status = "higher" if curr_val >= base_val else "lower"
            return (
                f"Threatening Shot Count is projected at {int(curr_val)}, "
                f"which is **{status} than the historical baseline**. "
                f"This feature counts extracted Goal, Saved and Post shot events."
            )

        # Defensive duel count
        elif feat == "tackles_successful":
            status = "higher" if curr_val >= base_val else "lower"
            return (
                f"Defensive Duel Count is projected at {int(curr_val)}, "
                f"representing **{status} defensive-duel activity** relative "
                f"to the historical baseline."
            )

        # Interceptions
        elif feat == "interceptions":
            status = "higher" if curr_val >= base_val else "lower"
            return (
                f"Interception Count is projected at {int(curr_val)}, "
                f"representing **{status} interception activity** relative "
                f"to the historical baseline."
            )

        # 50/50 duel share
        elif feat == "aerial_duels_won_pct":
            return (
                f"50/50 Duel Share is projected at {curr_val:.1f}%. "
                f"This represents the team's relative share of recorded "
                f"50/50 duel activity and is **not an aerial-duel win rate**."
            )

        return (
            f"Metric value is {curr_val:.1f} "
            f"({diff:+.1f} from the historical baseline)."
        )

    # Extract the feature importances from our trained Random Forest model. 
    # We need to know which statistical pillars actually dictate the outcome today.
    rf_importances = model.feature_importances_
    current_vals = input_vector[0]

    contributions = []
    
    # Calculate the weighted impact of each tactical metric. 
    # We multiply the raw deviation by the model's feature importance to find our true game-changers.
    for i, feat in enumerate(tactical_features):
        base_val = team_baseline[feat] 
        curr_val = current_vals[i]
        imp = rf_importances[i]

        diff = (base_val - curr_val) if feat == 'ppda' else (curr_val - base_val)
        score = diff * imp
        contributions.append((feat, curr_val, score))

    # Sort the contributions to isolate the top 3 match-winning advantages and the top 3 critical vulnerabilities.
    contributions.sort(key=lambda x: x[2], reverse=True)
    top_positives = [c for c in contributions if c[2] > 0][:3]
    top_negatives = [c for c in contributions if c[2] < 0][-3:]

    with st.container(border=True):
        st.markdown(f"### 🏟️ Match Preview: {home_team} vs {away_team}")
        st.caption(
            f"**Formation**: {home_formation} | "
            f"**Style**: {tactical_style.split(' ')[0]} | "
            f"**Scenario Win Estimate**: {mc_win_pct:.1f}%"
        )
        
        col_pos, col_neg = st.columns(2)
        
        with col_pos:
            st.markdown("##### ✅ Tactical Advantages (Primary Attack Vectors)")
            if top_positives:
                for feat, val, score in top_positives:
                    base_val = team_baseline[feat] 
                    advice = get_dynamic_advice(feat, val, base_val, True, tactical_style)
                    display_feat = FEATURE_LABELS.get(feat, feat)
                    st.success(
                    f"**{display_feat}** (Value: `{val:.1f}`)\n\n"
                    f"*Analysis: {advice}*"
                )
            else:
                st.caption("No distinct statistical advantages found.")

        with col_neg:
            st.markdown("##### ⚠️ Achilles' Heel (Areas to Defend)")
            if top_negatives:
                for feat, val, score in top_negatives:
                    base_val = team_baseline[feat] 
                    advice = get_dynamic_advice(feat, val, base_val, False, tactical_style)
                    display_feat = FEATURE_LABELS.get(feat, feat)
                    st.error(f"**{display_feat}** (Value: `{val:.1f}`)\n\n"f"*Warning: {advice}*")
            else:
                # 🚀 Advanced Empty State Design: 
                # What if the manager crafts a flawless tactic with no statistical weaknesses?
                # Instead of leaving a confusing blank space, we celebrate the tactical balance and reassure the coaching staff.
                st.success("🛡️ **Perfectly Balanced (Zero Tactical Deficits)**")
                st.caption("All core metrics are performing at or above your historical baseline. No distinct statistical vulnerabilities detected for this setup.")
                
                # Dynamically identify the unsung heroes—metrics that didn't make the top 3 advantages, 
                # but are performing safely above the historical baseline to secure our foundation.
                managed_feats = [c[0] for c in contributions if c[2] >= 0 and c not in top_positives]
                
                if managed_feats:
                    safe_str = ", ".join(
                        FEATURE_LABELS.get(f, f)
                        for f in managed_feats
                    )
                    st.info(f"**✅ Secured Metrics**: \n\n`{safe_str}` \n\nThese areas remain highly stable under the current tactic. Your defensive and transition phases are fully covered.")
       
elif app_mode == "📊 4. Model Comparison":
    st.subheader(
        "📊 Model Comparison Dashboard"
    )

    st.success(
        "🏆 **Selected Final Model: Random Forest** — "
        "Random Forest achieved the highest mean Accuracy (54.78%), "
        "the highest Macro Precision (0.6052), and the lowest Accuracy "
        "variation (SD = 0.0522) during development-stage grouped "
        "cross-validation. Although MLP and XGBoost achieved marginally "
        "higher Macro F1-Scores (0.5162 vs. 0.5155), the difference was "
        "very small. Random Forest was retained because of its competitive "
        "overall performance and its direct support for the global "
        "feature-importance mechanism used by the prototype."
    )

    st.info(
        "📌 **Evaluation stages:** 54.78% is Random Forest's mean Accuracy "
        "during development-stage Stratified Group 5-Fold Cross-Validation "
        "on the 2018–2020 data. After model selection was completed, "
        "Random Forest was evaluated once on the later 2022–2024 "
        "chronological test set, where it achieved 58.50% Accuracy."
    )

    st.caption(
    "Supplementary stability check — Accuracy SD across the five folds: "
    "Random Forest = 0.0522, MLP = 0.1203, XGBoost = 0.0567. "
    "SD is shown as supporting evidence and is not one of the assignment's "
    "required evaluation metrics."
    )
    
    st.caption(
        "Random Forest is the selected final model. "
        "MLP and XGBoost are retained as comparison "
        "models to demonstrate differences in validation "
        "performance and model-estimated probabilities."
    )
    # =====================================================
    # PREPARE CV METRICS
    # =====================================================

    metrics_df = comparison_metrics.copy()

    # Remove DummyClassifier from the dashboard comparison
    metrics_df = metrics_df[
        metrics_df["Model"].isin(
            [
                "Random Forest",
                "MLP",
                "XGBoost"
            ]
        )
    ].copy()

    metrics_df = metrics_df.set_index(
        "Model"
    )


    # =====================================================
    # TABS
    # =====================================================

    tab1, tab2, tab3 = st.tabs(
        [
            "🌲 RF vs MLP",
            "🌲 RF vs XGBoost",
            "⚽ Same Tactical Scenario"
        ]
    )


    # =====================================================
    # TAB 1 — RF vs MLP
    # =====================================================

    with tab1:

        st.markdown(
            "### Random Forest vs MLP"
        )

        rf_mlp = metrics_df.loc[
            [
                "Random Forest",
                "MLP"
            ],
            [
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1"
            ]
        ].copy()

        st.dataframe(
            rf_mlp.style.format(
                "{:.4f}"
            ),
            use_container_width=True
        )

        st.info(
            "**Interpretation:** Random Forest achieved "
            "higher mean Accuracy and Macro Precision. "
            "MLP achieved a marginally higher Macro F1, "
            "but showed substantially greater accuracy "
            "variation across the grouped cross-validation "
            "folds."
        )


    # =====================================================
    # TAB 2 — RF vs XGBOOST
    # =====================================================

    with tab2:

        st.markdown(
            "### Random Forest vs XGBoost"
        )

        rf_xgb = metrics_df.loc[
            [
                "Random Forest",
                "XGBoost"
            ],
            [
                "Accuracy",
                "Macro Precision",
                "Macro Recall",
                "Macro F1"
            ]
        ].copy()

        st.dataframe(
            rf_xgb.style.format(
                "{:.4f}"
            ),
            use_container_width=True
        )

        st.info(
            "**Interpretation:** Random Forest and "
            "XGBoost produced very similar overall "
            "validation performance. Random Forest "
            "achieved slightly higher Accuracy and "
            "Macro Precision, while XGBoost achieved "
            "slightly higher Macro Recall and Macro F1."
        )


    # =====================================================
    # TAB 3 — SAME TACTICAL SCENARIO
    # =====================================================

    with tab3:

        st.markdown(
            "### Same Tactical Scenario — "
            "Three-Model Prediction Comparison"
        )

        st.caption(
            "The live comparison below uses deployment versions of Random Forest, "
            "MLP and XGBoost trained on the available historical dataset. All three "
            "receive exactly the same scenario-adjusted seven-feature input and the "
            "same Monte Carlo perturbations. These live outputs are for tactical "
            "scenario comparison and are separate from the development-stage "
            "cross-validation metrics shown above."
        )

        with st.expander("ℹ️ What exactly is being compared?"):

            st.markdown(
                """
                - **Formation is not a direct machine-learning feature.**
                - The selected formation, opponent formation and
                  tactical philosophy first modify the team's historical baseline
                  through predefined scenario rules.
                - This produces the same seven numerical inputs for all three models.
                - The three models then independently estimate Win, Draw and Loss
                  probabilities from that identical scenario.
                """
            )
        # -------------------------------------------------
        # Retrieve current tactical setup
        # -------------------------------------------------

        tactical_style = st.session_state.selected_philosophy_value

        comparison_stats = apply_tactical_style(
            mapped_stats_base,
            tactical_style
        )


        # -------------------------------------------------
        # Construct same 7-feature vector
        # -------------------------------------------------

        comparison_vector = np.array([[
            comparison_stats["xg"],
            comparison_stats["possession"],
            comparison_stats["shots_on_target"],
            comparison_stats["ppda"],
            comparison_stats["tackles_successful"],
            comparison_stats["interceptions"],
            comparison_stats["aerial_duels_won_pct"]
        ]])


        # -------------------------------------------------
        # Same Monte Carlo noise for ALL models
        # -------------------------------------------------

        min_bounds = np.array([
            STAT_LIMITS[f][0]
            for f in tactical_features
        ])

        max_bounds = np.array([
            STAT_LIMITS[f][1]
            for f in tactical_features
        ])

        N_COMPARE_SIM = 1000

        np.random.seed(42)

        comparison_noise = np.random.normal(
            0,
            1,
            (N_COMPARE_SIM, 7)
        )

        comparison_scale = np.array([
            0.15,
            2.5,
            0.8,
            0.8,
            1.2,
            1.0,
            2.0
        ])

        comparison_inputs = np.clip(
            comparison_vector
            + comparison_noise
            * comparison_scale,

            a_min=min_bounds,
            a_max=max_bounds
        )


        # =================================================
        # RANDOM FOREST
        # =================================================

        rf_probs_raw = model.predict_proba(
            comparison_inputs
        )

        rf_classes = list(
            model.classes_
        )

        rf_probability = {
            outcome:
            np.mean(
                rf_probs_raw[
                    :,
                    rf_classes.index(outcome)
                ]
            )
            for outcome in [
                "Win",
                "Draw",
                "Loss"
            ]
        }


        # =================================================
        # MLP
        # =================================================

        mlp_probs_raw = (
            mlp_model.predict_proba(
                comparison_inputs
            )
        )

        mlp_classes = list(
            mlp_model.classes_
        )

        mlp_probability = {
            outcome:
            np.mean(
                mlp_probs_raw[
                    :,
                    mlp_classes.index(outcome)
                ]
            )
            for outcome in [
                "Win",
                "Draw",
                "Loss"
            ]
        }


        # =================================================
        # XGBOOST
        # =================================================

        xgb_probs_raw = (
            xgb_model.predict_proba(
                comparison_inputs
            )
        )

        xgb_probability = {
            outcome:
            np.mean(
                xgb_probs_raw[
                    :,
                    xgb_classes.index(outcome)
                ]
            )
            for outcome in [
                "Win",
                "Draw",
                "Loss"
            ]
        }


        # =================================================
        # DISPLAY CURRENT SCENARIO
        # =================================================

        st.markdown(
            f"**Current Scenario:** "
            f"{home_team} vs {away_team} | "
            f"{home_formation} | "
            f"{tactical_style}"
        )


        # =================================================
        # RESULT TABLE
        # =================================================

        live_comparison_df = pd.DataFrame(
            {
                "Random Forest": [
                    rf_probability["Win"],
                    rf_probability["Draw"],
                    rf_probability["Loss"]
                ],

                "MLP": [
                    mlp_probability["Win"],
                    mlp_probability["Draw"],
                    mlp_probability["Loss"]
                ],

                "XGBoost": [
                    xgb_probability["Win"],
                    xgb_probability["Draw"],
                    xgb_probability["Loss"]
                ]
            },

            index=[
                "Win",
                "Draw",
                "Loss"
            ]
        )


        # Convert to %
        live_percentage_df = (
            live_comparison_df * 100
        )


        st.dataframe(
            live_percentage_df.style.format(
                "{:.1f}%"
            ),
            use_container_width=True
        )

        # =================================================
        # BAR CHART
        # =================================================

        st.markdown(
            "#### Outcome Probability Comparison"
        )

        st.bar_chart(
            live_percentage_df
        )


        # =================================================
        # SIMPLE SUMMARY
        # =================================================

        rf_top = max(
            rf_probability,
            key=rf_probability.get
        )

        mlp_top = max(
            mlp_probability,
            key=mlp_probability.get
        )

        xgb_top = max(
            xgb_probability,
            key=xgb_probability.get
        )


        st.markdown(
            "#### Model Decisions"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "🌲 Random Forest",
            f"{rf_top} — {rf_probability[rf_top] * 100:.1f}%"
        )

        c2.metric(
            "🧠 MLP",
            f"{mlp_top} — {mlp_probability[mlp_top] * 100:.1f}%"
        )    

        c3.metric(
            "🚀 XGBoost",
            f"{xgb_top} — {xgb_probability[xgb_top] * 100:.1f}%"
        )


        if rf_top == mlp_top == xgb_top:
            st.success(
                f"✅ All three models currently agree that **{rf_top}** "
                "is the most likely outcome under this tactical scenario."
            )
        else:
            st.warning(
                "⚠️ The three models do not fully agree on the most likely "
                "outcome. This illustrates model disagreement under the same "
                "tactical scenario."
            )

        st.info(
            "ℹ️ The displayed Win, Draw and Loss percentages are "
            "**model-estimated probabilities**, not calibrated probabilities. "
            "No separate probability-calibration procedure was applied."
        )
