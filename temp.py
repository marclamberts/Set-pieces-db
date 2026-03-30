import os
from io import BytesIO
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Allsvenskan Set Piece Studio Pro",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

FILE_NAME = "Allsvenskan - Corners 2025.xlsx"
LOGIN_NAME = "Admin"
LOGIN_PASSWORD = "Football2026"

# =========================================================
# DESIGN
# =========================================================
BG = "#07111f"
BG_2 = "#0b1730"
CARD = "#101a2b"
CARD_2 = "#16243a"
TEXT = "#f3f7fc"
MUTED = "#99adc7"
MUTED_2 = "#6b87a8"
ACCENT = "#5da8ff"
SUCCESS = "#34d399"
WARNING = "#fbbf24"
DANGER = "#fb7185"
PURPLE = "#a78bfa"
ORANGE = "#fb923c"
BORDER = "rgba(255,255,255,0.08)"
PITCH = "#133d24"
PITCH_LINE = "rgba(255,255,255,0.65)"

QUAL_PALETTE = [
    ACCENT, SUCCESS, WARNING, DANGER, PURPLE, ORANGE,
    "#8ad6ff", "#6ee7b7", "#fde68a", "#fda4af"
]

px.defaults.template = "plotly_dark"

CSS = f"""
<style>
body, .stApp {{
    background:
        radial-gradient(ellipse 900px 600px at 90% -10%, rgba(93,168,255,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 700px 500px at -5% 20%, rgba(52,211,153,0.07) 0%, transparent 55%),
        linear-gradient(180deg, {BG} 0%, {BG_2} 100%);
    color: {TEXT};
}}
.block-container {{
    max-width: 1680px;
    padding-top: 0.8rem;
    padding-bottom: 1.5rem;
}}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #080f1d 0%, #060e1a 100%);
    border-right: 1px solid {BORDER};
}}
.hero-wrap {{
    background: linear-gradient(135deg, rgba(93,168,255,0.14) 0%, rgba(93,168,255,0.04) 60%, rgba(52,211,153,0.06) 100%);
    border: 1px solid rgba(93,168,255,0.18);
    border-radius: 26px;
    padding: 24px 28px 20px 28px;
    margin-bottom: 14px;
    box-shadow: 0 16px 48px rgba(0,0,0,0.22);
}}
.hero-title {{
    font-size: 2.1rem;
    font-weight: 900;
    line-height: 1.0;
    margin-bottom: 0.35rem;
    letter-spacing: -0.02em;
}}
.hero-title span {{ color: {ACCENT}; }}
.hero-sub {{
    color: {MUTED};
    font-size: 0.97rem;
    line-height: 1.5;
}}
.pill {{
    display: inline-block;
    padding: 0.28rem 0.68rem;
    border-radius: 999px;
    background: rgba(93,168,255,0.12);
    color: #d4e8ff;
    border: 1px solid rgba(93,168,255,0.20);
    font-size: 0.76rem;
    margin-right: 0.4rem;
    margin-top: 0.45rem;
    font-weight: 500;
}}
.kpi-card {{
    background: linear-gradient(160deg, {CARD} 0%, {CARD_2} 100%);
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 16px 16px 12px 16px;
    min-height: 108px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18);
}}
.kpi-label {{
    color: {MUTED};
    text-transform: uppercase;
    font-size: 0.70rem;
    letter-spacing: 0.10em;
    margin-bottom: 8px;
    font-weight: 600;
}}
.kpi-value {{
    color: {TEXT};
    font-weight: 900;
    font-size: 1.8rem;
    line-height: 1.0;
}}
.kpi-foot {{
    margin-top: 8px;
    color: {MUTED};
    font-size: 0.80rem;
}}
.section-title {{
    font-size: 1.10rem;
    font-weight: 800;
    margin: 0.1rem 0 0.18rem 0;
    color: {TEXT};
}}
.section-sub {{
    color: {MUTED};
    font-size: 0.90rem;
    margin-bottom: 0.85rem;
    line-height: 1.4;
}}
.insight-box {{
    background: linear-gradient(135deg, rgba(93,168,255,0.09), rgba(93,168,255,0.04));
    border: 1px solid rgba(93,168,255,0.18);
    border-radius: 16px;
    padding: 14px 16px;
    margin-bottom: 8px;
    min-height: 76px;
}}
.empty-state {{
    text-align: center;
    padding: 56px 24px;
    color: {MUTED};
    font-size: 0.94rem;
    border: 1px dashed rgba(255,255,255,0.10);
    border-radius: 16px;
    background: rgba(255,255,255,0.015);
}}
.footer-note {{
    color: {MUTED_2};
    font-size: 0.82rem;
    margin-top: 0.8rem;
    padding-top: 12px;
    border-top: 1px solid {BORDER};
}}
div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 14px;
    overflow: hidden;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 4px;
    background: rgba(255,255,255,0.025);
    border-radius: 14px;
    padding: 4px;
    border: 1px solid {BORDER};
}}
.stTabs [aria-selected="true"] {{
    background: rgba(93,168,255,0.16) !important;
    color: #d4e8ff !important;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

import plotly.graph_objects as go
import numpy as np

def shotmap_figure(df_shots, title="Shotmap", side_focus="Both"):
    fig = draw_pitch(go.Figure(), title=title, height=600, half=True)
    if df_shots.empty: 
        return fig

    # 1. Robust Column Discovery
    # Finds 'shot.statsbomb_xg' or 'shot_statsbomb_xg' or anything containing 'statsbomb_xg'
    xg_col = next((c for c in df_shots.columns if "statsbomb_xg" in c), None)
    # Finds the team column (usually 'pass_team_name' or 'corner_team')
    team_col = next((c for c in df_shots.columns if "team" in c.lower()), df_shots.columns[0])
    
    plot = df_shots.copy()

    # 2. Handle xG sizing safely
    if xg_col:
        # Convert to numeric just in case it's stored as a string
        plot[xg_col] = pd.to_numeric(plot[xg_col], errors='coerce').fillna(0)
        plot["_size"] = np.clip(plot[xg_col] * 100 + 10, 10, 40)
    else:
        plot["_size"] = 15 # Default size if column is missing

    # 3. Plotting
    for group, sub in plot.groupby(team_col):
        fig.add_trace(go.Scatter(
            x=80 - sub["shot_location_y"], 
            y=sub["shot_location_x"],
            mode="markers",
            name=str(group),
            marker=dict(size=sub["_size"], opacity=0.7, line=dict(color="white", width=1)),
            text=[
                f"<b>Player:</b> {r.get('Shooter', 'N/A')}<br>"
                f"<b>xG:</b> {r[xg_col]:.3f}<br>" if xg_col else "<b>xG:</b> N/A<br>"
                f"<b>Outcome:</b> {r.get('SP_outcome', 'N/A')}"
                for _, r in sub.iterrows()
            ],
            hovertemplate="%{text}<extra></extra>"
        ))
    return fig

def delivery_map_figure(df_events, color_col="pass_team_name", title="Delivery Map", side_focus="Both"):
    fig = draw_pitch(go.Figure(), title=title, height=700, half=False)
    
    # Filter for corners with landing points
    plot = df_events.dropna(subset=["pass_end_location_x", "pass_end_location_y"]).copy()
    if plot.empty: 
        return fig

    # Group by the team that took the corner
    group_col = color_col if color_col in plot.columns else "pass_team_name"

    for team, sub in plot.groupby(group_col):
        fig.add_trace(go.Scatter(
            x=80 - sub["pass_end_location_y"], 
            y=sub["pass_end_location_x"],
            mode="markers", # DOTS ONLY
            name=str(team),
            marker=dict(size=14, opacity=0.8, line=dict(width=1, color='white')),
            text=[
                f"<b>Outcome:</b> {r['SP_outcome']}<br>"
                f"<b>Taker:</b> {r.get('Taker', 'N/A')}<br>"
                f"<b>Match:</b> {r.get('Match', 'N/A')}"
                for _, r in sub.iterrows()
            ],
            hovertemplate="%{text}<extra></extra>"
        ))
    return fig
# =========================================================
# LOGIN
# =========================================================
def login_screen():
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-title">⚽ <span>Allsvenskan</span> Set Piece Studio Pro</div>
            <div class="hero-sub">A cleaner corner analysis workspace for coaches and analysts — faster reads, clearer visuals, better side filtering.</div>
            <div>
                <span class="pill">2025 Season</span>
                <span class="pill">Team Intel</span>
                <span class="pill">Visual Studio</span>
                <span class="pill">Exports</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, c, _ = st.columns([1, 1.4, 1])
    with c:
        with st.form("login_form"):
            st.markdown("### 🔐 Sign in")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)
            if submitted:
                if username == LOGIN_NAME and password == LOGIN_PASSWORD:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials.")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_screen()
    st.stop()

# =========================================================
# HELPERS
# =========================================================
def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def human_pct(v, decimals=1):
    return "—" if pd.isna(v) else f"{v*100:.{decimals}f}%"

def human_val(v, decimals=2):
    return "—" if pd.isna(v) else f"{v:.{decimals}f}"

def find_col(df, candidates):
    lower_map = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    for c in df.columns:
        c_low = str(c).strip().lower()
        for cand in candidates:
            if cand.lower() in c_low:
                return c
    return None

def split_match_name(m):
    if not isinstance(m, str):
        return None, None
    for sep in [" - ", " vs ", " v "]:
        if sep in m:
            l, r = m.split(sep, 1)
            return l.strip(), r.strip()
    return None, None

def classify_outcome(text):
    s = str(text).lower().strip()
    if "first contact - shot within 3 seconds" in s:
        return "Shot ≤3s"
    if "first contact" in s and "shot" in s:
        return "First Contact Shot"
    if "shot" in s:
        return "Shot"
    if "no first contact" in s:
        return "No First Contact"
    if s in ["", "nan"]:
        return "Unknown"
    return "Other"

def side_from_y(y):
    if pd.isna(y):
        return "Unknown"
    return "Right" if y < 40 else "Left"

def delivery_zone_from_y(y):
    if pd.isna(y):
        return "Unknown"
    if y < 30:
        return "Near Post Zone"
    if y <= 50:
        return "Central Zone"
    return "Far Post Zone"

def delivery_length(x0, y0, x1, y1):
    if any(pd.isna(v) for v in [x0, y0, x1, y1]):
        return np.nan
    return float(np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2))

def zone_from_end_location(x, y):
    if pd.isna(x) or pd.isna(y):
        return "Unknown"
    if x >= 114 and 30 <= y <= 50:
        return "6-yard box"
    if x >= 108 and 18 <= y <= 62:
        return "Penalty area"
    if x >= 100 and 18 <= y <= 62:
        return "Deep box"
    return "Outside danger zone"

def infer_delivery_type(technique, height):
    t = str(technique).lower()
    h = str(height).lower()
    if "inswing" in t:
        return "Inswinger"
    if "outswing" in t:
        return "Outswinger"
    if "short" in t:
        return "Short"
    if "straight" in t:
        return "Straight"
    if "high" in h:
        return "High Ball"
    if "low" in h or "ground" in h:
        return "Low Ball"
    return "Other"

def xg_category(xg):
    if pd.isna(xg):
        return "No shot"
    if xg >= 0.20:
        return "Big Chance (xG≥0.20)"
    if xg >= 0.10:
        return "Good Chance (xG≥0.10)"
    if xg >= 0.05:
        return "Half Chance (xG≥0.05)"
    return "Low xG (<0.05)"

def percentile_rank(series, value):
    s = series.dropna()
    if len(s) == 0 or pd.isna(value):
        return np.nan
    return float((s <= value).mean() * 100)

def safe_range_slider(label, min_val, max_val):
    if min_val < max_val:
        return st.slider(label, min_val, max_val, (min_val, max_val))
    st.caption(f"{label}: {min_val}")
    return (min_val, max_val)

# =========================================================
# UI HELPERS
# =========================================================
def section_header(title, sub=""):
    st.markdown(
        f'<div class="section-title">{title}</div>'
        + (f'<div class="section-sub">{sub}</div>' if sub else ""),
        unsafe_allow_html=True,
    )

def metric_card(label, value, foot=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def insight_box(title, body):
    st.markdown(
        f"""
        <div class="insight-box">
            <div style="font-weight:700;font-size:0.92rem">{title}</div>
            <div style="color:{MUTED};font-size:0.87rem;margin-top:4px;line-height:1.45">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def empty_state(msg="No data for current filters."):
    st.markdown(f'<div class="empty-state">{msg}</div>', unsafe_allow_html=True)

def figure_layout(fig, height=420, title=None):
    fig.update_layout(
        height=height,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=48 if title else 8, b=8),
        font=dict(color=TEXT),
        legend_title_text="",
        hoverlabel=dict(bgcolor="#0d1c31", font_color=TEXT),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    return fig

def draw_pitch(fig, title=None, height=700, half=False):
    # Vertical orientation: X=Width(0-80), Y=Length(0-120)
    fig.update_xaxes(range=[0, 80], visible=False)
    fig.update_yaxes(range=[60 if half else 0, 120], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=title, height=height, template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=10, r=10, t=40, b=10),
        shapes=[
            dict(type="rect", x0=0, y0=0, x1=80, y1=120, line=dict(color="white", width=2)),
            dict(type="line", x0=0, y0=60, x1=80, y1=60, line=dict(color="white", width=1.5)),
            dict(type="rect", x0=18, y0=102, x1=62, y1=120, line=dict(color="white", width=1.5)),
            dict(type="rect", x0=30, y0=114, x1=50, y1=120, line=dict(color="white", width=1.5)),
        ],
    )
    return fig
def filter_chips(team, match_count, taker_count, side_focus, venue_filter):
    chips = []
    if team != "All Teams":
        chips.append(f"Team: {team}")
    if match_count:
        chips.append(f"Matches: {match_count}")
    if taker_count:
        chips.append(f"Takers: {taker_count}")
    if side_focus != "Both":
        chips.append(f"Side: {side_focus}")
    if set(venue_filter) != {"Home", "Away", "Unknown"}:
        chips.append("Venue filtered")
    if not chips:
        chips = ["All teams", "All matches", "Both sides"]

    html = "".join([f'<span class="pill">{c}</span>' for c in chips])
    st.markdown(f'<div style="margin:0.05rem 0 0.75rem 0">{html}</div>', unsafe_allow_html=True)

def visual_context_note(side_focus, n_events):
    label = "Both sides" if side_focus == "Both" else f"{side_focus} side only"
    st.caption(f"Showing: {label} · {n_events:,} events in current view")

def annotate_side(fig, side_focus):
    txt = "Showing both sides" if side_focus == "Both" else f"Showing {side_focus.lower()} side only"
    fig.add_annotation(
        x=0.99,
        y=1.08,
        xref="paper",
        yref="paper",
        text=txt,
        showarrow=False,
        xanchor="right",
        font=dict(size=12, color="#d4e8ff"),
        bgcolor="rgba(93,168,255,0.12)",
        bordercolor="rgba(93,168,255,0.25)",
        borderwidth=1,
        borderpad=6,
    )
    return fig

def top_insights(team_df):
    if team_df.empty:
        return []
    insights = []
    best_sr = team_df.sort_values("shot_rate", ascending=False).iloc[0]
    best_xg = team_df.sort_values("xg_per_match", ascending=False).iloc[0]
    best_6y = team_df.sort_values("six_yard_delivery_rate", ascending=False).iloc[0]
    best_short = team_df.sort_values("short_corner_rate", ascending=False).iloc[0]

    insights.append(("Best Shot Rate", f"{best_sr['team']} convert {human_pct(best_sr['shot_rate'])} of corners into shots."))
    insights.append(("Highest xG/Match", f"{best_xg['team']} create {human_val(best_xg['xg_per_match'], 3)} xG per match from corners."))
    insights.append(("Best 6-yard Targeting", f"{best_6y['team']} hit the 6-yard box on {human_pct(best_6y['six_yard_delivery_rate'])} of corners."))
    insights.append(("Most Short Corners", f"{best_short['team']} use short routines {human_pct(best_short['short_corner_rate'])} of the time."))
    return insights[:4]

# =========================================================
# DATA LOAD
# =========================================================
@st.cache_data
def load_data():
    possible_files = [FILE_NAME, "Allsvenskan - Corners 2025 (1).xlsx"]
    for f in possible_files:
        if os.path.exists(f):
            return pd.read_excel(f)
    raise FileNotFoundError(f"{FILE_NAME} not found.")

@st.cache_data
def prepare_data(raw_df):
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    _f = lambda *c: find_col(df, list(c))

    match_id_col = _f("match_id", "match id")
    match_col = _f("match")
    team_col = _f("pass_team_name", "team", "team_name")
    minute_col = _f("minute")
    second_col = _f("second")
    outcome_col = _f("sp_outcome", "outcome")
    xg_col = _f("shot.statsbomb_xg", "shot_xg", "xg")
    taker_col = _f("taker")
    shooter_col = _f("shooter")
    def_setup_col = _f("defensive_setup")
    pass_x_col = _f("pass_location_x")
    pass_y_col = _f("pass_location_y")
    pass_ex_col = _f("pass_end_location_x")
    pass_ey_col = _f("pass_end_location_y")
    shot_x_col = _f("shot_location_x")
    shot_y_col = _f("shot_location_y")
    shot_z_col = _f("shot_location_z")
    tech_col = _f("pass.technique.name", "pass_technique")
    height_col = _f("pass.height.name", "pass_height")
    body_col = _f("pass.body_part.name", "pass_body_part")
    shot_body_col = _f("shot.body_part.name", "shot_body_part")
    shot_outcome_col = _f("shot.outcome.name", "shot_outcome")

    required = {
        "match_id": match_id_col,
        "match": match_col,
        "team": team_col,
        "minute": minute_col,
        "second": second_col,
    }
    missing = [k for k, v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    rename_map = {
        match_id_col: "match_id",
        match_col: "Match",
        team_col: "corner_team",
        minute_col: "Minute",
        second_col: "Second",
    }

    optional_map = {
        outcome_col: "SP_outcome",
        xg_col: "shot_xg",
        taker_col: "Taker",
        shooter_col: "Shooter",
        def_setup_col: "Defensive_setup",
        pass_x_col: "pass_location_x",
        pass_y_col: "pass_location_y",
        pass_ex_col: "pass_end_location_x",
        pass_ey_col: "pass_end_location_y",
        shot_x_col: "shot_location_x",
        shot_y_col: "shot_location_y",
        shot_z_col: "shot_location_z",
        tech_col: "pass_technique",
        height_col: "pass_height",
        body_col: "pass_body_part",
        shot_body_col: "shot_body_part",
        shot_outcome_col: "shot_outcome",
    }

    for src, dst in optional_map.items():
        if src is not None:
            rename_map[src] = dst

    df = df.rename(columns=rename_map)

    defaults = [
        "SP_outcome", "shot_xg", "Taker", "Shooter", "Defensive_setup",
        "pass_location_x", "pass_location_y", "pass_end_location_x", "pass_end_location_y",
        "shot_location_x", "shot_location_y", "shot_location_z",
        "pass_technique", "pass_height", "pass_body_part",
        "shot_body_part", "shot_outcome"
    ]
    for c in defaults:
        if c not in df.columns:
            df[c] = np.nan

    for c in [
        "Minute", "Second", "shot_xg",
        "pass_location_x", "pass_location_y",
        "pass_end_location_x", "pass_end_location_y",
        "shot_location_x", "shot_location_y", "shot_location_z"
    ]:
        df[c] = safe_numeric(df[c])

    df["corner_team"] = df["corner_team"].astype(str).str.strip()
    df["event_minute"] = df["Minute"].fillna(0) + df["Second"].fillna(0) / 60

    homes, aways = zip(*[split_match_name(m) for m in df["Match"]])
    df["home_team"] = list(homes)
    df["away_team"] = list(aways)
    df["is_home_corner"] = df["corner_team"] == df["home_team"]
    df["is_away_corner"] = df["corner_team"] == df["away_team"]
    df["venue_split"] = np.where(df["is_home_corner"], "Home", np.where(df["is_away_corner"], "Away", "Unknown"))

    sp = df["SP_outcome"].astype(str)
    df["led_to_shot"] = sp.str.contains("shot", case=False, na=False)
    df["led_to_shot"] = sp.str.contains("shot", case=False, na=False) & ~sp.str.contains("no shot", case=False, na=False)
    df["is_fast_shot"] = sp.str.contains("within 3 seconds", case=False, na=False)
    df["outcome_bucket"] = sp.apply(classify_outcome)

    df["is_inswinger"] = df["pass_technique"].astype(str).str.contains("inswing", case=False, na=False)
    df["is_outswinger"] = df["pass_technique"].astype(str).str.contains("outswing", case=False, na=False)
    df["is_short_corner"] = df["pass_technique"].astype(str).str.contains("short", case=False, na=False)

    df["side"] = df["pass_location_y"].apply(side_from_y)
    df["delivery_zone"] = df["pass_end_location_y"].apply(delivery_zone_from_y)
    df["delivery_length"] = df.apply(
        lambda r: delivery_length(
            r["pass_location_x"], r["pass_location_y"],
            r["pass_end_location_x"], r["pass_end_location_y"]
        ),
        axis=1
    )
    df["end_zone"] = df.apply(
        lambda r: zone_from_end_location(r["pass_end_location_x"], r["pass_end_location_y"]),
        axis=1,
    )
    df["is_six_yard_delivery"] = df["end_zone"].eq("6-yard box")
    df["is_penalty_area_delivery"] = df["end_zone"].eq("Penalty area")
    df["delivery_type"] = df.apply(
        lambda r: infer_delivery_type(r["pass_technique"], r["pass_height"]),
        axis=1,
    )
    df["xg_category"] = df["shot_xg"].apply(xg_category)
    df["goal_from_corner"] = df["shot_outcome"].astype(str).str.contains("goal", case=False, na=False)

    df["phase"] = pd.cut(
        df["event_minute"],
        bins=[-0.1, 15, 30, 45, 60, 75, 120],
        labels=["0-15", "16-30", "31-45", "46-60", "61-75", "76+"],
        right=True,
    ).astype(str)

    match_summary = (
        df.groupby(["match_id", "Match", "home_team", "away_team"], dropna=False)
        .agg(
            total_corners=("match_id", "size"),
            shots_from_corners=("led_to_shot", "sum"),
            fast_shots=("is_fast_shot", "sum"),
            total_xg=("shot_xg", "sum"),
            unique_takers=("Taker", pd.Series.nunique),
        )
        .reset_index()
    )

    def count_team_corners(row, team_col_name):
        team_name = row[team_col_name]
        if pd.isna(team_name):
            return np.nan
        return int(((df["match_id"] == row["match_id"]) & (df["corner_team"] == team_name)).sum())

    match_summary["home_corners"] = match_summary.apply(lambda r: count_team_corners(r, "home_team"), axis=1)
    match_summary["away_corners"] = match_summary.apply(lambda r: count_team_corners(r, "away_team"), axis=1)
    match_summary["shot_rate"] = match_summary["shots_from_corners"] / match_summary["total_corners"].replace(0, np.nan)
    match_summary["xg_per_corner"] = match_summary["total_xg"] / match_summary["total_corners"].replace(0, np.nan)
    match_summary["corner_diff"] = match_summary["home_corners"] - match_summary["away_corners"]

    return df, match_summary

def build_team_summary(df):
    if df.empty:
        return pd.DataFrame(columns=[
            "team", "corners_taken", "matches", "shots_from_corners", "fast_shots",
            "total_xg", "taker_variety", "inswingers", "outswingers", "short_corners",
            "six_yard_deliveries", "penalty_area_deliveries", "corners_per_match",
            "shot_rate", "fast_shot_rate", "xg_per_match", "xg_per_corner",
            "six_yard_delivery_rate", "penalty_area_delivery_rate",
            "short_corner_rate", "inswinger_rate", "outswinger_rate"
        ])

    out = (
        df.groupby("corner_team", dropna=False)
        .agg(
            corners_taken=("match_id", "size"),
            matches=("match_id", pd.Series.nunique),
            shots_from_corners=("led_to_shot", "sum"),
            fast_shots=("is_fast_shot", "sum"),
            total_xg=("shot_xg", "sum"),
            taker_variety=("Taker", pd.Series.nunique),
            inswingers=("is_inswinger", "sum"),
            outswingers=("is_outswinger", "sum"),
            short_corners=("is_short_corner", "sum"),
            six_yard_deliveries=("is_six_yard_delivery", "sum"),
            penalty_area_deliveries=("is_penalty_area_delivery", "sum"),
        )
        .reset_index()
        .rename(columns={"corner_team": "team"})
    )

    denom = out["corners_taken"].replace(0, np.nan)
    out["corners_per_match"] = out["corners_taken"] / out["matches"].replace(0, np.nan)
    out["shot_rate"] = out["shots_from_corners"] / denom
    out["fast_shot_rate"] = out["fast_shots"] / denom
    out["xg_per_match"] = out["total_xg"] / out["matches"].replace(0, np.nan)
    out["xg_per_corner"] = out["total_xg"] / denom
    out["six_yard_delivery_rate"] = out["six_yard_deliveries"] / denom
    out["penalty_area_delivery_rate"] = out["penalty_area_deliveries"] / denom
    out["short_corner_rate"] = out["short_corners"] / denom
    out["inswinger_rate"] = out["inswingers"] / denom
    out["outswinger_rate"] = out["outswingers"] / denom
    return out

def taker_summary(df):
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby(["corner_team", "Taker"], dropna=False)
        .agg(
            corners=("match_id", "size"),
            shots=("led_to_shot", "sum"),
            fast_shots=("is_fast_shot", "sum"),
            goals=("goal_from_corner", "sum"),
            total_xg=("shot_xg", "sum"),
            inswingers=("is_inswinger", "sum"),
            short_corners=("is_short_corner", "sum"),
            six_yard_deliveries=("is_six_yard_delivery", "sum"),
        )
        .reset_index()
    )
    d = out["corners"].replace(0, np.nan)
    out["shot_rate"] = out["shots"] / d
    out["xg_per_corner"] = out["total_xg"] / d
    out["goal_rate"] = out["goals"] / d
    out["inswinger_rate"] = out["inswingers"] / d
    return out.sort_values(["corners", "total_xg"], ascending=False)

# =========================================================
# CHARTS
# =========================================================
def delivery_map_figure(df_events, color_col="delivery_zone", title="Delivery Map", side_focus="Both"):
    fig = draw_pitch(go.Figure(), title=title, height=700, half=False)
    
    # Filter for delivery end locations
    plot = df_events.dropna(subset=["pass_end_location_x", "pass_end_location_y"]).copy()
    if plot.empty: return fig

    for category, sub in plot.groupby(color_col, dropna=False):
        fig.add_trace(go.Scatter(
            x=80 - sub["pass_end_location_y"], # Width (Horizontal on screen)
            y=sub["pass_end_location_x"],      # Length (Vertical on screen)
            mode="markers",                    # NO LINES - ONLY END LOCATION
            name=str(category),
            marker=dict(size=14, opacity=0.8, line=dict(width=1, color='white')),
            text=[
                f"<b>Match:</b> {r.get('Match','')}<br>"
                f"<b>Outcome:</b> {r.get('SP_outcome', 'N/A')}<br>" # Added the label here
                f"<b>Taker:</b> {r.get('Taker','')}"
                for _, r in sub.iterrows()
            ],
            hovertemplate="%{text}<extra></extra>"
        ))
    return fig

def delivery_map_figure(df_events, color_col="delivery_zone", title="Delivery Map", side_focus="Both"):
    fig = draw_pitch(go.Figure(), title=title, height=700, half=False)
    plot = df_events.dropna(
        subset=["pass_location_x", "pass_location_y", "pass_end_location_x", "pass_end_location_y"]
    ).copy()

    if plot.empty:
        return annotate_side(fig, side_focus)

    legend_added = set()
    for _, row in plot.iterrows():
        category = str(row.get(color_col, "Unknown"))
        show = category not in legend_added
        if show: legend_added.add(category)

        fig.add_trace(
            go.Scatter(
                x=[80 - row["pass_end_location_y"]],
                y=[row["pass_end_location_x"]],
                mode="markers",
                name=category,
                showlegend=show,
                legendgroup=category,
                marker=dict(size=12, opacity=0.8, line=dict(width=1, color='white')),
                # UPDATED TEXT HERE
                text=(
                    f"<b>{row.get('Match','')}</b><br>"
                    f"Team: {row.get('corner_team','')}<br>"
                    f"Outcome: {row.get('SP_outcome', 'N/A')}<br>" # This adds the label
                    f"Taker: {row.get('Taker','')}<br>"
                    f"Side: {row.get('side','')}"
                ),
                hovertemplate="%{text}<extra></extra>",
            )
        )
    return annotate_side(fig, side_focus)

def outcome_pie(df, title="Outcome Split"):
    if df.empty:
        return go.Figure()
    s = df.groupby("outcome_bucket", dropna=False).size().reset_index(name="n")
    fig = px.pie(s, names="outcome_bucket", values="n", hole=0.55, title=title, color_discrete_sequence=QUAL_PALETTE)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return figure_layout(fig, 380, title)

def technique_pie(df, title="Technique Split"):
    if df.empty:
        return go.Figure()
    s = df.groupby("pass_technique", dropna=False).size().reset_index(name="n")
    fig = px.pie(s, names="pass_technique", values="n", hole=0.55, title=title, color_discrete_sequence=QUAL_PALETTE)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return figure_layout(fig, 380, title)

def team_scatter(df, x, y, size, title):
    if df.empty:
        return go.Figure()
    fig = px.scatter(
        df, x=x, y=y, size=size, text="team", hover_name="team",
        title=title, color_discrete_sequence=[ACCENT],
    )
    fig.update_traces(textposition="top center")
    if not df[x].dropna().empty:
        fig.add_vline(x=df[x].median(), line_dash="dot", line_color="rgba(255,255,255,0.18)")
    if not df[y].dropna().empty:
        fig.add_hline(y=df[y].median(), line_dash="dot", line_color="rgba(255,255,255,0.18)")
    return figure_layout(fig, 420, title)

def cumulative_line(df, title="Cumulative Corners", group_col="corner_team"):
    if df.empty:
        return go.Figure()
    base = (
        df.groupby(["Minute", group_col], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values([group_col, "Minute"])
    )
    base["cum"] = base.groupby(group_col)["n"].cumsum()
    fig = px.line(base, x="Minute", y="cum", color=group_col, markers=True, title=title)
    return figure_layout(fig, 380, title)

def minute_histogram(df, title="Minute Distribution", color_col=None):
    if df.empty:
        return go.Figure()
    if color_col:
        fig = px.histogram(df, x="Minute", color=color_col, nbins=24, barmode="stack", title=title, color_discrete_sequence=QUAL_PALETTE)
    else:
        fig = px.histogram(df, x="Minute", nbins=24, title=title, color_discrete_sequence=[ACCENT])
    fig.update_traces(opacity=0.85)
    return figure_layout(fig, 360, title)

def phase_heatmap(df, title="Phase Heatmap"):
    if df.empty:
        return go.Figure()
    tmp = df.groupby(["corner_team", "phase"], dropna=False).size().reset_index(name="n")
    pivot = tmp.pivot(index="corner_team", columns="phase", values="n").fillna(0)
    order = [p for p in ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"] if p in pivot.columns]
    pivot = pivot.reindex(columns=order)
    fig = px.imshow(pivot, aspect="auto", title=title, color_continuous_scale="Blues", text_auto=True)
    return figure_layout(fig, max(380, 42 * len(pivot)), title)

def taker_bar_chart(df, metric, title, top_n=12, min_corners=3):
    if df.empty or metric not in df.columns:
        return go.Figure()
    plot = df[df["corners"] >= min_corners].copy()
    if plot.empty:
        return go.Figure()
    plot["label"] = plot["Taker"].astype(str) + " (" + plot["corner_team"].astype(str) + ")"
    plot = plot.sort_values(metric, ascending=False).head(top_n)
    fig = px.bar(
        plot,
        x="label",
        y=metric,
        color=metric,
        color_continuous_scale="Blues",
        title=title,
        hover_data=["corners", "shots", "total_xg"],
    )
    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35)
    return figure_layout(fig, 380, title)

def rolling_shot_rate(df, window=5, title="Rolling Shot Rate"):
    if df.empty:
        return go.Figure()
    base = (
        df.groupby(["Match", "corner_team"], dropna=False)
        .agg(corners=("match_id", "size"), shots=("led_to_shot", "sum"))
        .reset_index()
        .sort_values(["corner_team", "Match"])
    )
    base["shot_rate"] = base["shots"] / base["corners"].replace(0, np.nan)
    fig = go.Figure()
    for team, grp in base.groupby("corner_team"):
        grp = grp.copy().reset_index(drop=True)
        grp["rolling"] = grp["shot_rate"].rolling(min(window, len(grp)), min_periods=1).mean()
        fig.add_trace(
            go.Scatter(
                x=grp["Match"],
                y=grp["rolling"],
                mode="lines+markers",
                name=str(team),
                hovertemplate=f"<b>{team}</b><br>Match: %{{x}}<br>Rolling SR: %{{y:.1%}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=title,
        height=400,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=48, b=8),
        font=dict(color=TEXT),
        hoverlabel=dict(bgcolor="#0d1c31", font_color=TEXT),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickformat=".0%")
    return fig

# =========================================================
# EXPORTS
# =========================================================
def download_excel_workbook(events_df, team_df, match_df, taker_df):
    buf = BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            events_df.to_excel(writer, sheet_name="Events", index=False)
            team_df.to_excel(writer, sheet_name="Teams", index=False)
            match_df.to_excel(writer, sheet_name="Matches", index=False)
            taker_df.to_excel(writer, sheet_name="Takers", index=False)
        return buf.getvalue()
    except Exception:
        return None

# =========================================================
# LOAD DATA
# =========================================================
try:
    raw_df = load_data()
    df, match_summary = prepare_data(raw_df)
except Exception as e:
    st.error("Failed to load data.")
    st.exception(e)
    st.stop()

# =========================================================
# HEADER
# =========================================================
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-title">⚽ <span>Allsvenskan</span> Set Piece Studio Pro</div>
        <div class="hero-sub">A cleaner corner analysis workspace for coaches and analysts — faster reads, clearer visuals, better side filtering.</div>
        <div>
            <span class="pill">Executive</span>
            <span class="pill">Visuals</span>
            <span class="pill">Teams</span>
            <span class="pill">Matches</span>
            <span class="pill">Scouting</span>
            <span class="pill">Exports</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR
# =========================================================
all_teams = sorted([t for t in df["corner_team"].dropna().unique() if str(t).strip()])
all_takers = sorted([str(t) for t in df["Taker"].dropna().astype(str).unique() if str(t).strip()])
all_matches = sorted(df["Match"].dropna().astype(str).unique().tolist())
all_setups = sorted([str(x) for x in df["Defensive_setup"].dropna().astype(str).unique() if str(x).strip()])

with st.sidebar:
    st.markdown("### 🎛 Studio Controls")

    page = st.radio(
        "Workspace",
        [
            "🏠 Executive Dashboard",
            "📊 Visualisation Studio",
            "🏟 Team Analysis",
            "🔍 Match Explorer",
            "👤 Scouting Center",
            "📈 Trend Lab",
            "🗂 Data Hub",
        ],
    )

    st.markdown("---")
    st.markdown("**Quick Filters**")

    sel_team = st.selectbox("Team", ["All Teams"] + all_teams)
    sel_matches = st.multiselect("Matches", all_matches)
    side_focus = st.radio(
        "Corner Side",
        ["Both", "Left", "Right"],
        horizontal=True,
        help="Filter corners by the side they were taken from, and show that state on visuals.",
    )

    with st.expander("Advanced filters", expanded=False):
        sel_takers = st.multiselect("Taker(s)", all_takers)

        minute_min = int(df["Minute"].min()) if not df["Minute"].dropna().empty else 0
        minute_max = int(df["Minute"].max()) if not df["Minute"].dropna().empty else 0
        min_range = safe_range_slider("Minute Range", minute_min, minute_max)

        match_corner_min = int(match_summary["total_corners"].min()) if not match_summary.empty else 0
        match_corner_max = int(match_summary["total_corners"].max()) if not match_summary.empty else 0
        corner_range = safe_range_slider("Match Corner Range", match_corner_min, match_corner_max)

        zone_filter = st.multiselect(
            "Delivery Zone",
            ["Near Post Zone", "Central Zone", "Far Post Zone", "Unknown"],
            default=["Near Post Zone", "Central Zone", "Far Post Zone", "Unknown"],
        )
        end_zone_filter = st.multiselect(
            "End Zone",
            ["6-yard box", "Penalty area", "Deep box", "Outside danger zone", "Unknown"],
            default=["6-yard box", "Penalty area", "Deep box", "Outside danger zone", "Unknown"],
        )
        venue_filter = st.multiselect(
            "Home / Away",
            ["Home", "Away", "Unknown"],
            default=["Home", "Away", "Unknown"],
        )
        phase_filter = st.multiselect(
            "Phase",
            ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"],
            default=["0-15", "16-30", "31-45", "46-60", "61-75", "76+"],
        )
        setup_filter = st.multiselect("Defensive Setup", all_setups)

        st.markdown("**Event type**")
        shot_only = st.checkbox("Shot outcomes only")
        inswing_only = st.checkbox("Inswingers only")
        outswing_only = st.checkbox("Outswingers only")
        short_only = st.checkbox("Short corners only")

# defaults when expander state not interacted with yet
minute_min = int(df["Minute"].min()) if not df["Minute"].dropna().empty else 0
minute_max = int(df["Minute"].max()) if not df["Minute"].dropna().empty else 0
min_range = locals().get("min_range", (minute_min, minute_max))
match_corner_min = int(match_summary["total_corners"].min()) if not match_summary.empty else 0
match_corner_max = int(match_summary["total_corners"].max()) if not match_summary.empty else 0
corner_range = locals().get("corner_range", (match_corner_min, match_corner_max))
sel_takers = locals().get("sel_takers", [])
zone_filter = locals().get("zone_filter", ["Near Post Zone", "Central Zone", "Far Post Zone", "Unknown"])
end_zone_filter = locals().get("end_zone_filter", ["6-yard box", "Penalty area", "Deep box", "Outside danger zone", "Unknown"])
venue_filter = locals().get("venue_filter", ["Home", "Away", "Unknown"])
phase_filter = locals().get("phase_filter", ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"])
setup_filter = locals().get("setup_filter", [])
shot_only = locals().get("shot_only", False)
inswing_only = locals().get("inswing_only", False)
outswing_only = locals().get("outswing_only", False)
short_only = locals().get("short_only", False)

# =========================================================
# FILTERS
# =========================================================
def apply_filters(events, matches):
    out_matches = matches[
        matches["total_corners"].between(corner_range[0], corner_range[1])
    ].copy()

    out_events = events[events["match_id"].isin(out_matches["match_id"].unique())].copy()
    out_events = out_events[out_events["Minute"].fillna(0).between(min_range[0], min_range[1])]

    effective_side_filter = ["Left", "Right", "Unknown"]
    if side_focus == "Left":
        effective_side_filter = ["Left"]
    elif side_focus == "Right":
        effective_side_filter = ["Right"]

    filter_map = {
        "corner_team": None if sel_team == "All Teams" else [sel_team],
        "Taker": sel_takers if sel_takers else None,
        "Match": sel_matches if sel_matches else None,
        "side": effective_side_filter,
        "delivery_zone": zone_filter if zone_filter else None,
        "end_zone": end_zone_filter if end_zone_filter else None,
        "venue_split": venue_filter if venue_filter else None,
        "phase": phase_filter if phase_filter else None,
        "Defensive_setup": setup_filter if setup_filter else None,
    }

    for col, allowed in filter_map.items():
        if allowed is not None:
            out_events = out_events[out_events[col].astype(str).isin([str(x) for x in allowed])]

    if shot_only:
        out_events = out_events[out_events["led_to_shot"]]
    if short_only:
        out_events = out_events[out_events["is_short_corner"]]
    if inswing_only and not outswing_only:
        out_events = out_events[out_events["is_inswinger"]]
    if outswing_only and not inswing_only:
        out_events = out_events[out_events["is_outswinger"]]

    out_matches = out_matches[out_matches["match_id"].isin(out_events["match_id"].unique())]
    out_teams = build_team_summary(out_events)
    out_takers = taker_summary(out_events)
    return out_events, out_matches, out_teams, out_takers

league_event_df, league_match_df, league_team_df, league_taker_df = apply_filters(df, match_summary)

filter_chips(
    sel_team,
    len(sel_matches),
    len(sel_takers),
    side_focus,
    venue_filter,
)

# =========================================================
# KPI ROW
# =========================================================
def render_kpis(events):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_xg = events["shot_xg"].fillna(0).sum() if not events.empty else 0
    shot_rate = events["led_to_shot"].mean() if not events.empty else 0
    goals = int(events["goal_from_corner"].sum()) if not events.empty else 0

    with c1:
        metric_card("Events", f"{len(events):,}", "Corner actions")
    with c2:
        metric_card("Matches", f"{events['match_id'].nunique() if not events.empty else 0:,}", "Unique matches")
    with c3:
        metric_card("Shots", f"{int(events['led_to_shot'].sum()) if not events.empty else 0:,}", "From corners")
    with c4:
        metric_card("Total xG", f"{total_xg:.2f}", "xG generated")
    with c5:
        metric_card("Shot Rate", f"{shot_rate*100:.1f}%", "Shots / corner")
    with c6:
        metric_card("Goals", f"{goals}", "From corners")

# =========================================================
# PAGES
# =========================================================
if page == "🏠 Executive Dashboard":
    render_kpis(league_event_df)
    st.markdown("<br>", unsafe_allow_html=True)

    insights = top_insights(league_team_df)
    if insights:
        cols = st.columns(len(insights))
        for i, (title, body) in enumerate(insights):
            with cols[i]:
                insight_box(title, body)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        section_header("Corner Volume by Team")
        if not league_team_df.empty:
            fig = px.bar(
                league_team_df.sort_values("corners_taken", ascending=False),
                x="team",
                y="corners_taken",
                color="corners_per_match",
                color_continuous_scale="Blues",
                labels={"team": "", "corners_taken": "Corners"},
                hover_data=["matches", "shot_rate", "xg_per_match"],
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(figure_layout(fig, 400, "Corner Volume by Team"), use_container_width=True)
        else:
            empty_state()

    with c2:
        section_header("Efficiency Map")
        if not league_team_df.empty:
            st.plotly_chart(
                team_scatter(league_team_df, "shot_rate", "xg_per_match", "corners_taken", "Shot Rate vs xG/Match"),
                use_container_width=True,
            )
        else:
            empty_state()

    section_header("Executive Match Board")
    if not league_match_df.empty:
        show_cols = [
            c for c in [
                "Match", "home_team", "away_team", "home_corners", "away_corners",
                "total_corners", "shots_from_corners", "fast_shots", "total_xg",
                "shot_rate", "xg_per_corner", "unique_takers", "corner_diff"
            ] if c in league_match_df.columns
        ]
        st.dataframe(
            league_match_df[show_cols].sort_values(["total_xg", "total_corners"], ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=420,
        )
    else:
        empty_state()

    with st.expander("More breakdowns", expanded=False):
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(outcome_pie(league_event_df), use_container_width=True)
        with c4:
            st.plotly_chart(technique_pie(league_event_df), use_container_width=True)

elif page == "📊 Visualisation Studio":
    section_header("Visualisation Studio", "Cleaner match visuals with side-aware filtering.")
    visual_context_note(side_focus, len(league_event_df))

    tabs = st.tabs(["🎯 Shots", "🏹 Deliveries", "↔ Side Comparison", "⏱ Timing"])

    with tabs[0]:
        shot_df = league_event_df.dropna(subset=["shot_location_x", "shot_location_y"])
        if shot_df.empty:
            st.info("No shot data available.")
        else:
            st.plotly_chart(
                shotmap_figure(shot_df, title="Shot Locations & xG", side_focus=side_focus),
                use_container_width=True,
            )

    with tabs[1]:
        delivery_df = league_event_df.dropna(subset=["pass_end_location_x", "pass_end_location_y"])
        if delivery_df.empty:
            st.info("No delivery landing data available.")
        else:
            st.plotly_chart(
                delivery_map_figure(delivery_df, title="Delivery Landing Points", side_focus=side_focus),
                use_container_width=True,
            )
    with tabs[2]:
        left_df = league_event_df[league_event_df["side"] == "Left"]
        right_df = league_event_df[league_event_df["side"] == "Right"]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Left-side corners")
            if left_df.empty:
                empty_state("No left-side corners in current filters.")
            else:
                st.plotly_chart(
                    delivery_map_figure(left_df, "delivery_zone", "Left-side Delivery Map", side_focus="Left"),
                    use_container_width=True,
                )
        with c2:
            st.markdown("#### Right-side corners")
            if right_df.empty:
                empty_state("No right-side corners in current filters.")
            else:
                st.plotly_chart(
                    delivery_map_figure(right_df, "delivery_zone", "Right-side Delivery Map", side_focus="Right"),
                    use_container_width=True,
                )

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(cumulative_line(league_event_df, "Cumulative Corners by Team"), use_container_width=True)
        with c2:
            st.plotly_chart(minute_histogram(league_event_df, "Minute Distribution by Team", color_col="corner_team"), use_container_width=True)
        st.plotly_chart(phase_heatmap(league_event_df, "Team × Phase Heatmap"), use_container_width=True)

elif page == "🏟 Team Analysis":
    if sel_team == "All Teams":
        st.info("Select a team in the sidebar for full team analysis.")
        if not league_team_df.empty:
            st.dataframe(league_team_df.sort_values("xg_per_match", ascending=False).reset_index(drop=True), use_container_width=True, height=560)
        else:
            empty_state()
    else:
        team_ev = league_event_df[league_event_df["corner_team"] == sel_team].copy()
        team_row = league_team_df[league_team_df["team"] == sel_team]
        team_takers = taker_summary(team_ev)

        render_kpis(team_ev)
        tabs = st.tabs(["📊 Overview", "🎯 Visuals", "👤 Takers", "📋 Matches", "🏆 Report Card"])

        with tabs[0]:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(outcome_pie(team_ev, f"{sel_team} — Outcome Split"), use_container_width=True)
            with c2:
                st.plotly_chart(technique_pie(team_ev, f"{sel_team} — Technique Split"), use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                zone_df = team_ev.groupby("end_zone", dropna=False).size().reset_index(name="n")
                fig = px.bar(zone_df, x="end_zone", y="n", color="end_zone", title="End-Zone Profile", color_discrete_sequence=QUAL_PALETTE)
                st.plotly_chart(figure_layout(fig, 360, "End-Zone Profile"), use_container_width=True)
            with c4:
                side_df = team_ev.groupby("side", dropna=False).size().reset_index(name="n")
                fig = px.pie(side_df, names="side", values="n", title="Side Split", hole=0.55, color_discrete_sequence=QUAL_PALETTE)
                fig.update_traces(textposition="outside", textinfo="percent+label")
                st.plotly_chart(figure_layout(fig, 360, "Side Split"), use_container_width=True)

        with tabs[1]:
            shot_df = team_ev.dropna(subset=["shot_location_x", "shot_location_y"])
            del_df = team_ev.dropna(subset=["pass_location_x", "pass_location_y", "pass_end_location_x", "pass_end_location_y"])
            c1, c2 = st.columns(2)
            with c1:
                if shot_df.empty:
                    empty_state("No shot location data.")
                else:
                    st.plotly_chart(shotmap_figure(shot_df, "Taker", f"Shotmap — {sel_team}", side_focus=side_focus), use_container_width=True)
            with c2:
                if del_df.empty:
                    empty_state("No delivery coordinate data.")
                else:
                    st.plotly_chart(delivery_map_figure(del_df, "delivery_zone", f"Delivery Map — {sel_team}", side_focus=side_focus), use_container_width=True)

        with tabs[2]:
            if team_takers.empty:
                empty_state("No taker data.")
            else:
                st.dataframe(team_takers.reset_index(drop=True), use_container_width=True, height=380)
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(taker_bar_chart(team_takers, "xg_per_corner", f"Takers by xG/Corner — {sel_team}", min_corners=2), use_container_width=True)
                with c2:
                    st.plotly_chart(taker_bar_chart(team_takers, "shot_rate", f"Takers by Shot Rate — {sel_team}", min_corners=2), use_container_width=True)

        with tabs[3]:
            match_view = (
                team_ev.groupby(["Match", "venue_split"], dropna=False)
                .agg(
                    corners=("match_id", "size"),
                    shots=("led_to_shot", "sum"),
                    total_xg=("shot_xg", "sum"),
                    fast_shots=("is_fast_shot", "sum"),
                )
                .reset_index()
            )
            match_view["shot_rate"] = match_view["shots"] / match_view["corners"].replace(0, np.nan)
            match_view["xg_per_corner"] = match_view["total_xg"] / match_view["corners"].replace(0, np.nan)
            st.dataframe(match_view.sort_values("total_xg", ascending=False).reset_index(drop=True), use_container_width=True, height=420)
            st.plotly_chart(rolling_shot_rate(team_ev, title=f"Rolling Shot Rate — {sel_team}"), use_container_width=True)

        with tabs[4]:
            if team_row.empty:
                empty_state()
            else:
                row = team_row.iloc[0]
                c1, c2, c3 = st.columns(3)
                with c1:
                    metric_card("Corners/Match", human_val(row.get("corners_per_match")), "Volume")
                with c2:
                    metric_card("Shot Rate", human_pct(row.get("shot_rate")), "Shots per corner")
                with c3:
                    metric_card("xG/Match", human_val(row.get("xg_per_match"), 3), "Chance quality")

                st.markdown("<br>", unsafe_allow_html=True)
                metric_map = [
                    ("Corners/Match", "corners_per_match"),
                    ("Shot Rate", "shot_rate"),
                    ("xG/Match", "xg_per_match"),
                    ("6Y Delivery Rate", "six_yard_delivery_rate"),
                    ("Short Corner Rate", "short_corner_rate"),
                    ("Inswinger Rate", "inswinger_rate"),
                ]
                for label, col in metric_map:
                    val = row.get(col)
                    pct = percentile_rank(league_team_df[col], val) if col in league_team_df.columns else np.nan
                    display = human_pct(val) if "Rate" in label else human_val(val, 3)
                    st.markdown(
                        f"**{label}**: {display} · {pct:.0f}th percentile"
                        if not pd.isna(pct) else f"**{label}**: {display}"
                    )

elif page == "🔍 Match Explorer":
    match_options = sorted(league_match_df["Match"].dropna().unique().tolist()) if not league_match_df.empty else []
    sel_match = st.selectbox("Select Match", ["All Matches"] + match_options)

    match_ev = league_event_df.copy()
    match_m = league_match_df.copy()

    if sel_match != "All Matches":
        match_m = match_m[match_m["Match"] == sel_match]
        match_ev = match_ev[match_ev["match_id"].isin(match_m["match_id"].unique())]

    render_kpis(match_ev)
    tabs = st.tabs(["📋 Summary", "⏱ Timeline", "🎯 Shotmap", "🏹 Delivery", "🔴 Event Feed"])

    with tabs[0]:
        if match_m.empty:
            empty_state()
        else:
            st.dataframe(match_m.reset_index(drop=True), use_container_width=True, height=280)
            breakdown = build_team_summary(match_ev)
            if not breakdown.empty:
                st.dataframe(breakdown.reset_index(drop=True), use_container_width=True, height=260)

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(minute_histogram(match_ev, "Minute Distribution", color_col="corner_team"), use_container_width=True)
        with c2:
            st.plotly_chart(cumulative_line(match_ev, "Cumulative Corners"), use_container_width=True)

    with tabs[2]:
        shot_df = match_ev.dropna(subset=["shot_location_x", "shot_location_y"])
        if shot_df.empty:
            empty_state("No shot data.")
        else:
            st.plotly_chart(shotmap_figure(shot_df, "corner_team", f"Shotmap — {sel_match}", side_focus=side_focus), use_container_width=True)

    with tabs[3]:
        del_df = match_ev.dropna(subset=["pass_location_x", "pass_location_y", "pass_end_location_x", "pass_end_location_y"])
        if del_df.empty:
            empty_state("No delivery data.")
        else:
            st.plotly_chart(delivery_map_figure(del_df, "delivery_zone", f"Delivery Map — {sel_match}", side_focus=side_focus), use_container_width=True)

    with tabs[4]:
        show_cols = [
            c for c in [
                "Match", "corner_team", "Taker", "Shooter", "Minute", "Second",
                "SP_outcome", "shot_xg", "pass_technique", "side",
                "delivery_zone", "end_zone", "phase"
            ] if c in match_ev.columns
        ]
        st.dataframe(match_ev[show_cols].sort_values(["Minute", "Second"]).reset_index(drop=True), use_container_width=True, height=560)

elif page == "👤 Scouting Center":
    tabs = st.tabs(["🏅 Teams", "👤 Takers", "🔀 Compare", "🛡 Scout Report"])

    with tabs[0]:
        if league_team_df.empty:
            empty_state()
        else:
            ranked = league_team_df.sort_values(["xg_per_match", "shot_rate"], ascending=False).reset_index(drop=True)
            ranked.index += 1
            ranked.insert(0, "#", ranked.index)
            st.dataframe(ranked, use_container_width=True, height=520)

    with tabs[1]:
        if league_taker_df.empty:
            empty_state("No taker data.")
        else:
            min_corners_scout = st.number_input("Minimum corners", 1, 50, 5)
            filt = league_taker_df[league_taker_df["corners"] >= min_corners_scout]
            st.dataframe(filt.reset_index(drop=True), use_container_width=True, height=460)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(taker_bar_chart(filt, "xg_per_corner", "Top Takers — xG/Corner", min_corners=min_corners_scout), use_container_width=True)
            with c2:
                st.plotly_chart(taker_bar_chart(filt, "shot_rate", "Top Takers — Shot Rate", min_corners=min_corners_scout), use_container_width=True)

    with tabs[2]:
        teams = sorted(league_team_df["team"].dropna().unique().tolist()) if not league_team_df.empty else []
        if len(teams) < 2:
            empty_state("Need at least 2 teams.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                team_a = st.selectbox("Team A", teams, key="cmp_a")
            with c2:
                team_b = st.selectbox("Team B", teams, index=min(1, len(teams) - 1), key="cmp_b")

            a = league_team_df[league_team_df["team"] == team_a].iloc[0]
            b = league_team_df[league_team_df["team"] == team_b].iloc[0]

            comp = pd.DataFrame({
                "Metric": ["Corners/Match", "Shot Rate", "xG/Match", "6Y Delivery Rate", "Short Corner Rate", "Inswinger Rate"],
                team_a: [
                    a["corners_per_match"], a["shot_rate"], a["xg_per_match"],
                    a["six_yard_delivery_rate"], a["short_corner_rate"], a["inswinger_rate"],
                ],
                team_b: [
                    b["corners_per_match"], b["shot_rate"], b["xg_per_match"],
                    b["six_yard_delivery_rate"], b["short_corner_rate"], b["inswinger_rate"],
                ],
            })
            st.dataframe(comp, use_container_width=True, height=300)

            ev_a = league_event_df[league_event_df["corner_team"] == team_a]
            ev_b = league_event_df[league_event_df["corner_team"] == team_b]
            c3, c4 = st.columns(2)
            with c3:
                shot_a = ev_a.dropna(subset=["shot_location_x", "shot_location_y"])
                if shot_a.empty:
                    empty_state(f"No shots — {team_a}")
                else:
                    st.plotly_chart(shotmap_figure(shot_a, "Taker", f"Shotmap — {team_a}", side_focus=side_focus), use_container_width=True)
            with c4:
                shot_b = ev_b.dropna(subset=["shot_location_x", "shot_location_y"])
                if shot_b.empty:
                    empty_state(f"No shots — {team_b}")
                else:
                    st.plotly_chart(shotmap_figure(shot_b, "Taker", f"Shotmap — {team_b}", side_focus=side_focus), use_container_width=True)

    with tabs[3]:
        scout_teams = sorted(league_team_df["team"].dropna().unique().tolist()) if not league_team_df.empty else []
        if not scout_teams:
            empty_state()
        else:
            opp = st.selectbox("Select Opposition", scout_teams)
            opp_ev = league_event_df[league_event_df["corner_team"] == opp]
            opp_row = league_team_df[league_team_df["team"] == opp]
            opp_takers = taker_summary(opp_ev)

            if opp_ev.empty or opp_row.empty:
                empty_state("Not enough data.")
            else:
                r = opp_row.iloc[0]
                dom_tech = opp_ev["pass_technique"].value_counts()
                dom_zone = opp_ev["end_zone"].value_counts()
                dom_side = opp_ev["side"].value_counts()
                top_taker = opp_takers.iloc[0]["Taker"] if not opp_takers.empty else "Unknown"

                zone_recommendation = (
                    "Protect the 6-yard area aggressively."
                    if (not dom_zone.empty and dom_zone.index[0] == "6-yard box")
                    else "Standard zonal coverage is acceptable."
                )
                short_recommendation = (
                    " Watch for short corners."
                    if r.get("short_corner_rate", 0) > 0.15
                    else ""
                )
                inswing_recommendation = (
                    " Prepare for inswinging delivery patterns."
                    if r.get("inswinger_rate", 0) > 0.30
                    else ""
                )

                st.markdown(
                    f"""
                    <div class="insight-box">
                        <div style="font-size:1.2rem;font-weight:800;margin-bottom:10px">Scout Report — {opp}</div>
                        <div style="line-height:1.8;color:{MUTED}">
                            <b>Threat level:</b> {opp} generate <b>{human_pct(r.get('shot_rate'))}</b> shot rate and
                            <b>{human_val(r.get('xg_per_match'), 3)} xG/match</b> from corners.<br><br>
                            <b>Primary delivery:</b> {dom_tech.index[0] if not dom_tech.empty else 'Unknown'}<br>
                            <b>Main target zone:</b> {dom_zone.index[0] if not dom_zone.empty else 'Unknown'}<br>
                            <b>Preferred side:</b> {dom_side.index[0] if not dom_side.empty else 'Unknown'}<br>
                            <b>Primary taker:</b> {top_taker}<br><br>
                            <b>Recommendations:</b>
                            {zone_recommendation}{short_recommendation}{inswing_recommendation}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

elif page == "📈 Trend Lab":
    tabs = st.tabs(["📉 Rolling", "📊 Match Trends", "🕐 Phase Trends", "🧮 Tests"])

    with tabs[0]:
        window = st.slider("Rolling window", 2, 10, 5)
        st.plotly_chart(rolling_shot_rate(league_event_df, window=window, title=f"Rolling Shot Rate ({window}-match)"), use_container_width=True)

    with tabs[1]:
        if league_event_df.empty:
            empty_state()
        else:
            mv = (
                league_event_df.groupby(["Match", "corner_team"], dropna=False)
                .size()
                .reset_index(name="corners")
                .sort_values("Match")
            )
            fig = px.line(mv, x="Match", y="corners", color="corner_team", markers=True, title="Corners per Match", color_discrete_sequence=QUAL_PALETTE)
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(figure_layout(fig, 400, "Corners per Match"), use_container_width=True)

    with tabs[2]:
        st.plotly_chart(phase_heatmap(league_event_df, "Corner Phase Pattern"), use_container_width=True)
        ph_sr = (
            league_event_df.groupby(["phase", "corner_team"], dropna=False)
            .agg(corners=("match_id", "size"), shots=("led_to_shot", "sum"))
            .reset_index()
        )
        ph_sr["shot_rate"] = ph_sr["shots"] / ph_sr["corners"].replace(0, np.nan)
        fig = px.line(ph_sr, x="phase", y="shot_rate", color="corner_team", markers=True, title="Shot Rate Across Phases", color_discrete_sequence=QUAL_PALETTE)
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(figure_layout(fig, 380, "Shot Rate Across Phases"), use_container_width=True)

    with tabs[3]:
        section_header("Statistical Tests")
        ct_df = league_event_df[league_event_df["pass_technique"].astype(str).str.contains("swing|short", case=False, na=False)].copy()
        if len(ct_df) >= 10:
            ct_df["swing_type"] = np.where(ct_df["is_inswinger"], "Inswinger", "Outswinger/Other")
            ct_df["shot_yn"] = np.where(ct_df["led_to_shot"], "Shot", "No Shot")
            cont = pd.crosstab(ct_df["swing_type"], ct_df["shot_yn"])
            st.dataframe(cont)
            try:
                chi2, p, dof, _ = scipy_stats.chi2_contingency(cont)
                st.write(f"Chi-square = {chi2:.3f} | p = {p:.4f} | df = {dof}")
            except Exception as e:
                st.warning(f"Test failed: {e}")
        else:
            empty_state("Not enough data for chi-square test.")

elif page == "🗂 Data Hub":
    tabs = st.tabs(["📄 Events", "🏟 Teams", "📋 Matches", "👤 Takers", "⬇ Downloads"])

    with tabs[0]:
        st.dataframe(league_event_df.reset_index(drop=True), use_container_width=True, height=600)

    with tabs[1]:
        st.dataframe(league_team_df.reset_index(drop=True), use_container_width=True, height=600)

    with tabs[2]:
        st.dataframe(league_match_df.reset_index(drop=True), use_container_width=True, height=600)

    with tabs[3]:
        st.dataframe(league_taker_df.reset_index(drop=True), use_container_width=True, height=600)

    with tabs[4]:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.download_button(
                "⬇ Events CSV",
                league_event_df.to_csv(index=False).encode(),
                "events.csv",
                "text/csv",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "⬇ Teams CSV",
                league_team_df.to_csv(index=False).encode(),
                "teams.csv",
                "text/csv",
                use_container_width=True,
            )
        with c3:
            st.download_button(
                "⬇ Matches CSV",
                league_match_df.to_csv(index=False).encode(),
                "matches.csv",
                "text/csv",
                use_container_width=True,
            )

        wb = download_excel_workbook(league_event_df, league_team_df, league_match_df, league_taker_df)
        if wb:
            st.download_button(
                "⬇ Full Excel Workbook",
                wb,
                "allsvenskan_corners.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    f"""
    <div class="footer-note">
        ⚽ Allsvenskan Set Piece Studio Pro · Cleaner UX build · 2025 Season · Streamlit + Plotly
    </div>
    """,
    unsafe_allow_html=True,
)
