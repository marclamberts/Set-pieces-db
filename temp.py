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

FILE_NAME = "SWE SP.xlsx"
HOPS_FILE_NAME = "duel_hops_rating_summary.xlsx"
DELAY_FILE_NAME = "corner_delays.xlsx"

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

# =========================================================
# LOGIN
# =========================================================
def login_screen():
    st.markdown(
        """
        <div class="hero-wrap">
            <div class="hero-title">⚽ <span>Allsvenskan</span> Set Piece Studio Pro</div>
            <div class="hero-sub">Corners, free kicks and throw-ins in one analysis workspace.</div>
            <div>
                <span class="pill">2025 Season</span>
                <span class="pill">Corners</span>
                <span class="pill">Free Kicks</span>
                <span class="pill">Throw-Ins</span>
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
    if "goal" in s:
        return "Goal"
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

def infer_delivery_type(set_piece_type, height, start_x=None):
    sp = str(set_piece_type).lower()
    h = str(height).lower()
    if "corner" in sp:
        if "high" in h:
            return "Corner - High"
        if "ground" in h or "low" in h:
            return "Corner - Low"
        return "Corner - Other"
    if "free kick" in sp:
        if pd.notna(start_x) and start_x >= 95:
            return "Direct Free Kick"
        if "high" in h:
            return "Free Kick Cross"
        return "Free Kick Short"
    if "throw" in sp:
        if pd.notna(start_x) and start_x >= 85:
            return "Long Throw"
        return "Short Throw"
    return "Other"

def set_piece_bucket(sp_type):
    s = str(sp_type).lower().strip()
    if "corner" in s:
        return "Corner"
    if "free kick" in s:
        return "Free Kick"
    if "throw" in s:
        return "Throw-In"
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
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0 or pd.isna(value):
        return np.nan
    return float((s <= value).mean() * 100)

def safe_range_slider(label, min_val, max_val):
    if min_val < max_val:
        return st.slider(label, min_val, max_val, (min_val, max_val))
    st.caption(f"{label}: {min_val}")
    return (min_val, max_val)

def parse_xy(cell, idx=0):
    if pd.isna(cell):
        return np.nan
    try:
        parts = [float(str(x).strip()) for x in str(cell).split(",")]
        return parts[idx] if len(parts) > idx else np.nan
    except Exception:
        return np.nan

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

def filter_chips(team, match_count, taker_count, side_focus, venue_filter, sp_filter):
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
    if set(sp_filter) != {"Corner", "Free Kick", "Throw-In", "Other"}:
        chips.append("SP type filtered")
    if not chips:
        chips = ["All teams", "All matches", "All set-piece types"]
    html = "".join([f'<span class="pill">{c}</span>' for c in chips])
    st.markdown(f'<div style="margin:0.05rem 0 0.75rem 0">{html}</div>', unsafe_allow_html=True)

def visual_context_note(side_focus, n_events):
    label = "Both sides" if side_focus == "Both" else f"{side_focus} side only"
    st.caption(f"Showing: {label} · {n_events:,} events in current view")

# =========================================================
# PITCH & ANNOTATION HELPERS
# =========================================================
def annotate_side(fig, side_focus):
    txt = "Showing both sides" if side_focus == "Both" else f"Showing {side_focus.lower()} side only"
    fig.add_annotation(
        x=0.99, y=1.08,
        xref="paper", yref="paper",
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

def draw_pitch(fig, title=None, height=700, half=False):
    y_min = 60 if half else 0
    fig.update_xaxes(range=[0, 80], visible=False)
    fig.update_yaxes(range=[y_min, 120], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=title, height=height, template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(l=10, r=10, t=40, b=10),
        shapes=[
            dict(type="rect", x0=0, y0=y_min, x1=80, y1=120, line=dict(color="white", width=2)),
            dict(type="line", x0=0, y0=60, x1=80, y1=60, line=dict(color="white", width=1.5)) if not half else {},
            dict(type="rect", x0=18, y0=102, x1=62, y1=120, line=dict(color="white", width=1.5)),
            dict(type="rect", x0=30, y0=114, x1=50, y1=120, line=dict(color="white", width=1.5)),
            dict(type="circle", x0=39.6, y0=107.6, x1=40.4, y1=108.4, fillcolor="white", line=dict(color="white")),
            dict(type="line", x0=36, y0=120, x1=44, y1=120, line=dict(color="#00FF00", width=4)),
        ],
    )
    return fig

# =========================================================
# CHART FUNCTIONS
# =========================================================
def shotmap_figure(df_shots, color_col="set_piece_type", title="Shotmap", side_focus="Both"):
    fig = draw_pitch(go.Figure(), title=title, height=600, half=True)
    if df_shots.empty:
        return annotate_side(fig, side_focus)

    df = df_shots.copy()
    df["shot_xg"] = pd.to_numeric(df.get("shot_xg", np.nan), errors="coerce").fillna(0)
    plot_df = df.dropna(subset=["shot_location_x", "shot_location_y"])
    if plot_df.empty:
        return annotate_side(fig, side_focus)

    for grp_name, sub in plot_df.groupby(color_col, dropna=False):
        sizes = np.clip(sub["shot_xg"].fillna(0).values * 150 + 10, 10, 55)
        hover_texts = [
            f"<b>Player:</b> {row.get('Shooter', 'N/A')}<br>"
            f"<b>Team:</b> {row.get('corner_team', 'N/A')}<br>"
            f"<b>Set Piece:</b> {row.get('set_piece_type', 'N/A')}<br>"
            f"<b>xG:</b> {row.get('shot_xg', 0):.3f}"
            for _, row in sub.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=sub["shot_location_y"],
            y=sub["shot_location_x"],
            mode="markers",
            name=str(grp_name),
            marker=dict(size=sizes, opacity=0.75, line=dict(color="white", width=1)),
            text=hover_texts,
            hovertemplate="%{text}<extra></extra>",
        ))
    return annotate_side(fig, side_focus)

def delivery_map_figure(df_events, color_col="set_piece_type", title="Delivery Map", side_focus="Both"):
    fig = draw_pitch(go.Figure(), title=title, height=700, half=False)
    plot = df_events.dropna(
        subset=["pass_location_x", "pass_location_y"]
    ).copy()

    if plot.empty:
        return annotate_side(fig, side_focus)

    plot["pass_end_location_x"] = plot["pass_end_location_x"].fillna(plot["shot_location_x"])
    plot["pass_end_location_y"] = plot["pass_end_location_y"].fillna(plot["shot_location_y"])

    plot = plot.dropna(subset=["pass_end_location_x", "pass_end_location_y"])
    if plot.empty:
        return annotate_side(fig, side_focus)

    legend_added = set()
    for _, row in plot.iterrows():
        category = str(row.get(color_col, "Unknown"))
        show = category not in legend_added
        if show:
            legend_added.add(category)

        fig.add_trace(go.Scatter(
            x=[80 - row["pass_end_location_y"]],
            y=[row["pass_end_location_x"]],
            mode="markers",
            name=category,
            showlegend=show,
            legendgroup=category,
            marker=dict(size=12, opacity=0.8, line=dict(width=1, color="white")),
            text=(
                f"<b>Player:</b> {row.get('Taker', 'N/A')}<br>"
                f"<b>Team:</b> {row.get('corner_team', 'N/A')}<br>"
                f"<b>Set Piece:</b> {row.get('set_piece_type', 'N/A')}"
            ),
            hovertemplate="%{text}<extra></extra>",
        ))
    return annotate_side(fig, side_focus)

def outcome_pie(df, title="Outcome Split"):
    if df.empty:
        return go.Figure()
    s = df.groupby("outcome_bucket", dropna=False).size().reset_index(name="n")
    fig = px.pie(s, names="outcome_bucket", values="n", hole=0.55, title=title, color_discrete_sequence=QUAL_PALETTE)
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return figure_layout(fig, 380, title)

def sp_type_pie(df, title="Set Piece Split"):
    if df.empty:
        return go.Figure()
    s = df.groupby("set_piece_type", dropna=False).size().reset_index(name="n")
    fig = px.pie(s, names="set_piece_type", values="n", hole=0.55, title=title, color_discrete_sequence=QUAL_PALETTE)
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

def cumulative_line(df, title="Cumulative Set Pieces", group_col="corner_team"):
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

def taker_bar_chart(df, metric, title, top_n=12, min_events=3):
    if df.empty or metric not in df.columns:
        return go.Figure()
    plot = df[df["events"] >= min_events].copy()
    if plot.empty:
        return go.Figure()
    plot["label"] = plot["Taker"].astype(str) + " (" + plot["corner_team"].astype(str) + ")"
    plot = plot.sort_values(metric, ascending=False).head(top_n)
    fig = px.bar(
        plot, x="label", y=metric,
        color=metric, color_continuous_scale="Blues",
        title=title, hover_data=["events", "shots", "total_xg"],
    )
    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35)
    return figure_layout(fig, 380, title)

# =========================================================
# DATA LOAD
# =========================================================
@st.cache_data
def load_data():
    possible_files = [
        FILE_NAME,
        "/mnt/data/SWE SP.xlsx",
        "Allsvenskan - Corners 2025.xlsx",
        "Allsvenskan - Corners 2025 (1).xlsx",
    ]
    for f in possible_files:
        if os.path.exists(f):
            return pd.read_excel(f)
    raise FileNotFoundError(f"{FILE_NAME} not found.")

@st.cache_data
def load_hops_data():
    possible_files = [HOPS_FILE_NAME, "/mnt/data/duel_hops_rating_summary.xlsx"]
    hops = None
    for f in possible_files:
        if os.path.exists(f):
            try:
                hops = pd.read_excel(f)
                break
            except Exception:
                continue

    if hops is None:
        return pd.DataFrame(columns=["Player", "Team", "Rating", "z_score", "Percentile", "Rank"])

    hops.columns = [str(c).strip() for c in hops.columns]

    player_col = find_col(hops, ["Player"])
    team_col = find_col(hops, ["Team"])
    rating_col = find_col(hops, ["Rating"])

    if any(x is None for x in [player_col, team_col, rating_col]):
        return pd.DataFrame(columns=["Player", "Team", "Rating", "z_score", "Percentile", "Rank"])

    hops = hops.rename(columns={player_col: "Player", team_col: "Team", rating_col: "Rating"})
    hops = hops[["Player", "Team", "Rating"]].copy()
    hops["Player"] = hops["Player"].astype(str).str.strip()
    hops["Team"] = hops["Team"].astype(str).str.strip()
    hops["Rating"] = pd.to_numeric(hops["Rating"], errors="coerce")
    hops = hops.dropna(subset=["Player", "Team", "Rating"]).reset_index(drop=True)

    rating_std = hops["Rating"].std()
    hops["z_score"] = 0.0 if pd.isna(rating_std) or rating_std == 0 else (hops["Rating"] - hops["Rating"].mean()) / rating_std
    hops["Percentile"] = hops["Rating"].rank(pct=True) * 100
    hops = hops.sort_values(["Rating", "Player"], ascending=[False, True]).reset_index(drop=True)
    hops["Rank"] = np.arange(1, len(hops) + 1)
    return hops

@st.cache_data
def load_delay_data():
    possible_files = [DELAY_FILE_NAME, "/mnt/data/corner_delays.xlsx"]
    delay_summary = None

    for f in possible_files:
        if os.path.exists(f):
            try:
                delay_summary = pd.read_excel(f, sheet_name="Summary")
                break
            except Exception:
                continue

    if delay_summary is None:
        return pd.DataFrame()

    delay_summary.columns = [str(c).strip() for c in delay_summary.columns]
    expected_cols = [
        "match", "corners_found", "corners_matched",
        "avg_delay_sec", "median_delay_sec", "min_delay_sec", "max_delay_sec"
    ]
    for c in expected_cols:
        if c not in delay_summary.columns:
            delay_summary[c] = np.nan

    for c in ["corners_found", "corners_matched", "avg_delay_sec", "median_delay_sec", "min_delay_sec", "max_delay_sec"]:
        delay_summary[c] = pd.to_numeric(delay_summary[c], errors="coerce")

    delay_summary["match"] = delay_summary["match"].astype(str).str.strip()
    delay_summary["match_label"] = (
        delay_summary["match"]
        .str.replace(".csv.gz", "", regex=False)
        .str.replace("events_match_", "", regex=False)
    )
    delay_summary["match_rate"] = delay_summary["corners_matched"] / delay_summary["corners_found"].replace(0, np.nan)

    return delay_summary.sort_values(
        ["avg_delay_sec", "median_delay_sec"],
        ascending=[False, False],
        na_position="last"
    ).reset_index(drop=True)

@st.cache_data
def prepare_data(raw_df):
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    _f = lambda *c: find_col(df, list(c))

    match_id_col = _f("match_id", "match id")
    match_col = _f("match", "match_name")
    team_col = _f("team.name", "pass_team_name", "team", "team_name")
    minute_col = _f("minute")
    second_col = _f("second")
    outcome_col = _f("sp_outcome", "outcome")
    xg_col = _f("shot.statsbomb_xg", "shot_xg", "xg")
    taker_col = _f("taker")
    shooter_col = _f("shooter")
    pass_loc_col = _f("location.pass")
    shot_loc_col = _f("location.shot")
    shot_x_col = _f("shot_x", "shot_location_x")
    shot_y_col = _f("shot_y", "shot_location_y")
    tech_col = _f("pass.technique.name", "pass_technique")
    height_col = _f("pass.height.name", "pass_height")
    shot_outcome_col = _f("shot.outcome.name", "shot_outcome")
    sp_type_col = _f("SP_Type", "sp_type", "set_piece_type")
    possession_col = _f("possession")
    timestamp_col = _f("timestamp")

    if match_id_col is None:
        raise ValueError("Missing required column: match_id")
    if team_col is None:
        raise ValueError("Missing required column: team.name/team")

    if minute_col is None and timestamp_col is not None:
        ts = df[timestamp_col].astype(str).str.split(":", expand=True)
        if ts.shape[1] >= 3:
            df["Minute__tmp"] = pd.to_numeric(ts[1], errors="coerce")
            df["Second__tmp"] = pd.to_numeric(ts[2].str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
            minute_col = "Minute__tmp"
            second_col = "Second__tmp"

    if match_col is None:
        match_col = "Match__tmp"
        if possession_col is not None:
            df[match_col] = "Match " + df[match_id_col].astype(str)
        else:
            df[match_col] = "Match " + df[match_id_col].astype(str)

    if minute_col is None:
        df["Minute__fallback"] = 0
        minute_col = "Minute__fallback"
    if second_col is None:
        df["Second__fallback"] = 0
        second_col = "Second__fallback"

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
        shot_x_col: "shot_location_x",
        shot_y_col: "shot_location_y",
        tech_col: "pass_technique",
        height_col: "pass_height",
        shot_outcome_col: "shot_outcome",
        sp_type_col: "set_piece_type_raw",
        pass_loc_col: "pass_location_raw",
        shot_loc_col: "shot_location_raw",
    }
    for src, dst in optional_map.items():
        if src is not None:
            rename_map[src] = dst
    df = df.rename(columns=rename_map)

    for c in ["SP_outcome", "shot_xg", "Taker", "Shooter", "shot_location_x", "shot_location_y",
              "pass_technique", "pass_height", "shot_outcome", "set_piece_type_raw",
              "pass_location_raw", "shot_location_raw"]:
        if c not in df.columns:
            df[c] = np.nan

    df["Minute"] = safe_numeric(df["Minute"]).fillna(0)
    df["Second"] = safe_numeric(df["Second"]).fillna(0)
    df["shot_xg"] = safe_numeric(df["shot_xg"])
    df["corner_team"] = df["corner_team"].astype(str).str.strip()
    df["Match"] = df["Match"].astype(str).str.strip()

    if "pass_location_raw" in df.columns:
        df["pass_location_x"] = df["pass_location_raw"].apply(lambda x: parse_xy(x, 0))
        df["pass_location_y"] = df["pass_location_raw"].apply(lambda x: parse_xy(x, 1))
    else:
        df["pass_location_x"] = np.nan
        df["pass_location_y"] = np.nan

    if "shot_location_raw" in df.columns and df["shot_location_x"].isna().all():
        df["shot_location_x"] = df["shot_location_raw"].apply(lambda x: parse_xy(x, 0))
    if "shot_location_raw" in df.columns and df["shot_location_y"].isna().all():
        df["shot_location_y"] = df["shot_location_raw"].apply(lambda x: parse_xy(x, 1))

    df["pass_end_location_x"] = df["shot_location_x"]
    df["pass_end_location_y"] = df["shot_location_y"]

    df["event_minute"] = df["Minute"].fillna(0) + df["Second"].fillna(0) / 60
    homes, aways = zip(*[split_match_name(m) for m in df["Match"]])
    df["home_team"] = list(homes)
    df["away_team"] = list(aways)
    df["is_home_corner"] = df["corner_team"] == df["home_team"]
    df["is_away_corner"] = df["corner_team"] == df["away_team"]
    df["venue_split"] = np.where(df["is_home_corner"], "Home", np.where(df["is_away_corner"], "Away", "Unknown"))

    df["set_piece_type"] = df["set_piece_type_raw"].apply(set_piece_bucket)
    sp = df["SP_outcome"].astype(str)
    df["led_to_shot"] = df["shot_xg"].fillna(0).gt(0) | df["shot_outcome"].notna() | sp.str.contains("shot|goal", case=False, na=False)
    df["is_fast_shot"] = sp.str.contains("within 3 seconds", case=False, na=False)
    df["outcome_bucket"] = np.where(df["shot_outcome"].notna(), df["shot_outcome"].astype(str), sp.apply(classify_outcome))

    tech = df["pass_technique"].astype(str)
    df["is_inswinger"] = tech.str.contains("inswing", case=False, na=False)
    df["is_outswinger"] = tech.str.contains("outswing", case=False, na=False)
    df["is_short_corner"] = tech.str.contains("short", case=False, na=False)
    df["side"] = df["pass_location_y"].apply(side_from_y)
    df["delivery_zone"] = df["pass_end_location_y"].apply(delivery_zone_from_y)
    df["delivery_length"] = df.apply(
        lambda r: delivery_length(r["pass_location_x"], r["pass_location_y"], r["pass_end_location_x"], r["pass_end_location_y"]), axis=1
    )
    df["end_zone"] = df.apply(lambda r: zone_from_end_location(r["pass_end_location_x"], r["pass_end_location_y"]), axis=1)
    df["is_six_yard_delivery"] = df["end_zone"].eq("6-yard box")
    df["is_penalty_area_delivery"] = df["end_zone"].eq("Penalty area")
    df["delivery_type"] = df.apply(
        lambda r: infer_delivery_type(r["set_piece_type"], r["pass_height"], r["pass_location_x"]), axis=1
    )
    df["xg_category"] = df["shot_xg"].apply(xg_category)
    df["goal_from_corner"] = df["shot_outcome"].astype(str).str.contains("goal", case=False, na=False)

    df["phase"] = pd.cut(
        df["event_minute"], bins=[-0.1, 15, 30, 45, 60, 75, 120],
        labels=["0-15", "16-30", "31-45", "46-60", "61-75", "76+"], right=True
    ).astype(str)

    match_summary = (
        df.groupby(["match_id", "Match", "home_team", "away_team"], dropna=False)
        .agg(
            total_events=("match_id", "size"),
            shots_from_set_pieces=("led_to_shot", "sum"),
            fast_shots=("is_fast_shot", "sum"),
            total_xg=("shot_xg", "sum"),
            unique_takers=("Taker", pd.Series.nunique),
            corners=("set_piece_type", lambda s: (s == "Corner").sum()),
            free_kicks=("set_piece_type", lambda s: (s == "Free Kick").sum()),
            throw_ins=("set_piece_type", lambda s: (s == "Throw-In").sum()),
        )
        .reset_index()
    )
    match_summary["shot_rate"] = match_summary["shots_from_set_pieces"] / match_summary["total_events"].replace(0, np.nan)
    match_summary["xg_per_event"] = match_summary["total_xg"] / match_summary["total_events"].replace(0, np.nan)

    return df, match_summary

# =========================================================
# SUMMARY BUILDERS
# =========================================================
def build_team_summary(df):
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("corner_team", dropna=False)
        .agg(
            events=("match_id", "size"),
            matches=("match_id", pd.Series.nunique),
            shots_from_set_pieces=("led_to_shot", "sum"),
            fast_shots=("is_fast_shot", "sum"),
            total_xg=("shot_xg", "sum"),
            taker_variety=("Taker", pd.Series.nunique),
            corners=("set_piece_type", lambda s: (s == "Corner").sum()),
            free_kicks=("set_piece_type", lambda s: (s == "Free Kick").sum()),
            throw_ins=("set_piece_type", lambda s: (s == "Throw-In").sum()),
            six_yard_deliveries=("is_six_yard_delivery", "sum"),
            penalty_area_deliveries=("is_penalty_area_delivery", "sum"),
        )
        .reset_index()
        .rename(columns={"corner_team": "team"})
    )
    denom = out["events"].replace(0, np.nan)
    out["events_per_match"] = out["events"] / out["matches"].replace(0, np.nan)
    out["shot_rate"] = out["shots_from_set_pieces"] / denom
    out["fast_shot_rate"] = out["fast_shots"] / denom
    out["xg_per_match"] = out["total_xg"] / out["matches"].replace(0, np.nan)
    out["xg_per_event"] = out["total_xg"] / denom
    out["six_yard_delivery_rate"] = out["six_yard_deliveries"] / denom
    out["penalty_area_delivery_rate"] = out["penalty_area_deliveries"] / denom
    out["corner_rate"] = out["corners"] / denom
    out["free_kick_rate"] = out["free_kicks"] / denom
    out["throw_in_rate"] = out["throw_ins"] / denom
    return out

def taker_summary(df):
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby(["corner_team", "Taker"], dropna=False)
        .agg(
            events=("match_id", "size"),
            shots=("led_to_shot", "sum"),
            fast_shots=("is_fast_shot", "sum"),
            goals=("goal_from_corner", "sum"),
            total_xg=("shot_xg", "sum"),
            corners=("set_piece_type", lambda s: (s == "Corner").sum()),
            free_kicks=("set_piece_type", lambda s: (s == "Free Kick").sum()),
            throw_ins=("set_piece_type", lambda s: (s == "Throw-In").sum()),
        )
        .reset_index()
    )
    d = out["events"].replace(0, np.nan)
    out["shot_rate"] = out["shots"] / d
    out["xg_per_event"] = out["total_xg"] / d
    out["goal_rate"] = out["goals"] / d
    return out.sort_values(["events", "total_xg"], ascending=False)

def build_sp_type_summary(df):
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("set_piece_type", dropna=False)
        .agg(
            events=("match_id", "size"),
            shots=("led_to_shot", "sum"),
            goals=("goal_from_corner", "sum"),
            total_xg=("shot_xg", "sum"),
            matches=("match_id", pd.Series.nunique),
        )
        .reset_index()
    )
    out["shot_rate"] = out["shots"] / out["events"].replace(0, np.nan)
    out["xg_per_event"] = out["total_xg"] / out["events"].replace(0, np.nan)
    out["events_per_match"] = out["events"] / out["matches"].replace(0, np.nan)
    return out.sort_values("events", ascending=False)

# =========================================================
# EXPORTS
# =========================================================
def download_excel_workbook(events_df, team_df, match_df, taker_df, hops_df, delay_df, sp_df):
    buf = BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            events_df.to_excel(writer, sheet_name="Events", index=False)
            team_df.to_excel(writer, sheet_name="Teams", index=False)
            match_df.to_excel(writer, sheet_name="Matches", index=False)
            taker_df.to_excel(writer, sheet_name="Takers", index=False)
            sp_df.to_excel(writer, sheet_name="SP Types", index=False)
            hops_df.to_excel(writer, sheet_name="HOPS", index=False)
            delay_df.to_excel(writer, sheet_name="Delay", index=False)
        return buf.getvalue()
    except Exception:
        return None

# =========================================================
# LOAD DATA
# =========================================================
try:
    raw_df = load_data()
    df, match_summary = prepare_data(raw_df)
    hops_df = load_hops_data()
    delay_df = load_delay_data()
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
        <div class="hero-sub">Corners, free kicks and throw-ins in one cleaner analysis workspace.</div>
        <div>
            <span class="pill">Executive</span>
            <span class="pill">Visuals</span>
            <span class="pill">Teams</span>
            <span class="pill">Matches</span>
            <span class="pill">Scouting</span>
            <span class="pill">Set Piece Types</span>
            <span class="pill">HOPS</span>
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
all_sp_types = ["Corner", "Free Kick", "Throw-In", "Other"]

with st.sidebar:
    st.markdown("### 🎛 Studio Controls")
    page = st.radio("Workspace", [
        "🏠 Executive Dashboard",
        "📊 Visualisation Studio",
        "🏟 Team Analysis",
        "🔍 Match Explorer",
        "👤 Scouting Center",
        "📈 Trend Lab",
        "⚽ Set Piece Types",
        "⏱ Delay Time",
        "🦘 HOPS",
        "🗂 Data Hub",
    ])
    st.markdown("---")
    st.markdown("**Quick Filters**")
    sel_team = st.selectbox("Team", ["All Teams"] + all_teams)
    sel_matches = st.multiselect("Matches", all_matches)
    sel_sp_types = st.multiselect("Set Piece Type", all_sp_types, default=["Corner", "Free Kick", "Throw-In"])
    side_focus = st.radio("Delivery Side", ["Both", "Left", "Right"], horizontal=True)

    with st.expander("Advanced filters", expanded=False):
        sel_takers = st.multiselect("Taker(s)", all_takers)
        minute_min = int(df["Minute"].min()) if not df["Minute"].dropna().empty else 0
        minute_max = int(df["Minute"].max()) if not df["Minute"].dropna().empty else 0
        min_range = safe_range_slider("Minute Range", minute_min, minute_max)

        match_event_min = int(match_summary["total_events"].min()) if not match_summary.empty else 0
        match_event_max = int(match_summary["total_events"].max()) if not match_summary.empty else 0
        event_range = safe_range_slider("Match Event Range", match_event_min, match_event_max)

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
        venue_filter = st.multiselect("Home / Away", ["Home", "Away", "Unknown"], default=["Home", "Away", "Unknown"])
        phase_filter = st.multiselect(
            "Phase", ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"],
            default=["0-15", "16-30", "31-45", "46-60", "61-75", "76+"]
        )
        shot_only = st.checkbox("Shot outcomes only")
        corners_only = st.checkbox("Corners only")
        free_kicks_only = st.checkbox("Free kicks only")
        throw_ins_only = st.checkbox("Throw-ins only")

# =========================================================
# FILTERS
# =========================================================
def apply_filters(events, matches, team_override="__USE_SELECTED_TEAM__"):
    out_matches = matches[matches["total_events"].between(event_range[0], event_range[1])].copy()
    out_events = events[events["match_id"].isin(out_matches["match_id"].unique())].copy()
    out_events = out_events[out_events["Minute"].fillna(0).between(min_range[0], min_range[1])]

    effective_side_filter = {"Left": ["Left"], "Right": ["Right"]}.get(side_focus, ["Left", "Right", "Unknown"])

    if team_override == "__USE_SELECTED_TEAM__":
        selected_team_filter = None if sel_team == "All Teams" else [sel_team]
    else:
        selected_team_filter = team_override

    filter_map = {
        "corner_team": selected_team_filter,
        "Taker": sel_takers if sel_takers else None,
        "Match": sel_matches if sel_matches else None,
        "side": effective_side_filter,
        "delivery_zone": zone_filter if zone_filter else None,
        "end_zone": end_zone_filter if end_zone_filter else None,
        "venue_split": venue_filter if venue_filter else None,
        "phase": phase_filter if phase_filter else None,
        "set_piece_type": sel_sp_types if sel_sp_types else None,
    }
    for col, allowed in filter_map.items():
        if allowed is not None:
            out_events = out_events[out_events[col].astype(str).isin([str(x) for x in allowed])]

    if shot_only:
        out_events = out_events[out_events["led_to_shot"]]
    if corners_only:
        out_events = out_events[out_events["set_piece_type"] == "Corner"]
    if free_kicks_only:
        out_events = out_events[out_events["set_piece_type"] == "Free Kick"]
    if throw_ins_only:
        out_events = out_events[out_events["set_piece_type"] == "Throw-In"]

    out_matches = out_matches[out_matches["match_id"].isin(out_events["match_id"].unique())]
    return out_events, out_matches, build_team_summary(out_events), taker_summary(out_events), build_sp_type_summary(out_events)

league_event_df, league_match_df, league_team_df, league_taker_df, league_sp_df = apply_filters(df, match_summary)
comparison_event_df, comparison_match_df, comparison_team_df, comparison_taker_df, comparison_sp_df = apply_filters(df, match_summary, team_override=None)
filter_chips(sel_team, len(sel_matches), len(sel_takers), side_focus, venue_filter, sel_sp_types)

# =========================================================
# KPI ROW
# =========================================================
def render_kpis(events):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    total_xg = events["shot_xg"].fillna(0).sum() if not events.empty else 0
    shot_rate = events["led_to_shot"].mean() if not events.empty else 0
    goals = int(events["goal_from_corner"].sum()) if not events.empty else 0
    with c1: metric_card("Events", f"{len(events):,}", "Set-piece actions")
    with c2: metric_card("Matches", f"{events['match_id'].nunique() if not events.empty else 0:,}", "Unique matches")
    with c3: metric_card("Shots", f"{int(events['led_to_shot'].sum()) if not events.empty else 0:,}", "From set pieces")
    with c4: metric_card("Total xG", f"{total_xg:.2f}", "xG generated")
    with c5: metric_card("Shot Rate", f"{shot_rate*100:.1f}%", "Shots / event")
    with c6: metric_card("Goals", f"{goals}", "From set pieces")

# =========================================================
# PAGES
# =========================================================
if page == "🏠 Executive Dashboard":
    render_kpis(league_event_df)
    st.markdown("<br>", unsafe_allow_html=True)

    c0, c00 = st.columns(2)
    with c0:
        section_header("Set Piece Mix")
        st.plotly_chart(sp_type_pie(league_event_df, "Set Piece Type Split"), use_container_width=True)
    with c00:
        section_header("Efficiency by Set Piece Type")
        if not league_sp_df.empty:
            fig = px.bar(
                league_sp_df.sort_values("events", ascending=False),
                x="set_piece_type", y="shot_rate", color="xg_per_event",
                title="Shot Rate by Set Piece Type", color_continuous_scale="Blues",
                hover_data=["events", "total_xg", "goals"],
            )
            fig.update_layout(coloraxis_showscale=False)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(figure_layout(fig, 400, "Shot Rate by Set Piece Type"), use_container_width=True)
        else:
            empty_state()

    c1, c2 = st.columns(2)
    with c1:
        section_header("Set Piece Volume by Team")
        if not league_team_df.empty:
            fig = px.bar(
                league_team_df.sort_values("events", ascending=False),
                x="team", y="events",
                color="events_per_match", color_continuous_scale="Blues",
                labels={"team": "", "events": "Set pieces"},
                hover_data=["matches", "shot_rate", "xg_per_match"],
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(figure_layout(fig, 400, "Set Piece Volume by Team"), use_container_width=True)
        else:
            empty_state()
    with c2:
        section_header("Efficiency Map")
        if not league_team_df.empty:
            st.plotly_chart(
                team_scatter(league_team_df, "shot_rate", "xg_per_match", "events", "Shot Rate vs xG/Match"),
                use_container_width=True,
            )
        else:
            empty_state()

    section_header("Executive Match Board")
    if not league_match_df.empty:
        show_cols = [c for c in [
            "Match", "home_team", "away_team", "total_events", "corners", "free_kicks", "throw_ins",
            "shots_from_set_pieces", "fast_shots", "total_xg", "shot_rate", "xg_per_event", "unique_takers"
        ] if c in league_match_df.columns]
        st.dataframe(
            league_match_df[show_cols].sort_values(["total_xg", "total_events"], ascending=False).reset_index(drop=True),
            use_container_width=True, height=420,
        )
    else:
        empty_state()

elif page == "📊 Visualisation Studio":
    section_header("Visualisation Studio", "Shots and deliveries across corners, free kicks and throw-ins.")
    visual_context_note(side_focus, len(league_event_df))
    tabs = st.tabs(["🎯 Shots", "🏹 Deliveries", "⚽ Type Comparison", "⏱ Timing"])

    with tabs[0]:
        shot_df = league_event_df.dropna(subset=["shot_location_x", "shot_location_y"])
        if shot_df.empty:
            st.info("No shot data available for this selection.")
        else:
            st.plotly_chart(
                shotmap_figure(shot_df, color_col="set_piece_type", title="Attacking Shot Map", side_focus=side_focus),
                use_container_width=True,
            )

    with tabs[1]:
        del_df = league_event_df.dropna(subset=["pass_location_x", "pass_location_y"])
        if del_df.empty:
            empty_state("No delivery data.")
        else:
            st.plotly_chart(
                delivery_map_figure(del_df, "set_piece_type", "Delivery Map", side_focus=side_focus),
                use_container_width=True,
            )

    with tabs[2]:
        if league_sp_df.empty:
            empty_state()
        else:
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(
                    league_sp_df, x="set_piece_type", y="events",
                    color="set_piece_type", title="Volume by Type", color_discrete_sequence=QUAL_PALETTE
                )
                st.plotly_chart(figure_layout(fig, 360, "Volume by Type"), use_container_width=True)
            with c2:
                fig = px.bar(
                    league_sp_df, x="set_piece_type", y="xg_per_event",
                    color="set_piece_type", title="xG per Event by Type", color_discrete_sequence=QUAL_PALETTE
                )
                st.plotly_chart(figure_layout(fig, 360, "xG per Event by Type"), use_container_width=True)

    with tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(cumulative_line(league_event_df, "Cumulative Set Pieces by Team"), use_container_width=True)
        with c2:
            st.plotly_chart(minute_histogram(league_event_df, "Minute Distribution by Set Piece Type", color_col="set_piece_type"), use_container_width=True)
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
                st.plotly_chart(sp_type_pie(team_ev, f"{sel_team} — Set Piece Split"), use_container_width=True)

        with tabs[1]:
            shot_df = team_ev.dropna(subset=["shot_location_x", "shot_location_y"])
            del_df = team_ev.dropna(subset=["pass_location_x", "pass_location_y"])
            c1, c2 = st.columns(2)
            with c1:
                if shot_df.empty:
                    empty_state("No shot location data.")
                else:
                    st.plotly_chart(
                        shotmap_figure(shot_df, color_col="set_piece_type", title=f"Shotmap — {sel_team}", side_focus=side_focus),
                        use_container_width=True,
                    )
            with c2:
                if del_df.empty:
                    empty_state("No delivery coordinate data.")
                else:
                    st.plotly_chart(
                        delivery_map_figure(del_df, "set_piece_type", f"Delivery Map — {sel_team}", side_focus=side_focus),
                        use_container_width=True,
                    )

        with tabs[2]:
            if team_takers.empty:
                empty_state("No taker data.")
            else:
                st.dataframe(team_takers.reset_index(drop=True), use_container_width=True, height=380)
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(taker_bar_chart(team_takers, "xg_per_event", f"Takers by xG/Event — {sel_team}", min_events=2), use_container_width=True)
                with c2:
                    st.plotly_chart(taker_bar_chart(team_takers, "shot_rate", f"Takers by Shot Rate — {sel_team}", min_events=2), use_container_width=True)

        with tabs[3]:
            match_view = (
                team_ev.groupby(["Match", "venue_split"], dropna=False)
                .agg(events=("match_id", "size"), shots=("led_to_shot", "sum"),
                     total_xg=("shot_xg", "sum"), goals=("goal_from_corner", "sum"))
                .reset_index()
            )
            match_view["shot_rate"] = match_view["shots"] / match_view["events"].replace(0, np.nan)
            match_view["xg_per_event"] = match_view["total_xg"] / match_view["events"].replace(0, np.nan)
            st.dataframe(match_view.sort_values("total_xg", ascending=False).reset_index(drop=True), use_container_width=True, height=420)

        with tabs[4]:
            if team_row.empty:
                empty_state()
            else:
                row = team_row.iloc[0]
                c1, c2, c3 = st.columns(3)
                with c1: metric_card("Events/Match", human_val(row.get("events_per_match")), "Volume")
                with c2: metric_card("Shot Rate", human_pct(row.get("shot_rate")), "Shots per event")
                with c3: metric_card("xG/Match", human_val(row.get("xg_per_match"), 3), "Chance quality")

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
            breakdown = build_sp_type_summary(match_ev)
            if not breakdown.empty:
                st.dataframe(breakdown.reset_index(drop=True), use_container_width=True, height=220)

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(minute_histogram(match_ev, "Minute Distribution", color_col="set_piece_type"), use_container_width=True)
        with c2:
            st.plotly_chart(cumulative_line(match_ev, "Cumulative Set Pieces"), use_container_width=True)

    with tabs[2]:
        shot_df = match_ev.dropna(subset=["shot_location_x", "shot_location_y"])
        if shot_df.empty:
            empty_state("No shot data.")
        else:
            st.plotly_chart(
                shotmap_figure(shot_df, color_col="set_piece_type", title=f"Shotmap — {sel_match}", side_focus=side_focus),
                use_container_width=True,
            )

    with tabs[3]:
        del_df = match_ev.dropna(subset=["pass_location_x", "pass_location_y"])
        if del_df.empty:
            empty_state("No delivery data.")
        else:
            st.plotly_chart(
                delivery_map_figure(del_df, "set_piece_type", f"Delivery Map — {sel_match}", side_focus=side_focus),
                use_container_width=True,
            )

    with tabs[4]:
        show_cols = [c for c in [
            "Match", "corner_team", "set_piece_type", "Taker", "Shooter", "Minute", "Second",
            "shot_outcome", "shot_xg", "pass_height", "side", "delivery_zone", "end_zone", "phase"
        ] if c in match_ev.columns]
        st.dataframe(match_ev[show_cols].sort_values(["Minute", "Second"]).reset_index(drop=True), use_container_width=True, height=560)

elif page == "👤 Scouting Center":
    tabs = st.tabs(["🏅 Teams", "👤 Takers", "⚽ Types"])

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
            min_events_scout = st.number_input("Minimum events", 1, 100, 5)
            filt = league_taker_df[league_taker_df["events"] >= min_events_scout]
            st.dataframe(filt.reset_index(drop=True), use_container_width=True, height=460)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(taker_bar_chart(filt, "xg_per_event", "Top Takers — xG/Event", min_events=min_events_scout), use_container_width=True)
            with c2:
                st.plotly_chart(taker_bar_chart(filt, "shot_rate", "Top Takers — Shot Rate", min_events=min_events_scout), use_container_width=True)

    with tabs[2]:
        if league_sp_df.empty:
            empty_state()
        else:
            st.dataframe(league_sp_df.reset_index(drop=True), use_container_width=True, height=320)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(league_sp_df, x="set_piece_type", y="events", color="set_piece_type", color_discrete_sequence=QUAL_PALETTE)
                st.plotly_chart(figure_layout(fig, 360, "Volume by Set Piece Type"), use_container_width=True)
            with c2:
                fig = px.bar(league_sp_df, x="set_piece_type", y="shot_rate", color="set_piece_type", color_discrete_sequence=QUAL_PALETTE)
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(figure_layout(fig, 360, "Shot Rate by Set Piece Type"), use_container_width=True)

elif page == "📈 Trend Lab":
    tabs = st.tabs(["📊 Match Trends", "🕐 Phase Trends", "🧮 Tests"])

    with tabs[0]:
        if league_event_df.empty:
            empty_state()
        else:
            mv = league_event_df.groupby(["Match", "set_piece_type"], dropna=False).size().reset_index(name="events").sort_values("Match")
            fig = px.line(mv, x="Match", y="events", color="set_piece_type", markers=True, title="Set Pieces per Match", color_discrete_sequence=QUAL_PALETTE)
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(figure_layout(fig, 400, "Set Pieces per Match"), use_container_width=True)

    with tabs[1]:
        st.plotly_chart(phase_heatmap(league_event_df, "Set Piece Phase Pattern"), use_container_width=True)
        ph_sr = (
            league_event_df.groupby(["phase", "set_piece_type"], dropna=False)
            .agg(events=("match_id", "size"), shots=("led_to_shot", "sum")).reset_index()
        )
        ph_sr["shot_rate"] = ph_sr["shots"] / ph_sr["events"].replace(0, np.nan)
        fig = px.line(ph_sr, x="phase", y="shot_rate", color="set_piece_type", markers=True, title="Shot Rate Across Phases", color_discrete_sequence=QUAL_PALETTE)
        fig.update_layout(yaxis_tickformat=".0%")
        st.plotly_chart(figure_layout(fig, 380, "Shot Rate Across Phases"), use_container_width=True)

    with tabs[2]:
        section_header("Statistical Tests")
        ct_df = league_event_df[league_event_df["set_piece_type"].isin(["Corner", "Free Kick", "Throw-In"])].copy()
        if len(ct_df) >= 10:
            ct_df["shot_yn"] = np.where(ct_df["led_to_shot"], "Shot", "No Shot")
            cont = pd.crosstab(ct_df["set_piece_type"], ct_df["shot_yn"])
            st.dataframe(cont)
            try:
                chi2, p, dof, _ = scipy_stats.chi2_contingency(cont)
                st.write(f"Chi-square = {chi2:.3f} | p = {p:.4f} | df = {dof}")
            except Exception as e:
                st.warning(f"Test failed: {e}")
        else:
            empty_state("Not enough data for chi-square test.")

elif page == "⚽ Set Piece Types":
    section_header("Set Piece Types", "Dedicated comparison for corners, free kicks and throw-ins.")
    if league_sp_df.empty:
        empty_state()
    else:
        render_kpis(league_event_df)
        st.dataframe(league_sp_df.reset_index(drop=True), use_container_width=True, height=260)
        c1, c2, c3 = st.columns(3)
        with c1:
            fig = px.bar(league_sp_df, x="set_piece_type", y="events_per_match", color="set_piece_type", color_discrete_sequence=QUAL_PALETTE)
            st.plotly_chart(figure_layout(fig, 340, "Events per Match"), use_container_width=True)
        with c2:
            fig = px.bar(league_sp_df, x="set_piece_type", y="shot_rate", color="set_piece_type", color_discrete_sequence=QUAL_PALETTE)
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(figure_layout(fig, 340, "Shot Rate"), use_container_width=True)
        with c3:
            fig = px.bar(league_sp_df, x="set_piece_type", y="xg_per_event", color="set_piece_type", color_discrete_sequence=QUAL_PALETTE)
            st.plotly_chart(figure_layout(fig, 340, "xG per Event"), use_container_width=True)

elif page == "⏱ Delay Time":
    section_header("Delay Time", "Delay between detected corner events and matched events from the uploaded delay workbook.")
    if delay_df.empty:
        empty_state("No delay workbook data found.")
    else:
        st.dataframe(delay_df.reset_index(drop=True), use_container_width=True, height=520)

elif page == "🦘 HOPS":
    section_header("HOPS", "Player duel HOPS ratings based on the uploaded summary workbook.")
    if hops_df.empty:
        empty_state("No HOPS workbook found.")
    else:
        hops_view = hops_df.copy()
        if sel_team != "All Teams":
            hops_view = hops_view[hops_view["Team"] == sel_team]
        if hops_view.empty:
            empty_state("No HOPS data for the current filters.")
        else:
            avg_rating = hops_view["Rating"].mean()
            top_rating = hops_view["Rating"].max()
            top_player = hops_view.loc[hops_view["Rating"].idxmax(), "Player"]
            unique_players = hops_view["Player"].nunique()
            unique_teams = hops_view["Team"].nunique()

            c1, c2, c3, c4, c5 = st.columns(5)
            with c1: metric_card("Players", f"{unique_players:,}", "In current view")
            with c2: metric_card("Teams", f"{unique_teams:,}", "Represented")
            with c3: metric_card("Avg Rating", f"{avg_rating:.3f}", "Mean HOPS")
            with c4: metric_card("Best Rating", f"{top_rating:.3f}", "Top score")
            with c5: metric_card("Top Player", top_player, "Highest HOPS")

            st.dataframe(hops_view[["Rank", "Player", "Team", "Rating", "Percentile"]].reset_index(drop=True), use_container_width=True, height=560)

elif page == "🗂 Data Hub":
    tabs = st.tabs(["📄 Events", "🏟 Teams", "📋 Matches", "👤 Takers", "⚽ SP Types", "🦘 HOPS", "⏱ Delay", "⬇ Downloads"])
    with tabs[0]:
        st.dataframe(league_event_df.reset_index(drop=True), use_container_width=True, height=600)
    with tabs[1]:
        st.dataframe(league_team_df.reset_index(drop=True), use_container_width=True, height=600)
    with tabs[2]:
        st.dataframe(league_match_df.reset_index(drop=True), use_container_width=True, height=600)
    with tabs[3]:
        st.dataframe(league_taker_df.reset_index(drop=True), use_container_width=True, height=600)
    with tabs[4]:
        st.dataframe(league_sp_df.reset_index(drop=True), use_container_width=True, height=420)
    with tabs[5]:
        st.dataframe(hops_df.reset_index(drop=True), use_container_width=True, height=600)
    with tabs[6]:
        st.dataframe(delay_df.reset_index(drop=True), use_container_width=True, height=600)

    with tabs[7]:
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        with c1:
            st.download_button("⬇ Events CSV", league_event_df.to_csv(index=False).encode(), "events.csv", "text/csv", use_container_width=True)
        with c2:
            st.download_button("⬇ Teams CSV", league_team_df.to_csv(index=False).encode(), "teams.csv", "text/csv", use_container_width=True)
        with c3:
            st.download_button("⬇ Matches CSV", league_match_df.to_csv(index=False).encode(), "matches.csv", "text/csv", use_container_width=True)
        with c4:
            st.download_button("⬇ Set Piece Types CSV", league_sp_df.to_csv(index=False).encode(), "sp_types.csv", "text/csv", use_container_width=True)
        with c5:
            st.download_button("⬇ HOPS CSV", hops_df.to_csv(index=False).encode(), "hops.csv", "text/csv", use_container_width=True)
        with c6:
            st.download_button("⬇ Delay CSV", delay_df.to_csv(index=False).encode(), "delay.csv", "text/csv", use_container_width=True)

        wb = download_excel_workbook(league_event_df, league_team_df, league_match_df, league_taker_df, hops_df, delay_df, league_sp_df)
        if wb:
            st.download_button(
                "⬇ Full Excel Workbook", wb, "allsvenskan_set_pieces.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

st.markdown(
    f'<div class="footer-note">⚽ Allsvenskan Set Piece Studio Pro · Set-piece build · Corners + Free Kicks + Throw-Ins · Streamlit + Plotly</div>',
    unsafe_allow_html=True,
)
