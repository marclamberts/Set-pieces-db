import io
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Allsvenskan Set Piece Studio",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# THEME
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

TYPE_COLORS = {
    "Corner": ACCENT,
    "Free Kick": SUCCESS,
    "Throw-In": ORANGE,
    "Other": PURPLE,
}

QUAL_PALETTE = [ACCENT, SUCCESS, WARNING, DANGER, PURPLE, ORANGE, "#8ad6ff", "#6ee7b7"]
px.defaults.template = "plotly_dark"

CSS = f"""
<style>
body, .stApp {{
    background:
        radial-gradient(ellipse 1200px 700px at 90% -10%, rgba(93,168,255,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 900px 600px at -10% 20%, rgba(52,211,153,0.08) 0%, transparent 55%),
        linear-gradient(180deg, {BG} 0%, {BG_2} 100%);
    color: {TEXT};
}}
.block-container {{
    max-width: 1580px;
    padding-top: 1rem;
    padding-bottom: 2rem;
}}
header[data-testid="stHeader"] {{
    background: rgba(0,0,0,0);
}}
#MainMenu, footer {{
    visibility: hidden;
}}
.hero {{
    background: linear-gradient(135deg, rgba(93,168,255,0.18) 0%, rgba(93,168,255,0.05) 55%, rgba(52,211,153,0.08) 100%);
    border: 1px solid rgba(93,168,255,0.18);
    border-radius: 30px;
    padding: 34px 34px 26px 34px;
    box-shadow: 0 16px 48px rgba(0,0,0,0.22);
    margin-bottom: 18px;
}}
.hero-title {{
    font-size: 2.7rem;
    font-weight: 900;
    line-height: 1.0;
    letter-spacing: -0.03em;
    margin-bottom: 0.55rem;
}}
.hero-title span {{
    color: {ACCENT};
}}
.hero-sub {{
    color: {MUTED};
    font-size: 1.02rem;
    line-height: 1.6;
    max-width: 980px;
}}
.upload-bar {{
    background: linear-gradient(160deg, {CARD} 0%, {CARD_2} 100%);
    border: 1px solid {BORDER};
    border-radius: 22px;
    padding: 16px 18px;
    margin-bottom: 18px;
}}
.segment-card {{
    background: linear-gradient(160deg, {CARD} 0%, {CARD_2} 100%);
    border: 1px solid {BORDER};
    border-radius: 26px;
    padding: 22px 22px 18px 22px;
    min-height: 245px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}}
.segment-pill {{
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 0.9rem;
}}
.segment-title {{
    font-size: 1.5rem;
    font-weight: 900;
    margin-bottom: 0.45rem;
}}
.segment-sub {{
    color: {MUTED};
    font-size: 0.95rem;
    line-height: 1.58;
    min-height: 74px;
}}
.panel {{
    background: rgba(255,255,255,0.02);
    border: 1px solid {BORDER};
    border-radius: 22px;
    padding: 18px 18px 12px 18px;
    margin-bottom: 14px;
}}
.kpi {{
    background: linear-gradient(160deg, {CARD} 0%, {CARD_2} 100%);
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 16px 16px 12px 16px;
    min-height: 102px;
}}
.kpi-label {{
    color: {MUTED};
    text-transform: uppercase;
    font-size: 0.68rem;
    letter-spacing: 0.12em;
    font-weight: 700;
}}
.kpi-value {{
    margin-top: 10px;
    font-size: 1.75rem;
    font-weight: 900;
    line-height: 1.0;
}}
.kpi-foot {{
    margin-top: 8px;
    font-size: 0.82rem;
    color: {MUTED};
}}
.section-title {{
    font-size: 1.12rem;
    font-weight: 850;
    margin: 0.1rem 0 0.22rem 0;
}}
.section-sub {{
    color: {MUTED};
    font-size: 0.92rem;
    margin-bottom: 0.85rem;
}}
.empty-state {{
    text-align: center;
    padding: 56px 24px;
    color: {MUTED};
    font-size: 0.94rem;
    border: 1px dashed rgba(255,255,255,0.10);
    border-radius: 18px;
    background: rgba(255,255,255,0.015);
}}
.pitch-wrap {{
    border: 1px solid {BORDER};
    border-radius: 22px;
    padding: 12px;
    background: rgba(255,255,255,0.02);
}}
.footer-note {{
    color: {MUTED_2};
    font-size: 0.82rem;
    margin-top: 1rem;
    padding-top: 12px;
    border-top: 1px solid {BORDER};
}}
div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER};
    border-radius: 14px;
    overflow: hidden;
}}
.stTabs [data-baseweb="tab-list"] {{
    gap: 6px;
    background: rgba(255,255,255,0.03);
    border-radius: 14px;
    padding: 4px;
    border: 1px solid {BORDER};
}}
.stTabs [aria-selected="true"] {{
    background: rgba(93,168,255,0.18) !important;
    color: #d4e8ff !important;
}}
div.stButton > button {{
    width: 100%;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.03);
    color: {TEXT};
    font-weight: 700;
    padding: 0.7rem 0.9rem;
}}
div.stButton > button:hover {{
    border-color: rgba(93,168,255,0.30);
    background: rgba(93,168,255,0.10);
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

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

def parse_xy(cell, idx=0):
    if pd.isna(cell):
        return np.nan
    try:
        parts = [float(str(x).strip()) for x in str(cell).split(",")]
        return parts[idx] if len(parts) > idx else np.nan
    except Exception:
        return np.nan

def set_piece_bucket(sp_type):
    s = str(sp_type).lower().strip()
    if "corner" in s:
        return "Corner"
    if "free kick" in s:
        return "Free Kick"
    if "throw" in s:
        return "Throw-In"
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

def section_header(title, sub=""):
    st.markdown(
        f'<div class="section-title">{title}</div>' +
        (f'<div class="section-sub">{sub}</div>' if sub else ""),
        unsafe_allow_html=True,
    )

def metric_card(label, value, foot=""):
    st.markdown(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-foot">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def empty_state(msg="No data for current selection."):
    st.markdown(f'<div class="empty-state">{msg}</div>', unsafe_allow_html=True)

# =========================================================
# DEMO DATA
# =========================================================
def build_demo_data():
    rows = [
        {
            "match_id": "1", "Match": "Malmö FF - AIK", "team": "Malmö FF", "Minute": 12, "Second": 14,
            "Taker": "Player A", "Shooter": "Player B", "set_piece_type": "Corner", "shot_xg": 0.11,
            "led_to_shot": True, "goal": False, "side": "Left", "delivery_zone": "Central Zone",
            "end_zone": "Penalty area", "phase": "0-15", "shot_location_x": 109, "shot_location_y": 39,
            "pass_location_x": 120, "pass_location_y": 64, "pass_end_location_x": 109, "pass_end_location_y": 39,
        },
        {
            "match_id": "1", "Match": "Malmö FF - AIK", "team": "AIK", "Minute": 28, "Second": 2,
            "Taker": "Player C", "Shooter": "Player D", "set_piece_type": "Free Kick", "shot_xg": 0.07,
            "led_to_shot": True, "goal": False, "side": "Right", "delivery_zone": "Near Post Zone",
            "end_zone": "Deep box", "phase": "16-30", "shot_location_x": 103, "shot_location_y": 25,
            "pass_location_x": 92, "pass_location_y": 21, "pass_end_location_x": 103, "pass_end_location_y": 25,
        },
        {
            "match_id": "2", "Match": "Hammarby - Djurgården", "team": "Hammarby", "Minute": 53, "Second": 44,
            "Taker": "Player E", "Shooter": "Player F", "set_piece_type": "Throw-In", "shot_xg": 0.18,
            "led_to_shot": True, "goal": True, "side": "Left", "delivery_zone": "Far Post Zone",
            "end_zone": "6-yard box", "phase": "46-60", "shot_location_x": 116, "shot_location_y": 52,
            "pass_location_x": 96, "pass_location_y": 62, "pass_end_location_x": 116, "pass_end_location_y": 52,
        },
        {
            "match_id": "2", "Match": "Hammarby - Djurgården", "team": "Djurgården", "Minute": 72, "Second": 10,
            "Taker": "Player G", "Shooter": "", "set_piece_type": "Corner", "shot_xg": 0.00,
            "led_to_shot": False, "goal": False, "side": "Right", "delivery_zone": "Near Post Zone",
            "end_zone": "Outside danger zone", "phase": "61-75", "pass_location_x": 120,
            "pass_location_y": 18, "pass_end_location_x": 101, "pass_end_location_y": 24,
        },
        {
            "match_id": "3", "Match": "Elfsborg - IFK Göteborg", "team": "Elfsborg", "Minute": 81, "Second": 5,
            "Taker": "Player H", "Shooter": "Player I", "set_piece_type": "Free Kick", "shot_xg": 0.22,
            "led_to_shot": True, "goal": True, "side": "Left", "delivery_zone": "Central Zone",
            "end_zone": "6-yard box", "phase": "76+", "shot_location_x": 115, "shot_location_y": 40,
            "pass_location_x": 88, "pass_location_y": 60, "pass_end_location_x": 115, "pass_end_location_y": 40,
        },
        {
            "match_id": "3", "Match": "Elfsborg - IFK Göteborg", "team": "IFK Göteborg", "Minute": 9, "Second": 20,
            "Taker": "Player J", "Shooter": "", "set_piece_type": "Throw-In", "shot_xg": 0.00,
            "led_to_shot": False, "goal": False, "side": "Right", "delivery_zone": "Central Zone",
            "end_zone": "Penalty area", "phase": "0-15", "pass_location_x": 94,
            "pass_location_y": 17, "pass_end_location_x": 108, "pass_end_location_y": 37,
        },
    ]
    return pd.DataFrame(rows)

# =========================================================
# DATA INGEST
# =========================================================
def parse_csv_like(df):
    data = df.copy()
    data.columns = [str(c).strip() for c in data.columns]
    _f = lambda *c: find_col(data, list(c))

    match_id_col = _f("match_id")
    match_col = _f("match", "Match")
    team_col = _f("team", "team.name")
    minute_col = _f("minute", "Minute")
    second_col = _f("second", "Second")
    sp_type_col = _f("SP_Type", "set_piece_type")
    xg_col = _f("shot_xg", "shot.statsbomb_xg")
    taker_col = _f("Taker", "taker")
    shooter_col = _f("Shooter", "shooter")
    pass_x_col = _f("pass_location_x")
    pass_y_col = _f("pass_location_y")
    pass_end_x_col = _f("pass_end_location_x")
    pass_end_y_col = _f("pass_end_location_y")
    shot_x_col = _f("shot_location_x")
    shot_y_col = _f("shot_location_y")
    side_col = _f("side")
    delivery_zone_col = _f("delivery_zone")
    end_zone_col = _f("end_zone")
    phase_col = _f("phase")
    led_to_shot_col = _f("led_to_shot")
    goal_col = _f("goal")
    pass_raw_col = _f("location.pass")
    shot_raw_col = _f("location.shot")
    outcome_col = _f("SP_outcome", "sp_outcome")
    shot_outcome_col = _f("shot_outcome", "shot.outcome.name")
    timestamp_col = _f("timestamp")

    if match_id_col is None:
        data["match_id"] = np.arange(1, len(data) + 1).astype(str)
        match_id_col = "match_id"

    if match_col is None:
        data["Match"] = "Match " + data[match_id_col].astype(str)
        match_col = "Match"

    if team_col is None or sp_type_col is None:
        raise ValueError("Dataset needs at least team/team.name and SP_Type/set_piece_type columns.")

    if minute_col is None and timestamp_col is not None:
        ts = data[timestamp_col].astype(str).str.split(":", expand=True)
        if ts.shape[1] >= 3:
            data["Minute_tmp"] = pd.to_numeric(ts[1], errors="coerce")
            data["Second_tmp"] = pd.to_numeric(ts[2].str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
            minute_col = "Minute_tmp"
            second_col = "Second_tmp"

    if minute_col is None:
        data["Minute"] = 0
        minute_col = "Minute"
    if second_col is None:
        data["Second"] = 0
        second_col = "Second"

    out = pd.DataFrame()
    out["match_id"] = data[match_id_col].astype(str)
    out["Match"] = data[match_col].astype(str)
    out["team"] = data[team_col].astype(str)
    out["Minute"] = safe_numeric(data[minute_col]).fillna(0)
    out["Second"] = safe_numeric(data[second_col]).fillna(0)
    out["Taker"] = data[taker_col].astype(str) if taker_col else ""
    out["Shooter"] = data[shooter_col].astype(str) if shooter_col else ""
    out["set_piece_type"] = data[sp_type_col].apply(set_piece_bucket)
    out["shot_xg"] = safe_numeric(data[xg_col]).fillna(0) if xg_col else 0.0

    if pass_raw_col:
        out["pass_location_x"] = data[pass_raw_col].apply(lambda x: parse_xy(x, 0))
        out["pass_location_y"] = data[pass_raw_col].apply(lambda x: parse_xy(x, 1))
    else:
        out["pass_location_x"] = safe_numeric(data[pass_x_col]).fillna(np.nan) if pass_x_col else np.nan
        out["pass_location_y"] = safe_numeric(data[pass_y_col]).fillna(np.nan) if pass_y_col else np.nan

    if shot_raw_col:
        out["shot_location_x"] = data[shot_raw_col].apply(lambda x: parse_xy(x, 0))
        out["shot_location_y"] = data[shot_raw_col].apply(lambda x: parse_xy(x, 1))
    else:
        out["shot_location_x"] = safe_numeric(data[shot_x_col]).fillna(np.nan) if shot_x_col else np.nan
        out["shot_location_y"] = safe_numeric(data[shot_y_col]).fillna(np.nan) if shot_y_col else np.nan

    out["pass_end_location_x"] = safe_numeric(data[pass_end_x_col]).fillna(np.nan) if pass_end_x_col else out["shot_location_x"]
    out["pass_end_location_y"] = safe_numeric(data[pass_end_y_col]).fillna(np.nan) if pass_end_y_col else out["shot_location_y"]

    if led_to_shot_col:
        out["led_to_shot"] = data[led_to_shot_col].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        sp_text = data[outcome_col].astype(str) if outcome_col else ""
        shot_text = data[shot_outcome_col].astype(str) if shot_outcome_col else ""
        out["led_to_shot"] = (out["shot_xg"] > 0) | sp_text.str.contains("shot|goal", case=False, na=False) | shot_text.str.contains("shot|goal", case=False, na=False)

    if goal_col:
        out["goal"] = data[goal_col].astype(str).str.lower().isin(["true", "1", "yes"])
    else:
        sp_text = data[outcome_col].astype(str) if outcome_col else ""
        shot_text = data[shot_outcome_col].astype(str) if shot_outcome_col else ""
        out["goal"] = sp_text.str.contains("goal", case=False, na=False) | shot_text.str.contains("goal", case=False, na=False)

    out["side"] = data[side_col].astype(str) if side_col else out["pass_location_y"].apply(side_from_y)
    out["delivery_zone"] = data[delivery_zone_col].astype(str) if delivery_zone_col else out["pass_end_location_y"].apply(delivery_zone_from_y)
    out["end_zone"] = data[end_zone_col].astype(str) if end_zone_col else out.apply(lambda r: zone_from_end_location(r["pass_end_location_x"], r["pass_end_location_y"]), axis=1)

    if phase_col:
        out["phase"] = data[phase_col].astype(str)
    else:
        event_minute = out["Minute"] + out["Second"] / 60
        out["phase"] = pd.cut(
            event_minute,
            bins=[-0.1, 15, 30, 45, 60, 75, 120],
            labels=["0-15", "16-30", "31-45", "46-60", "61-75", "76+"],
            right=True,
        ).astype(str)

    return out

@st.cache_data
def load_xlsx_bytes(file_bytes):
    raw = pd.read_excel(io.BytesIO(file_bytes))
    return parse_csv_like(raw)

@st.cache_data
def load_csv_bytes(file_bytes):
    raw = pd.read_csv(io.BytesIO(file_bytes))
    return parse_csv_like(raw)

@st.cache_data
def load_default_file_if_present():
    possible_files = ["SWE SP.xlsx", "/mnt/data/SWE SP.xlsx"]
    for f in possible_files:
        if os.path.exists(f):
            raw = pd.read_excel(f)
            return parse_csv_like(raw)
    return build_demo_data()

# =========================================================
# SUMMARY BUILDERS
# =========================================================
def build_type_summary(df):
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("set_piece_type", dropna=False)
        .agg(
            events=("match_id", "size"),
            matches=("match_id", pd.Series.nunique),
            shots=("led_to_shot", "sum"),
            goals=("goal", "sum"),
            total_xg=("shot_xg", "sum"),
        )
        .reset_index()
    )
    out["events_per_match"] = out["events"] / out["matches"].replace(0, np.nan)
    out["shot_rate"] = out["shots"] / out["events"].replace(0, np.nan)
    out["xg_per_event"] = out["total_xg"] / out["events"].replace(0, np.nan)
    return out.sort_values("events", ascending=False)

def build_team_summary(df):
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("team", dropna=False)
        .agg(
            events=("match_id", "size"),
            matches=("match_id", pd.Series.nunique),
            shots=("led_to_shot", "sum"),
            goals=("goal", "sum"),
            total_xg=("shot_xg", "sum"),
            takers=("Taker", pd.Series.nunique),
            six_yard=("end_zone", lambda s: (s == "6-yard box").sum()),
        )
        .reset_index()
    )
    out["events_per_match"] = out["events"] / out["matches"].replace(0, np.nan)
    out["shot_rate"] = out["shots"] / out["events"].replace(0, np.nan)
    out["xg_per_event"] = out["total_xg"] / out["events"].replace(0, np.nan)
    out["six_yard_rate"] = out["six_yard"] / out["events"].replace(0, np.nan)
    return out.sort_values(["xg_per_event", "shot_rate"], ascending=False)

def build_taker_summary(df):
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby(["team", "Taker"], dropna=False)
        .agg(
            events=("match_id", "size"),
            shots=("led_to_shot", "sum"),
            goals=("goal", "sum"),
            total_xg=("shot_xg", "sum"),
        )
        .reset_index()
    )
    out["shot_rate"] = out["shots"] / out["events"].replace(0, np.nan)
    out["xg_per_event"] = out["total_xg"] / out["events"].replace(0, np.nan)
    return out.sort_values(["events", "xg_per_event"], ascending=False)

def build_match_summary(df):
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("Match", dropna=False)
        .agg(
            events=("match_id", "size"),
            shots=("led_to_shot", "sum"),
            goals=("goal", "sum"),
            total_xg=("shot_xg", "sum"),
        )
        .reset_index()
    )
    out["shot_rate"] = out["shots"] / out["events"].replace(0, np.nan)
    out["xg_per_event"] = out["total_xg"] / out["events"].replace(0, np.nan)
    return out.sort_values(["total_xg", "events"], ascending=False)

# =========================================================
# VISUALS
# =========================================================
def draw_pitch(fig, title=None, height=650, half=False):
    y_min = 60 if half else 0
    fig.update_xaxes(range=[0, 80], visible=False)
    fig.update_yaxes(range=[y_min, 120], visible=False, scaleanchor="x", scaleratio=1)
    shapes = [
        dict(type="rect", x0=0, y0=y_min, x1=80, y1=120, line=dict(color="white", width=2)),
        dict(type="rect", x0=18, y0=102, x1=62, y1=120, line=dict(color="white", width=1.5)),
        dict(type="rect", x0=30, y0=114, x1=50, y1=120, line=dict(color="white", width=1.5)),
        dict(type="circle", x0=39.6, y0=107.6, x1=40.4, y1=108.4, fillcolor="white", line=dict(color="white")),
        dict(type="line", x0=36, y0=120, x1=44, y1=120, line=dict(color="#00FF00", width=4)),
    ]
    if not half:
        shapes.insert(1, dict(type="line", x0=0, y0=60, x1=80, y1=60, line=dict(color="white", width=1.5)))
    fig.update_layout(
        title=title,
        height=height,
        template="plotly_dark",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        margin=dict(l=10, r=10, t=40, b=10),
        shapes=shapes,
    )
    return fig

def shotmap_figure(df_shots, title="Shotmap"):
    fig = draw_pitch(go.Figure(), title=title, height=560, half=True)
    plot_df = df_shots.dropna(subset=["shot_location_x", "shot_location_y"]).copy()
    if plot_df.empty:
        return fig

    plot_df["shot_xg"] = pd.to_numeric(plot_df["shot_xg"], errors="coerce").fillna(0)
    sizes = np.clip(plot_df["shot_xg"] * 160 + 12, 12, 55)

    fig.add_trace(go.Scatter(
        x=plot_df["shot_location_y"],
        y=plot_df["shot_location_x"],
        mode="markers",
        marker=dict(size=sizes, opacity=0.78, line=dict(color="white", width=1.1)),
        text=[
            f"<b>Team:</b> {r.get('team','N/A')}<br><b>Taker:</b> {r.get('Taker','N/A')}<br><b>Shooter:</b> {r.get('Shooter','N/A')}<br><b>xG:</b> {r.get('shot_xg',0):.3f}"
            for _, r in plot_df.iterrows()
        ],
        hovertemplate="%{text}<extra></extra>",
        name="Shots",
    ))
    return fig

def delivery_map_figure(df_events, title="Delivery Map"):
    fig = draw_pitch(go.Figure(), title=title, height=650, half=False)
    plot = df_events.dropna(subset=["pass_end_location_x", "pass_end_location_y"]).copy()
    if plot.empty:
        return fig

    for zone, sub in plot.groupby("delivery_zone", dropna=False):
        fig.add_trace(go.Scatter(
            x=80 - sub["pass_end_location_y"],
            y=sub["pass_end_location_x"],
            mode="markers",
            name=str(zone),
            marker=dict(size=11, opacity=0.82, line=dict(width=1, color="white")),
            text=[
                f"<b>Team:</b> {r.get('team','N/A')}<br><b>Taker:</b> {r.get('Taker','N/A')}<br><b>Zone:</b> {r.get('delivery_zone','N/A')}"
                for _, r in sub.iterrows()
            ],
            hovertemplate="%{text}<extra></extra>",
        ))
    return fig

# =========================================================
# APP STATE
# =========================================================
if "segment" not in st.session_state:
    st.session_state["segment"] = None

def go_home():
    st.session_state["segment"] = None

def choose_segment(segment):
    st.session_state["segment"] = segment

# =========================================================
# DATA SOURCE
# =========================================================
def get_data():
    uploaded = st.file_uploader("Upload CSV or Excel dataset", type=["csv", "xlsx"], label_visibility="collapsed")
    if uploaded is not None:
        try:
            file_bytes = uploaded.getvalue()
            if uploaded.name.lower().endswith(".csv"):
                df = load_csv_bytes(file_bytes)
            else:
                df = load_xlsx_bytes(file_bytes)
            return df, f"Loaded: {uploaded.name}"
        except Exception as e:
            st.error(f"Could not parse uploaded file: {e}")
            return build_demo_data(), "Fallback: demo dataset"
    else:
        default_df = load_default_file_if_present()
        label = "Loaded default SWE SP.xlsx" if len(default_df) > 6 else "Using demo dataset"
        return default_df, label

data_df, data_label = get_data()

# =========================================================
# LANDING PAGE
# =========================================================
def landing_page(df):
    st.markdown(
        """
        <div class="hero">
            <div class="hero-title">Allsvenskan <span>Set Piece</span> Studio</div>
            <div class="hero-sub">
                A totally new Streamlit app built around one simple landing page.
                No login. No crowded navigation. Just three clear entry points:
                <b>Free Kick</b>, <b>Corner</b>, and <b>Throw-In</b>.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="upload-bar">
            <div style="font-size:0.82rem;color:{MUTED};text-transform:uppercase;letter-spacing:0.12em;font-weight:700;">Dataset status</div>
            <div style="font-size:1rem;color:{TEXT};font-weight:700;margin-top:6px;">{data_label}</div>
            <div style="font-size:0.88rem;color:{MUTED};margin-top:4px;">Upload a CSV or Excel file at the top to replace the current dataset.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary = build_type_summary(df)
    if not summary.empty:
        c1, c2, c3, c4 = st.columns(4)
        for col, label in zip([c1, c2, c3], ["Free Kick", "Corner", "Throw-In"]):
            with col:
                row = summary[summary["set_piece_type"] == label]
                if row.empty:
                    metric_card(label, "0", "No events")
                else:
                    r = row.iloc[0]
                    metric_card(label, f"{int(r['events']):,}", f"{human_pct(r['shot_rate'])} shot rate")
        with c4:
            metric_card("Matches", f"{df['match_id'].nunique():,}", "Across current dataset")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    cards = [
        ("Free Kick", SUCCESS, "Direct and indirect free-kick routines, shot quality, delivery zones, and match patterns."),
        ("Corner", ACCENT, "Corner volume, end-zone targeting, taker impact, and shot creation from wide dead balls."),
        ("Throw-In", ORANGE, "Attacking throw-ins, long-throw patterns, end zones, and team usage."),
    ]
    cols = st.columns(3)
    for col, (label, color, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="segment-card">
                    <div class="segment-pill" style="background:{color}22;color:{TEXT};border-color:{color}55;">Segment</div>
                    <div class="segment-title">{label}</div>
                    <div class="segment-sub">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Open {label}", key=f"open_{label}"):
                choose_segment(label)
                st.rerun()

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    section_header("League Snapshot", "Top-level comparison across the three segments")
    if summary.empty:
        empty_state()
    else:
        c5, c6 = st.columns(2)
        with c5:
            fig = px.bar(
                summary,
                x="set_piece_type",
                y="events",
                color="set_piece_type",
                color_discrete_map=TYPE_COLORS,
                title="Volume by Segment",
                text="events",
            )
            st.plotly_chart(figure_layout(fig, 360, "Volume by Segment"), use_container_width=True)
        with c6:
            fig = px.bar(
                summary,
                x="set_piece_type",
                y="xg_per_event",
                color="set_piece_type",
                color_discrete_map=TYPE_COLORS,
                title="xG per Event",
                text_auto=".3f",
            )
            st.plotly_chart(figure_layout(fig, 360, "xG per Event"), use_container_width=True)

# =========================================================
# SEGMENT PAGE
# =========================================================
def render_segment(df, segment_name):
    seg_df = df[df["set_piece_type"] == segment_name].copy()

    c1, c2 = st.columns([1, 6])
    with c1:
        if st.button("← Home"):
            go_home()
            st.rerun()
    with c2:
        st.markdown(
            f"""
            <div class="hero" style="padding:24px 26px 20px 26px;">
                <div class="hero-title" style="font-size:2.25rem;">{segment_name} <span>Studio</span></div>
                <div class="hero-sub">Focused analysis workspace for {segment_name.lower()} events only.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if seg_df.empty:
        empty_state(f"No {segment_name.lower()} data found.")
        return

    all_teams = sorted(seg_df["team"].dropna().astype(str).unique().tolist())
    all_matches = sorted(seg_df["Match"].dropna().astype(str).unique().tolist())
    all_takers = sorted([str(x) for x in seg_df["Taker"].dropna().astype(str).unique() if str(x).strip()])

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    section_header("Filters", "Keep the segment view clean and focused")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        team_filter = st.selectbox("Team", ["All Teams"] + all_teams, key=f"team_{segment_name}")
    with f2:
        side_filter = st.selectbox("Side", ["Both", "Left", "Right", "Unknown"], key=f"side_{segment_name}")
    with f3:
        match_filter = st.multiselect("Matches", all_matches, key=f"match_{segment_name}")
    with f4:
        taker_filter = st.multiselect("Takers", all_takers, key=f"taker_{segment_name}")
    with f5:
        shots_only = st.checkbox("Shots only", key=f"shots_only_{segment_name}")
    st.markdown("</div>", unsafe_allow_html=True)

    work = seg_df.copy()
    if team_filter != "All Teams":
        work = work[work["team"] == team_filter]
    if side_filter != "Both":
        work = work[work["side"] == side_filter]
    if match_filter:
        work = work[work["Match"].isin(match_filter)]
    if taker_filter:
        work = work[work["Taker"].astype(str).isin([str(x) for x in taker_filter])]
    if shots_only:
        work = work[work["led_to_shot"]]

    if work.empty:
        empty_state("No events match the current filters.")
        return

    team_summary = build_team_summary(work)
    taker_summary = build_taker_summary(work)
    match_summary = build_match_summary(work)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        metric_card("Events", f"{len(work):,}", segment_name)
    with k2:
        metric_card("Matches", f"{work['match_id'].nunique():,}", "Current filtered view")
    with k3:
        metric_card("Shots", f"{int(work['led_to_shot'].sum()):,}", "From set pieces")
    with k4:
        metric_card("Goals", f"{int(work['goal'].sum()):,}", "From set pieces")
    with k5:
        metric_card("Shot Rate", human_pct(work["led_to_shot"].mean()), f"{human_val(work['shot_xg'].sum(), 2)} total xG")

    tabs = st.tabs(["Overview", "Visuals", "Teams", "Takers", "Matches", "Data"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            zone_df = work.groupby("end_zone", dropna=False).size().reset_index(name="events")
            fig = px.bar(
                zone_df.sort_values("events", ascending=False),
                x="end_zone",
                y="events",
                color="end_zone",
                color_discrete_sequence=QUAL_PALETTE,
                title="End-Zone Distribution",
            )
            st.plotly_chart(figure_layout(fig, 360, "End-Zone Distribution"), use_container_width=True)
        with c2:
            phase_df = work.groupby("phase", dropna=False).size().reset_index(name="events")
            fig = px.line(
                phase_df,
                x="phase",
                y="events",
                markers=True,
                title="Event Timing by Match Phase",
                color_discrete_sequence=[TYPE_COLORS.get(segment_name, ACCENT)],
            )
            st.plotly_chart(figure_layout(fig, 360, "Event Timing by Match Phase"), use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            fig = px.histogram(
                work,
                x="Minute",
                nbins=20,
                title="Minute Distribution",
                color_discrete_sequence=[TYPE_COLORS.get(segment_name, ACCENT)],
            )
            st.plotly_chart(figure_layout(fig, 360, "Minute Distribution"), use_container_width=True)
        with c4:
            scatter_df = team_summary.copy()
            fig = px.scatter(
                scatter_df,
                x="shot_rate",
                y="xg_per_event",
                size="events",
                text="team",
                title="Team Efficiency Map",
                color_discrete_sequence=[TYPE_COLORS.get(segment_name, ACCENT)],
            )
            fig.update_traces(textposition="top center")
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(figure_layout(fig, 360, "Team Efficiency Map"), use_container_width=True)

    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            shot_df = work.dropna(subset=["shot_location_x", "shot_location_y"])
            if shot_df.empty:
                empty_state("No shot locations in this view.")
            else:
                st.plotly_chart(shotmap_figure(shot_df, f"{segment_name} Shotmap"), use_container_width=True)
        with c2:
            del_df = work.dropna(subset=["pass_end_location_x", "pass_end_location_y"])
            if del_df.empty:
                empty_state("No delivery coordinates in this view.")
            else:
                st.plotly_chart(delivery_map_figure(del_df, f"{segment_name} Delivery Map"), use_container_width=True)

    with tabs[2]:
        if team_summary.empty:
            empty_state("No team summary available.")
        else:
            st.dataframe(team_summary.reset_index(drop=True), use_container_width=True, height=420)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(
                    team_summary.head(12),
                    x="team",
                    y="events_per_match",
                    color="xg_per_event",
                    color_continuous_scale="Blues",
                    title="Events per Match",
                    hover_data=["shot_rate", "events", "matches"],
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(figure_layout(fig, 360, "Events per Match"), use_container_width=True)
            with c2:
                fig = px.bar(
                    team_summary.head(12),
                    x="team",
                    y="shot_rate",
                    color="six_yard_rate",
                    color_continuous_scale="Blues",
                    title="Shot Rate by Team",
                    hover_data=["events", "total_xg"],
                )
                fig.update_layout(coloraxis_showscale=False)
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(figure_layout(fig, 360, "Shot Rate by Team"), use_container_width=True)

    with tabs[3]:
        if taker_summary.empty:
            empty_state("No taker summary available.")
        else:
            st.dataframe(taker_summary.reset_index(drop=True), use_container_width=True, height=420)
            plot = taker_summary[taker_summary["events"] >= 1].head(12).copy()
            if not plot.empty:
                plot["label"] = plot["Taker"].astype(str) + " (" + plot["team"].astype(str) + ")"
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(
                        plot.sort_values("xg_per_event", ascending=False),
                        x="label",
                        y="xg_per_event",
                        color="xg_per_event",
                        color_continuous_scale="Blues",
                        title="Top Takers by xG/Event",
                    )
                    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35)
                    st.plotly_chart(figure_layout(fig, 360, "Top Takers by xG/Event"), use_container_width=True)
                with c2:
                    fig = px.bar(
                        plot.sort_values("shot_rate", ascending=False),
                        x="label",
                        y="shot_rate",
                        color="shot_rate",
                        color_continuous_scale="Blues",
                        title="Top Takers by Shot Rate",
                    )
                    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35)
                    fig.update_yaxes(tickformat=".0%")
                    st.plotly_chart(figure_layout(fig, 360, "Top Takers by Shot Rate"), use_container_width=True)

    with tabs[4]:
        if match_summary.empty:
            empty_state("No match summary available.")
        else:
            st.dataframe(match_summary.reset_index(drop=True), use_container_width=True, height=460)

    with tabs[5]:
        st.dataframe(work.reset_index(drop=True), use_container_width=True, height=520)
        csv_bytes = work.to_csv(index=False).encode()
        st.download_button(
            f"Download {segment_name} CSV",
            csv_bytes,
            f"{segment_name.lower().replace('-', '_').replace(' ', '_')}_events.csv",
            "text/csv",
            use_container_width=True,
        )

# =========================================================
# ROUTER
# =========================================================
if st.session_state["segment"] is None:
    landing_page(data_df)
else:
    render_segment(data_df, st.session_state["segment"])

st.markdown(
    '<div class="footer-note">⚽ Allsvenskan Set Piece Studio · Brand new Streamlit build · Free Kick + Corner + Throw-In</div>',
    unsafe_allow_html=True,
)
