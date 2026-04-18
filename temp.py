import os
from io import BytesIO
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Allsvenskan Set Piece Studio",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FILE_NAME = "SWE SP.xlsx"

BG = "#07111f"
BG_2 = "#0b1730"
CARD = "#101a2b"
CARD_2 = "#16243a"
TEXT = "#f3f7fc"
MUTED = "#99adc7"
ACCENT = "#5da8ff"
SUCCESS = "#34d399"
WARNING = "#fbbf24"
PURPLE = "#a78bfa"
ORANGE = "#fb923c"
BORDER = "rgba(255,255,255,0.08)"

TYPE_COLORS = {
    "Corner": ACCENT,
    "Free Kick": SUCCESS,
    "Throw-In": ORANGE,
    "Other": PURPLE,
}

QUAL_PALETTE = [ACCENT, SUCCESS, WARNING, PURPLE, ORANGE, "#8ad6ff", "#6ee7b7"]
px.defaults.template = "plotly_dark"

CSS = f'''
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
    padding-top: 1.1rem;
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
    max-width: 920px;
}}
.segment-card {{
    background: linear-gradient(160deg, {CARD} 0%, {CARD_2} 100%);
    border: 1px solid {BORDER};
    border-radius: 26px;
    padding: 20px 20px 18px 20px;
    min-height: 220px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
}}
.segment-pill {{
    display: inline-block;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
    border: 1px solid rgba(255,255,255,0.12);
    margin-bottom: 0.8rem;
}}
.segment-title {{
    font-size: 1.45rem;
    font-weight: 900;
    margin-bottom: 0.4rem;
}}
.segment-sub {{
    color: {MUTED};
    font-size: 0.94rem;
    line-height: 1.55;
    min-height: 72px;
}}
.kpi {{
    background: linear-gradient(160deg, {CARD} 0%, {CARD_2} 100%);
    border: 1px solid {BORDER};
    border-radius: 18px;
    padding: 14px 14px 12px 14px;
    min-height: 100px;
}}
.kpi-label {{
    color: {MUTED};
    text-transform: uppercase;
    font-size: 0.68rem;
    letter-spacing: 0.10em;
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
    margin: 0.15rem 0 0.25rem 0;
}}
.section-sub {{
    color: {MUTED};
    font-size: 0.92rem;
    margin-bottom: 0.85rem;
}}
.panel {{
    background: rgba(255,255,255,0.02);
    border: 1px solid {BORDER};
    border-radius: 22px;
    padding: 18px 18px 10px 18px;
    margin-bottom: 14px;
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
.footer-note {{
    color: #6b87a8;
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
    padding: 0.65rem 0.85rem;
}}
div.stButton > button:hover {{
    border-color: rgba(93,168,255,0.30);
    background: rgba(93,168,255,0.10);
}}
</style>
'''
st.markdown(CSS, unsafe_allow_html=True)

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
        unsafe_allow_html=True
    )

def metric_card(label, value, foot=""):
    st.markdown(
        f'''
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-foot">{foot}</div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

def empty_state(msg="No data for current selection."):
    st.markdown(f'<div class="empty-state">{msg}</div>', unsafe_allow_html=True)

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
    if df_shots.empty:
        return fig

    plot_df = df_shots.dropna(subset=["shot_location_x", "shot_location_y"]).copy()
    if plot_df.empty:
        return fig

    plot_df["shot_xg"] = pd.to_numeric(plot_df["shot_xg"], errors="coerce").fillna(0)
    size = np.clip(plot_df["shot_xg"] * 160 + 12, 12, 55)

    fig.add_trace(go.Scatter(
        x=plot_df["shot_location_y"],
        y=plot_df["shot_location_x"],
        mode="markers",
        marker=dict(size=size, opacity=0.78, line=dict(color="white", width=1.1)),
        text=[
            f"<b>Team:</b> {r.get('team','N/A')}<br><b>Taker:</b> {r.get('Taker','N/A')}<br><b>xG:</b> {r.get('shot_xg',0):.3f}"
            for _, r in plot_df.iterrows()
        ],
        hovertemplate="%{text}<extra></extra>",
        name="Shots",
    ))
    return fig

def delivery_map_figure(df_events, title="Delivery Map"):
    fig = draw_pitch(go.Figure(), title=title, height=650, half=False)
    plot = df_events.dropna(subset=["pass_location_x", "pass_location_y", "pass_end_location_x", "pass_end_location_y"]).copy()
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

def build_type_summary(df):
    if df.empty:
        return pd.DataFrame()
    out = (
        df.groupby("set_piece_type", dropna=False)
        .agg(events=("match_id", "size"), matches=("match_id", pd.Series.nunique), shots=("led_to_shot", "sum"), total_xg=("shot_xg", "sum"), goals=("goal", "sum"))
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
        .agg(events=("match_id", "size"), matches=("match_id", pd.Series.nunique), shots=("led_to_shot", "sum"), total_xg=("shot_xg", "sum"), takers=("Taker", pd.Series.nunique), six_yard=("end_zone", lambda s: (s == "6-yard box").sum()))
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
        .agg(events=("match_id", "size"), shots=("led_to_shot", "sum"), total_xg=("shot_xg", "sum"), goals=("goal", "sum"))
        .reset_index()
    )
    out["shot_rate"] = out["shots"] / out["events"].replace(0, np.nan)
    out["xg_per_event"] = out["total_xg"] / out["events"].replace(0, np.nan)
    return out.sort_values(["events", "xg_per_event"], ascending=False)

@st.cache_data
def load_data():
    possible_files = [FILE_NAME, "/mnt/data/SWE SP.xlsx"]
    for f in possible_files:
        if os.path.exists(f):
            return pd.read_excel(f)
    raise FileNotFoundError(f"{FILE_NAME} not found.")

@st.cache_data
def prepare_data(raw_df):
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    _f = lambda *c: find_col(df, list(c))

    match_id_col = _f("match_id")
    match_col = _f("match", "match_name")
    team_col = _f("team.name", "team")
    minute_col = _f("minute")
    second_col = _f("second")
    timestamp_col = _f("timestamp")
    sp_type_col = _f("SP_Type", "set_piece_type")
    xg_col = _f("shot.statsbomb_xg", "shot_xg")
    taker_col = _f("taker")
    shooter_col = _f("shooter")
    outcome_col = _f("sp_outcome", "outcome")
    shot_outcome_col = _f("shot.outcome.name", "shot_outcome")
    pass_loc_col = _f("location.pass")
    shot_loc_col = _f("location.shot")

    if match_id_col is None:
        raise ValueError("Missing match_id column")
    if team_col is None:
        raise ValueError("Missing team column")
    if sp_type_col is None:
        raise ValueError("Missing SP_Type column")

    if minute_col is None and timestamp_col is not None:
        ts = df[timestamp_col].astype(str).str.split(":", expand=True)
        if ts.shape[1] >= 3:
            df["Minute_tmp"] = pd.to_numeric(ts[1], errors="coerce")
            df["Second_tmp"] = pd.to_numeric(ts[2].str.replace(r"[^0-9.]", "", regex=True), errors="coerce")
            minute_col = "Minute_tmp"
            second_col = "Second_tmp"

    if minute_col is None:
        df["Minute_tmp"] = 0
        minute_col = "Minute_tmp"
    if second_col is None:
        df["Second_tmp"] = 0
        second_col = "Second_tmp"
    if match_col is None:
        df["Match_tmp"] = "Match " + df[match_id_col].astype(str)
        match_col = "Match_tmp"

    rename_map = {
        match_id_col: "match_id",
        match_col: "Match",
        team_col: "team",
        minute_col: "Minute",
        second_col: "Second",
        sp_type_col: "set_piece_type_raw",
    }
    optional = {
        xg_col: "shot_xg",
        taker_col: "Taker",
        shooter_col: "Shooter",
        outcome_col: "SP_outcome",
        shot_outcome_col: "shot_outcome",
        pass_loc_col: "pass_location_raw",
        shot_loc_col: "shot_location_raw",
    }
    for src, dst in optional.items():
        if src is not None:
            rename_map[src] = dst
    df = df.rename(columns=rename_map)

    for c in ["shot_xg", "Taker", "Shooter", "SP_outcome", "shot_outcome", "pass_location_raw", "shot_location_raw"]:
        if c not in df.columns:
            df[c] = np.nan

    df["Minute"] = safe_numeric(df["Minute"]).fillna(0)
    df["Second"] = safe_numeric(df["Second"]).fillna(0)
    df["shot_xg"] = safe_numeric(df["shot_xg"]).fillna(0)
    df["team"] = df["team"].astype(str).str.strip()
    df["Match"] = df["Match"].astype(str).str.strip()
    df["set_piece_type"] = df["set_piece_type_raw"].apply(set_piece_bucket)
    df["event_minute"] = df["Minute"] + df["Second"] / 60

    df["pass_location_x"] = df["pass_location_raw"].apply(lambda x: parse_xy(x, 0))
    df["pass_location_y"] = df["pass_location_raw"].apply(lambda x: parse_xy(x, 1))
    df["shot_location_x"] = df["shot_location_raw"].apply(lambda x: parse_xy(x, 0))
    df["shot_location_y"] = df["shot_location_raw"].apply(lambda x: parse_xy(x, 1))
    df["pass_end_location_x"] = df["shot_location_x"]
    df["pass_end_location_y"] = df["shot_location_y"]

    sp_text = df["SP_outcome"].astype(str)
    shot_text = df["shot_outcome"].astype(str)
    df["led_to_shot"] = (df["shot_xg"] > 0) | sp_text.str.contains("shot|goal", case=False, na=False) | shot_text.ne("nan")
    df["goal"] = shot_text.str.contains("goal", case=False, na=False) | sp_text.str.contains("goal", case=False, na=False)

    df["side"] = df["pass_location_y"].apply(side_from_y)
    df["delivery_zone"] = df["pass_end_location_y"].apply(delivery_zone_from_y)
    df["end_zone"] = df.apply(lambda r: zone_from_end_location(r["pass_end_location_x"], r["pass_end_location_y"]), axis=1)

    df["phase"] = pd.cut(
        df["event_minute"],
        bins=[-0.1, 15, 30, 45, 60, 75, 120],
        labels=["0-15", "16-30", "31-45", "46-60", "61-75", "76+"],
        right=True,
    ).astype(str)

    return df

df = prepare_data(load_data())

if "segment" not in st.session_state:
    st.session_state["segment"] = None

def go_home():
    st.session_state["segment"] = None

def choose_segment(segment):
    st.session_state["segment"] = segment

def landing_page():
    st.markdown(
        '''
        <div class="hero">
            <div class="hero-title">Allsvenskan <span>Set Piece</span> Studio</div>
            <div class="hero-sub">
                A redesigned analysis workspace built around three clear entry points.
                Pick the set-piece segment you want to study and jump straight into focused visuals,
                team performance, taker profiles, and match-level breakdowns.
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    type_summary = build_type_summary(df)

    if not type_summary.empty:
        cols = st.columns(4)
        for i, label in enumerate(["Corner", "Free Kick", "Throw-In"]):
            with cols[i]:
                row = type_summary[type_summary["set_piece_type"] == label]
                if row.empty:
                    metric_card(label, "0", "No events")
                else:
                    r = row.iloc[0]
                    metric_card(label, f"{int(r['events']):,}", f"{human_pct(r['shot_rate'])} shot rate")
        with cols[3]:
            metric_card("Matches", f"{df['match_id'].nunique():,}", "Across all set pieces")

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    cards = [
        ("Free Kick", SUCCESS, "Direct and indirect free-kick routines, shot quality, delivery zones, and match patterns."),
        ("Corner", ACCENT, "Corner volume, end-zone targeting, taker impact, and shot creation from wide dead balls."),
        ("Throw-In", ORANGE, "Attacking throw-in sequences, long-throw profiles, location outcomes, and delivery focus."),
    ]

    for col, (label, color, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f'''
                <div class="segment-card">
                    <div class="segment-pill" style="background:{color}22;color:{TEXT};border-color:{color}55;">{label}</div>
                    <div class="segment-title">{label}</div>
                    <div class="segment-sub">{desc}</div>
                </div>
                ''',
                unsafe_allow_html=True,
            )
            if st.button(f"Open {label}", key=f"open_{label}"):
                choose_segment(label)
                st.rerun()

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    section_header("League Snapshot", "Top-level comparison across all three segments")
    if type_summary.empty:
        empty_state()
    else:
        c4, c5 = st.columns(2)
        with c4:
            fig = px.bar(
                type_summary,
                x="set_piece_type",
                y="events",
                color="set_piece_type",
                color_discrete_map=TYPE_COLORS,
                title="Volume by Segment",
                text="events",
            )
            st.plotly_chart(figure_layout(fig, 360, "Volume by Segment"), use_container_width=True)
        with c5:
            fig = px.bar(
                type_summary,
                x="set_piece_type",
                y="xg_per_event",
                color="set_piece_type",
                color_discrete_map=TYPE_COLORS,
                title="xG per Event",
                text_auto=".3f",
            )
            st.plotly_chart(figure_layout(fig, 360, "xG per Event"), use_container_width=True)

def render_segment(segment_name):
    seg_df = df[df["set_piece_type"] == segment_name].copy()

    top_bar_1, top_bar_2 = st.columns([1, 6])
    with top_bar_1:
        if st.button("← Home"):
            go_home()
            st.rerun()
    with top_bar_2:
        st.markdown(
            f'''
            <div class="hero" style="padding:24px 26px 20px 26px;">
                <div class="hero-title" style="font-size:2.25rem;">{segment_name} <span>Studio</span></div>
                <div class="hero-sub">Focused analysis for {segment_name.lower()} events only.</div>
            </div>
            ''',
            unsafe_allow_html=True,
        )

    if seg_df.empty:
        empty_state(f"No {segment_name.lower()} data found.")
        return

    teams = sorted(seg_df["team"].dropna().unique().tolist())
    matches = sorted(seg_df["Match"].dropna().unique().tolist())
    takers = sorted([str(x) for x in seg_df["Taker"].dropna().astype(str).unique() if str(x).strip()])

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    section_header("Filters", "Keep each segment simple and focused")
    f1, f2, f3, f4, f5 = st.columns(5)
    with f1:
        team_filter = st.selectbox("Team", ["All Teams"] + teams, key=f"team_{segment_name}")
    with f2:
        side_filter = st.selectbox("Side", ["Both", "Left", "Right"], key=f"side_{segment_name}")
    with f3:
        match_filter = st.multiselect("Matches", matches, key=f"match_{segment_name}")
    with f4:
        taker_filter = st.multiselect("Takers", takers, key=f"taker_{segment_name}")
    with f5:
        shot_only = st.checkbox("Shots only", key=f"shots_only_{segment_name}")
    st.markdown('</div>', unsafe_allow_html=True)

    work = seg_df.copy()
    if team_filter != "All Teams":
        work = work[work["team"] == team_filter]
    if side_filter != "Both":
        work = work[work["side"] == side_filter]
    if match_filter:
        work = work[work["Match"].isin(match_filter)]
    if taker_filter:
        work = work[work["Taker"].astype(str).isin([str(x) for x in taker_filter])]
    if shot_only:
        work = work[work["led_to_shot"]]

    if work.empty:
        empty_state("No events match the current filters.")
        return

    team_summary = build_team_summary(work)
    taker_summary_df = build_taker_summary(work)

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        metric_card("Events", f"{len(work):,}", segment_name)
    with k2:
        metric_card("Matches", f"{work['match_id'].nunique():,}", "Current view")
    with k3:
        metric_card("Shots", f"{int(work['led_to_shot'].sum()):,}", "From set pieces")
    with k4:
        metric_card("Shot Rate", human_pct(work["led_to_shot"].mean()), "Shots / event")
    with k5:
        metric_card("Total xG", human_val(work["shot_xg"].sum(), 2), "Generated")

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
            summary = work.groupby("team", dropna=False).agg(events=("match_id", "size"), shots=("led_to_shot", "sum"), total_xg=("shot_xg", "sum")).reset_index()
            summary["shot_rate"] = summary["shots"] / summary["events"].replace(0, np.nan)
            fig = px.scatter(
                summary,
                x="shot_rate",
                y="total_xg",
                size="events",
                text="team",
                title="Team Efficiency Map",
                color_discrete_sequence=[TYPE_COLORS.get(segment_name, ACCENT)],
            )
            fig.update_traces(textposition="top center")
            fig.update_yaxes(title="Total xG")
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
            delivery_df = work.dropna(subset=["pass_location_x", "pass_location_y", "pass_end_location_x", "pass_end_location_y"])
            if delivery_df.empty:
                empty_state("No delivery coordinates in this view.")
            else:
                st.plotly_chart(delivery_map_figure(delivery_df, f"{segment_name} Delivery Map"), use_container_width=True)

    with tabs[2]:
        if team_summary.empty:
            empty_state("No team summary available.")
        else:
            st.dataframe(team_summary.reset_index(drop=True), use_container_width=True, height=430)
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
        if taker_summary_df.empty:
            empty_state("No taker summary available.")
        else:
            st.dataframe(taker_summary_df.reset_index(drop=True), use_container_width=True, height=430)
            plot = taker_summary_df[taker_summary_df["events"] >= 2].head(12).copy()
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
        match_view = work.groupby("Match", dropna=False).agg(events=("match_id", "size"), shots=("led_to_shot", "sum"), total_xg=("shot_xg", "sum"), goals=("goal", "sum")).reset_index().sort_values(["total_xg", "events"], ascending=False)
        match_view["shot_rate"] = match_view["shots"] / match_view["events"].replace(0, np.nan)
        match_view["xg_per_event"] = match_view["total_xg"] / match_view["events"].replace(0, np.nan)
        st.dataframe(match_view.reset_index(drop=True), use_container_width=True, height=450)

    with tabs[5]:
        st.dataframe(work.reset_index(drop=True), use_container_width=True, height=520)
        csv = work.to_csv(index=False).encode()
        st.download_button(
            f"Download {segment_name} CSV",
            csv,
            f"{segment_name.lower().replace('-', '_').replace(' ', '_')}_events.csv",
            "text/csv",
            use_container_width=True,
        )

if st.session_state["segment"] is None:
    landing_page()
else:
    render_segment(st.session_state["segment"])

st.markdown(
    '<div class="footer-note">⚽ Allsvenskan Set Piece Studio · Redesigned landing experience · Free Kick + Corner + Throw-In</div>',
    unsafe_allow_html=True,
)
