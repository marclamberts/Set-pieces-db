import os
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Set Piece Studio",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

FILE_PATHS = [
    "SWE SP.xlsx",
    "/mnt/data/SWE SP.xlsx",
]

# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
    .block-container {
        max-width: 1500px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .kpi {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 16px;
        background: rgba(255,255,255,0.03);
    }
    .hero {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 24px;
        padding: 24px;
        background: linear-gradient(135deg, rgba(93,168,255,0.14), rgba(52,211,153,0.06));
        margin-bottom: 16px;
    }
    .segment-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 20px;
        padding: 20px;
        background: rgba(255,255,255,0.03);
        min-height: 220px;
    }
    .status-card {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px;
        background: rgba(255,255,255,0.03);
        margin-bottom: 16px;
    }
    .panel {
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 16px;
        background: rgba(255,255,255,0.03);
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================
def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")


def human_pct(v, decimals=1):
    if pd.isna(v):
        return "—"
    return f"{v*100:.{decimals}f}%"


def find_col(df, names):
    cols = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    for c in df.columns:
        c_low = str(c).strip().lower()
        for n in names:
            if n.lower() in c_low:
                return c
    return None


def parse_xy(cell, i):
    if pd.isna(cell):
        return np.nan
    try:
        parts = [float(x.strip()) for x in str(cell).split(",")]
        return parts[i] if len(parts) > i else np.nan
    except Exception:
        return np.nan


def sp_type_map(x):
    s = str(x).lower()
    if "corner" in s:
        return "Corner"
    if "free" in s:
        return "Free Kick"
    if "throw" in s:
        return "Throw-In"
    return "Other"


def metric_card(label, value, foot=""):
    st.markdown(
        f"""
        <div class="kpi">
            <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; color:#9aa9bf; font-weight:700;">{label}</div>
            <div style="font-size:1.8rem; font-weight:900; margin-top:8px;">{value}</div>
            <div style="font-size:0.82rem; color:#9aa9bf; margin-top:6px;">{foot}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(msg="No data for current selection."):
    st.markdown(
        f"""
        <div style="
            text-align:center;
            padding:56px 24px;
            color:#9aa9bf;
            font-size:0.95rem;
            border:1px dashed rgba(255,255,255,0.10);
            border-radius:18px;
            background:rgba(255,255,255,0.015);
        ">
            {msg}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# DATA LOAD
# =========================================================
@st.cache_data(show_spinner=False)
def load_raw_data():
    for path in FILE_PATHS:
        if os.path.exists(path):
            return pd.read_excel(path), path
    raise FileNotFoundError(
        "Could not find SWE SP.xlsx. Put it next to the app or in /mnt/data/"
    )


@st.cache_data(show_spinner=False)
def prepare(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    team = find_col(df, ["team", "team.name"])
    sp = find_col(df, ["SP_Type", "set_piece_type"])
    xg = find_col(df, ["shot_xg", "shot.statsbomb_xg"])
    minute = find_col(df, ["minute"])
    second = find_col(df, ["second"])
    match = find_col(df, ["match"])
    match_id = find_col(df, ["match_id"])
    shot_loc = find_col(df, ["location.shot"])
    pass_loc = find_col(df, ["location.pass"])
    taker = find_col(df, ["Taker", "taker"])
    shooter = find_col(df, ["Shooter", "shooter"])
    outcome = find_col(df, ["SP_outcome", "sp_outcome"])
    shot_outcome = find_col(df, ["shot_outcome", "shot.outcome.name"])

    if team is None or sp is None:
        raise ValueError(
            "Workbook must contain at least team/team.name and SP_Type/set_piece_type columns."
        )

    out = pd.DataFrame()
    out["team"] = df[team].astype(str)
    out["Match"] = df[match].astype(str) if match else "Match"
    out["match_id"] = (
        df[match_id].astype(str)
        if match_id
        else pd.Series(np.arange(1, len(df) + 1).astype(str))
    )
    out["Minute"] = safe_numeric(df[minute]).fillna(0) if minute else 0
    out["Second"] = safe_numeric(df[second]).fillna(0) if second else 0
    out["Taker"] = df[taker].astype(str) if taker else ""
    out["Shooter"] = df[shooter].astype(str) if shooter else ""
    out["type"] = df[sp].apply(sp_type_map)
    out["shot_xg"] = safe_numeric(df[xg]).fillna(0) if xg else 0.0

    if shot_loc:
        out["x"] = df[shot_loc].apply(lambda x: parse_xy(x, 0))
        out["y"] = df[shot_loc].apply(lambda x: parse_xy(x, 1))
    else:
        out["x"] = np.nan
        out["y"] = np.nan

    if pass_loc:
        out["pass_x"] = df[pass_loc].apply(lambda x: parse_xy(x, 0))
        out["pass_y"] = df[pass_loc].apply(lambda x: parse_xy(x, 1))
    else:
        out["pass_x"] = np.nan
        out["pass_y"] = np.nan

    sp_text = df[outcome].astype(str) if outcome else ""
    shot_text = df[shot_outcome].astype(str) if shot_outcome else ""

    out["shot"] = (
        (out["shot_xg"] > 0)
        | sp_text.str.contains("shot|goal", case=False, na=False)
        | shot_text.str.contains("shot|goal", case=False, na=False)
    )
    out["goal"] = (
        sp_text.str.contains("goal", case=False, na=False)
        | shot_text.str.contains("goal", case=False, na=False)
    )

    out["side"] = np.where(
        out["pass_y"].isna(),
        "Unknown",
        np.where(out["pass_y"] < 40, "Right", "Left"),
    )

    event_minute = out["Minute"] + out["Second"] / 60
    out["phase"] = pd.cut(
        event_minute,
        bins=[-0.1, 15, 30, 45, 60, 75, 120],
        labels=["0-15", "16-30", "31-45", "46-60", "61-75", "76+"],
        right=True,
    ).astype(str)

    return out


@st.cache_data(show_spinner=False)
def build_aggregates(df):
    type_summary = (
        df.groupby("type", dropna=False)
        .agg(
            events=("type", "size"),
            matches=("match_id", pd.Series.nunique),
            shots=("shot", "sum"),
            goals=("goal", "sum"),
            xg=("shot_xg", "sum"),
        )
        .reset_index()
    )
    type_summary["shot_rate"] = (
        type_summary["shots"] / type_summary["events"].replace(0, np.nan)
    )
    type_summary["xg_per_event"] = (
        type_summary["xg"] / type_summary["events"].replace(0, np.nan)
    )

    team_summary = (
        df.groupby(["type", "team"], dropna=False)
        .agg(
            events=("team", "size"),
            matches=("match_id", pd.Series.nunique),
            shots=("shot", "sum"),
            goals=("goal", "sum"),
            xg=("shot_xg", "sum"),
        )
        .reset_index()
    )
    team_summary["shot_rate"] = (
        team_summary["shots"] / team_summary["events"].replace(0, np.nan)
    )
    team_summary["xg_per_event"] = (
        team_summary["xg"] / team_summary["events"].replace(0, np.nan)
    )
    team_summary["events_per_match"] = (
        team_summary["events"] / team_summary["matches"].replace(0, np.nan)
    )

    taker_summary = (
        df.groupby(["type", "team", "Taker"], dropna=False)
        .agg(
            events=("Taker", "size"),
            shots=("shot", "sum"),
            goals=("goal", "sum"),
            xg=("shot_xg", "sum"),
        )
        .reset_index()
    )
    taker_summary["shot_rate"] = (
        taker_summary["shots"] / taker_summary["events"].replace(0, np.nan)
    )
    taker_summary["xg_per_event"] = (
        taker_summary["xg"] / taker_summary["events"].replace(0, np.nan)
    )

    match_summary = (
        df.groupby(["type", "Match"], dropna=False)
        .agg(
            events=("Match", "size"),
            shots=("shot", "sum"),
            goals=("goal", "sum"),
            xg=("shot_xg", "sum"),
        )
        .reset_index()
    )
    match_summary["shot_rate"] = (
        match_summary["shots"] / match_summary["events"].replace(0, np.nan)
    )

    phase_summary = (
        df.groupby(["type", "phase"], dropna=False)
        .size()
        .reset_index(name="events")
    )

    return type_summary, team_summary, taker_summary, match_summary, phase_summary


# =========================================================
# FIGURES
# =========================================================
def shotmap(data, title):
    fig = go.Figure()
    d = data.dropna(subset=["x", "y"]).copy()

    if not d.empty:
        fig.add_trace(
            go.Scattergl(
                x=d["y"],
                y=d["x"],
                mode="markers",
                marker=dict(
                    size=np.clip(d["shot_xg"].fillna(0) * 120 + 8, 8, 28),
                    opacity=0.75,
                    line=dict(width=1, color="white"),
                ),
                text=[
                    f"{r['team']} | {r['Shooter'] or 'Unknown'} | xG {r['shot_xg']:.3f}"
                    for _, r in d.iterrows()
                ],
                hovertemplate="%{text}<extra></extra>",
                name="Shots",
            )
        )

    fig.update_xaxes(range=[0, 80], visible=False)
    fig.update_yaxes(range=[60, 120], visible=False)
    fig.update_layout(
        title=title,
        height=520,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#0e1117",
        margin=dict(l=10, r=10, t=45, b=10),
        shapes=[
            dict(type="rect", x0=0, y0=60, x1=80, y1=120, line=dict(color="white", width=2)),
            dict(type="rect", x0=18, y0=102, x1=62, y1=120, line=dict(color="white", width=1.4)),
            dict(type="rect", x0=30, y0=114, x1=50, y1=120, line=dict(color="white", width=1.4)),
            dict(type="circle", x0=39.6, y0=107.6, x1=40.4, y1=108.4, fillcolor="white", line=dict(color="white")),
        ],
        font=dict(color="white"),
    )
    return fig


# =========================================================
# STATE
# =========================================================
if "page" not in st.session_state:
    st.session_state["page"] = "home"


def go(page):
    st.session_state["page"] = page


# =========================================================
# LOAD APP DATA
# =========================================================
try:
    raw_df, path_used = load_raw_data()
    df = prepare(raw_df)
    type_summary, team_summary, taker_summary, match_summary, phase_summary = build_aggregates(df)
except Exception as e:
    st.error("Failed to load workbook.")
    st.exception(e)
    st.stop()


# =========================================================
# HOME
# =========================================================
if st.session_state["page"] == "home":
    st.markdown(
        """
        <div class="hero">
            <div style="font-size:0.8rem; letter-spacing:0.16em; text-transform:uppercase; color:#9aa9bf; font-weight:700;">Fast mode</div>
            <div style="font-size:2.5rem; font-weight:900; margin-top:8px;">⚽ Set Piece Studio</div>
            <div style="font-size:1rem; color:#9aa9bf; margin-top:10px; max-width:900px;">
                Optimized to load once, cache aggressively, and avoid recalculating summaries on every click.
                The app reads the workbook directly on startup and opens into three clear segment choices.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-card">
            <div style="font-size:0.8rem; letter-spacing:0.16em; text-transform:uppercase; color:#9aa9bf; font-weight:700;">Workbook</div>
            <div style="font-size:1rem; font-weight:800; margin-top:6px;">{path_used}</div>
            <div style="font-size:0.88rem; color:#9aa9bf; margin-top:4px;">Rows loaded: {len(df):,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    for col, label in zip([c1, c2, c3], ["Free Kick", "Corner", "Throw-In"]):
        with col:
            row = type_summary[type_summary["type"] == label]
            if row.empty:
                metric_card(label, "0", "No events")
            else:
                r = row.iloc[0]
                metric_card(label, f"{int(r['events']):,}", f"{human_pct(r['shot_rate'])} shot rate")
    with c4:
        metric_card("Matches", f"{df['match_id'].nunique():,}", "Across workbook")

    cards = [
        ("Free Kick", "Direct and indirect routines, shot quality, and match context."),
        ("Corner", "Corner volume, output, takers, and shot production."),
        ("Throw-In", "Attacking throw-ins, long-throw usage, and end results."),
    ]
    cols = st.columns(3)
    for col, (label, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="segment-card">
                    <div style="font-size:0.78rem; text-transform:uppercase; letter-spacing:0.12em; color:#9aa9bf; font-weight:700;">Segment</div>
                    <div style="font-size:1.5rem; font-weight:900; margin-top:10px;">{label}</div>
                    <div style="font-size:0.95rem; color:#9aa9bf; line-height:1.55; margin-top:10px;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button(f"Open {label}", key=f"open_{label}"):
                go(label)
                st.rerun()

    st.subheader("League Snapshot")
    c5, c6 = st.columns(2)
    with c5:
        st.bar_chart(type_summary.set_index("type")["events"])
    with c6:
        st.line_chart(type_summary.set_index("type")["xg_per_event"])


# =========================================================
# SEGMENT VIEW
# =========================================================
else:
    page = st.session_state["page"]
    seg = df[df["type"] == page].copy()

    if st.button("← Back"):
        go("home")
        st.rerun()

    st.markdown(
        f"""
        <div class="hero">
            <div style="font-size:2.35rem; font-weight:900;">{page} <span style="color:#5da8ff;">Studio</span></div>
            <div style="font-size:1rem; color:#9aa9bf; margin-top:10px;">Fast filtered view for {page.lower()} events only.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    teams = sorted(seg["team"].dropna().astype(str).unique().tolist())
    matches = sorted(seg["Match"].dropna().astype(str).unique().tolist())
    takers = sorted([str(x) for x in seg["Taker"].dropna().astype(str).unique() if str(x).strip()])

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.05rem;font-weight:800;margin-bottom:10px;">Filters</div>', unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        team_filter = st.selectbox("Team", ["All Teams"] + teams)
    with f2:
        side_filter = st.selectbox("Side", ["Both", "Left", "Right", "Unknown"])
    with f3:
        match_filter = st.selectbox("Match", ["All Matches"] + matches)
    with f4:
        taker_filter = st.selectbox("Taker", ["All Takers"] + takers)
    st.markdown("</div>", unsafe_allow_html=True)

    work = seg
    if team_filter != "All Teams":
        work = work[work["team"] == team_filter]
    if side_filter != "Both":
        work = work[work["side"] == side_filter]
    if match_filter != "All Matches":
        work = work[work["Match"] == match_filter]
    if taker_filter != "All Takers":
        work = work[work["Taker"] == taker_filter]

    if work.empty:
        empty_state("No events match the current filters.")
        st.stop()

    seg_team = team_summary[team_summary["type"] == page].copy()
    seg_taker = taker_summary[taker_summary["type"] == page].copy()
    seg_match = match_summary[match_summary["type"] == page].copy()

    if team_filter != "All Teams":
        seg_team = seg_team[seg_team["team"] == team_filter]
        seg_taker = seg_taker[seg_taker["team"] == team_filter]
    if taker_filter != "All Takers":
        seg_taker = seg_taker[seg_taker["Taker"] == taker_filter]
    if match_filter != "All Matches":
        seg_match = seg_match[seg_match["Match"] == match_filter]

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        metric_card("Events", f"{len(work):,}", page)
    with k2:
        metric_card("Matches", f"{work['match_id'].nunique():,}", "Current view")
    with k3:
        metric_card("Shots", f"{int(work['shot'].sum()):,}", "From set pieces")
    with k4:
        metric_card("Goals", f"{int(work['goal'].sum()):,}", "Scored")
    with k5:
        metric_card("xG", f"{work['shot_xg'].sum():.2f}", human_pct(work["shot"].mean()))

    tabs = st.tabs(["Overview", "Shotmap", "Teams", "Takers", "Matches", "Data"])

    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            phase_view = (
                work.groupby("phase", dropna=False)
                .size()
                .reset_index(name="events")
                .set_index("phase")["events"]
            )
            st.bar_chart(phase_view)
        with c2:
            team_view = (
                work.groupby("team", dropna=False)["shot_xg"]
                .sum()
                .sort_values(ascending=False)
            )
            st.bar_chart(team_view)

    with tabs[1]:
        st.plotly_chart(shotmap(work, f"{page} Shotmap"), use_container_width=True)

    with tabs[2]:
        st.dataframe(
            seg_team.sort_values(["xg_per_event", "shot_rate"], ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=420,
        )

    with tabs[3]:
        st.dataframe(
            seg_taker.sort_values(["events", "xg_per_event"], ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=420,
        )

    with tabs[4]:
        st.dataframe(
            seg_match.sort_values(["xg", "events"], ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=420,
        )

    with tabs[5]:
        show_cols = ["Match", "team", "Minute", "Taker", "Shooter", "shot_xg", "shot", "goal", "side", "phase"]
        st.dataframe(work[show_cols].reset_index(drop=True), use_container_width=True, height=520)
