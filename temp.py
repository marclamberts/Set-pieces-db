import os
import pandas as pd
import numpy as np
import streamlit as st

# Optional interactive charts
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

st.set_page_config(page_title="Set Pieces — Corners", layout="wide")

DEFAULT_FILE = "Allsvenskan - Corners 2025.xlsx"
DEFAULT_SHEET = "Sheet 1"

# -----------------------------
# HARD UI OVERRIDES (app shell)
# -----------------------------
st.markdown(
    """
<style>
/* Kill Streamlit chrome */
#MainMenu {visibility:hidden;}
header {visibility:hidden;}
footer {visibility:hidden;}
.stDeployButton {display:none;}
.stApp {background:#f6f7fb;}

/* Remove default padding so our shell aligns */
.block-container {padding-top:0.5rem; padding-bottom:0.5rem; max-width: 1600px;}

/* App shell layout */
.app-shell {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 14px;
  align-items: start;
}

/* Topbar */
.topbar {
  position: sticky;
  top: 0;
  z-index: 999;
  background: rgba(246,247,251,0.85);
  backdrop-filter: blur(10px);
  padding: 10px 0 12px 0;
}

/* Topbar card */
.topbar-card {
  background: #ffffff;
  border: 1px solid #e8ecf4;
  box-shadow: 0 10px 30px rgba(15,23,42,0.06);
  border-radius: 16px;
  padding: 12px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}

/* Brand */
.brand {
  display:flex; gap:10px; align-items:center;
}
.badge {
  width:34px; height:34px;
  border-radius: 10px;
  background: linear-gradient(135deg, #111827 0%, #334155 100%);
  box-shadow: 0 10px 22px rgba(17,24,39,0.18);
}
.brand h1 {
  margin:0;
  font-size: 16px;
  letter-spacing: -0.02em;
  font-weight: 800;
}
.brand p {
  margin:0;
  color:#6b7280;
  font-size: 12px;
}

/* Left rail */
.rail {
  position: sticky;
  top: 74px;
  height: calc(100vh - 92px);
  overflow: auto;
  background:#ffffff;
  border: 1px solid #e8ecf4;
  box-shadow: 0 10px 30px rgba(15,23,42,0.06);
  border-radius: 16px;
  padding: 12px;
}

/* Content */
.content {
  min-height: 80vh;
}

/* Module cards */
.module {
  background:#ffffff;
  border: 1px solid #e8ecf4;
  box-shadow: 0 10px 30px rgba(15,23,42,0.06);
  border-radius: 16px;
  padding: 14px;
  margin-bottom: 14px;
}
.module-title {
  font-size: 13px;
  font-weight: 800;
  letter-spacing: -0.01em;
  margin: 0 0 10px 0;
}
.module-sub {
  color:#6b7280;
  font-size: 12px;
  margin-top: -6px;
  margin-bottom: 10px;
}

/* KPI tiles */
.kpis {
  display:grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px;
}
.kpi {
  border: 1px solid #eef2f7;
  border-radius: 14px;
  padding: 10px 12px;
  background: #ffffff;
}
.kpi .k {
  color:#6b7280;
  font-size: 11px;
  margin-bottom: 6px;
}
.kpi .v {
  font-size: 18px;
  font-weight: 850;
  letter-spacing: -0.02em;
}
.kpi .h {
  color:#94a3b8;
  font-size: 11px;
  margin-top: 4px;
}

/* Two column inside content modules */
.grid2 {
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
}

/* Section divider */
.hr {
  border:none;
  border-top: 1px solid #eef2f7;
  margin: 10px 0;
}

/* Make widgets compact/dense */
label[data-testid="stWidgetLabel"] {font-size: 11px; color:#6b7280;}
div[data-baseweb="select"] {font-size: 12px;}
div[data-baseweb="tag"] {background:#0f172a0f; border-radius:999px;}
.stTextInput input {border-radius: 12px;}
.stMultiSelect div[data-baseweb="select"] {border-radius: 12px;}
.stSelectbox div[data-baseweb="select"] {border-radius: 12px;}
.stSlider {padding-top: 0.2rem; padding-bottom: 0.2rem;}
.stToggle {padding: 0.15rem 0;}
.stCheckbox {padding: 0.15rem 0;}

/* Radio as tabs (nav) */
div[role="radiogroup"] > label {
  background:#ffffff;
  border:1px solid #e8ecf4;
  border-radius: 12px;
  padding: 8px 10px;
  margin-right: 8px;
  box-shadow: 0 10px 25px rgba(15,23,42,0.05);
}
div[role="radiogroup"] > label:has(input:checked) {
  border-color: #c7d2fe;
  box-shadow: 0 12px 28px rgba(99,102,241,0.12);
}

/* Dataframe */
div[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid #e8ecf4;
}

/* Buttons */
.stDownloadButton button, .stButton button {
  background:#111827 !important;
  color:#fff !important;
  border:none !important;
  border-radius: 12px !important;
  padding: 0.55em 1.05em !important;
  box-shadow: 0 10px 22px rgba(17,24,39,0.18) !important;
}
.stDownloadButton button:hover, .stButton button:hover {
  background:#0b1220 !important;
}

/* Hide Streamlit "empty" spacing artifacts */
.stMarkdown {margin-bottom: 0.2rem;}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Data loader
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data_from_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name=DEFAULT_SHEET)
    df.columns = [str(c).strip() for c in df.columns]

    df["Minute_num"] = pd.to_numeric(df["Minute"], errors="coerce") if "Minute" in df.columns else np.nan
    df["Second_num"] = pd.to_numeric(df["Second"], errors="coerce") if "Second" in df.columns else np.nan

    shot_ts = df["shot_timestamp"].notna() if "shot_timestamp" in df.columns else pd.Series(False, index=df.index)
    shot_out = df["shot.outcome.name"].notna() if "shot.outcome.name" in df.columns else pd.Series(False, index=df.index)
    df["is_shot"] = shot_ts | shot_out

    if "shot.statsbomb_xg" in df.columns:
        df["xg"] = pd.to_numeric(df["shot.statsbomb_xg"], errors="coerce").fillna(0.0)
    else:
        df["xg"] = 0.0

    # Normalized keys used across UI
    df["team"] = df["pass_team_name"] if "pass_team_name" in df.columns else None
    df["match"] = df["Match"] if "Match" in df.columns else None
    df["taker"] = df["Taker"] if "Taker" in df.columns else None
    df["technique"] = df["pass.technique.name"] if "pass.technique.name" in df.columns else None
    df["height"] = df["pass.height.name"] if "pass.height.name" in df.columns else None
    df["shot_outcome"] = df["shot.outcome.name"] if "shot.outcome.name" in df.columns else None
    df["sp_outcome"] = df["SP_outcome"] if "SP_outcome" in df.columns else None

    return df


# -----------------------------
# Load file (no Streamlit sidebar)
# -----------------------------
file_col1, file_col2 = st.columns([1, 2])
with file_col1:
    uploaded = st.file_uploader("Data (.xlsx)", type=["xlsx"], label_visibility="collapsed")

if uploaded is not None:
    data_file = uploaded
else:
    if os.path.exists(DEFAULT_FILE):
        data_file = DEFAULT_FILE
    else:
        st.error(f"Place `{DEFAULT_FILE}` next to this script or upload it.")
        st.stop()

df = load_data_from_excel(data_file)

# -----------------------------
# TOPBAR
# -----------------------------
st.markdown(
    """
<div class="topbar">
  <div class="topbar-card">
    <div class="brand">
      <div class="badge"></div>
      <div>
        <h1>Set Pieces • Corners</h1>
        <p>Allsvenskan 2025 • Analysis workspace</p>
      </div>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
      <span style="color:#6b7280; font-size:12px;">Platform UI • Dense mode</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# SHELL: left rail + content
# -----------------------------
st.markdown('<div class="app-shell">', unsafe_allow_html=True)

rail = st.container()
content = st.container()

# -----------------------------
# LEFT RAIL (filters + entity drilldowns)
# -----------------------------
with rail:
    st.markdown('<div class="rail">', unsafe_allow_html=True)

    st.markdown("#### Navigation")
    view = st.radio(
        "view",
        ["League", "Team", "Match", "Player"],
        horizontal=True,
        label_visibility="collapsed",
    )

    st.markdown("<hr class='hr'>", unsafe_allow_html=True)
    st.markdown("#### Filter rail")

    q = st.text_input("Search", value="", placeholder="Team / match / taker…")

    teams = sorted(pd.Series(df["team"]).dropna().unique().tolist())
    sel_teams = st.multiselect("Teams", teams, default=teams)

    df_team = df[df["team"].isin(sel_teams)] if sel_teams else df
    matches = sorted(pd.Series(df_team["match"]).dropna().unique().tolist())
    sel_matches = st.multiselect("Matches", matches, default=matches)

    df_match = df_team[df_team["match"].isin(sel_matches)] if sel_matches else df_team
    takers = sorted(pd.Series(df_match["taker"]).dropna().unique().tolist())
    sel_takers = st.multiselect("Takers", takers, default=takers)

    techniques = sorted(pd.Series(df_match["technique"]).dropna().unique().tolist())
    sel_techniques = st.multiselect("Technique", techniques, default=techniques)

    heights = sorted(pd.Series(df_match["height"]).dropna().unique().tolist())
    sel_heights = st.multiselect("Delivery height", heights, default=heights)

    # SAFE minute control
    minute_series = pd.to_numeric(df_match.get("Minute_num", pd.Series(dtype=float)), errors="coerce").dropna()
    minute_unique = np.sort(minute_series.unique()) if len(minute_series) else np.array([])

    st.markdown("<hr class='hr'>", unsafe_allow_html=True)
    st.markdown("#### Time window")

    minute_mode = "disabled"
    minute_range = None
    minute_single_value = None
    apply_single_minute = False

    if len(minute_unique) == 0:
        st.caption("No minute values in current selection.")
    elif len(minute_unique) == 1:
        minute_single_value = int(minute_unique[0])
        apply_single_minute = st.checkbox(f"Only minute {minute_single_value}", value=False)
        minute_mode = "single"
    else:
        minute_range = st.slider(
            "Minutes",
            min_value=int(minute_unique.min()),
            max_value=int(minute_unique.max()),
            value=(int(minute_unique.min()), int(minute_unique.max())),
        )
        minute_mode = "range"

    st.markdown("<hr class='hr'>", unsafe_allow_html=True)
    st.markdown("#### Output")
    only_shots = st.toggle("Only corners → shot", value=False)
    show_table = st.toggle("Show data table", value=True)

    # Context drilldown selectors (feel like platform)
    st.markdown("<hr class='hr'>", unsafe_allow_html=True)
    st.markdown("#### Focus entity")

    focus_team = None
    focus_match = None
    focus_player = None

    if view == "Team":
        team_list = sorted(pd.Series(df_team["team"]).dropna().unique().tolist())
        focus_team = st.selectbox("Team", team_list if team_list else ["(none)"])
    elif view == "Match":
        match_list = sorted(pd.Series(df_match["match"]).dropna().unique().tolist())
        focus_match = st.selectbox("Match", match_list if match_list else ["(none)"])
    elif view == "Player":
        player_list = sorted(pd.Series(df_match["taker"]).dropna().unique().tolist())
        focus_player = st.selectbox("Taker", player_list if player_list else ["(none)"])

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# Apply filters
# -----------------------------
f = df.copy()

if sel_teams:
    f = f[f["team"].isin(sel_teams)]
if sel_matches:
    f = f[f["match"].isin(sel_matches)]
if sel_takers:
    f = f[f["taker"].isin(sel_takers)]
if sel_techniques:
    f = f[f["technique"].isin(sel_techniques)]
if sel_heights:
    f = f[f["height"].isin(sel_heights)]

if q.strip():
    qq = q.strip().lower()
    mask = (
        f["team"].fillna("").astype(str).str.lower().str.contains(qq)
        | f["match"].fillna("").astype(str).str.lower().str.contains(qq)
        | f["taker"].fillna("").astype(str).str.lower().str.contains(qq)
    )
    f = f[mask]

if "Minute_num" in f.columns:
    f = f.copy()
    f["Minute_num"] = pd.to_numeric(f["Minute_num"], errors="coerce")
    if minute_mode == "range" and minute_range is not None:
        f = f[f["Minute_num"].between(minute_range[0], minute_range[1], inclusive="both")]
    elif minute_mode == "single" and apply_single_minute and minute_single_value is not None:
        f = f[f["Minute_num"] == minute_single_value]

if only_shots and "is_shot" in f.columns:
    f = f[f["is_shot"]]

# Entity focus filtering inside view (platform behavior)
if view == "Team" and focus_team and focus_team != "(none)":
    f_view = f[f["team"] == focus_team].copy()
elif view == "Match" and focus_match and focus_match != "(none)":
    f_view = f[f["match"] == focus_match].copy()
elif view == "Player" and focus_player and focus_player != "(none)":
    f_view = f[f["taker"] == focus_player].copy()
else:
    f_view = f.copy()

# -----------------------------
# CONTENT AREA
# -----------------------------
with content:
    st.markdown('<div class="content">', unsafe_allow_html=True)

    # KPI tiles (custom, not Streamlit metric widgets)
    total_corners = int(len(f_view))
    n_matches = int(f_view["match_id"].nunique()) if "match_id" in f_view.columns else int(pd.Series(f_view["match"]).nunique())
    corners_per_match = (total_corners / n_matches) if n_matches else 0.0
    shot_corners = int(f_view["is_shot"].sum()) if "is_shot" in f_view.columns else 0
    shot_rate = (shot_corners / total_corners) if total_corners else 0.0
    total_xg = float(pd.Series(f_view.get("xg", 0)).sum())
    sp_txt = pd.Series(f_view.get("sp_outcome", "")).fillna("").astype(str)
    shots_3s = int(sp_txt.str.contains("shot within 3 seconds", case=False, na=False).sum())

    st.markdown(
        f"""
<div class="module">
  <div class="module-title">Overview</div>
  <div class="module-sub">Current view: <b>{view}</b> • Rows: <b>{total_corners:,}</b></div>
  <div class="kpis">
    <div class="kpi"><div class="k">Corner events</div><div class="v">{total_corners:,}</div><div class="h">Filtered rows</div></div>
    <div class="kpi"><div class="k">Matches</div><div class="v">{n_matches:,}</div><div class="h">Unique match count</div></div>
    <div class="kpi"><div class="k">Corners / match</div><div class="v">{corners_per_match:.2f}</div><div class="h">Volume intensity</div></div>
    <div class="kpi"><div class="k">Shot rate</div><div class="v">{shot_rate*100:.1f}%</div><div class="h">Corners → shot</div></div>
    <div class="kpi"><div class="k">Total xG</div><div class="v">{total_xg:.3f}</div><div class="h">From shots</div></div>
    <div class="kpi"><div class="k">Shots ≤ 3s</div><div class="v">{shots_3s:,}</div><div class="h">SP_outcome tag</div></div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Helper plot functions
    def plot_bar(df_, x, y, title, height=360):
        st.markdown(f"<div class='module'><div class='module-title'>{title}</div>", unsafe_allow_html=True)
        if len(df_) == 0:
            st.info("No data for current filters.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        if PLOTLY_OK:
            fig = px.bar(df_, x=x, y=y)
            fig.update_layout(
                height=height,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="",
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(df_, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    def plot_pie(df_, names, values, title, height=360):
        st.markdown(f"<div class='module'><div class='module-title'>{title}</div>", unsafe_allow_html=True)
        if len(df_) == 0:
            st.info("No data for current filters.")
            st.markdown("</div>", unsafe_allow_html=True)
            return
        if PLOTLY_OK:
            fig = px.pie(df_, names=names, values=values, hole=0.55)
            fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(df_, use_container_width=True, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Main modules by view (optA/statsbomb-ish “workspace”)
    if view == "League":
        left, right = st.columns(2)

        team_counts = (
            f_view.groupby("team", dropna=False)
            .size()
            .sort_values(ascending=False)
            .reset_index(name="corners")
        )
        tech_counts = (
            f_view.groupby("technique", dropna=False)
            .size()
            .sort_values(ascending=False)
            .reset_index(name="corners")
        )

        with left:
            plot_bar(team_counts, "team", "corners", "Corner volume by team")
        with right:
            plot_pie(tech_counts, "technique", "corners", "Delivery technique mix")

        left2, right2 = st.columns(2)
        shots = f_view[f_view.get("is_shot", False)].copy()
        shot_outcomes = (
            shots.groupby("shot_outcome", dropna=False)
            .size()
            .sort_values(ascending=False)
            .reset_index(name="shots")
        )
        xg_by_team = (
            f_view.groupby("team", dropna=False)["xg"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        with left2:
            plot_bar(shot_outcomes, "shot_outcome", "shots", "Shot outcomes from corners")
        with right2:
            plot_bar(xg_by_team, "team", "xg", "xG from corners by team")

    elif view == "Team":
        left, right = st.columns(2)

        taker_counts = (
            f_view.groupby("taker", dropna=False)
            .size()
            .sort_values(ascending=False)
            .head(15)
            .reset_index(name="corners")
        )
        xg_takers = (
            f_view.groupby("taker", dropna=False)["xg"]
            .sum()
            .sort_values(ascending=False)
            .head(15)
            .reset_index()
        )
        tech_counts = (
            f_view.groupby("technique", dropna=False)
            .size()
            .sort_values(ascending=False)
            .reset_index(name="corners")
        )

        with left:
            plot_bar(taker_counts, "taker", "corners", "Primary takers (volume)")
        with right:
            plot_bar(xg_takers, "taker", "xg", "Most dangerous takers (xG created)")

        plot_pie(tech_counts, "technique", "corners", "Technique profile")

    elif view == "Match":
        left, right = st.columns(2)

        by_team = (
            f_view.groupby("team", dropna=False)
            .size()
            .sort_values(ascending=False)
            .reset_index(name="corners")
        )

        with left:
            plot_bar(by_team, "team", "corners", "Corners by team (match)")

        with right:
            st.markdown("<div class='module'><div class='module-title'>Corner timing</div>", unsafe_allow_html=True)
            time_df = f_view[["Minute_num"]].copy() if "Minute_num" in f_view.columns else pd.DataFrame()
            time_df["Minute_num"] = pd.to_numeric(time_df.get("Minute_num", np.nan), errors="coerce")
            time_df = time_df.dropna()
            if len(time_df) == 0:
                st.info("No minute values in this selection.")
            else:
                if PLOTLY_OK:
                    fig = px.histogram(time_df, x="Minute_num", nbins=24)
                    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="", yaxis_title="")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.bar_chart(time_df["Minute_num"].value_counts().sort_index())
            st.markdown("</div>", unsafe_allow_html=True)

    elif view == "Player":
        left, right = st.columns(2)

        by_team = (
            f_view.groupby("team", dropna=False)
            .size()
            .sort_values(ascending=False)
            .reset_index(name="corners")
        )
        tech_counts = (
            f_view.groupby("technique", dropna=False)
            .size()
            .sort_values(ascending=False)
            .reset_index(name="corners")
        )

        with left:
            plot_bar(by_team, "team", "corners", "Corners taken by team")
        with right:
            plot_pie(tech_counts, "technique", "corners", "Technique profile")

        # player summary module
        sr = (int(f_view["is_shot"].sum()) / len(f_view)) if len(f_view) else 0.0
        st.markdown(
            f"""
<div class="module">
  <div class="module-title">Player summary</div>
  <div class="module-sub">What this taker produces in the current filtered context</div>
  <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px;">
    <div class="kpi"><div class="k">Corners</div><div class="v">{len(f_view):,}</div><div class="h">Volume</div></div>
    <div class="kpi"><div class="k">Shot rate</div><div class="v">{sr*100:.1f}%</div><div class="h">Corners → shot</div></div>
    <div class="kpi"><div class="k">Total xG</div><div class="v">{float(f_view["xg"].sum()):.3f}</div><div class="h">From shots</div></div>
    <div class="kpi"><div class="k">Top height</div><div class="v">{f_view["height"].fillna("Unknown").value_counts().idxmax() if len(f_view) else "—"}</div><div class="h">Delivery</div></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # Data table / export
    if show_table:
        st.markdown("<div class='module'><div class='module-title'>Data</div>", unsafe_allow_html=True)

        preferred = ["match", "team", "taker", "Minute_num", "Second_num", "technique", "height", "sp_outcome", "is_shot", "shot_outcome", "xg"]
        cols = [c for c in preferred if c in f_view.columns] + [c for c in f_view.columns if c not in preferred]
        st.dataframe(f_view[cols], use_container_width=True, hide_index=True)

        csv = f_view.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download CSV (current view)",
            data=csv,
            file_name="corners_filtered_view.csv",
            mime="text/csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# Close shell
st.markdown("</div>", unsafe_allow_html=True)
