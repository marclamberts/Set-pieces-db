import os
import pandas as pd
import numpy as np
import streamlit as st

# Optional interactive charts
try:
    import plotly.express as px
    import plotly.io as pio
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# -----------------------------
# Config
# -----------------------------
st.set_page_config(page_title="Set Pieces — Corners", layout="wide")

DEFAULT_FILE = "Allsvenskan - Corners 2025.xlsx"
DEFAULT_SHEET = "Sheet 1"

# -----------------------------
# Helpers
# -----------------------------
def _col(df: pd.DataFrame, name: str):
    return df[name] if name in df.columns else pd.Series([np.nan] * len(df), index=df.index)

def _safe_unique(series: pd.Series):
    return sorted(pd.Series(series).dropna().astype(str).unique().tolist())

def _to_num(s: pd.Series):
    return pd.to_numeric(s, errors="coerce")

def _contains(s: pd.Series, q: str):
    return s.fillna("").astype(str).str.lower().str.contains(q, na=False)

# -----------------------------
# Theme + Shell CSS
# -----------------------------
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

def inject_css(dark: bool):
    # Design tokens
    if not dark:
        tokens = {
            "bg": "#F6F7FB",
            "panel": "#FFFFFF",
            "panel2": "#FBFCFF",
            "text": "#0F172A",
            "muted": "#64748B",
            "muted2": "#94A3B8",
            "border": "#E6EAF2",
            "border2": "#EEF2F7",
            "shadow": "0 12px 32px rgba(15, 23, 42, 0.08)",
            "shadow2": "0 10px 24px rgba(15, 23, 42, 0.06)",
            "accent": "#4F46E5",
            "accent2": "#7C3AED",
            "chip": "rgba(79,70,229,0.09)",
            "success": "#16A34A",
            "warning": "#F59E0B",
        }
        plotly_template = "plotly_white"
    else:
        tokens = {
            "bg": "#0B1220",
            "panel": "#0F172A",
            "panel2": "#111C33",
            "text": "#E5E7EB",
            "muted": "#A3AEC2",
            "muted2": "#7C8AA3",
            "border": "#22304A",
            "border2": "#1A2740",
            "shadow": "0 14px 36px rgba(0, 0, 0, 0.45)",
            "shadow2": "0 10px 26px rgba(0, 0, 0, 0.35)",
            "accent": "#818CF8",
            "accent2": "#A78BFA",
            "chip": "rgba(129,140,248,0.14)",
            "success": "#22C55E",
            "warning": "#FBBF24",
        }
        plotly_template = "plotly_dark"

    if PLOTLY_OK:
        pio.templates.default = plotly_template

    st.markdown(
        f"""
<style>
:root {{
  --bg: {tokens["bg"]};
  --panel: {tokens["panel"]};
  --panel2: {tokens["panel2"]};
  --text: {tokens["text"]};
  --muted: {tokens["muted"]};
  --muted2: {tokens["muted2"]};
  --border: {tokens["border"]};
  --border2: {tokens["border2"]};
  --shadow: {tokens["shadow"]};
  --shadow2: {tokens["shadow2"]};
  --accent: {tokens["accent"]};
  --accent2: {tokens["accent2"]};
  --chip: {tokens["chip"]};
  --success: {tokens["success"]};
  --warning: {tokens["warning"]};
  --r: 16px;
  --r2: 14px;
}}

html, body, .stApp {{
  background: var(--bg) !important;
  color: var(--text) !important;
}}

/* Kill Streamlit chrome */
#MainMenu {{visibility:hidden;}}
header {{visibility:hidden;}}
footer {{visibility:hidden;}}
.stDeployButton {{display:none;}}

/* Width + padding */
.block-container {{
  padding-top: 0.6rem;
  padding-bottom: 1.0rem;
  max-width: 1600px;
}}

/* App shell */
.app-shell {{
  display: grid;
  grid-template-columns: 330px 1fr;
  gap: 14px;
  align-items: start;
}}
@media (max-width: 1100px) {{
  .app-shell {{
    grid-template-columns: 1fr;
  }}
  .rail {{
    position: relative !important;
    top: auto !important;
    height: auto !important;
  }}
}}

/* Topbar */
.topbar {{
  position: sticky;
  top: 0;
  z-index: 999;
  background: color-mix(in srgb, var(--bg) 84%, transparent);
  backdrop-filter: blur(10px);
  padding: 10px 0 12px 0;
}}
.topbar-card {{
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: var(--shadow2);
  border-radius: var(--r);
  padding: 12px 14px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
}}

/* Brand */
.brand {{
  display:flex; gap:10px; align-items:center;
}}
.badge {{
  width:34px; height:34px;
  border-radius: 10px;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
  box-shadow: 0 14px 28px rgba(79,70,229,0.25);
}}
.brand h1 {{
  margin:0;
  font-size: 15px;
  letter-spacing: -0.02em;
  font-weight: 900;
  color: var(--text);
}}
.brand p {{
  margin:0;
  color: var(--muted);
  font-size: 12px;
}}

/* Left rail */
.rail {{
  position: sticky;
  top: 78px;
  height: calc(100vh - 96px);
  overflow: auto;
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: var(--shadow2);
  border-radius: var(--r);
  padding: 12px;
}}
.rail h4, .rail h3 {{
  margin: 0.2rem 0 0.6rem 0;
}}

/* Content */
.content {{
  min-height: 80vh;
}}

/* Module cards */
.module {{
  background: var(--panel);
  border: 1px solid var(--border);
  box-shadow: var(--shadow2);
  border-radius: var(--r);
  padding: 14px;
  margin-bottom: 14px;
}}
.module-title {{
  font-size: 13px;
  font-weight: 900;
  letter-spacing: -0.01em;
  margin: 0 0 10px 0;
  color: var(--text);
}}
.module-sub {{
  color: var(--muted);
  font-size: 12px;
  margin-top: -6px;
  margin-bottom: 10px;
}}

/* KPIs */
.kpis {{
  display:grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 10px;
}}
@media (max-width: 1200px) {{
  .kpis {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
}}
@media (max-width: 700px) {{
  .kpis {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
}}
.kpi {{
  border: 1px solid var(--border2);
  border-radius: var(--r2);
  padding: 10px 12px;
  background: var(--panel2);
}}
.kpi .k {{
  color: var(--muted);
  font-size: 11px;
  margin-bottom: 6px;
}}
.kpi .v {{
  font-size: 18px;
  font-weight: 950;
  letter-spacing: -0.02em;
  color: var(--text);
}}
.kpi .h {{
  color: var(--muted2);
  font-size: 11px;
  margin-top: 4px;
}}

/* Divider */
.hr {{
  border:none;
  border-top: 1px solid var(--border2);
  margin: 10px 0;
}}

/* Widgets */
label[data-testid="stWidgetLabel"] {{
  font-size: 11px;
  color: var(--muted);
}}
.stTextInput input {{
  border-radius: 12px !important;
}}
div[data-baseweb="select"] {{
  font-size: 12px;
}}
div[data-baseweb="tag"] {{
  background: var(--chip);
  border-radius: 999px;
  border: 1px solid color-mix(in srgb, var(--accent) 25%, var(--border));
}}
/* Radio as tabs */
div[role="radiogroup"] > label {{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 8px 10px;
  margin-right: 8px;
  box-shadow: var(--shadow2);
}}
div[role="radiogroup"] > label:has(input:checked) {{
  border-color: color-mix(in srgb, var(--accent) 45%, var(--border));
  box-shadow: 0 14px 30px rgba(79,70,229,0.18);
}}

/* Dataframe */
div[data-testid="stDataFrame"] {{
  border-radius: var(--r2);
  overflow: hidden;
  border: 1px solid var(--border);
}}

/* Buttons */
.stDownloadButton button, .stButton button {{
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 0.55em 1.05em !important;
  box-shadow: 0 12px 26px rgba(79,70,229,0.25) !important;
}}
.stDownloadButton button:hover, .stButton button:hover {{
  filter: brightness(0.98);
}}

/* Reduce Streamlit markdown spacing noise */
.stMarkdown {{ margin-bottom: 0.25rem; }}
</style>
""",
        unsafe_allow_html=True,
    )

inject_css(st.session_state.dark_mode)

# -----------------------------
# Data loader
# -----------------------------
@st.cache_data(show_spinner=False)
def load_data_from_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name=DEFAULT_SHEET)
    df.columns = [str(c).strip() for c in df.columns]

    # Time
    df["Minute_num"] = _to_num(_col(df, "Minute"))
    df["Second_num"] = _to_num(_col(df, "Second"))

    # Shots
    shot_ts = _col(df, "shot_timestamp").notna()
    shot_out = _col(df, "shot.outcome.name").notna()
    df["is_shot"] = (shot_ts | shot_out).fillna(False)

    # xG
    if "shot.statsbomb_xg" in df.columns:
        df["xg"] = _to_num(df["shot.statsbomb_xg"]).fillna(0.0)
    else:
        df["xg"] = 0.0

    # Normalized keys used across UI
    df["team"] = _col(df, "pass_team_name")
    df["match"] = _col(df, "Match")
    df["taker"] = _col(df, "Taker")
    df["technique"] = _col(df, "pass.technique.name")
    df["height"] = _col(df, "pass.height.name")
    df["shot_outcome"] = _col(df, "shot.outcome.name")
    df["sp_outcome"] = _col(df, "SP_outcome")

    return df

# -----------------------------
# File loader row (no sidebar)
# -----------------------------
file_col1, file_col2 = st.columns([1, 2], vertical_alignment="center")
with file_col1:
    uploaded = st.file_uploader("Data (.xlsx)", type=["xlsx"], label_visibility="collapsed")

with file_col2:
    c1, c2, c3 = st.columns([1, 1, 1], vertical_alignment="center")
    with c1:
        st.toggle("Dark mode", value=st.session_state.dark_mode, key="dark_mode")
    with c2:
        st.caption("Dense • Platform shell")
    with c3:
        st.caption("Allsvenskan 2025 • Corners")

# Re-inject after toggle change
inject_css(st.session_state.dark_mode)

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
        <p>Analysis workspace</p>
      </div>
    </div>
    <div style="display:flex; gap:10px; align-items:center;">
      <span style="font-size:12px; color:var(--muted);">Filters → Views → Export</span>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# SHELL
# -----------------------------
st.markdown('<div class="app-shell">', unsafe_allow_html=True)
rail = st.container()
content = st.container()

# -----------------------------
# LEFT RAIL
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

    # Session-state backed filters so reset works cleanly
    if "q" not in st.session_state:
        st.session_state.q = ""
    if "only_shots" not in st.session_state:
        st.session_state.only_shots = False
    if "show_table" not in st.session_state:
        st.session_state.show_table = True

    q = st.text_input("Search", value=st.session_state.q, placeholder="Team / match / taker…", key="q")

    # Global lists
    teams_all = _safe_unique(df["team"])
    sel_teams = st.multiselect("Teams", teams_all, default=teams_all)

    df_team = df[df["team"].isin(sel_teams)] if sel_teams else df
    matches_all = _safe_unique(df_team["match"])
    sel_matches = st.multiselect("Matches", matches_all, default=matches_all)

    df_match = df_team[df_team["match"].isin(sel_matches)] if sel_matches else df_team
    takers_all = _safe_unique(df_match["taker"])
    sel_takers = st.multiselect("Takers", takers_all, default=takers_all)

    techniques_all = _safe_unique(df_match["technique"])
    sel_techniques = st.multiselect("Technique", techniques_all, default=techniques_all)

    heights_all = _safe_unique(df_match["height"])
    sel_heights = st.multiselect("Delivery height", heights_all, default=heights_all)

    # SAFE minute control
    minute_series = _to_num(_col(df_match, "Minute_num")).dropna()
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
    only_shots = st.toggle("Only corners → shot", value=st.session_state.only_shots, key="only_shots")
    show_table = st.toggle("Show data table", value=st.session_state.show_table, key="show_table")

    st.markdown("<hr class='hr'>", unsafe_allow_html=True)
    st.markdown("#### Focus entity")

    focus_team = focus_match = focus_player = None
    if view == "Team":
        focus_team = st.selectbox("Team", teams_all if teams_all else ["(none)"])
    elif view == "Match":
        focus_match = st.selectbox("Match", matches_all if matches_all else ["(none)"])
    elif view == "Player":
        focus_player = st.selectbox("Taker", takers_all if takers_all else ["(none)"])

    st.markdown("<hr class='hr'>", unsafe_allow_html=True)

    # Reset button
    if st.button("Reset filters"):
        st.session_state.q = ""
        st.session_state.only_shots = False
        st.session_state.show_table = True
        st.rerun()

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
        _contains(f["team"], qq)
        | _contains(f["match"], qq)
        | _contains(f["taker"], qq)
    )
    f = f[mask]

# minute filter
f["Minute_num"] = _to_num(_col(f, "Minute_num"))
if minute_mode == "range" and minute_range is not None:
    f = f[f["Minute_num"].between(minute_range[0], minute_range[1], inclusive="both")]
elif minute_mode == "single" and apply_single_minute and minute_single_value is not None:
    f = f[f["Minute_num"] == minute_single_value]

# only shots
if only_shots:
    f = f[f["is_shot"] == True]

# Entity focus filtering inside view
if view == "Team" and focus_team and focus_team != "(none)":
    f_view = f[f["team"].astype(str) == str(focus_team)].copy()
elif view == "Match" and focus_match and focus_match != "(none)":
    f_view = f[f["match"].astype(str) == str(focus_match)].copy()
elif view == "Player" and focus_player and focus_player != "(none)":
    f_view = f[f["taker"].astype(str) == str(focus_player)].copy()
else:
    f_view = f.copy()

# -----------------------------
# Plot helpers
# -----------------------------
def plot_bar(df_, x, y, title, height=360):
    st.markdown(f"<div class='module'><div class='module-title'>{title}</div>", unsafe_allow_html=True)
    if df_ is None or len(df_) == 0:
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
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(df_, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

def plot_pie(df_, names, values, title, height=360):
    st.markdown(f"<div class='module'><div class='module-title'>{title}</div>", unsafe_allow_html=True)
    if df_ is None or len(df_) == 0:
        st.info("No data for current filters.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if PLOTLY_OK:
        fig = px.pie(df_, names=names, values=values, hole=0.6)
        fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(df_, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# CONTENT
# -----------------------------
with content:
    st.markdown('<div class="content">', unsafe_allow_html=True)

    # KPIs
    total_corners = int(len(f_view))
    if "match_id" in f_view.columns:
        n_matches = int(pd.Series(f_view["match_id"]).nunique())
    else:
        n_matches = int(pd.Series(f_view["match"]).astype(str).replace("nan", np.nan).dropna().nunique())

    corners_per_match = (total_corners / n_matches) if n_matches else 0.0
    shot_corners = int(pd.Series(f_view["is_shot"]).fillna(False).sum())
    shot_rate = (shot_corners / total_corners) if total_corners else 0.0
    total_xg = float(pd.Series(f_view.get("xg", 0)).fillna(0).sum())

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

    # Main modules by view
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
        shots = f_view[f_view["is_shot"] == True].copy()
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

            time_df = pd.DataFrame({"Minute_num": _to_num(_col(f_view, "Minute_num"))}).dropna()
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

        sr = (int(pd.Series(f_view["is_shot"]).fillna(False).sum()) / len(f_view)) if len(f_view) else 0.0
        top_height = "—"
        if len(f_view) and "height" in f_view.columns:
            vc = f_view["height"].fillna("Unknown").astype(str).value_counts()
            if len(vc):
                top_height = vc.index[0]

        st.markdown(
            f"""
<div class="module">
  <div class="module-title">Player summary</div>
  <div class="module-sub">What this taker produces in the current filtered context</div>
  <div style="display:grid; grid-template-columns: 1fr 1fr; gap: 10px;">
    <div class="kpi"><div class="k">Corners</div><div class="v">{len(f_view):,}</div><div class="h">Volume</div></div>
    <div class="kpi"><div class="k">Shot rate</div><div class="v">{sr*100:.1f}%</div><div class="h">Corners → shot</div></div>
    <div class="kpi"><div class="k">Total xG</div><div class="v">{float(pd.Series(f_view.get("xg", 0)).fillna(0).sum()):.3f}</div><div class="h">From shots</div></div>
    <div class="kpi"><div class="k">Top height</div><div class="v">{top_height}</div><div class="h">Delivery</div></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

    # Data table / export
    if show_table:
        st.markdown("<div class='module'><div class='module-title'>Data</div>", unsafe_allow_html=True)

        preferred = [
            "match", "team", "taker", "Minute_num", "Second_num",
            "technique", "height", "sp_outcome", "is_shot", "shot_outcome", "xg"
        ]
        cols = [c for c in preferred if c in f_view.columns] + [c for c in f_view.columns if c not in preferred]

        st.dataframe(f_view[cols], use_container_width=True, hide_index=True)

        csv = f_view[cols].to_csv(index=False).encode("utf-8")
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
