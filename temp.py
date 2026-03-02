"""
Allsvenskan 2025 – Corner Kick Analysis
Professional analytics dashboard
"""

import os
import pandas as pd
import numpy as np
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    import plotly.io as pio
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Corners · Allsvenskan 2025",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

DEFAULT_FILE = "Allsvenskan - Corners 2025.xlsx"
DEFAULT_SHEET = "Sheet 1"

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _col(df, name):
    return df[name] if name in df.columns else pd.Series([np.nan] * len(df), index=df.index)

def _safe_unique(series):
    return sorted(pd.Series(series).dropna().astype(str).unique().tolist())

def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def _contains(s, q):
    return s.fillna("").astype(str).str.lower().str.contains(q, na=False)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* ── Reset Streamlit chrome ── */
#MainMenu, header, footer, .stDeployButton { visibility: hidden; display: none; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1600px !important; }

/* ── Root tokens ── */
:root {
  --bg:        #09090f;
  --surface:   #111118;
  --card:      #16161e;
  --card2:     #1b1b24;
  --border:    rgba(255,255,255,0.07);
  --border2:   rgba(255,255,255,0.04);
  --text:      #f0f0f8;
  --sub:       #9898b8;
  --muted:     #5a5a7a;
  --accent:    #6366f1;
  --accent2:   #a855f7;
  --green:     #22d3a0;
  --orange:    #f97316;
  --red:       #f43f5e;
  --r:         12px;
  --r2:        8px;
}

html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; }

/* ── Topbar ── */
.topbar {
  position: sticky; top: 0; z-index: 999;
  padding: 14px 0 16px 0;
  background: linear-gradient(to bottom, #09090f 85%, transparent);
}
.topbar-inner {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 14px 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  box-shadow: 0 24px 48px rgba(0,0,0,.5);
}
.logo-dot {
  width: 36px; height: 36px; border-radius: 10px;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
  box-shadow: 0 8px 24px rgba(99,102,241,.45);
  flex-shrink: 0;
}
.brand-title {
  font-size: 15px; font-weight: 800;
  letter-spacing: -.03em; color: var(--text); margin: 0;
}
.brand-sub { font-size: 11px; color: var(--muted); margin: 0; }
.badge-pill {
  background: rgba(99,102,241,.12);
  border: 1px solid rgba(99,102,241,.25);
  border-radius: 999px;
  padding: 4px 12px;
  font-size: 11px; font-weight: 600;
  color: #a5b4fc;
}

/* ── Layout ── */
.shell { display: grid; grid-template-columns: 300px 1fr; gap: 16px; align-items: start; margin-top: 16px; }

/* ── Sidebar rail ── */
.rail {
  position: sticky; top: 90px;
  height: calc(100vh - 108px);
  overflow-y: auto;
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px;
  box-shadow: 0 20px 40px rgba(0,0,0,.35);
  scrollbar-width: thin;
  scrollbar-color: var(--border) transparent;
}
.rail::-webkit-scrollbar { width: 4px; }
.rail::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

.rail-section { font-size: 10px; font-weight: 700; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); margin: 16px 0 8px 0; }
.rail-section:first-child { margin-top: 0; }

/* ── Cards ── */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 20px;
  margin-bottom: 16px;
  box-shadow: 0 16px 32px rgba(0,0,0,.25);
}
.card-title {
  font-size: 13px; font-weight: 800; letter-spacing: -.02em;
  color: var(--text); margin: 0 0 4px 0;
}
.card-sub { font-size: 11px; color: var(--sub); margin: 0 0 16px 0; }

/* ── KPI row ── */
.kpis { display: grid; grid-template-columns: repeat(6, 1fr); gap: 10px; }
@media(max-width:1300px){ .kpis { grid-template-columns: repeat(3,1fr); } }

.kpi {
  background: var(--card2);
  border: 1px solid var(--border2);
  border-radius: var(--r);
  padding: 14px 16px;
  position: relative;
  overflow: hidden;
}
.kpi::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.kpi-label { font-size: 10px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--muted); margin-bottom: 8px; }
.kpi-value { font-size: 24px; font-weight: 900; letter-spacing: -.04em; color: var(--text); line-height: 1; }
.kpi-hint { font-size: 10px; color: var(--muted); margin-top: 6px; }

/* ── Section header ── */
.section-row {
  display: flex; align-items: baseline; gap: 10px; margin-bottom: 14px;
}
.section-title { font-size: 13px; font-weight: 800; color: var(--text); letter-spacing: -.02em; }
.section-count {
  font-size: 10px; font-weight: 700; color: var(--accent);
  background: rgba(99,102,241,.1); border-radius: 999px; padding: 2px 8px;
}

/* ── Table ── */
div[data-testid="stDataFrame"] { border-radius: var(--r) !important; overflow: hidden; border: 1px solid var(--border) !important; }
div[data-testid="stDataFrame"] th { background: var(--card2) !important; }

/* ── Widgets ── */
label[data-testid="stWidgetLabel"] { font-size: 11px !important; color: var(--sub) !important; font-weight: 500 !important; }
.stTextInput input {
  background: var(--card2) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r2) !important; color: var(--text) !important;
}
div[data-baseweb="select"] { font-size: 12px !important; }
div[data-baseweb="select"] > div {
  background: var(--card2) !important; border-color: var(--border) !important;
  border-radius: var(--r2) !important;
}
div[data-baseweb="tag"] { background: rgba(99,102,241,.15) !important; border-radius: 999px !important; }
div[role="radiogroup"] > label {
  background: var(--card2) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; padding: 8px 12px !important; margin-right: 6px !important;
  font-size: 12px !important; font-weight: 600 !important;
}
div[role="radiogroup"] > label:has(input:checked) {
  border-color: var(--accent) !important;
  background: rgba(99,102,241,.12) !important;
  color: #a5b4fc !important;
}
.stSlider [data-baseweb="slider"] { padding: 0 !important; }
.stToggle [data-baseweb="toggle"] span { background: var(--accent) !important; }

/* ── Buttons ── */
.stButton button, .stDownloadButton button {
  background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
  color: #fff !important; border: none !important; border-radius: 10px !important;
  font-size: 12px !important; font-weight: 700 !important;
  padding: 0.5em 1.2em !important;
  box-shadow: 0 8px 20px rgba(99,102,241,.3) !important;
  transition: filter .15s !important;
}
.stButton button:hover, .stDownloadButton button:hover { filter: brightness(1.08) !important; }

/* ── Divider ── */
.divider { border: none; border-top: 1px solid var(--border2); margin: 12px 0; }

/* ── Stat table ── */
.stat-row { display: flex; justify-content: space-between; align-items: center; padding: 6px 0; border-bottom: 1px solid var(--border2); }
.stat-row:last-child { border-bottom: none; }
.stat-label { font-size: 12px; color: var(--sub); }
.stat-val { font-size: 12px; font-weight: 700; color: var(--text); }

.stMarkdown { margin-bottom: 0 !important; }

/* ── Upload area ── */
[data-testid="stFileUploader"] { background: var(--card2) !important; border-radius: var(--r) !important; border: 1px dashed var(--border) !important; }

/* ── Alerts ── */
.stAlert { background: var(--card2) !important; border-radius: var(--r) !important; }

/* ── Plotly bg fix ── */
.js-plotly-plot .plotly, .plot-container { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

inject_css()

# ─────────────────────────────────────────────
# Data loader
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name=DEFAULT_SHEET)
    df.columns = [str(c).strip() for c in df.columns]

    df["Minute_num"] = _to_num(_col(df, "Minute"))
    df["Second_num"] = _to_num(_col(df, "Second"))

    shot_ts = _col(df, "shot_timestamp").notna()
    shot_out = _col(df, "shot.outcome.name").notna()
    df["is_shot"] = (shot_ts | shot_out).fillna(False)

    df["xg"] = _to_num(_col(df, "shot.statsbomb_xg")).fillna(0.0) if "shot.statsbomb_xg" in df.columns else 0.0

    df["team"]         = _col(df, "pass_team_name")
    df["match"]        = _col(df, "Match")
    df["taker"]        = _col(df, "Taker")
    df["technique"]    = _col(df, "pass.technique.name")
    df["height"]       = _col(df, "pass.height.name")
    df["shot_outcome"] = _col(df, "shot.outcome.name")
    df["sp_outcome"]   = _col(df, "SP_outcome")

    return df

# ─────────────────────────────────────────────
# Plotly chart helpers
# ─────────────────────────────────────────────
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#9898b8", size=11),
    margin=dict(l=8, r=8, t=8, b=8),
    xaxis=dict(showgrid=False, zeroline=False, showline=False, tickfont=dict(size=10)),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.04)", zeroline=False, showline=False, tickfont=dict(size=10)),
    hoverlabel=dict(bgcolor="#1b1b24", font_size=12, font_family="Inter"),
    showlegend=False,
)

ACCENT_COLORS = ["#6366f1","#a855f7","#22d3a0","#f97316","#f43f5e","#facc15","#38bdf8","#fb7185","#34d399","#818cf8"]

def styled_bar(df_, x, y, color=None, orientation="v", height=320):
    if df_ is None or len(df_) == 0:
        st.info("No data for this selection.")
        return
    if not PLOTLY_OK:
        st.dataframe(df_, use_container_width=True, hide_index=True)
        return
    if color and color in df_.columns:
        fig = px.bar(df_, x=x, y=y, color=color, color_discrete_sequence=ACCENT_COLORS, orientation=orientation)
    else:
        fig = px.bar(df_, x=x, y=y, orientation=orientation, color_discrete_sequence=[ACCENT_COLORS[0]])
    fig.update_traces(
        marker_line_width=0,
        marker_color=ACCENT_COLORS[0] if not (color and color in df_.columns) else None,
        hovertemplate="<b>%{x}</b><br>%{y}<extra></extra>" if orientation == "v" else "<b>%{y}</b><br>%{x}<extra></extra>",
    )
    fig.update_layout(height=height, **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def styled_donut(df_, names, values, height=320):
    if df_ is None or len(df_) == 0:
        st.info("No data for this selection.")
        return
    if not PLOTLY_OK:
        st.dataframe(df_, use_container_width=True, hide_index=True)
        return
    fig = go.Figure(go.Pie(
        labels=df_[names], values=df_[values],
        hole=0.65, textinfo="percent",
        textfont=dict(size=10, color="#9898b8"),
        marker=dict(colors=ACCENT_COLORS, line=dict(color="#09090f", width=2)),
        hovertemplate="<b>%{label}</b><br>%{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#9898b8"),
        margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def styled_histogram(series, nbins=24, height=300, color="#6366f1"):
    if not PLOTLY_OK or len(series.dropna()) == 0:
        st.info("No data.")
        return
    fig = px.histogram(series.dropna().to_frame("x"), x="x", nbins=nbins, color_discrete_sequence=[color])
    fig.update_traces(marker_line_width=0, hovertemplate="Minute: %{x}<br>Count: %{y}<extra></extra>")
    fig.update_layout(height=height, **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def styled_scatter(df_, x, y, color=None, text=None, height=360):
    if not PLOTLY_OK or len(df_) == 0:
        st.info("No data.")
        return
    fig = px.scatter(
        df_, x=x, y=y, text=text,
        color=color, color_discrete_sequence=ACCENT_COLORS,
        size_max=18,
    )
    fig.update_traces(
        marker=dict(size=9, line=dict(width=0)),
        textfont=dict(size=9, color="#9898b8"),
        textposition="top center",
    )
    fig.update_layout(height=height, **CHART_LAYOUT)
    if color:
        fig.update_layout(showlegend=True, legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

# ─────────────────────────────────────────────
# File loading row
# ─────────────────────────────────────────────
top_c1, top_c2 = st.columns([3, 1])
with top_c1:
    st.markdown("""
<div class="topbar">
  <div class="topbar-inner">
    <div style="display:flex;gap:12px;align-items:center;">
      <div class="logo-dot"></div>
      <div>
        <p class="brand-title">Corner Kick Analytics</p>
        <p class="brand-sub">Allsvenskan 2025 · StatsBomb Data</p>
      </div>
    </div>
    <div style="display:flex;gap:8px;align-items:center;">
      <span class="badge-pill">⚽ Live Filters</span>
      <span class="badge-pill">📊 xG Powered</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

with top_c2:
    uploaded = st.file_uploader("Upload data (.xlsx)", type=["xlsx"], label_visibility="collapsed")

if uploaded is not None:
    data_file = uploaded
elif os.path.exists(DEFAULT_FILE):
    data_file = DEFAULT_FILE
else:
    st.markdown("""
<div class="card" style="text-align:center;padding:40px;">
  <div style="font-size:40px;margin-bottom:12px;">📂</div>
  <div class="card-title" style="font-size:16px;margin-bottom:8px;">Upload your data file</div>
  <div class="card-sub">Drop <code>Allsvenskan - Corners 2025.xlsx</code> above or place it next to this script</div>
</div>
""", unsafe_allow_html=True)
    st.stop()

with st.spinner("Loading dataset…"):
    df = load_data(data_file)

# ─────────────────────────────────────────────
# Two-column shell
# ─────────────────────────────────────────────
st.markdown('<div class="shell">', unsafe_allow_html=True)
rail_cont = st.container()
main_cont = st.container()

# ─────────────────────────────────────────────
# LEFT RAIL
# ─────────────────────────────────────────────
with rail_cont:
    st.markdown('<div class="rail">', unsafe_allow_html=True)

    st.markdown('<div class="rail-section">View</div>', unsafe_allow_html=True)
    view = st.radio("view", ["League", "Team", "Match", "Player"],
                    horizontal=False, label_visibility="collapsed")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="rail-section">Search</div>', unsafe_allow_html=True)
    if "q" not in st.session_state:
        st.session_state.q = ""
    q = st.text_input("Quick search", placeholder="Team / match / player…",
                      key="q", label_visibility="collapsed")

    st.markdown('<div class="rail-section">Teams</div>', unsafe_allow_html=True)
    teams_all = _safe_unique(df["team"])
    sel_teams = st.multiselect("Teams", teams_all, default=teams_all,
                               label_visibility="collapsed")

    df_t = df[df["team"].isin(sel_teams)] if sel_teams else df
    matches_all = _safe_unique(df_t["match"])

    st.markdown('<div class="rail-section">Matches</div>', unsafe_allow_html=True)
    sel_matches = st.multiselect("Matches", matches_all, default=matches_all,
                                 label_visibility="collapsed")

    df_m = df_t[df_t["match"].isin(sel_matches)] if sel_matches else df_t

    st.markdown('<div class="rail-section">Takers</div>', unsafe_allow_html=True)
    takers_all = _safe_unique(df_m["taker"])
    sel_takers = st.multiselect("Takers", takers_all, default=takers_all,
                                label_visibility="collapsed")

    st.markdown('<div class="rail-section">Technique</div>', unsafe_allow_html=True)
    techniques_all = _safe_unique(df_m["technique"])
    sel_techniques = st.multiselect("Technique", techniques_all, default=techniques_all,
                                    label_visibility="collapsed")

    st.markdown('<div class="rail-section">Delivery Height</div>', unsafe_allow_html=True)
    heights_all = _safe_unique(df_m["height"])
    sel_heights = st.multiselect("Height", heights_all, default=heights_all,
                                 label_visibility="collapsed")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="rail-section">Time Window</div>', unsafe_allow_html=True)

    minute_series = _to_num(df_m["Minute_num"]).dropna()
    minute_unique = np.sort(minute_series.unique()) if len(minute_series) else np.array([])

    minute_mode = "disabled"
    minute_range = None
    minute_single = None
    apply_single = False

    if len(minute_unique) == 0:
        st.caption("No minute data in selection")
    elif len(minute_unique) == 1:
        minute_single = int(minute_unique[0])
        apply_single = st.checkbox(f"Only minute {minute_single}")
        minute_mode = "single"
    else:
        minute_range = st.slider(
            "Minutes",
            min_value=int(minute_unique.min()),
            max_value=int(minute_unique.max()),
            value=(int(minute_unique.min()), int(minute_unique.max())),
            label_visibility="collapsed",
        )
        minute_mode = "range"

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown('<div class="rail-section">Display</div>', unsafe_allow_html=True)
    if "only_shots" not in st.session_state:
        st.session_state.only_shots = False
    if "show_table" not in st.session_state:
        st.session_state.show_table = True

    only_shots = st.toggle("Corners → shot only", key="only_shots")
    show_table  = st.toggle("Show data table",    key="show_table")

    # Focus entity
    focus_team = focus_match = focus_player = None
    if view in ["Team", "Match", "Player"]:
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown(f'<div class="rail-section">Focus {view}</div>', unsafe_allow_html=True)
        if view == "Team":
            focus_team = st.selectbox("Team", teams_all or ["(none)"],
                                      label_visibility="collapsed")
        elif view == "Match":
            focus_match = st.selectbox("Match", matches_all or ["(none)"],
                                       label_visibility="collapsed")
        elif view == "Player":
            focus_player = st.selectbox("Taker", takers_all or ["(none)"],
                                        label_visibility="collapsed")

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    if st.button("↺ Reset all filters"):
        for k in ["q", "only_shots", "show_table"]:
            if k in st.session_state:
                del st.session_state[k]
        st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # .rail

# ─────────────────────────────────────────────
# Apply filters
# ─────────────────────────────────────────────
f = df.copy()
if sel_teams:      f = f[f["team"].isin(sel_teams)]
if sel_matches:    f = f[f["match"].isin(sel_matches)]
if sel_takers:     f = f[f["taker"].isin(sel_takers)]
if sel_techniques: f = f[f["technique"].isin(sel_techniques)]
if sel_heights:    f = f[f["height"].isin(sel_heights)]

if q.strip():
    qq = q.strip().lower()
    mask = (_contains(f["team"], qq) | _contains(f["match"], qq)
            | _contains(f["taker"], qq))
    f = f[mask]

f["Minute_num"] = _to_num(f["Minute_num"])
if minute_mode == "range" and minute_range:
    f = f[f["Minute_num"].between(minute_range[0], minute_range[1])]
elif minute_mode == "single" and apply_single and minute_single is not None:
    f = f[f["Minute_num"] == minute_single]

if only_shots:
    f = f[f["is_shot"] == True]

# Entity focus
if view == "Team" and focus_team and focus_team != "(none)":
    f_view = f[f["team"].astype(str) == str(focus_team)].copy()
elif view == "Match" and focus_match and focus_match != "(none)":
    f_view = f[f["match"].astype(str) == str(focus_match)].copy()
elif view == "Player" and focus_player and focus_player != "(none)":
    f_view = f[f["taker"].astype(str) == str(focus_player)].copy()
else:
    f_view = f.copy()

# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
with main_cont:
    st.markdown('<div class="content">', unsafe_allow_html=True)

    # ── KPI strip ──
    total  = len(f_view)
    n_mat  = (pd.Series(f_view["match"]).astype(str)
              .replace("nan", np.nan).dropna().nunique())
    cpm    = total / n_mat if n_mat else 0
    shots  = int(f_view["is_shot"].fillna(False).sum())
    sr     = shots / total if total else 0
    xg_tot = (float(f_view["xg"].fillna(0).sum())
               if "xg" in f_view.columns else 0.0)
    sp_txt = (f_view.get("sp_outcome", pd.Series(dtype=str))
              .fillna("").astype(str))
    s3s    = int(sp_txt.str.contains("shot within 3 seconds",
                                      case=False, na=False).sum())
    goals  = int(
        f_view.get("shot_outcome", pd.Series(dtype=str))
        .fillna("").astype(str)
        .str.contains("Goal", case=False, na=False).sum()
    )

    st.markdown(f"""
<div class="card">
  <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:16px;">
    <div class="card-title" style="font-size:14px;">Overview</div>
    <span style="font-size:11px;color:var(--accent);background:rgba(99,102,241,.1);
                 border-radius:999px;padding:2px 10px;font-weight:700;">
      {view} view · {total:,} events
    </span>
  </div>
  <div class="kpis">
    <div class="kpi">
      <div class="kpi-label">Corner events</div>
      <div class="kpi-value">{total:,}</div>
      <div class="kpi-hint">In selection</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Matches</div>
      <div class="kpi-value">{n_mat:,}</div>
      <div class="kpi-hint">Unique</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Corners / match</div>
      <div class="kpi-value">{cpm:.1f}</div>
      <div class="kpi-hint">Volume intensity</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Shot rate</div>
      <div class="kpi-value">{sr*100:.1f}%</div>
      <div class="kpi-hint">Corners → shot</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Total xG</div>
      <div class="kpi-value">{xg_tot:.3f}</div>
      <div class="kpi-hint">From shots</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Goals</div>
      <div class="kpi-value">{goals}</div>
      <div class="kpi-hint">Direct from corner</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # LEAGUE VIEW
    # ═══════════════════════════════════════════
    if view == "League":

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Corner Volume by Team</div>"
                        "<div class='card-sub'>Total corners taken — sorted ascending</div>",
                        unsafe_allow_html=True)
            team_counts = (f_view.groupby("team", dropna=False).size()
                           .sort_values(ascending=True).reset_index(name="corners"))
            styled_bar(team_counts, x="corners", y="team", orientation="h", height=380)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Delivery Technique Mix</div>"
                        "<div class='card-sub'>League-wide distribution</div>",
                        unsafe_allow_html=True)
            tech_counts = (f_view.groupby("technique", dropna=False).size()
                           .sort_values(ascending=False).reset_index(name="corners"))
            styled_donut(tech_counts, "technique", "corners", height=380)
            st.markdown("</div>", unsafe_allow_html=True)

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>xG from Corners by Team</div>"
                        "<div class='card-sub'>Total expected goals generated</div>",
                        unsafe_allow_html=True)
            xg_team = (f_view.groupby("team", dropna=False)["xg"].sum()
                       .sort_values(ascending=True).reset_index())
            styled_bar(xg_team, x="xg", y="team", orientation="h", height=380)
            st.markdown("</div>", unsafe_allow_html=True)

        with c4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Shot Outcomes</div>"
                        "<div class='card-sub'>All shots from corners</div>",
                        unsafe_allow_html=True)
            shots_df = f_view[f_view["is_shot"] == True]
            shot_out = (shots_df.groupby("shot_outcome", dropna=False).size()
                        .sort_values(ascending=False).reset_index(name="shots"))
            styled_donut(shot_out, "shot_outcome", "shots", height=380)
            st.markdown("</div>", unsafe_allow_html=True)

        # Efficiency scatter
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Team Efficiency: Shot Rate vs xG / Shot</div>"
                    "<div class='card-sub'>Horizontal = how often a corner becomes a shot · "
                    "Vertical = quality of those shots · ideal = top-right</div>",
                    unsafe_allow_html=True)
        eff = f_view.groupby("team", dropna=False).agg(
            corners=("is_shot", "count"),
            shot_count=("is_shot", "sum"),
            total_xg=("xg", "sum"),
        ).reset_index()
        eff["shot_rate"]   = eff["shot_count"] / eff["corners"].replace(0, np.nan)
        eff["xg_per_shot"] = eff["total_xg"]   / eff["shot_count"].replace(0, np.nan)
        eff = eff.dropna(subset=["shot_rate", "xg_per_shot"])
        styled_scatter(eff, x="shot_rate", y="xg_per_shot", text="team", height=340)
        st.markdown("</div>", unsafe_allow_html=True)

        # Corner timing
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Corner Timing Distribution</div>"
                    "<div class='card-sub'>When in the match are corners awarded?</div>",
                    unsafe_allow_html=True)
        styled_histogram(_to_num(f_view["Minute_num"]), nbins=30, height=260)
        st.markdown("</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # TEAM VIEW
    # ═══════════════════════════════════════════
    elif view == "Team":
        team_label = focus_team if focus_team else "All teams"

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-title'>Primary Takers — {team_label}</div>"
                        "<div class='card-sub'>Corner volume per taker (top 15)</div>",
                        unsafe_allow_html=True)
            tk = (f_view.groupby("taker", dropna=False).size()
                  .sort_values(ascending=True).head(15).reset_index(name="corners"))
            styled_bar(tk, x="corners", y="taker", orientation="h", height=340)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-title'>xG by Taker — {team_label}</div>"
                        "<div class='card-sub'>Expected goals created per taker</div>",
                        unsafe_allow_html=True)
            xg_tk = (f_view.groupby("taker", dropna=False)["xg"].sum()
                     .sort_values(ascending=True).head(15).reset_index())
            styled_bar(xg_tk, x="xg", y="taker", orientation="h", height=340)
            st.markdown("</div>", unsafe_allow_html=True)

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Technique Profile</div>", unsafe_allow_html=True)
            tech = f_view.groupby("technique", dropna=False).size().reset_index(name="n")
            styled_donut(tech, "technique", "n", height=300)
            st.markdown("</div>", unsafe_allow_html=True)

        with c4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Delivery Height Profile</div>", unsafe_allow_html=True)
            ht = f_view.groupby("height", dropna=False).size().reset_index(name="n")
            styled_donut(ht, "height", "n", height=300)
            st.markdown("</div>", unsafe_allow_html=True)

        # SP outcome breakdown
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Set Piece Outcomes</div>"
                    "<div class='card-sub'>SP_outcome breakdown for this team</div>",
                    unsafe_allow_html=True)
        sp = (f_view.groupby("sp_outcome", dropna=False).size()
              .sort_values(ascending=True).reset_index(name="count"))
        styled_bar(sp, x="count", y="sp_outcome", orientation="h", height=280)
        st.markdown("</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # MATCH VIEW
    # ═══════════════════════════════════════════
    elif view == "Match":

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Corners by Team</div>", unsafe_allow_html=True)
            by_team = (f_view.groupby("team", dropna=False).size()
                       .sort_values(ascending=False).reset_index(name="corners"))
            styled_bar(by_team, x="team", y="corners", height=320)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Corner Timing</div>"
                        "<div class='card-sub'>Minute distribution</div>",
                        unsafe_allow_html=True)
            styled_histogram(_to_num(f_view["Minute_num"]), nbins=24, height=320)
            st.markdown("</div>", unsafe_allow_html=True)

        c3, c4 = st.columns(2)

        with c3:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Technique Distribution</div>", unsafe_allow_html=True)
            tech = f_view.groupby("technique", dropna=False).size().reset_index(name="n")
            styled_donut(tech, "technique", "n", height=280)
            st.markdown("</div>", unsafe_allow_html=True)

        with c4:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Shot Outcomes</div>", unsafe_allow_html=True)
            sout = (f_view[f_view["is_shot"] == True]
                    .groupby("shot_outcome", dropna=False).size()
                    .reset_index(name="shots"))
            styled_donut(sout, "shot_outcome", "shots", height=280)
            st.markdown("</div>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════
    # PLAYER VIEW
    # ═══════════════════════════════════════════
    elif view == "Player":
        player_label = focus_player if focus_player else "All takers"

        p_total = len(f_view)
        p_shots = int(f_view["is_shot"].fillna(False).sum())
        p_sr    = p_shots / p_total if p_total else 0
        p_xg    = (float(f_view["xg"].fillna(0).sum())
                   if "xg" in f_view.columns else 0.0)
        p_xg_c  = p_xg / p_total if p_total else 0
        top_ht  = "—"
        if p_total and "height" in f_view.columns:
            vc = f_view["height"].fillna("Unknown").astype(str).value_counts()
            if len(vc):
                top_ht = vc.index[0]

        st.markdown(f"""
<div class="card">
  <div class="card-title">{player_label} — Player Card</div>
  <div class="card-sub">Performance summary for current filters</div>
  <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px;">
    <div class="kpi">
      <div class="kpi-label">Corners</div>
      <div class="kpi-value">{p_total:,}</div>
      <div class="kpi-hint">Volume</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Shot rate</div>
      <div class="kpi-value">{p_sr*100:.1f}%</div>
      <div class="kpi-hint">→ shot</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">Total xG</div>
      <div class="kpi-value">{p_xg:.3f}</div>
      <div class="kpi-hint">From shots</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">xG / corner</div>
      <div class="kpi-value">{p_xg_c:.4f}</div>
      <div class="kpi-hint">Efficiency</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Technique Mix</div>", unsafe_allow_html=True)
            tech = f_view.groupby("technique", dropna=False).size().reset_index(name="n")
            styled_donut(tech, "technique", "n", height=280)
            st.markdown("</div>", unsafe_allow_html=True)

        with c2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='card-title'>Delivery Height Mix</div>", unsafe_allow_html=True)
            ht = f_view.groupby("height", dropna=False).size().reset_index(name="n")
            styled_donut(ht, "height", "n", height=280)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='card-title'>Corner Timing</div>"
                    "<div class='card-sub'>Minute distribution</div>",
                    unsafe_allow_html=True)
        styled_histogram(_to_num(f_view["Minute_num"]), nbins=20, height=240)
        st.markdown("</div>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # DATA TABLE
    # ─────────────────────────────────────────────
    if show_table:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-title'>Data Table</div>"
                    f"<div class='card-sub'>{len(f_view):,} rows · current view</div>",
                    unsafe_allow_html=True)

        preferred = ["match", "team", "taker", "Minute_num", "Second_num",
                     "technique", "height", "sp_outcome", "is_shot",
                     "shot_outcome", "xg"]
        cols = ([c for c in preferred if c in f_view.columns]
                + [c for c in f_view.columns if c not in preferred])

        st.dataframe(
            f_view[cols].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
            height=420,
        )

        csv = f_view[cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇ Download CSV (current view)",
            data=csv,
            file_name=f"corners_{view.lower()}_view.csv",
            mime="text/csv",
        )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)  # .content

st.markdown("</div>", unsafe_allow_html=True)  # .shell
