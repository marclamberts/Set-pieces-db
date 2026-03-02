import os
import pandas as pd
import numpy as np
import streamlit as st

DEFAULT_FILE = "Allsvenskan - Corners 2025.xlsx"
DEFAULT_SHEET = "Sheet 1"

ACCENT_COLORS = [
    "#6366f1","#a855f7","#22d3a0","#f97316",
    "#f43f5e","#facc15","#38bdf8","#fb7185",
    "#34d399","#818cf8",
]

def _col(df, name):
    return df[name] if name in df.columns else pd.Series([np.nan]*len(df), index=df.index)

def _safe_unique(series):
    return sorted(pd.Series(series).dropna().astype(str).unique().tolist())

def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def _contains(s, q):
    return s.fillna("").astype(str).str.lower().str.contains(q, na=False)

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not os.path.exists(DEFAULT_FILE):
        st.error(f"Data file not found. Place `{DEFAULT_FILE}` next to Home.py and restart.")
        st.stop()
    df = pd.read_excel(DEFAULT_FILE, sheet_name=DEFAULT_SHEET)
    df.columns = [str(c).strip() for c in df.columns]
    df["Minute_num"] = _to_num(_col(df, "Minute"))
    df["Second_num"] = _to_num(_col(df, "Second"))
    shot_ts  = _col(df, "shot_timestamp").notna()
    shot_out = _col(df, "shot.outcome.name").notna()
    df["is_shot"]      = (shot_ts | shot_out).fillna(False)
    df["xg"]           = (_to_num(_col(df, "shot.statsbomb_xg")).fillna(0.0)
                          if "shot.statsbomb_xg" in df.columns else 0.0)
    df["team"]         = _col(df, "pass_team_name")
    df["match"]        = _col(df, "Match")
    df["taker"]        = _col(df, "Taker")
    df["technique"]    = _col(df, "pass.technique.name")
    df["height"]       = _col(df, "pass.height.name")
    df["shot_outcome"] = _col(df, "shot.outcome.name")
    df["sp_outcome"]   = _col(df, "SP_outcome")
    return df

# ── Plotly chart config ──
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'DM Sans', sans-serif", color="#9898b8", size=11),
    margin=dict(l=8, r=8, t=8, b=8),
    xaxis=dict(showgrid=False, zeroline=False, showline=False, tickfont=dict(size=10)),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.05)",
               zeroline=False, showline=False, tickfont=dict(size=10)),
    hoverlabel=dict(bgcolor="#1b1b24", font_size=12,
                    font_family="'DM Sans', sans-serif"),
    showlegend=False,
)

def inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700;800;900&family=DM+Mono:wght@400;500&display=swap');

* { font-family: 'DM Sans', sans-serif !important; }
code, .stCode { font-family: 'DM Mono', monospace !important; }

#MainMenu, header, footer, .stDeployButton { visibility:hidden; display:none; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width:1560px !important; }

:root {
  --bg:      #08080e;
  --card:    #111119;
  --card2:   #17171f;
  --border:  rgba(255,255,255,0.06);
  --border2: rgba(255,255,255,0.03);
  --text:    #ededf5;
  --sub:     #9494b0;
  --muted:   #505070;
  --acc:     #6366f1;
  --acc2:    #a855f7;
  --green:   #22d3a0;
  --orange:  #f97316;
  --r:       14px;
  --r2:      9px;
}

html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--card) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebarNavItems"] a {
  border-radius: 10px !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 8px 12px !important;
  margin: 2px 0 !important;
  transition: background .15s !important;
}
[data-testid="stSidebarNavItems"] a:hover {
  background: rgba(99,102,241,.12) !important;
}
[data-testid="stSidebarNavItems"] a[aria-current="page"] {
  background: rgba(99,102,241,.18) !important;
  color: #a5b4fc !important;
}

/* ── Page header ── */
.page-header {
  padding: 28px 0 24px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 24px;
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 16px;
}
.page-header-left {}
.page-eyebrow {
  font-size: 10px; font-weight: 700;
  letter-spacing: .12em; text-transform: uppercase;
  color: var(--acc); margin-bottom: 6px;
}
.page-title {
  font-size: 26px; font-weight: 900;
  letter-spacing: -.04em; color: var(--text);
  margin: 0; line-height: 1.1;
}
.page-sub { font-size: 13px; color: var(--sub); margin-top: 5px; }

/* ── Hero (homepage only) ── */
.hero {
  position: relative;
  padding: 70px 0 60px 0;
  overflow: hidden;
  margin-bottom: 36px;
}
.hero-bg {
  position: absolute; inset: 0;
  background: radial-gradient(ellipse 80% 60% at 60% 40%,
    rgba(99,102,241,.18) 0%,
    rgba(168,85,247,.10) 40%,
    transparent 70%);
  pointer-events: none;
}
.hero-orb {
  position: absolute;
  border-radius: 50%;
  filter: blur(80px);
  pointer-events: none;
}
.hero-orb1 {
  width: 400px; height: 400px;
  background: rgba(99,102,241,.15);
  top: -80px; right: 10%;
}
.hero-orb2 {
  width: 280px; height: 280px;
  background: rgba(168,85,247,.12);
  bottom: -40px; right: 25%;
}
.hero-content { position: relative; z-index: 1; max-width: 680px; }
.hero-eyebrow {
  font-size: 10px; font-weight: 700;
  letter-spacing: .14em; text-transform: uppercase;
  color: var(--acc); margin-bottom: 14px;
  display: flex; align-items: center; gap: 8px;
}
.hero-eyebrow::before {
  content: '';
  display: inline-block;
  width: 20px; height: 2px;
  background: var(--acc);
}
.hero-title {
  font-size: clamp(42px, 6vw, 72px);
  font-weight: 900;
  letter-spacing: -.05em;
  color: var(--text);
  margin: 0 0 18px 0;
  line-height: .95;
}
.hero-accent {
  background: linear-gradient(135deg, var(--acc), var(--acc2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-desc {
  font-size: 15px; color: var(--sub);
  line-height: 1.65; margin-bottom: 24px;
  max-width: 540px;
}
.hero-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.pill {
  background: rgba(99,102,241,.1);
  border: 1px solid rgba(99,102,241,.2);
  border-radius: 999px;
  padding: 5px 14px;
  font-size: 11px; font-weight: 600;
  color: #a5b4fc;
}

/* ── Stat grid (homepage) ── */
.stat-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 12px;
  margin-bottom: 40px;
}
@media(max-width:1200px){ .stat-grid { grid-template-columns: repeat(3,1fr); } }

.stat-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 20px;
  position: relative;
  overflow: hidden;
  transition: border-color .2s, transform .2s;
}
.stat-card::after {
  content: '';
  position: absolute; top:0; left:0; right:0; height:2px;
  background: linear-gradient(90deg, var(--acc), var(--acc2));
}
.stat-card:hover {
  border-color: rgba(99,102,241,.3);
  transform: translateY(-2px);
}
.stat-icon { font-size: 20px; margin-bottom: 10px; }
.stat-val {
  font-size: 28px; font-weight: 900;
  letter-spacing: -.05em; color: var(--text);
  line-height: 1;
}
.stat-label {
  font-size: 12px; font-weight: 600;
  color: var(--sub); margin-top: 6px;
}
.stat-sub { font-size: 10px; color: var(--muted); margin-top: 3px; }

/* ── Nav cards (homepage) ── */
.nav-section-title {
  font-size: 11px; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 14px;
}
.nav-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
  margin-bottom: 40px;
}
@media(max-width:1100px){ .nav-grid { grid-template-columns: repeat(2,1fr); } }

.nav-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 20px;
  text-decoration: none !important;
  display: flex;
  align-items: flex-start;
  gap: 14px;
  transition: border-color .2s, transform .2s, box-shadow .2s;
  cursor: pointer;
}
.nav-card:hover {
  border-color: rgba(99,102,241,.35);
  transform: translateY(-2px);
  box-shadow: 0 16px 40px rgba(99,102,241,.12);
}
.nav-card-icon { font-size: 24px; flex-shrink: 0; margin-top: 1px; }
.nav-card-body { flex: 1; }
.nav-card-title {
  font-size: 14px; font-weight: 800;
  color: var(--text); margin-bottom: 5px;
  letter-spacing: -.02em;
}
.nav-card-desc { font-size: 12px; color: var(--sub); line-height: 1.55; }
.nav-card-arrow {
  color: var(--acc); font-size: 18px;
  align-self: center; flex-shrink: 0;
  opacity: 0;
  transition: opacity .2s, transform .2s;
}
.nav-card:hover .nav-card-arrow { opacity: 1; transform: translateX(3px); }

/* ── Module cards ── */
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 20px;
  margin-bottom: 14px;
}
.card-title {
  font-size: 13px; font-weight: 800;
  letter-spacing: -.02em; color: var(--text);
  margin: 0 0 3px 0;
}
.card-sub { font-size: 11px; color: var(--sub); margin: 0 0 14px 0; }

/* ── KPI strip ── */
.kpis { display: grid; grid-template-columns: repeat(6,1fr); gap:10px; }
@media(max-width:1300px){ .kpis { grid-template-columns: repeat(3,1fr); } }

.kpi {
  background: var(--card2); border: 1px solid var(--border2);
  border-radius: var(--r2); padding: 14px 16px;
  position: relative; overflow: hidden;
}
.kpi::before {
  content: ''; position:absolute; top:0; left:0; right:0; height:2px;
  background: linear-gradient(90deg, var(--acc), var(--acc2));
}
.kpi-label {
  font-size: 10px; font-weight: 600;
  letter-spacing:.06em; text-transform:uppercase;
  color: var(--muted); margin-bottom: 7px;
}
.kpi-value {
  font-size: 22px; font-weight: 900;
  letter-spacing: -.04em; color: var(--text); line-height: 1;
}
.kpi-hint { font-size: 10px; color: var(--muted); margin-top: 5px; }

/* ── Widgets ── */
label[data-testid="stWidgetLabel"] {
  font-size: 11px !important; color: var(--sub) !important; font-weight: 600 !important;
}
.stTextInput input {
  background: var(--card2) !important; border: 1px solid var(--border) !important;
  border-radius: var(--r2) !important; color: var(--text) !important; font-size:13px !important;
}
div[data-baseweb="select"] { font-size: 13px !important; }
div[data-baseweb="select"] > div {
  background: var(--card2) !important; border-color: var(--border) !important;
  border-radius: var(--r2) !important; color: var(--text) !important;
}
div[data-baseweb="tag"] { background: rgba(99,102,241,.15) !important; border-radius:999px !important; }

/* ── Buttons ── */
.stButton button, .stDownloadButton button {
  background: linear-gradient(135deg, var(--acc), var(--acc2)) !important;
  color: #fff !important; border: none !important; border-radius: 10px !important;
  font-size: 12px !important; font-weight: 700 !important;
  padding: 0.5em 1.4em !important;
  box-shadow: 0 8px 20px rgba(99,102,241,.28) !important;
}
.stButton button:hover, .stDownloadButton button:hover { filter: brightness(1.08) !important; }

/* ── Table ── */
div[data-testid="stDataFrame"] {
  border-radius: var(--r) !important; overflow:hidden;
  border: 1px solid var(--border) !important;
}

/* ── Divider ── */
.divider { border:none; border-top: 1px solid var(--border2); margin: 14px 0; }

/* ── Filter row ── */
.filter-bar {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 14px 16px;
  margin-bottom: 18px;
}

/* ── Footer ── */
.footer {
  text-align: center;
  font-size: 11px; color: var(--muted);
  padding: 32px 0 16px 0;
  border-top: 1px solid var(--border2);
  margin-top: 24px;
}

/* ── Misc ── */
.stMarkdown { margin-bottom: 0 !important; }
.stAlert { background: var(--card2) !important; border-radius: var(--r) !important; }
.js-plotly-plot .plotly, .plot-container { background: transparent !important; }

/* ── Sidebar toggle ── */
div[role="radiogroup"] > label {
  background: var(--card2) !important; border: 1px solid var(--border) !important;
  border-radius: 9px !important; padding: 7px 11px !important;
  font-size: 12px !important; font-weight: 600 !important; margin-right:5px !important;
}
div[role="radiogroup"] > label:has(input:checked) {
  border-color: var(--acc) !important;
  background: rgba(99,102,241,.12) !important;
  color: #a5b4fc !important;
}
</style>
""", unsafe_allow_html=True)


# ── Plotly helpers ──
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


def styled_bar(df_, x, y, orientation="v", height=320, color_col=None):
    if df_ is None or len(df_) == 0:
        st.info("No data for this selection.")
        return
    if not PLOTLY_OK:
        st.dataframe(df_, use_container_width=True, hide_index=True); return
    if color_col and color_col in df_.columns:
        fig = px.bar(df_, x=x, y=y, color=color_col,
                     color_discrete_sequence=ACCENT_COLORS, orientation=orientation)
    else:
        fig = px.bar(df_, x=x, y=y, orientation=orientation,
                     color_discrete_sequence=[ACCENT_COLORS[0]])
    fig.update_traces(
        marker_line_width=0,
        marker_color=ACCENT_COLORS[0] if not (color_col and color_col in df_.columns) else None,
        hovertemplate=("<b>%{x}</b><br>%{y}<extra></extra>"
                       if orientation == "v" else "<b>%{y}</b><br>%{x}<extra></extra>"),
    )
    fig.update_layout(height=height, **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def styled_donut(df_, names, values, height=320):
    if df_ is None or len(df_) == 0:
        st.info("No data."); return
    if not PLOTLY_OK:
        st.dataframe(df_, use_container_width=True, hide_index=True); return
    fig = go.Figure(go.Pie(
        labels=df_[names], values=df_[values],
        hole=0.65, textinfo="percent",
        textfont=dict(size=10, color="#9898b8"),
        marker=dict(colors=ACCENT_COLORS, line=dict(color="#08080e", width=2)),
        hovertemplate="<b>%{label}</b><br>%{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'DM Sans',sans-serif", color="#9898b8"),
        margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def styled_histogram(series, nbins=24, height=280):
    if not PLOTLY_OK or len(series.dropna()) == 0:
        st.info("No data."); return
    fig = px.histogram(series.dropna().to_frame("x"), x="x", nbins=nbins,
                       color_discrete_sequence=[ACCENT_COLORS[0]])
    fig.update_traces(marker_line_width=0,
                      hovertemplate="Minute: %{x}<br>Count: %{y}<extra></extra>")
    fig.update_layout(height=height, **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def styled_scatter(df_, x, y, text=None, height=340):
    if not PLOTLY_OK or len(df_) == 0:
        st.info("No data."); return
    fig = px.scatter(df_, x=x, y=y, text=text,
                     color_discrete_sequence=ACCENT_COLORS)
    fig.update_traces(
        marker=dict(size=10, line=dict(width=0), color=ACCENT_COLORS[0],
                    opacity=0.85),
        textfont=dict(size=9, color="#9898b8"),
        textposition="top center",
    )
    fig.update_layout(height=height, **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def page_header(eyebrow: str, title: str, sub: str = ""):
    st.markdown(f"""
<div class="page-header">
  <div class="page-header-left">
    <div class="page-eyebrow">{eyebrow}</div>
    <h1 class="page-title">{title}</h1>
    {"<div class='page-sub'>" + sub + "</div>" if sub else ""}
  </div>
</div>
""", unsafe_allow_html=True)


def kpi_strip(items: list):
    """items = list of (label, value, hint)"""
    cols_html = "".join(f"""
<div class="kpi">
  <div class="kpi-label">{lbl}</div>
  <div class="kpi-value">{val}</div>
  <div class="kpi-hint">{hint}</div>
</div>""" for lbl, val, hint in items)
    st.markdown(f"""
<div class="card">
  <div class="kpis">{cols_html}</div>
</div>""", unsafe_allow_html=True)
