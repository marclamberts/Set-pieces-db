import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from io import BytesIO
from scipy import stats as scipy_stats
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
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
# DESIGN TOKENS
# =========================================================
BG       = "#07111f"
BG_2     = "#0b1730"
CARD     = "#101a2b"
CARD_2   = "#16243a"
CARD_3   = "#1a2d47"
BORDER   = "rgba(255,255,255,0.08)"
BORDER_2 = "rgba(255,255,255,0.14)"
TEXT     = "#f3f7fc"
MUTED    = "#99adc7"
MUTED_2  = "#6b87a8"
ACCENT   = "#5da8ff"
ACCENT_2 = "#8ad6ff"
ACCENT_3 = "#c4e8ff"
SUCCESS  = "#34d399"
SUCCESS_2= "#6ee7b7"
WARNING  = "#fbbf24"
WARNING_2= "#fde68a"
DANGER   = "#fb7185"
DANGER_2 = "#fda4af"
PURPLE   = "#a78bfa"
ORANGE   = "#fb923c"
PITCH    = "#133d24"
PITCH_2  = "#1a5232"
PITCH_LINE = "rgba(255,255,255,0.65)"

QUAL_PALETTE = [ACCENT, SUCCESS, WARNING, DANGER, PURPLE, ORANGE,
                ACCENT_2, SUCCESS_2, WARNING_2, DANGER_2]

px.defaults.template = "plotly_dark"

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;0,800;0,900;1,400&family=JetBrains+Mono:wght@400;600&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}
body, .stApp {{ font-family: 'DM Sans', sans-serif !important; }}

.stApp {{
    background:
        radial-gradient(ellipse 900px 600px at 90% -10%, rgba(93,168,255,0.10) 0%, transparent 60%),
        radial-gradient(ellipse 700px 500px at -5% 20%, rgba(52,211,153,0.07) 0%, transparent 55%),
        radial-gradient(ellipse 600px 400px at 50% 110%, rgba(167,139,250,0.06) 0%, transparent 50%),
        linear-gradient(180deg, {BG} 0%, {BG_2} 100%);
    color: {TEXT};
}}
.block-container {{
    max-width: 1760px;
    padding-top: 0.8rem;
    padding-bottom: 1.5rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}}
[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, #080f1d 0%, #060e1a 100%);
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebarNav"] {{ display: none; }}

/* ---- Hero ---- */
.hero-wrap {{
    background: linear-gradient(135deg, rgba(93,168,255,0.14) 0%, rgba(93,168,255,0.04) 60%, rgba(52,211,153,0.06) 100%);
    border: 1px solid rgba(93,168,255,0.18);
    border-radius: 28px;
    padding: 24px 28px 20px 28px;
    margin-bottom: 16px;
    box-shadow: 0 16px 48px rgba(0,0,0,0.22), inset 0 1px 0 rgba(255,255,255,0.06);
    position: relative;
    overflow: hidden;
}}
.hero-wrap::before {{
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(93,168,255,0.12) 0%, transparent 70%);
    pointer-events: none;
}}
.hero-title {{
    font-size: 2.2rem; font-weight: 900; line-height: 1.0;
    margin-bottom: 0.35rem; color: {TEXT};
    letter-spacing: -0.02em;
}}
.hero-title span {{ color: {ACCENT}; }}
.hero-sub {{ color: {MUTED}; font-size: 0.97rem; max-width: 900px; line-height: 1.5; }}
.pill {{
    display: inline-block; padding: 0.28rem 0.68rem; border-radius: 999px;
    background: rgba(93,168,255,0.12); color: #d4e8ff;
    border: 1px solid rgba(93,168,255,0.20); font-size: 0.76rem;
    margin-right: 0.4rem; margin-top: 0.5rem; font-weight: 500;
}}
.pill-green {{
    background: rgba(52,211,153,0.12); color: #a7f3d0;
    border-color: rgba(52,211,153,0.20);
}}
.pill-orange {{
    background: rgba(251,146,60,0.12); color: #fed7aa;
    border-color: rgba(251,146,60,0.20);
}}

/* ---- KPI Cards ---- */
.kpi-card {{
    background: linear-gradient(160deg, {CARD} 0%, {CARD_2} 100%);
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 18px 18px 14px 18px;
    min-height: 116px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.18), inset 0 1px 0 rgba(255,255,255,0.04);
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s;
}}
.kpi-card:hover {{
    box-shadow: 0 12px 36px rgba(0,0,0,0.28);
    border-color: rgba(255,255,255,0.12);
}}
.kpi-label {{
    color: {MUTED}; text-transform: uppercase; font-size: 0.70rem;
    letter-spacing: 0.10em; margin-bottom: 8px; font-weight: 600;
}}
.kpi-value {{
    color: {TEXT}; font-weight: 900; font-size: 1.95rem; line-height: 1.0;
    letter-spacing: -0.02em;
}}
.kpi-suffix {{ font-size: 1.1rem; font-weight: 600; opacity: 0.75; margin-left: 2px; }}
.kpi-foot {{ margin-top: 8px; color: {MUTED}; font-size: 0.80rem; }}
.kpi-delta-pos {{ color: {SUCCESS}; font-size: 0.78rem; margin-top: 5px; font-weight: 600; }}
.kpi-delta-neg {{ color: {DANGER}; font-size: 0.78rem; margin-top: 5px; font-weight: 600; }}
.kpi-accent-bar {{
    position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
    border-radius: 0 0 20px 20px;
}}

/* ---- Section headers ---- */
.section-title {{
    font-size: 1.10rem; font-weight: 800; margin: 0.1rem 0 0.18rem 0;
    letter-spacing: -0.01em; color: {TEXT};
}}
.section-sub {{ color: {MUTED}; font-size: 0.90rem; margin-bottom: 0.85rem; line-height: 1.4; }}
.section-divider {{
    height: 1px; background: {BORDER};
    margin: 20px 0; border: none;
}}

/* ---- Insight boxes ---- */
.insight-box {{
    background: linear-gradient(135deg, rgba(93,168,255,0.09), rgba(93,168,255,0.04));
    border: 1px solid rgba(93,168,255,0.18); border-radius: 16px;
    padding: 14px 16px; margin-bottom: 8px; min-height: 76px;
}}
.insight-box-green {{
    background: linear-gradient(135deg, rgba(52,211,153,0.09), rgba(52,211,153,0.04));
    border-color: rgba(52,211,153,0.22);
}}
.insight-box-orange {{
    background: linear-gradient(135deg, rgba(251,191,36,0.09), rgba(251,191,36,0.04));
    border-color: rgba(251,191,36,0.22);
}}
.insight-box-red {{
    background: linear-gradient(135deg, rgba(251,113,133,0.09), rgba(251,113,133,0.04));
    border-color: rgba(251,113,133,0.22);
}}
.insight-box-purple {{
    background: linear-gradient(135deg, rgba(167,139,250,0.09), rgba(167,139,250,0.04));
    border-color: rgba(167,139,250,0.22);
}}
.insight-icon {{ font-size: 1.3rem; margin-right: 8px; vertical-align: middle; }}
.insight-title {{ font-weight: 700; font-size: 0.92rem; }}
.insight-body {{ color: {MUTED}; font-size: 0.87rem; margin-top: 4px; line-height: 1.45; }}

/* ---- Sub cards ---- */
.sub-card {{
    background: linear-gradient(160deg, rgba(255,255,255,0.035), rgba(255,255,255,0.018));
    border: 1px solid {BORDER}; border-radius: 16px; padding: 14px 16px;
}}
.sub-card-accent {{
    background: linear-gradient(135deg, rgba(93,168,255,0.08), rgba(93,168,255,0.03));
    border-color: rgba(93,168,255,0.16);
}}

/* ---- Data tables ---- */
div[data-testid="stDataFrame"] {{
    border: 1px solid {BORDER}; border-radius: 16px; overflow: hidden;
    box-shadow: 0 6px 20px rgba(0,0,0,0.16);
}}

/* ---- Tabs ---- */
.stTabs [data-baseweb="tab-list"] {{
    gap: 3px; background: rgba(255,255,255,0.025);
    border-radius: 14px; padding: 4px; border: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 10px; padding: 6px 14px;
    font-size: 0.85rem; font-weight: 500;
}}
.stTabs [aria-selected="true"] {{
    background: rgba(93,168,255,0.16) !important;
    color: {ACCENT_3} !important;
}}

/* ---- Sidebar ---- */
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stSlider"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stCheckbox"] label {{
    color: {MUTED}; font-size: 0.79rem;
    text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600;
}}

/* ---- Download buttons ---- */
.stDownloadButton button {{
    background: linear-gradient(135deg, rgba(93,168,255,0.18), rgba(93,168,255,0.08)) !important;
    border: 1px solid rgba(93,168,255,0.28) !important;
    color: {TEXT} !important; border-radius: 12px !important;
    font-weight: 600 !important; font-family: 'DM Sans', sans-serif !important;
    transition: all 0.2s !important;
}}
.stDownloadButton button:hover {{
    background: linear-gradient(135deg, rgba(93,168,255,0.28), rgba(93,168,255,0.16)) !important;
    border-color: rgba(93,168,255,0.42) !important;
    transform: translateY(-1px) !important;
}}

/* ---- Mono values ---- */
.mono {{ font-family: 'JetBrains Mono', monospace; }}

/* ---- Empty state ---- */
.empty-state {{
    text-align: center; padding: 56px 24px; color: {MUTED};
    font-size: 0.94rem; border: 1px dashed rgba(255,255,255,0.10);
    border-radius: 16px; background: rgba(255,255,255,0.015);
}}
.empty-state-icon {{ font-size: 2.5rem; margin-bottom: 10px; opacity: 0.5; }}

/* ---- Stat table ---- */
.stat-row {{
    display: flex; align-items: center;
    padding: 9px 0; border-bottom: 1px solid {BORDER};
}}
.stat-label {{ flex: 1.4; color: {MUTED}; font-size: 0.86rem; }}
.stat-val {{
    flex: 0.7; font-weight: 700; font-size: 0.94rem;
    color: {TEXT}; text-align: right; padding-right: 14px;
    font-family: 'JetBrains Mono', monospace;
}}
.stat-bar-wrap {{ flex: 2; }}
.stat-pct {{ flex: 0.5; font-size: 0.76rem; text-align: right; }}

/* ---- Compare card ---- */
.compare-col {{
    background: linear-gradient(160deg, {CARD}, {CARD_2});
    border: 1px solid {BORDER}; border-radius: 18px;
    padding: 16px 18px;
}}

/* ---- Scrollable table wrapper ---- */
.scroll-wrap {{ overflow-x: auto; }}

/* ---- Badge ---- */
.badge {{
    display: inline-block; padding: 2px 8px; border-radius: 6px;
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.04em;
}}
.badge-green {{ background: rgba(52,211,153,0.18); color: {SUCCESS}; }}
.badge-red {{ background: rgba(251,113,133,0.18); color: {DANGER}; }}
.badge-blue {{ background: rgba(93,168,255,0.18); color: {ACCENT}; }}
.badge-orange {{ background: rgba(251,146,60,0.18); color: {ORANGE}; }}

/* ---- Footer ---- */
.footer-note {{
    color: {MUTED_2}; font-size: 0.82rem; margin-top: 0.6rem;
    padding-top: 12px; border-top: 1px solid {BORDER};
}}

/* ---- Progress ring placeholder ---- */
.ring-wrap {{ text-align: center; padding: 8px 0; }}

/* ---- Heatmap table cell ---- */
.heatcell {{
    text-align: center; padding: 6px 8px;
    font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;
    border-radius: 6px; margin: 2px;
    display: inline-block; min-width: 44px;
}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# =========================================================
# LOGIN
# =========================================================
def login_screen():
    st.markdown(
        '<div class="hero-wrap">'
        '<div class="hero-title">⚽ <span>Allsvenskan</span> Set Piece Studio Pro</div>'
        '<div class="hero-sub">Premium corner analytics — executive intelligence, pitch visualisations, taker profiling, trend analysis, and full export workflows.</div>'
        '<div><span class="pill">2025 Season</span><span class="pill pill-green">Live Filters</span><span class="pill pill-orange">Export Ready</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    _, col_m, _ = st.columns([1, 1.6, 1])
    with col_m:
        st.markdown('<div class="sub-card sub-card-accent" style="padding:28px 32px;margin-top:20px">', unsafe_allow_html=True)
        st.markdown("#### 🔐 Sign in to Studio")
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter username")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            sub = st.form_submit_button("Sign In →", use_container_width=True)
            if sub:
                if username == LOGIN_NAME and password == LOGIN_PASSWORD:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials.")
        st.markdown("</div>", unsafe_allow_html=True)


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if not st.session_state["authenticated"]:
    login_screen()
    st.stop()

# =========================================================
# CORE HELPERS
# =========================================================
def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def human_pct(v, decimals=1):
    return "—" if pd.isna(v) else f"{v*100:.{decimals}f}%"

def human_val(v, decimals=2):
    return "—" if pd.isna(v) else f"{v:.{decimals}f}"

def fmt_int(v):
    return "—" if pd.isna(v) else f"{int(v):,}"

def color_for_pct(p):
    if pd.isna(p): return MUTED
    if p >= 80:    return SUCCESS
    if p >= 60:    return ACCENT
    if p >= 40:    return WARNING
    if p >= 20:    return ORANGE
    return DANGER

def percentile_rank(series, value):
    s = series.dropna()
    if len(s) == 0 or pd.isna(value): return np.nan
    return float((s <= value).mean() * 100)

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
    if not isinstance(m, str): return None, None
    for sep in [" - ", " vs ", " v "]:
        if sep in m:
            l, r = m.split(sep, 1)
            return l.strip(), r.strip()
    return None, None

def classify_outcome(text):
    s = str(text).lower().strip()
    if "first contact - shot within 3 seconds" in s: return "Shot ≤3s"
    if "first contact" in s and "shot" in s:          return "First Contact Shot"
    if "shot" in s:                                    return "Shot"
    if "no first contact" in s:                        return "No First Contact"
    if s in ["", "nan"]:                               return "Unknown"
    return "Other"

def left_right_from_y(y):
    if pd.isna(y): return "Unknown"
    if y < 30:     return "Near Post Zone"
    if y <= 50:    return "Central Zone"
    return "Far Post Zone"

def corner_side_from_start_y(y):
    if pd.isna(y): return "Unknown"
    return "Right Corner" if y < 40 else "Left Corner"

def delivery_length(x0, y0, x1, y1):
    if any(pd.isna(v) for v in [x0, y0, x1, y1]): return np.nan
    return float(np.sqrt((x1-x0)**2 + (y1-y0)**2))

def zone_from_end_location(x, y):
    if pd.isna(x) or pd.isna(y): return "Unknown"
    if x >= 114 and 30 <= y <= 50: return "6-yard box"
    if x >= 108 and 18 <= y <= 62: return "Penalty area"
    if x >= 100 and 18 <= y <= 62: return "Deep box"
    return "Outside danger zone"

def infer_delivery_type(technique, height, body_part):
    t = str(technique).lower()
    h = str(height).lower()
    b = str(body_part).lower()
    if "inswing" in t: return "Inswinger"
    if "outswing" in t: return "Outswinger"
    if "short" in t: return "Short"
    if "straight" in t: return "Straight"
    if "high" in h: return "High Ball"
    if "ground" in h or "low" in h: return "Low Ball"
    return "Other"

def xg_category(xg):
    if pd.isna(xg): return "No shot"
    if xg >= 0.2:  return "Big Chance (xG≥0.20)"
    if xg >= 0.10: return "Good Chance (xG≥0.10)"
    if xg >= 0.05: return "Half Chance (xG≥0.05)"
    return "Low xG (<0.05)"

# =========================================================
# CHART LAYOUT HELPERS
# =========================================================
def figure_layout(fig, height=420, title=None, margin=None):
    m = margin or dict(l=8, r=8, t=50 if title else 12, b=8)
    fig.update_layout(
        height=height, title=title,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=m, legend_title_text="",
        font=dict(color=TEXT, family="DM Sans, sans-serif"),
        hoverlabel=dict(bgcolor="#0d1c31", font_color=TEXT, font_family="DM Sans, sans-serif", bordercolor=BORDER_2),
        title_font=dict(size=13, color=TEXT, family="DM Sans, sans-serif"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)", zeroline=False, tickfont=dict(size=11))
    return fig

def draw_pitch(fig, title=None, height=560, half=False):
    xmax = 120 if not half else 65
    shapes = [
        dict(type="rect",   x0=0,   y0=0,  x1=120, y1=80,  line=dict(color=PITCH_LINE, width=2)),
        dict(type="line",   x0=60,  y0=0,  x1=60,  y1=80,  line=dict(color=PITCH_LINE, width=1.5)),
        dict(type="circle", x0=50,  y0=30, x1=70,  y1=50,  line=dict(color=PITCH_LINE, width=1.5)),
        dict(type="rect",   x0=0,   y0=18, x1=18,  y1=62,  line=dict(color=PITCH_LINE, width=1.5)),
        dict(type="rect",   x0=0,   y0=30, x1=6,   y1=50,  line=dict(color=PITCH_LINE, width=1.5)),
        dict(type="rect",   x0=102, y0=18, x1=120, y1=62,  line=dict(color=PITCH_LINE, width=1.5)),
        dict(type="rect",   x0=114, y0=30, x1=120, y1=50,  line=dict(color=PITCH_LINE, width=1.5)),
        dict(type="circle", x0=10,  y0=38, x1=14,  y1=42,  line=dict(color=PITCH_LINE, width=1.5)),
        dict(type="circle", x0=106, y0=38, x1=110, y1=42,  line=dict(color=PITCH_LINE, width=1.5)),
        dict(type="path",   path="M 18 33 Q 28 40 18 47", line=dict(color=PITCH_LINE, width=1.5)),
        dict(type="path",   path="M 102 33 Q 92 40 102 47", line=dict(color=PITCH_LINE, width=1.5)),
    ]
    fig.update_xaxes(range=[55 if half else 0, 120], visible=False)
    fig.update_yaxes(range=[0, 80], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=title,
        paper_bgcolor=PITCH, plot_bgcolor=PITCH,
        margin=dict(l=8, r=8, t=42 if title else 8, b=8),
        height=height, shapes=shapes, legend_title_text="",
        font=dict(color="white", family="DM Sans, sans-serif"),
        hoverlabel=dict(bgcolor="#0d1c31", font_color="white", bordercolor="rgba(255,255,255,0.2)"),
    )
    return fig

# =========================================================
# PLOT FACTORIES
# =========================================================
def shotmap_figure(df_shots, color_col="corner_team", title="Shotmap", half=True):
    fig = draw_pitch(go.Figure(), title=title, height=580, half=half)
    if df_shots.empty: return fig
    df = df_shots.copy()
    df["_xg"] = df["shot_xg"].fillna(0)
    df["_sz"] = np.clip(df["_xg"] * 90 + 11, 10, 38)
    cats = df[color_col].fillna("Unknown").astype(str).unique().tolist()
    cmap = {c: QUAL_PALETTE[i % len(QUAL_PALETTE)] for i, c in enumerate(cats)}
    for cat in cats:
        sub = df[df[color_col].fillna("Unknown").astype(str) == cat]
        fig.add_trace(go.Scatter(
            x=sub["shot_location_x"], y=sub["shot_location_y"],
            mode="markers", name=str(cat),
            marker=dict(size=sub["_sz"], color=cmap[cat], opacity=0.88,
                        line=dict(color="white", width=1.2)),
            text=[
                f"<b>{r.get('Match','')}</b><br>"
                f"Team: {r.get('corner_team','')}<br>"
                f"Taker: {r.get('Taker','')}<br>"
                f"Shooter: {r.get('Shooter','')}<br>"
                f"Body: {r.get('shot_body_part','')}<br>"
                f"xG: {0 if pd.isna(r.get('shot_xg')) else r.get('shot_xg',0):.3f}<br>"
                f"Result: {r.get('shot_outcome','')}<br>"
                f"Min: {int(r['Minute']) if pd.notna(r.get('Minute')) else ''}"
                for _, r in sub.iterrows()],
            hovertemplate="%{text}<extra></extra>",
        ))
    return fig

def delivery_map_figure(df_events, color_col="delivery_zone", title="Delivery Map"):
    fig = draw_pitch(go.Figure(), title=title, height=620)
    if df_events.empty: return fig
    plot_df = df_events.dropna(subset=["pass_location_x","pass_location_y","pass_end_location_x","pass_end_location_y"]).copy()
    if plot_df.empty: return fig
    cats = plot_df[color_col].fillna("Unknown").astype(str).unique().tolist()
    cmap = {c: QUAL_PALETTE[i % len(QUAL_PALETTE)] for i, c in enumerate(cats)}
    legend_added = set()
    for _, row in plot_df.iterrows():
        cat = str(row[color_col]) if pd.notna(row[color_col]) else "Unknown"
        show = cat not in legend_added
        if show: legend_added.add(cat)
        fig.add_trace(go.Scatter(
            x=[row["pass_location_x"], row["pass_end_location_x"]],
            y=[row["pass_location_y"], row["pass_end_location_y"]],
            mode="lines+markers",
            line=dict(color=cmap[cat], width=2.0),
            marker=dict(size=[5, 9], color=[cmap[cat], "white"], opacity=[0.7, 1.0]),
            name=str(cat), legendgroup=str(cat), showlegend=show,
            text=(
                f"<b>{row.get('Match','')}</b><br>"
                f"Team: {row.get('corner_team','')}<br>"
                f"Taker: {row.get('Taker','')}<br>"
                f"Tech: {row.get('pass_technique','')}<br>"
                f"Height: {row.get('pass_height','')}<br>"
                f"Zone: {row.get('delivery_zone','')}<br>"
                f"End: {row.get('end_zone','')}<br>"
                f"Side: {row.get('corner_side','')}<br>"
                f"Outcome: {row.get('SP_outcome','')}<br>"
                f"Min: {int(row['Minute']) if pd.notna(row.get('Minute')) else ''}"
            ),
            hovertemplate="%{text}<extra></extra>",
        ))
    return fig

def heatmap_density(df, x_col, y_col, title, nbinsx=20, nbinsy=14, height=520):
    plot = df.dropna(subset=[x_col, y_col]).copy()
    if plot.empty:
        fig = go.Figure()
        return figure_layout(fig, height, title)
    fig = draw_pitch(go.Figure(), title=title, height=height)
    fig.add_trace(go.Histogram2dContour(
        x=plot[x_col], y=plot[y_col],
        colorscale=[[0,"rgba(0,0,0,0)"],[0.2,"rgba(93,168,255,0.3)"],
                    [0.5,"rgba(52,211,153,0.55)"],[0.8,"rgba(251,191,36,0.75)"],
                    [1.0,"rgba(251,113,133,0.95)"]],
        ncontours=14, showscale=False,
        hoverinfo="skip", opacity=0.88,
    ))
    return fig

def outcome_pie(df, title="Outcome Split"):
    if df.empty: return go.Figure()
    s = df.groupby("outcome_bucket", dropna=False).size().reset_index(name="n")
    fig = px.pie(s, names="outcome_bucket", values="n", title=title, hole=0.56,
                 color_discrete_sequence=QUAL_PALETTE)
    fig.update_traces(textposition="outside", textinfo="percent+label", pull=[0.03]*len(s))
    return figure_layout(fig, 380, title)

def technique_pie(df, title="Delivery Technique"):
    if df.empty: return go.Figure()
    s = df.groupby("pass_technique", dropna=False).size().reset_index(name="n")
    fig = px.pie(s, names="pass_technique", values="n", title=title, hole=0.56,
                 color_discrete_sequence=QUAL_PALETTE[3:])
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return figure_layout(fig, 380, title)

def bar_team(df, y_col, title, color=ACCENT, label_y=None, h=380):
    if df.empty: return go.Figure()
    fig = px.bar(df.sort_values(y_col, ascending=False), x="team", y=y_col,
                 title=title, color_discrete_sequence=[color],
                 labels={"team":"", y_col: label_y or y_col})
    fig.update_traces(marker_line_width=0)
    return figure_layout(fig, h, title)

def team_scatter(df, x, y, sz, title, label_x=None, label_y=None):
    if df.empty: return go.Figure()
    hover = [c for c in ["corners_taken","matches","total_xg","fast_shot_rate","six_yard_delivery_rate"] if c in df.columns]
    fig = px.scatter(df, x=x, y=y, size=sz, hover_name="team", hover_data=hover,
                     title=title, text="team",
                     color_discrete_sequence=[ACCENT],
                     labels={x: label_x or x, y: label_y or y})
    fig.update_traces(textposition="top center", textfont=dict(size=9, color=TEXT))
    if not df[x].dropna().empty:
        fig.add_vline(x=df[x].median(), line_dash="dot", line_color="rgba(255,255,255,0.15)")
    if not df[y].dropna().empty:
        fig.add_hline(y=df[y].median(), line_dash="dot", line_color="rgba(255,255,255,0.15)")
    return figure_layout(fig, 420, title)

def phase_heatmap(df, title="Phase Heatmap", h=400):
    if df.empty: return go.Figure()
    tmp = df.groupby(["corner_team","phase"], dropna=False).size().reset_index(name="n")
    pivot = tmp.pivot(index="corner_team", columns="phase", values="n").fillna(0)
    order = [p for p in ["0-15","16-30","31-45","46-60","61-75","76+"] if p in pivot.columns]
    pivot = pivot.reindex(columns=order)
    fig = px.imshow(pivot, aspect="auto", title=title, color_continuous_scale="Blues",
                    labels=dict(x="Phase",y="Team",color="Corners"), text_auto=True)
    return figure_layout(fig, max(h, 40*max(6,len(pivot))), title)

def cumulative_line(df, color_col="corner_team", title="Cumulative Corners", h=400):
    if df.empty: return go.Figure()
    base = (df.groupby(["Minute",color_col], dropna=False).size().reset_index(name="n")
            .sort_values([color_col,"Minute"]))
    base["cum"] = base.groupby(color_col)["n"].cumsum()
    fig = px.line(base, x="Minute", y="cum", color=color_col, markers=True, title=title,
                  labels={"cum":"Cumulative Corners","Minute":"Match Minute"})
    return figure_layout(fig, h, title)

def minute_histogram(df, color_col=None, title="Minute Distribution", h=360):
    if df.empty: return go.Figure()
    if color_col:
        fig = px.histogram(df, x="Minute", color=color_col, nbins=24, title=title, barmode="stack",
                           color_discrete_sequence=QUAL_PALETTE)
    else:
        fig = px.histogram(df, x="Minute", nbins=24, title=title, color_discrete_sequence=[ACCENT])
    fig.update_traces(opacity=0.85)
    return figure_layout(fig, h, title)

def end_zone_bar(df, group_col="corner_team", title="End Zone Volume", h=420):
    if df.empty: return go.Figure()
    zone_order = ["6-yard box","Penalty area","Deep box","Outside danger zone","Unknown"]
    s = df.groupby([group_col,"end_zone"], dropna=False).size().reset_index(name="n")
    fig = px.bar(s, x=group_col, y="n", color="end_zone", title=title,
                 category_orders={"end_zone": zone_order},
                 color_discrete_sequence=QUAL_PALETTE,
                 labels={group_col:"","n":"Corners"})
    return figure_layout(fig, h, title)

def xg_accumulation(df, group_col=None, title="xG Accumulation", h=360):
    if df.empty: return go.Figure()
    df2 = df.dropna(subset=["shot_xg","Minute"]).sort_values("Minute")
    if df2.empty: return go.Figure()
    if group_col:
        fig = go.Figure()
        for grp, sub in df2.groupby(group_col):
            sub2 = sub.sort_values("Minute")
            sub2["cum_xg"] = sub2["shot_xg"].cumsum()
            fig.add_trace(go.Scatter(x=sub2["Minute"], y=sub2["cum_xg"],
                name=str(grp), mode="lines", line=dict(width=2.5)))
    else:
        df2["cum_xg"] = df2["shot_xg"].cumsum()
        fig = go.Figure(go.Scatter(x=df2["Minute"], y=df2["cum_xg"], mode="lines",
            fill="tozeroy", fillcolor="rgba(52,211,153,0.10)",
            line=dict(color=SUCCESS, width=2.5), name="xG"))
    fig.update_layout(
        title=title, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=h, legend_title_text="",
        font=dict(color=TEXT, family="DM Sans, sans-serif"),
        margin=dict(l=8,r=8,t=48,b=8),
        hoverlabel=dict(bgcolor="#0d1c31",font_color=TEXT),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.05)")
    return fig

def shot_xg_distribution(df, group_col=None, title="xG Distribution", h=360):
    """Violin / box plot of shot xG values."""
    plot = df.dropna(subset=["shot_xg"]).copy()
    if plot.empty: return go.Figure()
    if group_col and group_col in plot.columns:
        fig = px.violin(plot, y="shot_xg", x=group_col, box=True, points="outliers",
                        color=group_col, title=title,
                        color_discrete_sequence=QUAL_PALETTE,
                        labels={"shot_xg":"xG", group_col:""})
    else:
        fig = px.violin(plot, y="shot_xg", box=True, points="outliers",
                        title=title, color_discrete_sequence=[ACCENT],
                        labels={"shot_xg":"xG"})
    return figure_layout(fig, h, title)

def rolling_shot_rate(df, window=5, title="Rolling Shot Rate (5-match)"):
    """Rolling average shot rate per team across matches."""
    if df.empty: return go.Figure()
    match_team = (
        df.groupby(["Match","corner_team"], dropna=False)
        .agg(corners=("match_id","size"), shots=("led_to_shot","sum"))
        .reset_index()
    )
    match_team["shot_rate"] = match_team["shots"] / match_team["corners"].replace(0, np.nan)
    match_team = match_team.sort_values(["corner_team","Match"])
    fig = go.Figure()
    for team, grp in match_team.groupby("corner_team"):
        grp = grp.copy().reset_index(drop=True)
        grp["rolling"] = grp["shot_rate"].rolling(min(window, len(grp)), min_periods=1).mean()
        fig.add_trace(go.Scatter(x=grp["Match"], y=grp["rolling"],
            mode="lines+markers", name=str(team), line=dict(width=2.5),
            marker=dict(size=6), hovertemplate=f"<b>{team}</b><br>Match: %{{x}}<br>Rolling SR: %{{y:.1%}}<extra></extra>"))
    fig.update_layout(
        title=title, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=400, legend_title_text="",
        font=dict(color=TEXT, family="DM Sans, sans-serif"),
        margin=dict(l=8,r=8,t=48,b=8),
        hoverlabel=dict(bgcolor="#0d1c31",font_color=TEXT),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)", tickformat=".0%"),
    )
    return fig

def correlation_heatmap_teams(team_df, title="Team Metric Correlations", h=480):
    cols = [c for c in ["corners_per_match","shot_rate","xg_per_match","fast_shot_rate",
                         "six_yard_delivery_rate","short_corner_rate","inswinger_rate",
                         "penalty_area_delivery_rate","taker_variety"] if c in team_df.columns]
    if len(cols) < 3 or team_df.empty: return go.Figure()
    corr = team_df[cols].corr()
    fig = px.imshow(corr, aspect="auto", title=title, color_continuous_scale="RdBu_r",
                    range_color=[-1,1], text_auto=".2f")
    return figure_layout(fig, h, title)

def taker_bar_chart(taker_df, metric, title, h=380, top_n=15, min_corners=3):
    if taker_df.empty or metric not in taker_df.columns: return go.Figure()
    plot = taker_df[taker_df["corners"] >= min_corners].copy()
    plot["label"] = plot["Taker"].astype(str) + " (" + plot["corner_team"].astype(str) + ")"
    plot = plot.sort_values(metric, ascending=False).head(top_n)
    fig = px.bar(plot, x="label", y=metric, title=title,
                 hover_data=["corner_team","corners","shots","total_xg"],
                 color=metric, color_continuous_scale="Blues",
                 labels={"label":""})
    fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-35)
    return figure_layout(fig, h, title)

def taker_radar(taker_df_all, taker_name, h=380):
    if taker_df_all.empty: return go.Figure()
    row = taker_df_all[taker_df_all["Taker"].astype(str) == taker_name]
    if row.empty: return go.Figure()
    row = row.iloc[0]
    def _norm(val, col):
        s = taker_df_all[col].dropna()
        if s.max() == s.min() or pd.isna(val): return 0.0
        return float((val - s.min()) / (s.max() - s.min()))
    fc = lambda c: row.get(c, 0) / max(row.get("corners", 1), 1)
    dims = [
        ("Shot Rate",        _norm(row.get("shot_rate",0),       "shot_rate")),
        ("xG / Corner",      _norm(row.get("xg_per_corner",0),   "xg_per_corner")),
        ("Fast Shot Rate",   _norm(fc("fast_shots"),              "shot_rate")),
        ("6Y Delivery",      _norm(fc("six_yard_deliveries"),     "shot_rate")),
        ("Inswinger Rate",   _norm(fc("inswingers"),              "shot_rate")),
        ("Short Corner",     _norm(fc("short_corners"),           "shot_rate")),
        ("Goal Involvement", _norm(row.get("goal_rate",0),        "goal_rate" if "goal_rate" in taker_df_all.columns else "shot_rate")),
    ]
    labels = [d[0] for d in dims] + [dims[0][0]]
    vals   = [d[1] for d in dims] + [dims[0][1]]
    fig = go.Figure(go.Scatterpolar(
        r=vals, theta=labels, fill="toself",
        fillcolor="rgba(93,168,255,0.22)",
        line=dict(color=ACCENT, width=2.2), name=taker_name,
        hovertemplate="%{theta}: %{r:.2f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0,1], showticklabels=False,
                            gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                             linecolor="rgba(255,255,255,0.15)",
                             tickfont=dict(size=10, color=MUTED)),
        ),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=h, margin=dict(l=18,r=18,t=48,b=18),
        title=f"Profile — {taker_name}",
        font=dict(color=TEXT, family="DM Sans, sans-serif"), showlegend=False,
    )
    return fig

def xg_shot_scatter(df_shots, title="xG per Shot vs Distance from Goal"):
    if df_shots.empty: return go.Figure()
    df2 = df_shots.dropna(subset=["shot_xg","shot_location_x","shot_location_y"]).copy()
    if df2.empty: return go.Figure()
    df2["dist_to_goal"] = np.sqrt((df2["shot_location_x"] - 120)**2 + (df2["shot_location_y"] - 40)**2)
    fig = px.scatter(df2, x="dist_to_goal", y="shot_xg",
                     color="corner_team", size="shot_xg",
                     hover_name="Shooter",
                     hover_data=["Match","Taker","shot_outcome","Minute"],
                     title=title,
                     color_discrete_sequence=QUAL_PALETTE,
                     labels={"dist_to_goal":"Distance to Goal (m)", "shot_xg":"xG"})
    return figure_layout(fig, 400, title)

def delivery_length_hist(df, title="Delivery Length Distribution", h=360):
    if df.empty: return go.Figure()
    df2 = df.dropna(subset=["delivery_length"]).copy()
    fig = px.histogram(df2, x="delivery_length", color="corner_team", nbins=28,
                       title=title, barmode="overlay", opacity=0.7,
                       color_discrete_sequence=QUAL_PALETTE,
                       labels={"delivery_length":"Delivery Length (m)"})
    return figure_layout(fig, h, title)

def home_away_comparison(df_team, title="Home vs Away — Shot Rate & xG", h=380):
    if df_team.empty: return go.Figure()
    hvsa = (
        df_team.groupby(["corner_team","venue_split"], dropna=False)
        .agg(corners=("match_id","size"), shots=("led_to_shot","sum"), total_xg=("shot_xg","sum"))
        .reset_index()
    )
    hvsa["shot_rate"] = hvsa["shots"] / hvsa["corners"].replace(0, np.nan)
    hvsa["xg_per_corner"] = hvsa["total_xg"] / hvsa["corners"].replace(0, np.nan)
    fig = px.bar(hvsa[hvsa["venue_split"].isin(["Home","Away"])],
                 x="corner_team", y="shot_rate", color="venue_split", barmode="group",
                 title=title, color_discrete_sequence=[ACCENT, SUCCESS],
                 labels={"corner_team":"","shot_rate":"Shot Rate","venue_split":""})
    return figure_layout(fig, h, title)

def defensive_setup_scatter(def_df, title="Defensive Setup: Volume vs Shot Rate Allowed"):
    if def_df.empty: return go.Figure()
    fig = px.scatter(def_df, x="corners", y="shot_rate", size="total_xg",
                     hover_name="Defensive_setup",
                     color="xg_per_corner", color_continuous_scale="Reds",
                     title=title,
                     labels={"corners":"Sample Size","shot_rate":"Shot Rate Allowed"})
    return figure_layout(fig, 400, title)

def sankey_outcome_flow(df, title="Corner → Outcome Flow"):
    """Simplified sankey: team → delivery zone → outcome."""
    if df.empty: return go.Figure()
    # Build source→target→value lists
    teams = df["corner_team"].fillna("Unknown").astype(str).unique().tolist()
    zones = df["end_zone"].fillna("Unknown").astype(str).unique().tolist()
    outcomes = df["outcome_bucket"].fillna("Unknown").astype(str).unique().tolist()
    nodes = teams + zones + outcomes
    node_idx = {n: i for i, n in enumerate(nodes)}
    src, tgt, val = [], [], []
    for (team, zone), cnt in df.groupby(["corner_team","end_zone"], dropna=False).size().items():
        src.append(node_idx[str(team)]); tgt.append(node_idx[str(zone)]); val.append(int(cnt))
    for (zone, oc), cnt in df.groupby(["end_zone","outcome_bucket"], dropna=False).size().items():
        src.append(node_idx[str(zone)]); tgt.append(node_idx[str(oc)]); val.append(int(cnt))
    colors = QUAL_PALETTE * 10
    fig = go.Figure(go.Sankey(
        node=dict(label=nodes, pad=16, thickness=18,
                  color=[colors[i % len(colors)] for i in range(len(nodes))],
                  line=dict(color="rgba(0,0,0,0)", width=0)),
        link=dict(source=src, target=tgt, value=val,
                  color="rgba(255,255,255,0.06)"),
    ))
    fig.update_layout(
        title=title, height=520,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="DM Sans, sans-serif", size=11),
        margin=dict(l=8,r=8,t=48,b=8),
    )
    return fig

def shot_zone_grid(df_shots, title="Shot Zone Grid (Goal-Face Quadrants)"):
    """Plot shot_location_y vs shot_location_z to show goal-face quadrant."""
    if df_shots.empty: return go.Figure()
    df2 = df_shots.dropna(subset=["shot_location_y","shot_location_z"]).copy()
    if df2.empty:
        fig = go.Figure()
        return figure_layout(fig, 420, title + " (no z-data)")
    fig = px.scatter(df2, x="shot_location_y", y="shot_location_z",
                     color="shot_outcome", size="shot_xg",
                     hover_name="Shooter",
                     hover_data=["Match","Taker","corner_team","Minute"],
                     title=title,
                     color_discrete_sequence=QUAL_PALETTE,
                     labels={"shot_location_y":"Y (across goal)","shot_location_z":"Z (height)"})
    # Add goal frame
    fig.add_shape(type="rect", x0=30, y0=0, x1=50, y1=2.67,
                  line=dict(color="white", width=2, dash="dot"))
    fig.add_shape(type="line", x0=40, y0=0, x1=40, y1=2.67,
                  line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"))
    fig.add_shape(type="line", x0=30, y0=1.33, x1=50, y1=1.33,
                  line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash"))
    return figure_layout(fig, 420, title)

def xg_heatmap_pitch(df_shots, title="xG Heatmap on Pitch"):
    """Density of shot xG weighted by xG value."""
    if df_shots.empty: return go.Figure()
    df2 = df_shots.dropna(subset=["shot_location_x","shot_location_y","shot_xg"]).copy()
    if df2.empty:
        return draw_pitch(go.Figure(), title=title)
    fig = draw_pitch(go.Figure(), title=title, height=520)
    fig.add_trace(go.Histogram2dContour(
        x=df2["shot_location_x"], y=df2["shot_location_y"],
        z=df2["shot_xg"],
        histfunc="sum",
        colorscale=[[0,"rgba(0,0,0,0)"],[0.3,"rgba(93,168,255,0.35)"],
                    [0.65,"rgba(52,211,153,0.65)"],[0.85,"rgba(251,191,36,0.80)"],
                    [1.0,"rgba(251,113,133,0.95)"]],
        ncontours=12, showscale=False, opacity=0.85,
    ))
    return fig

# =========================================================
# DATA AGGREGATION
# =========================================================
def build_team_summary(df):
    if df.empty:
        return pd.DataFrame(columns=["team","corners_taken","matches","shots_from_corners",
            "first_contact_shots","fast_shots","total_xg","avg_xg_per_corner","taker_variety",
            "inswingers","outswingers","short_corners","target_box_deliveries","six_yard_deliveries",
            "penalty_area_deliveries","corners_per_match","shot_rate","first_contact_rate",
            "fast_shot_rate","xg_per_match","box_delivery_rate","six_yard_delivery_rate",
            "penalty_area_delivery_rate","short_corner_rate","inswinger_rate","outswinger_rate"])
    ts = (df.groupby("corner_team", dropna=False).agg(
        corners_taken=("match_id","size"),
        matches=("match_id", pd.Series.nunique),
        shots_from_corners=("led_to_shot","sum"),
        first_contact_shots=("is_first_contact_shot","sum"),
        fast_shots=("is_fast_shot","sum"),
        total_xg=("shot_xg","sum"),
        avg_xg_per_corner=("shot_xg","mean"),
        taker_variety=("Taker", pd.Series.nunique),
        inswingers=("is_inswinger","sum"),
        outswingers=("is_outswinger","sum"),
        short_corners=("is_short_corner","sum"),
        target_box_deliveries=("is_goal_kick_zone_delivery","sum"),
        six_yard_deliveries=("is_six_yard_delivery","sum"),
        penalty_area_deliveries=("is_penalty_area_delivery","sum"),
    ).reset_index().rename(columns={"corner_team":"team"}))
    denom = ts["corners_taken"].replace(0, np.nan)
    ts["corners_per_match"] = ts["corners_taken"] / ts["matches"].replace(0, np.nan)
    ts["shot_rate"]                = ts["shots_from_corners"] / denom
    ts["first_contact_rate"]       = ts["first_contact_shots"] / denom
    ts["fast_shot_rate"]           = ts["fast_shots"] / denom
    ts["xg_per_match"]             = ts["total_xg"] / ts["matches"].replace(0, np.nan)
    ts["box_delivery_rate"]        = ts["target_box_deliveries"] / denom
    ts["six_yard_delivery_rate"]   = ts["six_yard_deliveries"] / denom
    ts["penalty_area_delivery_rate"] = ts["penalty_area_deliveries"] / denom
    ts["short_corner_rate"]        = ts["short_corners"] / denom
    ts["inswinger_rate"]           = ts["inswingers"] / denom
    ts["outswinger_rate"]          = ts["outswingers"] / denom
    return ts

def add_advanced_features(df):
    d = df.copy()
    if d.empty:
        for c in ["venue_split","delivery_length_band","xg_created","goal_from_corner",
                   "delivery_success_proxy","delivery_type","xg_category"]:
            if c not in d.columns: d[c] = np.nan
        return d
    d["venue_split"] = np.where(d["is_home_corner"],"Home", np.where(d["is_away_corner"],"Away","Unknown"))
    d["delivery_length_band"] = pd.cut(d["delivery_length"], bins=[-0.1,8,16,28,200],
        labels=["Short","Medium","Long","Very Long"], right=True).astype(str)
    d["xg_created"]  = d["shot_xg"].fillna(0)
    d["goal_from_corner"] = d["shot_outcome"].astype(str).str.contains("goal", case=False, na=False)
    d["delivery_success_proxy"] = (d["led_to_shot"].fillna(False)
                                   | d["is_first_contact_shot"].fillna(False)
                                   | d["is_goal_kick_zone_delivery"].fillna(False))
    d["delivery_type"] = d.apply(lambda r: infer_delivery_type(
        r.get("pass_technique",""), r.get("pass_height",""), r.get("pass_body_part","")), axis=1)
    d["xg_category"] = d["shot_xg"].apply(xg_category)
    return d

def taker_summary_table(df):
    if df.empty: return pd.DataFrame()
    out = (df.groupby(["corner_team","Taker"], dropna=False).agg(
        corners=("match_id","size"),
        shots=("led_to_shot","sum"),
        fast_shots=("is_fast_shot","sum"),
        first_contact_shots=("is_first_contact_shot","sum"),
        goals=("goal_from_corner","sum"),
        total_xg=("xg_created","sum"),
        inswingers=("is_inswinger","sum"),
        outswingers=("is_outswinger","sum"),
        short_corners=("is_short_corner","sum"),
        six_yard_deliveries=("is_six_yard_delivery","sum"),
        penalty_area_deliveries=("is_penalty_area_delivery","sum"),
        matches=("match_id", pd.Series.nunique),
    ).reset_index())
    d = out["corners"].replace(0, np.nan)
    out["shot_rate"]         = out["shots"] / d
    out["xg_per_corner"]     = out["total_xg"] / d
    out["goal_rate"]         = out["goals"] / d
    out["fast_shot_rate"]    = out["fast_shots"] / d
    out["inswinger_rate"]    = out["inswingers"] / d
    out["six_yard_rate"]     = out["six_yard_deliveries"] / d
    return out.sort_values(["corners","total_xg"], ascending=False)

def team_insight_table(df):
    if df.empty: return pd.DataFrame()
    out = (df.groupby(["corner_team","delivery_zone","end_zone"], dropna=False).agg(
        corners=("match_id","size"),
        shots=("led_to_shot","sum"),
        total_xg=("shot_xg","sum"),
        fast_shots=("is_fast_shot","sum"),
    ).reset_index())
    out["shot_rate"]     = out["shots"] / out["corners"].replace(0, np.nan)
    out["xg_per_corner"] = out["total_xg"] / out["corners"].replace(0, np.nan)
    return out.sort_values(["corner_team","corners"], ascending=[True,False])

def match_pattern_table(df):
    if df.empty: return pd.DataFrame()
    out = (df.groupby(["Match","corner_team","venue_split"], dropna=False).agg(
        corners=("match_id","size"),
        shots=("led_to_shot","sum"),
        fast_shots=("is_fast_shot","sum"),
        goals=("goal_from_corner","sum"),
        total_xg=("xg_created","sum"),
        six_yard_deliveries=("is_six_yard_delivery","sum"),
        short_corners=("is_short_corner","sum"),
        inswingers=("is_inswinger","sum"),
    ).reset_index())
    out["shot_rate"]     = out["shots"] / out["corners"].replace(0, np.nan)
    out["xg_per_corner"] = out["total_xg"] / out["corners"].replace(0, np.nan)
    return out.sort_values(["total_xg","corners"], ascending=False)

def defensive_setup_table(df):
    if df.empty: return pd.DataFrame()
    out = (df.groupby("Defensive_setup", dropna=False).agg(
        corners=("match_id","size"),
        shots=("led_to_shot","sum"),
        total_xg=("shot_xg","sum"),
        fast_shots=("is_fast_shot","sum"),
        six_yard_deliveries=("is_six_yard_delivery","sum"),
    ).reset_index())
    out["shot_rate"]      = out["shots"] / out["corners"].replace(0, np.nan)
    out["xg_per_corner"]  = out["total_xg"] / out["corners"].replace(0, np.nan)
    out["fast_shot_rate"] = out["fast_shots"] / out["corners"].replace(0, np.nan)
    return out[out["Defensive_setup"].astype(str) != "nan"].sort_values("corners", ascending=False)

# =========================================================
# UI HELPERS
# =========================================================
def section_header(title, sub=""):
    st.markdown(
        f'<div class="section-title">{title}</div>'
        + (f'<div class="section-sub">{sub}</div>' if sub else ""),
        unsafe_allow_html=True)

def metric_card(label, value, suffix="", foot="", delta=None, accent_color=ACCENT):
    delta_html = ""
    if delta is not None:
        cls  = "kpi-delta-pos" if delta >= 0 else "kpi-delta-neg"
        sign = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="{cls}">{sign} {abs(delta):.2f} vs avg</div>'
    bar_color = accent_color
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-accent-bar" style="background:{bar_color};opacity:0.6;"></div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}<span class="kpi-suffix">{suffix}</span></div>
        <div class="kpi-foot">{foot}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

def insight_box(icon, title, body, variant="blue"):
    cls_map = {"blue":"insight-box","green":"insight-box insight-box-green",
               "orange":"insight-box insight-box-orange","red":"insight-box insight-box-red",
               "purple":"insight-box insight-box-purple"}
    cls = cls_map.get(variant, "insight-box")
    st.markdown(f"""
    <div class="{cls}">
        <span class="insight-icon">{icon}</span>
        <span class="insight-title">{title}</span>
        <div class="insight-body">{body}</div>
    </div>""", unsafe_allow_html=True)

def render_kpis(events, matches, league_avg=None):
    c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
    total_xg  = events["shot_xg"].fillna(0).sum()   if not events.empty else 0
    shot_rate = events["led_to_shot"].mean()*100      if len(events)>0 else 0
    fast_rate = events["is_fast_shot"].mean()*100     if len(events)>0 else 0
    goals     = int(events["goal_from_corner"].sum()) if not events.empty else 0
    with c1: metric_card("Events",        f"{len(events):,}",            foot="Corner actions",    accent_color=ACCENT)
    with c2: metric_card("Matches",       f"{events['match_id'].nunique() if not events.empty else 0:,}", foot="Unique matches", accent_color=ACCENT)
    with c3: metric_card("Avg C/Match",   f"{matches['total_corners'].mean():.2f}" if not matches.empty else "—", foot="Volume benchmark", accent_color=ACCENT_2)
    with c4: metric_card("Shots",         f"{int(events['led_to_shot'].sum()) if not events.empty else 0:,}", foot="From corners",   accent_color=SUCCESS)
    with c5: metric_card("Total xG",      f"{total_xg:.2f}",             foot="xG generated",      accent_color=SUCCESS)
    with c6: metric_card("Shot Rate",     f"{shot_rate:.1f}",    "%",    foot="Shots/corner",      accent_color=WARNING)
    with c7: metric_card("Fast Shot %",   f"{fast_rate:.1f}",    "%",    foot="Shot within 3s",    accent_color=ORANGE)
    with c8: metric_card("Goals",         f"{goals}",                    foot="From corners",      accent_color=DANGER)

def empty_state(msg="No data for current filters.", icon="🔍"):
    st.markdown(f'<div class="empty-state"><div class="empty-state-icon">{icon}</div>{msg}</div>', unsafe_allow_html=True)

def percentile_bars_html(row, league_df, metrics):
    """Render a styled percentile report card in HTML."""
    rows_html = ""
    for label, col, fmt, tip in metrics:
        val = row.get(col, np.nan)
        p   = percentile_rank(league_df[col] if col in league_df.columns else pd.Series(dtype=float), val)
        if pd.isna(val): display = "—"
        elif fmt == "pct": display = f"{val*100:.1f}%"
        elif fmt == "xg":  display = f"{val:.3f}"
        else:              display = f"{val:.2f}"
        bar_w = int(p) if not pd.isna(p) else 0
        pct_label = f"{p:.0f}th" if not pd.isna(p) else "—"
        bar_color = color_for_pct(p)
        rows_html += f"""
        <div class="stat-row">
            <div class="stat-label">{label}<br>
                <span style="font-size:0.73rem;color:rgba(153,173,199,0.55)">{tip}</span>
            </div>
            <div class="stat-val">{display}</div>
            <div class="stat-bar-wrap">
                <div style="height:7px;background:rgba(255,255,255,0.06);border-radius:99px;overflow:hidden">
                    <div style="height:100%;width:{bar_w}%;background:{bar_color};border-radius:99px"></div>
                </div>
            </div>
            <div class="stat-pct" style="color:{bar_color}">{pct_label}</div>
        </div>"""
    return rows_html

def add_download_buttons(events_df, team_df, match_df, prefix=""):
    c1,c2,c3 = st.columns(3)
    with c1:
        st.download_button("⬇ Events CSV", data=events_df.to_csv(index=False).encode(),
            file_name=f"{prefix}events.csv", mime="text/csv", use_container_width=True)
    with c2:
        st.download_button("⬇ Team Summary CSV", data=team_df.to_csv(index=False).encode(),
            file_name=f"{prefix}team_summary.csv", mime="text/csv", use_container_width=True)
    with c3:
        st.download_button("⬇ Match Summary CSV", data=match_df.to_csv(index=False).encode(),
            file_name=f"{prefix}match_summary.csv", mime="text/csv", use_container_width=True)

def download_excel_workbook(events_df, team_df, match_df, taker_df, zone_df, match_name="export"):
    buf = BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            events_df.to_excel(w, "Events",        index=False)
            team_df.to_excel(w,   "Team Summary",  index=False)
            match_df.to_excel(w,  "Match Summary", index=False)
            taker_df.to_excel(w,  "Taker Summary", index=False)
            zone_df.to_excel(w,   "Zone Table",    index=False)
        return buf.getvalue()
    except Exception:
        return None

def top_insights(events, teams_df):
    out = []
    if events.empty or teams_df.empty:
        return [("🔍","No Data","Apply different filters.",  "blue")]
    def _best(col, asc=False):
        return teams_df.sort_values(col, ascending=asc).iloc[0] if not teams_df.empty else None
    r = _best("shot_rate")
    if r is not None:
        out.append(("🎯","Best Shot Rate", f"<b>{r['team']}</b> — {human_pct(r['shot_rate'])} of corners generate shots.", "green"))
    r = _best("xg_per_match")
    if r is not None:
        out.append(("⚡","Highest xG/Match", f"<b>{r['team']}</b> — {r['xg_per_match']:.3f} xG per match from corners.", "blue"))
    r = _best("six_yard_delivery_rate")
    if r is not None:
        out.append(("📍","6-Yard Targeting", f"<b>{r['team']}</b> — {human_pct(r['six_yard_delivery_rate'])} of deliveries reach the 6-yard box.", "orange"))
    r = _best("fast_shot_rate")
    if r is not None:
        out.append(("⚡","Fast Transitions", f"<b>{r['team']}</b> — {human_pct(r['fast_shot_rate'])} fast shot rate (shot within 3s).", "green"))
    r = _best("short_corner_rate")
    if r is not None:
        out.append(("↗️","Short Corners", f"<b>{r['team']}</b> uses short routines {human_pct(r['short_corner_rate'])} of the time.", "purple"))
    return out[:4]

# =========================================================
# DATA LOAD / PREP
# =========================================================
@st.cache_data
def load_data():
    if not os.path.exists(FILE_NAME):
        raise FileNotFoundError(f"{FILE_NAME} not found.")
    return pd.read_excel(FILE_NAME)

@st.cache_data
def prepare_data(raw_df):
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # Column detection
    _f = lambda *c: find_col(df, list(c))
    match_id_col   = _f("match_id","match id")
    match_col      = _f("match")
    team_col       = _f("pass_team_name","team","team_name")
    minute_col     = _f("minute")
    second_col     = _f("second")
    outcome_col    = _f("sp_outcome","outcome")
    xg_col         = _f("shot.statsbomb_xg","shot_xg","xg")
    taker_col      = _f("taker")
    shooter_col    = _f("shooter")
    def_setup_col  = _f("defensive_setup")
    shot_team_col  = _f("shot_team_name")
    pass_x_col     = _f("pass_location_x")
    pass_y_col     = _f("pass_location_y")
    pass_ex_col    = _f("pass_end_location_x")
    pass_ey_col    = _f("pass_end_location_y")
    shot_x_col     = _f("shot_location_x")
    shot_y_col     = _f("shot_location_y")
    shot_z_col     = _f("shot_location_z")
    pt_col         = _f("pass.technique.name","pass_technique")
    ph_col         = _f("pass.height.name","pass_height")
    pb_col         = _f("pass.body_part.name","pass_body_part")
    sb_col         = _f("shot.body_part.name","shot_body_part")
    so_col         = _f("shot.outcome.name","shot_outcome")
    po_col         = _f("pass.outcome.name","pass_outcome")
    pp_col         = _f("pass_position")
    sp_col         = _f("shot_position")
    gk_col         = _f("goalkeeper","gk_name","goalkeeper_name")
    matchday_col   = _f("matchday","gameweek","round","week")
    stadium_col    = _f("stadium","venue","ground")

    required = {"match_id":match_id_col,"match":match_col,"pass_team_name":team_col,
                "minute":minute_col,"second":second_col}
    missing = [k for k,v in required.items() if v is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Got: {list(df.columns)}")

    rename_map = {match_id_col:"match_id", match_col:"Match",
                  team_col:"corner_team", minute_col:"Minute", second_col:"Second"}
    optional_map = {
        outcome_col:"SP_outcome", xg_col:"shot_xg", taker_col:"Taker",
        shooter_col:"Shooter", def_setup_col:"Defensive_setup", shot_team_col:"shot_team_name",
        pass_x_col:"pass_location_x", pass_y_col:"pass_location_y",
        pass_ex_col:"pass_end_location_x", pass_ey_col:"pass_end_location_y",
        shot_x_col:"shot_location_x", shot_y_col:"shot_location_y", shot_z_col:"shot_location_z",
        pt_col:"pass_technique", ph_col:"pass_height", pb_col:"pass_body_part",
        sb_col:"shot_body_part", so_col:"shot_outcome", po_col:"pass_outcome",
        pp_col:"pass_position", sp_col:"shot_position",
        gk_col:"goalkeeper_name", matchday_col:"matchday", stadium_col:"stadium",
    }
    for k, v in optional_map.items():
        if k is not None and k != v:
            rename_map[k] = v

    df = df.rename(columns=rename_map)

    defaults = {c: np.nan for c in [
        "SP_outcome","shot_xg","Taker","Shooter","Defensive_setup","shot_team_name",
        "pass_location_x","pass_location_y","pass_end_location_x","pass_end_location_y",
        "shot_location_x","shot_location_y","shot_location_z","pass_technique","pass_height",
        "pass_body_part","shot_body_part","shot_outcome","pass_outcome","pass_position",
        "shot_position","goalkeeper_name","matchday","stadium",
    ]}
    defaults["SP_outcome"] = ""
    for c, v in defaults.items():
        if c not in df.columns: df[c] = v

    for col in ["Minute","Second","shot_xg","pass_location_x","pass_location_y",
                "pass_end_location_x","pass_end_location_y","shot_location_x",
                "shot_location_y","shot_location_z"]:
        df[col] = safe_numeric(df[col])

    df["corner_team"]  = df["corner_team"].astype(str).str.strip()
    df["event_minute"] = df["Minute"].fillna(0) + df["Second"].fillna(0) / 60.0

    homes, aways = zip(*[split_match_name(m) for m in df["Match"]])
    df["home_team"], df["away_team"] = list(homes), list(aways)
    df["is_home_corner"] = df["corner_team"] == df["home_team"]
    df["is_away_corner"] = df["corner_team"] == df["away_team"]

    sp = df["SP_outcome"].astype(str)
    df["led_to_shot"]         = sp.str.contains("shot",         case=False, na=False)
    df["is_first_contact_shot"] = sp.str.contains("first contact", case=False, na=False)
    df["is_fast_shot"]        = sp.str.contains("within 3 seconds", case=False, na=False)
    df["outcome_bucket"]      = sp.apply(classify_outcome)
    df["is_inswinger"]        = df["pass_technique"].astype(str).str.contains("inswing",  case=False, na=False)
    df["is_outswinger"]       = df["pass_technique"].astype(str).str.contains("outswing", case=False, na=False)
    df["is_short_corner"]     = df["pass_technique"].astype(str).str.contains("short",    case=False, na=False)
    df["delivery_zone"]       = df["pass_end_location_y"].apply(left_right_from_y)
    df["corner_side"]         = df["pass_location_y"].apply(corner_side_from_start_y)
    df["delivery_length"]     = df.apply(
        lambda r: delivery_length(r["pass_location_x"],r["pass_location_y"],
                                   r["pass_end_location_x"],r["pass_end_location_y"]), axis=1)
    df["end_zone"]            = df.apply(
        lambda r: zone_from_end_location(r["pass_end_location_x"],r["pass_end_location_y"]), axis=1)
    df["is_goal_kick_zone_delivery"] = (
        df["pass_end_location_x"].between(114,120,inclusive="both") &
        df["pass_end_location_y"].between(30,50,inclusive="both"))
    df["is_six_yard_delivery"]   = df["end_zone"].eq("6-yard box")
    df["is_penalty_area_delivery"] = df["end_zone"].eq("Penalty area")
    df["phase"] = pd.cut(df["event_minute"],
        bins=[-0.1,15,30,45,60,75,120],
        labels=["0-15","16-30","31-45","46-60","61-75","76+"], right=True).astype(str)

    # Match summary
    match_summary = (df.groupby(["match_id","Match","home_team","away_team"], dropna=False).agg(
        total_corners=("match_id","size"),
        shots_from_corners=("led_to_shot","sum"),
        first_contact_shots=("is_first_contact_shot","sum"),
        fast_shots=("is_fast_shot","sum"),
        total_xg=("shot_xg","sum"),
        avg_xg=("shot_xg","mean"),
        unique_takers=("Taker", pd.Series.nunique),
        inswingers=("is_inswinger","sum"),
        outswingers=("is_outswinger","sum"),
        short_corners=("is_short_corner","sum"),
        six_yard_deliveries=("is_six_yard_delivery","sum"),
        penalty_area_deliveries=("is_penalty_area_delivery","sum"),
    ).reset_index())

    for team_type, col_name in [("home_team","home_corners"),("away_team","away_corners")]:
        def _count(r):
            tn = r[team_type]
            if pd.isna(tn): return np.nan
            return int(((df["match_id"]==r["match_id"]) & (df["corner_team"]==tn)).sum())
        match_summary[col_name] = match_summary.apply(_count, axis=1)

    match_summary["shot_rate"]    = match_summary["shots_from_corners"] / match_summary["total_corners"].replace(0,np.nan)
    match_summary["xg_per_corner"] = match_summary["total_xg"] / match_summary["total_corners"].replace(0,np.nan)
    match_summary["corner_diff"]  = match_summary["home_corners"] - match_summary["away_corners"]

    team_summary = build_team_summary(df)
    return df, match_summary, team_summary

# =========================================================
# LOAD
# =========================================================
try:
    raw_df = load_data()
    df, match_summary, team_summary = prepare_data(raw_df)
except Exception as e:
    st.error("Failed to load data.")
    st.exception(e)
    st.stop()

# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="hero-wrap">'
    '<div class="hero-title">⚽ <span>Allsvenskan</span> Set Piece Studio Pro</div>'
    '<div class="hero-sub">Premium corner analytics workspace — executive dashboards, pitch visualisations, taker profiling, trend analysis, opponent scouting, and full data exports.</div>'
    '<div>'
    '<span class="pill">Executive</span><span class="pill">Visualisation</span>'
    '<span class="pill">Team Intel</span><span class="pill">Match Explorer</span>'
    '<span class="pill">Scouting</span><span class="pill pill-green">Trends</span>'
    '<span class="pill pill-green">Opposition</span><span class="pill pill-orange">Data Hub</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True)

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("### 🎛 Studio Controls")
    st.markdown("---")
    page = st.radio("**Workspace**", [
        "🏠 Executive Dashboard",
        "📊 Visualisation Studio",
        "🏟 Team Analysis",
        "🔍 Match Explorer",
        "👤 Scouting Center",
        "📈 Trend Lab",
        "🛡 Opposition Intel",
        "📐 Advanced Analytics",
        "🗂 Data Hub",
    ])

    st.markdown("---")
    st.markdown("**Core Filters**")
    all_teams = sorted([t for t in team_summary["team"].dropna().unique() if str(t).strip()])
    sel_team  = st.selectbox("Team", ["All Teams"] + all_teams)
    all_takers_list = sorted([str(t) for t in df["Taker"].dropna().astype(str).unique() if str(t).strip()])
    sel_takers = st.multiselect("Taker(s)", all_takers_list)
    all_matches_list = sorted(df["Match"].dropna().astype(str).unique().tolist())
    sel_matches = st.multiselect("Matches", all_matches_list)

    st.markdown("**Time & Volume**")
    mmin = int(df["Minute"].min()) if not df["Minute"].dropna().empty else 0
    mmax = int(df["Minute"].max()) if not df["Minute"].dropna().empty else 120
    if mmax > mmin:
        min_range = st.slider("Minute Range", mmin, mmax, (mmin, mmax))
    else:
        min_range = (mmin, mmax)
    cmin = int(match_summary["total_corners"].min()) if len(match_summary) else 0
    cmax = int(match_summary["total_corners"].max()) if len(match_summary) else 0
    if cmax > cmin:
        corner_range = st.slider("Match Corner Range", cmin, cmax, (cmin, cmax))
    else:
        corner_range = (cmin, cmax)

    st.markdown("**Delivery Filters**")
    show_shot_only    = st.checkbox("Shot outcomes only")
    show_inswing_only = st.checkbox("Inswingers only")
    show_outswing_only= st.checkbox("Outswingers only")
    show_short_only   = st.checkbox("Short corners only")

    all_del_zones = [z for z in ["Near Post Zone","Central Zone","Far Post Zone","Unknown"]
                     if z in df["delivery_zone"].astype(str).unique()]
    sel_del_zones = st.multiselect("Delivery Zone", all_del_zones)
    all_end_zones = [z for z in ["6-yard box","Penalty area","Deep box","Outside danger zone","Unknown"]
                     if z in df["end_zone"].astype(str).unique()]
    sel_end_zones = st.multiselect("End Zone", all_end_zones)
    all_setups = sorted([str(x) for x in df["Defensive_setup"].dropna().astype(str).unique() if str(x).strip()])
    sel_setups = st.multiselect("Defensive Setup", all_setups)

    st.markdown("**Context Filters**")
    venue_filter   = st.multiselect("Home / Away", ["Home","Away","Unknown"], default=["Home","Away","Unknown"])
    phase_filter   = st.multiselect("Phase", ["0-15","16-30","31-45","46-60","61-75","76+"],
                                    default=["0-15","16-30","31-45","46-60","61-75","76+"])
    outcome_filter = st.multiselect("Outcome Bucket",
        ["Shot ≤3s","First Contact Shot","Shot","No First Contact","Other","Unknown"],
        default=["Shot ≤3s","First Contact Shot","Shot","No First Contact","Other","Unknown"])

    st.markdown("---")
    quick_mode = st.selectbox("⚡ Quick Mode",
        ["Balanced","Shot Creation","Delivery Zones","Short Corners","High Danger","Inswingers"])
    if quick_mode == "Shot Creation":   show_shot_only = True
    if quick_mode == "Short Corners":  show_short_only = True
    if quick_mode == "Inswingers":     show_inswing_only = True
    if quick_mode == "High Danger":    sel_end_zones = sel_end_zones or ["6-yard box","Penalty area"]

# =========================================================
# GLOBAL FILTER APPLICATION
# =========================================================
league_match_df = match_summary[
    (match_summary["total_corners"] >= corner_range[0]) &
    (match_summary["total_corners"] <= corner_range[1])
].copy()

league_event_df = df[df["match_id"].isin(league_match_df["match_id"].unique())].copy()
league_event_df = add_advanced_features(league_event_df)
league_event_df = league_event_df[
    (league_event_df["Minute"].fillna(0) >= min_range[0]) &
    (league_event_df["Minute"].fillna(0) <= min_range[1])
]

if sel_team != "All Teams":     league_event_df = league_event_df[league_event_df["corner_team"] == sel_team]
if sel_takers:                  league_event_df = league_event_df[league_event_df["Taker"].astype(str).isin(sel_takers)]
if sel_matches:                 league_event_df = league_event_df[league_event_df["Match"].astype(str).isin(sel_matches)]
if show_shot_only:              league_event_df = league_event_df[league_event_df["led_to_shot"]]
if show_inswing_only and not show_outswing_only: league_event_df = league_event_df[league_event_df["is_inswinger"]]
if show_outswing_only and not show_inswing_only: league_event_df = league_event_df[league_event_df["is_outswinger"]]
if show_short_only:             league_event_df = league_event_df[league_event_df["is_short_corner"]]
if sel_del_zones:               league_event_df = league_event_df[league_event_df["delivery_zone"].isin(sel_del_zones)]
if sel_end_zones:               league_event_df = league_event_df[league_event_df["end_zone"].isin(sel_end_zones)]
if sel_setups:                  league_event_df = league_event_df[league_event_df["Defensive_setup"].astype(str).isin(sel_setups)]
if venue_filter:                league_event_df = league_event_df[league_event_df["venue_split"].isin(venue_filter)]
if phase_filter:                league_event_df = league_event_df[league_event_df["phase"].isin(phase_filter)]
if outcome_filter:              league_event_df = league_event_df[league_event_df["outcome_bucket"].isin(outcome_filter)]

league_match_df = league_match_df[league_match_df["match_id"].isin(league_event_df["match_id"].unique())]
league_team_df  = build_team_summary(league_event_df)

# =========================================================
# PAGE: EXECUTIVE DASHBOARD
# =========================================================
if page == "🏠 Executive Dashboard":
    render_kpis(league_event_df, league_match_df)
    st.markdown("<br>", unsafe_allow_html=True)

    # Insights row
    insights = top_insights(league_event_df, league_team_df)
    cols = st.columns(len(insights))
    for i, (icon, title, body, variant) in enumerate(insights):
        with cols[i]: insight_box(icon, title, body, variant)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])
    with c1:
        section_header("Corner Volume by Team", "Total corners taken, coloured by corners/match rate")
        if not league_team_df.empty:
            fig = px.bar(league_team_df.sort_values("corners_taken", ascending=False),
                x="team", y="corners_taken", color="corners_per_match",
                color_continuous_scale="Blues",
                hover_data=["matches","corners_per_match","shots_from_corners","shot_rate","xg_per_match"],
                labels={"corners_taken":"Corners","team":""})
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(figure_layout(fig, 400), use_container_width=True)
        else: empty_state("No team data.")
    with c2:
        section_header("Efficiency Map", "Shot rate vs xG/match — bubble = volume")
        if not league_team_df.empty:
            st.plotly_chart(team_scatter(league_team_df,"shot_rate","xg_per_match","corners_taken",
                "Shot Rate vs xG/Match","Shot Rate","xG/Match"), use_container_width=True)
        else: empty_state()

    c3, c4, c5, c6 = st.columns(4)
    with c3: st.plotly_chart(outcome_pie(league_event_df, "Outcome Split"),              use_container_width=True)
    with c4: st.plotly_chart(technique_pie(league_event_df, "Technique Split"),          use_container_width=True)
    with c5:
        side_df = league_event_df.groupby("corner_side", dropna=False).size().reset_index(name="n")
        fig = px.pie(side_df, names="corner_side", values="n", title="Corner Side", hole=0.55,
                     color_discrete_sequence=QUAL_PALETTE[2:])
        fig.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(figure_layout(fig, 380, "Corner Side"), use_container_width=True)
    with c6:
        zone_pie = league_event_df.groupby("end_zone", dropna=False).size().reset_index(name="n")
        fig = px.pie(zone_pie, names="end_zone", values="n", title="End Zone Split", hole=0.55,
                     color_discrete_sequence=QUAL_PALETTE[4:])
        fig.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(figure_layout(fig, 380, "End Zone Split"), use_container_width=True)

    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(cumulative_line(league_event_df, title="Cumulative Corners by Team"), use_container_width=True)
    with c8:
        st.plotly_chart(minute_histogram(league_event_df, color_col="outcome_bucket", title="Corner Minute Distribution"), use_container_width=True)

    section_header("Executive Match Board", "Per-match breakdown — sortable")
    board_cols = [c for c in ["Match","home_team","away_team","home_corners","away_corners",
        "total_corners","shots_from_corners","fast_shots","total_xg","shot_rate",
        "xg_per_corner","unique_takers","corner_diff"] if c in league_match_df.columns]
    if not league_match_df.empty:
        st.dataframe(league_match_df[board_cols].sort_values(
            ["total_xg","total_corners"], ascending=False).reset_index(drop=True),
            use_container_width=True, height=420)
    else: empty_state("No match data for current filters.")

    # League xG leaderboard
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("League xG Leaderboard", "Teams ranked by total and per-match xG from corners")
    if not league_team_df.empty:
        ld = league_team_df[["team","total_xg","xg_per_match","corners_taken","shots_from_corners","shot_rate"]].copy()
        ld = ld.sort_values("total_xg", ascending=False).reset_index(drop=True)
        ld.index = ld.index + 1
        ld.insert(0,"#",ld.index)
        st.dataframe(ld, use_container_width=True, height=380)


# =========================================================
# PAGE: VISUALISATION STUDIO
# =========================================================
elif page == "📊 Visualisation Studio":
    section_header("Visualisation Studio", "Chart-first workspace with pitch maps, heatmaps, distributions, and flow diagrams.")
    viz_tabs = st.tabs(["🎯 Shotmaps","🏹 Deliveries","🔥 Density Maps","📊 Team Comparison",
                         "⏱ Timing","📍 Zones & Sides","📈 xG Analysis","🌊 Flow Diagrams",
                         "🎲 Distributions","🧩 Summary Board"])

    with viz_tabs[0]:
        ca, cb, cc = st.columns([1,1,2])
        with ca:
            shot_color = st.selectbox("Color by", ["corner_team","Shooter","Taker","pass_technique",
                "shot_body_part","shot_outcome","xg_category","outcome_bucket"], key="shot_color_vis")
        with cb:
            half_view = st.checkbox("Half-pitch view", value=True, key="half_shot")
        shot_df = league_event_df.dropna(subset=["shot_location_x","shot_location_y"])
        if shot_df.empty: empty_state("No shot location data.", "🎯")
        else:
            st.plotly_chart(shotmap_figure(shot_df, color_col=shot_color,
                title=f"Shotmap — {len(shot_df)} shots | Total xG: {shot_df['shot_xg'].fillna(0).sum():.3f}",
                half=half_view), use_container_width=True)
            st.markdown(f'<div class="footer-note">Bubble size = shot xG value. Hover for full detail.</div>', unsafe_allow_html=True)

        # xG heatmap on pitch alongside
        st.plotly_chart(xg_heatmap_pitch(shot_df, title="xG Density Heatmap on Pitch"), use_container_width=True)
        st.plotly_chart(shot_zone_grid(shot_df, title="Shot Placement — Goal Face View (Y vs Z)"), use_container_width=True)
        st.plotly_chart(xg_shot_scatter(shot_df, title="xG vs Distance to Goal"), use_container_width=True)

    with viz_tabs[1]:
        da, db = st.columns([1,3])
        with da:
            del_color = st.selectbox("Color by", ["delivery_zone","end_zone","pass_technique",
                "corner_team","Taker","corner_side","delivery_type"], key="del_color_vis")
        has_del = not league_event_df.dropna(
            subset=["pass_location_x","pass_location_y","pass_end_location_x","pass_end_location_y"]).empty
        if has_del:
            st.plotly_chart(delivery_map_figure(league_event_df, color_col=del_color,
                title="Delivery Map — All Corners"), use_container_width=True)
        else: empty_state("No delivery coordinate data.", "🏹")
        st.plotly_chart(delivery_length_hist(league_event_df, title="Delivery Length Distribution"), use_container_width=True)

    with viz_tabs[2]:
        c1,c2 = st.columns(2)
        with c1:
            has_end = not league_event_df.dropna(subset=["pass_end_location_x","pass_end_location_y"]).empty
            if has_end:
                st.plotly_chart(heatmap_density(league_event_df,"pass_end_location_x","pass_end_location_y",
                    "Delivery End-Location Density"), use_container_width=True)
            else: empty_state("No end-location data.")
        with c2:
            shot_df2 = league_event_df.dropna(subset=["shot_location_x","shot_location_y"])
            if not shot_df2.empty:
                st.plotly_chart(heatmap_density(shot_df2,"shot_location_x","shot_location_y",
                    "Shot Origin Density"), use_container_width=True)
            else: empty_state("No shot location data.")

    with viz_tabs[3]:
        r1c1,r1c2 = st.columns(2)
        with r1c1: st.plotly_chart(team_scatter(league_team_df,"corners_per_match","shot_rate",
            "corners_taken","Corners/Match vs Shot Rate"), use_container_width=True)
        with r1c2: st.plotly_chart(team_scatter(league_team_df,"six_yard_delivery_rate","xg_per_match",
            "shots_from_corners","6Y Rate vs xG/Match"), use_container_width=True)
        r2c1,r2c2 = st.columns(2)
        with r2c1: st.plotly_chart(team_scatter(league_team_df,"short_corner_rate","fast_shot_rate",
            "corners_taken","Short Corner % vs Fast Shot %"), use_container_width=True)
        with r2c2: st.plotly_chart(team_scatter(league_team_df,"inswinger_rate","shot_rate",
            "corners_taken","Inswinger % vs Shot Rate"), use_container_width=True)
        st.plotly_chart(correlation_heatmap_teams(league_team_df,
            title="Team Metric Correlation Matrix", h=520), use_container_width=True)
        st.plotly_chart(home_away_comparison(league_event_df,
            title="Home vs Away — Shot Rate Comparison"), use_container_width=True)

    with viz_tabs[4]:
        c1,c2 = st.columns(2)
        with c1: st.plotly_chart(cumulative_line(league_event_df, title="Cumulative Corners by Team"), use_container_width=True)
        with c2: st.plotly_chart(minute_histogram(league_event_df, color_col="corner_team", title="Minute Distribution by Team"), use_container_width=True)
        fig_phase = phase_heatmap(league_event_df, title="Corner Phase Heatmap (Team × Phase)", h=440)
        if len(fig_phase.data) > 0: st.plotly_chart(fig_phase, use_container_width=True)
        else: empty_state("No phase data.")
        # Phase bar chart
        ph_bar = league_event_df.groupby(["phase","corner_team"], dropna=False).size().reset_index(name="n")
        fig_phb = px.bar(ph_bar, x="phase", y="n", color="corner_team", barmode="group",
                          title="Corners per Phase by Team", color_discrete_sequence=QUAL_PALETTE,
                          labels={"phase":"","n":"Corners","corner_team":""})
        st.plotly_chart(figure_layout(fig_phb, 380, "Corners per Phase by Team"), use_container_width=True)

    with viz_tabs[5]:
        c1,c2 = st.columns(2)
        with c1: st.plotly_chart(end_zone_bar(league_event_df, title="End-Zone Volume by Team"), use_container_width=True)
        with c2:
            side_bar = league_event_df.groupby(["corner_team","corner_side"], dropna=False).size().reset_index(name="n")
            fig_sb = px.bar(side_bar, x="corner_team", y="n", color="corner_side", barmode="stack",
                title="Corner Side Distribution by Team", color_discrete_sequence=[ACCENT,SUCCESS],
                labels={"corner_team":"","n":"Corners","corner_side":""})
            st.plotly_chart(figure_layout(fig_sb, 420), use_container_width=True)
        c3,c4 = st.columns(2)
        with c3:
            band_df = league_event_df.groupby(["corner_team","delivery_length_band"], dropna=False).size().reset_index(name="n")
            fig_bd = px.bar(band_df, x="corner_team", y="n", color="delivery_length_band", barmode="stack",
                title="Delivery Length Band by Team", color_discrete_sequence=QUAL_PALETTE,
                labels={"corner_team":"","n":"Corners","delivery_length_band":""})
            st.plotly_chart(figure_layout(fig_bd, 400), use_container_width=True)
        with c4:
            body_df = league_event_df.groupby(["corner_team","pass_body_part"], dropna=False).size().reset_index(name="n")
            fig_bp = px.bar(body_df, x="corner_team", y="n", color="pass_body_part", barmode="stack",
                title="Pass Body Part by Team", color_discrete_sequence=QUAL_PALETTE[3:],
                labels={"corner_team":"","n":"Corners","pass_body_part":""})
            st.plotly_chart(figure_layout(fig_bp, 400), use_container_width=True)

    with viz_tabs[6]:
        c1,c2 = st.columns(2)
        with c1:
            st.plotly_chart(xg_accumulation(league_event_df, title="Total xG Accumulation"), use_container_width=True)
            st.plotly_chart(xg_accumulation(league_event_df, group_col="corner_team", title="xG Accumulation by Team"), use_container_width=True)
        with c2:
            xg_team = league_event_df.groupby("corner_team", dropna=False).agg(
                total_xg=("shot_xg","sum"), corners=("match_id","size")).reset_index()
            xg_team["xg_per_corner"] = xg_team["total_xg"] / xg_team["corners"].replace(0,np.nan)
            fig = px.bar(xg_team.sort_values("total_xg", ascending=False),
                x="corner_team", y="total_xg", color="xg_per_corner",
                color_continuous_scale="Greens", title="Total xG by Team",
                labels={"corner_team":"","total_xg":"Total xG"})
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(figure_layout(fig, 380), use_container_width=True)
            xg_oc = league_event_df.groupby("xg_category", dropna=False).size().reset_index(name="n")
            fig2 = px.bar(xg_oc, x="xg_category", y="n", color="xg_category",
                title="Shot Count by xG Category", color_discrete_sequence=QUAL_PALETTE,
                labels={"xg_category":"","n":"Shots"})
            st.plotly_chart(figure_layout(fig2, 360), use_container_width=True)
        shot_df3 = league_event_df.dropna(subset=["shot_xg"])
        if not shot_df3.empty:
            st.plotly_chart(shot_xg_distribution(shot_df3, group_col="corner_team",
                title="xG Distribution by Team (Violin)"), use_container_width=True)

    with viz_tabs[7]:
        if not league_event_df.empty:
            st.plotly_chart(sankey_outcome_flow(league_event_df, "Corner Flow: Team → Zone → Outcome"), use_container_width=True)
        else: empty_state("No data for flow diagram.", "🌊")

    with viz_tabs[8]:
        c1,c2 = st.columns(2)
        with c1: st.plotly_chart(outcome_pie(league_event_df), use_container_width=True)
        with c2: st.plotly_chart(technique_pie(league_event_df), use_container_width=True)
        c3,c4 = st.columns(2)
        with c3:
            ht = league_event_df.groupby("pass_height", dropna=False).size().reset_index(name="n")
            fig = px.pie(ht, names="pass_height", values="n", title="Pass Height Split", hole=0.54,
                         color_discrete_sequence=QUAL_PALETTE)
            st.plotly_chart(figure_layout(fig, 360, "Pass Height Split"), use_container_width=True)
        with c4:
            dt = league_event_df.groupby("delivery_type", dropna=False).size().reset_index(name="n")
            fig = px.pie(dt, names="delivery_type", values="n", title="Delivery Type Split", hole=0.54,
                         color_discrete_sequence=QUAL_PALETTE[2:])
            st.plotly_chart(figure_layout(fig, 360, "Delivery Type Split"), use_container_width=True)
        # Goal-face body part breakdown
        shot_4 = league_event_df.dropna(subset=["shot_body_part"])
        if not shot_4.empty:
            bp = shot_4.groupby(["shot_body_part","shot_outcome"], dropna=False).size().reset_index(name="n")
            fig = px.bar(bp, x="shot_body_part", y="n", color="shot_outcome", barmode="stack",
                title="Shot Body Part vs Outcome", color_discrete_sequence=QUAL_PALETTE,
                labels={"shot_body_part":"","n":"Shots","shot_outcome":""})
            st.plotly_chart(figure_layout(fig, 380), use_container_width=True)

    with viz_tabs[9]:
        c1,c2,c3,c4 = st.columns(4)
        total_xg = league_event_df["shot_xg"].fillna(0).sum()
        goals     = int(league_event_df["goal_from_corner"].sum()) if not league_event_df.empty else 0
        with c1: metric_card("Events",    f"{len(league_event_df):,}",  foot="Corner actions")
        with c2: metric_card("Total xG",  f"{total_xg:.2f}",            foot="From corners")
        with c3: metric_card("Goals",     f"{goals}",                   foot="From corners", accent_color=DANGER)
        with c4: metric_card("Shot Rate", f"{league_event_df['led_to_shot'].mean()*100:.1f}", "%", "Shots/corner")
        c5,c6 = st.columns(2)
        with c5: st.plotly_chart(outcome_pie(league_event_df), use_container_width=True)
        with c6: st.plotly_chart(technique_pie(league_event_df), use_container_width=True)
        st.plotly_chart(phase_heatmap(league_event_df, "Phase Heatmap"), use_container_width=True)


# =========================================================
# PAGE: TEAM ANALYSIS
# =========================================================
elif page == "🏟 Team Analysis":
    if sel_team == "All Teams":
        st.info("👈 Select a specific team in the sidebar for full team intelligence.")
        section_header("League Team Overview Table", "All teams — core metrics")
        if not league_team_df.empty:
            disp = league_team_df.copy()
            for pc in ["shot_rate","fast_shot_rate","six_yard_delivery_rate","short_corner_rate","inswinger_rate"]:
                if pc in disp.columns: disp[pc] = (disp[pc]*100).round(1).astype(str)+"%"
            disp["xg_per_match"] = disp["xg_per_match"].round(3)
            disp["corners_per_match"] = disp["corners_per_match"].round(2)
            st.dataframe(disp.reset_index(drop=True), use_container_width=True, height=560)
        else: empty_state()

        # Multi-metric radar for all teams
        if not league_team_df.empty and len(league_team_df) >= 2:
            section_header("Multi-Team Radar Comparison", "Normalised across all teams")
            radar_cols = ["shot_rate","xg_per_match","fast_shot_rate","six_yard_delivery_rate","short_corner_rate","inswinger_rate"]
            rc = [c for c in radar_cols if c in league_team_df.columns]
            fig_rad = go.Figure()
            for _, tr in league_team_df.iterrows():
                normed = []
                for c in rc:
                    s = league_team_df[c].dropna()
                    val = tr[c]
                    if s.max() == s.min() or pd.isna(val): normed.append(0.0)
                    else: normed.append((val - s.min())/(s.max()-s.min()))
                normed += [normed[0]]
                labels_r = rc + [rc[0]]
                fig_rad.add_trace(go.Scatterpolar(r=normed, theta=labels_r,
                    fill="toself", name=str(tr["team"]),
                    opacity=0.55, line=dict(width=1.8)))
            fig_rad.update_layout(
                polar=dict(bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0,1], showticklabels=False,
                                    gridcolor="rgba(255,255,255,0.08)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.08)",
                                     linecolor="rgba(255,255,255,0.15)"),
                ),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                height=560, font=dict(color=TEXT, family="DM Sans, sans-serif"),
                margin=dict(l=20,r=20,t=48,b=20), title="Multi-Team Radar",
            )
            st.plotly_chart(fig_rad, use_container_width=True)
    else:
        team_ev  = league_event_df[league_event_df["corner_team"] == sel_team].copy()
        team_mch = league_match_df[league_match_df["match_id"].isin(team_ev["match_id"].unique())].copy()
        team_row = league_team_df[league_team_df["team"] == sel_team]
        taker_df = taker_summary_table(team_ev)
        all_takers_league_df = taker_summary_table(league_event_df)

        team_tabs = st.tabs(["📊 Overview","🎯 Pitch Visuals","👤 Taker Intel",
                               "📋 Match Review","🏆 Report Card","⚖️ Home vs Away",
                               "🔬 Deep Dive","🗂 Raw Data"])

        with team_tabs[0]:
            section_header(f"{sel_team} — Overview", "Snapshot of corner profile")
            render_kpis(team_ev, team_mch)
            c1,c2 = st.columns(2)
            with c1:
                od = team_ev.groupby("outcome_bucket", dropna=False).size().reset_index(name="n").sort_values("n", ascending=False)
                fig = px.bar(od, x="outcome_bucket", y="n", color="outcome_bucket",
                    color_discrete_sequence=QUAL_PALETTE, title="Outcome Profile",
                    labels={"outcome_bucket":"","n":"Corners"})
                st.plotly_chart(figure_layout(fig, 380), use_container_width=True)
            with c2:
                zd = team_ev.groupby("end_zone", dropna=False).size().reset_index(name="n").sort_values("n", ascending=False)
                fig = px.bar(zd, x="end_zone", y="n", color="end_zone",
                    color_discrete_sequence=QUAL_PALETTE[2:], title="End-Zone Profile",
                    labels={"end_zone":"","n":"Corners"})
                st.plotly_chart(figure_layout(fig, 380), use_container_width=True)
            c3,c4 = st.columns(2)
            with c3:
                td = team_ev.groupby("pass_technique", dropna=False).size().reset_index(name="n")
                fig = px.pie(td, names="pass_technique", values="n", title="Technique Split", hole=0.54,
                             color_discrete_sequence=QUAL_PALETTE)
                st.plotly_chart(figure_layout(fig, 360, "Technique Split"), use_container_width=True)
            with c4:
                pd2 = team_ev.groupby("phase", dropna=False).size().reset_index(name="n")
                fig = px.bar(pd2, x="phase", y="n", color_discrete_sequence=[ACCENT],
                    title="Phase Distribution", labels={"phase":"","n":"Corners"})
                st.plotly_chart(figure_layout(fig, 360), use_container_width=True)
            c5,c6 = st.columns(2)
            with c5:
                st.plotly_chart(xg_accumulation(team_ev, title=f"xG Accumulation — {sel_team}"), use_container_width=True)
            with c6:
                match_xg = team_ev.groupby("Match", dropna=False).agg(
                    total_xg=("shot_xg","sum"), corners=("match_id","size")).reset_index()
                fig = px.bar(match_xg.sort_values("total_xg", ascending=False),
                    x="Match", y="total_xg", color="total_xg", color_continuous_scale="Blues",
                    title="xG per Match", labels={"Match":"","total_xg":"xG"})
                fig.update_layout(coloraxis_showscale=False, xaxis_tickangle=-30)
                st.plotly_chart(figure_layout(fig, 360), use_container_width=True)

        with team_tabs[1]:
            c1,c2 = st.columns(2)
            with c1:
                shot_t = team_ev.dropna(subset=["shot_location_x","shot_location_y"])
                if shot_t.empty: empty_state("No shot data.", "🎯")
                else: st.plotly_chart(shotmap_figure(shot_t, "Shooter", f"Shotmap — {sel_team}"), use_container_width=True)
            with c2:
                hd2 = not team_ev.dropna(subset=["pass_location_x","pass_location_y","pass_end_location_x","pass_end_location_y"]).empty
                if hd2: st.plotly_chart(delivery_map_figure(team_ev, "delivery_zone", f"Delivery Map — {sel_team}"), use_container_width=True)
                else: empty_state("No delivery coords.", "🏹")
            has_end2 = not team_ev.dropna(subset=["pass_end_location_x","pass_end_location_y"]).empty
            if has_end2:
                c3,c4 = st.columns(2)
                with c3:
                    st.plotly_chart(heatmap_density(team_ev,"pass_end_location_x","pass_end_location_y",
                        f"Delivery End-Location — {sel_team}"), use_container_width=True)
                with c4:
                    s2 = team_ev.dropna(subset=["shot_location_x","shot_location_y"])
                    if not s2.empty: st.plotly_chart(xg_heatmap_pitch(s2, f"xG Heatmap — {sel_team}"), use_container_width=True)
                    else: empty_state("No shot data.")
            shot_t2 = team_ev.dropna(subset=["shot_xg"])
            if not shot_t2.empty: st.plotly_chart(shot_xg_distribution(shot_t2, title=f"xG Distribution — {sel_team}"), use_container_width=True)

        with team_tabs[2]:
            section_header("Taker Intelligence", "All corner takers for this team")
            if taker_df.empty: empty_state("No taker data.")
            else:
                st.dataframe(taker_df.reset_index(drop=True), use_container_width=True, height=400)
                st.plotly_chart(taker_bar_chart(taker_df, "xg_per_corner", f"Takers by xG/Corner — {sel_team}", min_corners=2), use_container_width=True)
                st.plotly_chart(taker_bar_chart(taker_df, "shot_rate", f"Takers by Shot Rate — {sel_team}", min_corners=2), use_container_width=True)
                section_header("Individual Taker Radar", "Normalised vs all takers in dataset")
                ta_list = [str(t) for t in taker_df["Taker"].dropna().unique()]
                if ta_list:
                    sel_taker_r = st.selectbox("Select Taker", ta_list, key="team_taker_radar")
                    c1,c2 = st.columns([1.2,1])
                    with c1:
                        st.plotly_chart(taker_radar(all_takers_league_df, sel_taker_r), use_container_width=True)
                    with c2:
                        tr = taker_df[taker_df["Taker"].astype(str)==sel_taker_r]
                        if not tr.empty:
                            r = tr.iloc[0]
                            metric_card("Corners",     f"{int(r.get('corners',0)):,}",          foot="Sample size")
                            metric_card("Shot Rate",   human_pct(r.get("shot_rate")),            foot="Corners → shots")
                            metric_card("xG/Corner",   human_val(r.get("xg_per_corner"),4),      foot="Quality")
                            metric_card("Goals",       f"{int(r.get('goals',0))}",               foot="Direct goals", accent_color=DANGER)

        with team_tabs[3]:
            section_header("Match Review", "Per-match breakdown")
            mp = match_pattern_table(team_ev)
            if mp.empty: empty_state()
            else: st.dataframe(mp.reset_index(drop=True), use_container_width=True, height=480)
            section_header("Match-by-Match Shot Rate", "Rolling trend")
            st.plotly_chart(rolling_shot_rate(team_ev, title=f"Shot Rate per Match — {sel_team}"), use_container_width=True)

        with team_tabs[4]:
            if not team_row.empty:
                row = team_row.iloc[0]
                metrics = [
                    ("Corners/Match",    "corners_per_match",        "val",  "Volume ranking"),
                    ("Shot Rate",        "shot_rate",                 "pct",  "Shots per corner"),
                    ("xG/Match",         "xg_per_match",              "xg",   "Attacking quality"),
                    ("Fast Shot Rate",   "fast_shot_rate",            "pct",  "Shot within 3 seconds"),
                    ("6Y Delivery Rate", "six_yard_delivery_rate",    "pct",  "Balls into 6-yard box"),
                    ("Short Corner Rate","short_corner_rate",         "pct",  "Short routines used"),
                    ("Inswinger Rate",   "inswinger_rate",            "pct",  "Inswinging deliveries"),
                    ("Taker Variety",    "taker_variety",             "val",  "Unique takers used"),
                    ("Penalty Area %",   "penalty_area_delivery_rate","pct",  "Deliveries to penalty area"),
                ]
                rows_html = percentile_bars_html(row, league_team_df, metrics)
                st.markdown(f"""
                <div style="background:linear-gradient(160deg,{CARD},{CARD_2});border:1px solid {BORDER};
                     border-radius:22px;padding:22px 24px;">
                    <div style="font-size:1.12rem;font-weight:800;margin-bottom:16px">
                        🏆 {sel_team} — Report Card
                    </div>
                    {rows_html}
                    <div style="margin-top:14px;font-size:0.76rem;color:{MUTED_2}">
                        Percentile vs all teams in current filtered dataset.
                    </div>
                </div>""", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                c1,c2,c3,c4 = st.columns(4)
                def _pct(col): return f"{percentile_rank(league_team_df[col] if col in league_team_df.columns else pd.Series(dtype=float), row.get(col,np.nan)):.0f}" if not pd.isna(row.get(col,np.nan)) else "—"
                with c1: metric_card("Volume Pctile",    _pct("corners_per_match"), "th", "vs league")
                with c2: metric_card("Shot Rate Pctile", _pct("shot_rate"),          "th", "vs league")
                with c3: metric_card("xG/Match Pctile",  _pct("xg_per_match"),       "th", "vs league")
                with c4: metric_card("6Y Delivery",      human_pct(row.get("six_yard_delivery_rate")), "", "high-danger targeting")
            else: empty_state("No report card data.", "🏆")

        with team_tabs[5]:
            section_header("Home vs Away Analysis", "Split performance profile")
            hva = (team_ev.groupby("venue_split", dropna=False).agg(
                corners=("match_id","size"),
                shots=("led_to_shot","sum"),
                total_xg=("shot_xg","sum"),
                fast_shots=("is_fast_shot","sum"),
                six_yard=("is_six_yard_delivery","sum"),
                short=("is_short_corner","sum"),
            ).reset_index())
            hva["shot_rate"]   = hva["shots"] / hva["corners"].replace(0,np.nan)
            hva["xg_per_c"]    = hva["total_xg"] / hva["corners"].replace(0,np.nan)
            hva["fast_rate"]   = hva["fast_shots"] / hva["corners"].replace(0,np.nan)
            hva_f = hva[hva["venue_split"].isin(["Home","Away"])]
            if hva_f.empty: empty_state("No home/away split data.")
            else:
                st.dataframe(hva_f.reset_index(drop=True), use_container_width=True)
                c1,c2,c3 = st.columns(3)
                for col, label, colr in [
                    ("shot_rate","Shot Rate",[ACCENT,SUCCESS]),
                    ("xg_per_c","xG/Corner",[WARNING,ORANGE]),
                    ("fast_rate","Fast Shot Rate",[DANGER,PURPLE])
                ]:
                    with [c1,c2,c3][["shot_rate","xg_per_c","fast_rate"].index(col)]:
                        fig = px.bar(hva_f, x="venue_split", y=col, color="venue_split",
                            color_discrete_sequence=colr, title=label,
                            labels={"venue_split":"","col":label})
                        st.plotly_chart(figure_layout(fig, 320, label), use_container_width=True)
            st.plotly_chart(home_away_comparison(team_ev, f"Home vs Away — Shot Rate — {sel_team}"), use_container_width=True)

        with team_tabs[6]:
            section_header("Deep Dive Analytics", "Combination and cross-dimensional analysis")
            c1,c2 = st.columns(2)
            with c1:
                # technique × end zone heatmap
                tz = team_ev.groupby(["pass_technique","end_zone"], dropna=False).size().reset_index(name="n")
                if not tz.empty:
                    piv = tz.pivot(index="pass_technique", columns="end_zone", values="n").fillna(0)
                    fig = px.imshow(piv, aspect="auto", title="Technique × End Zone",
                        color_continuous_scale="Blues", text_auto=True)
                    st.plotly_chart(figure_layout(fig, 380, "Technique × End Zone"), use_container_width=True)
                else: empty_state("No cross-dimensional data.")
            with c2:
                # phase × outcome heatmap
                po = team_ev.groupby(["phase","outcome_bucket"], dropna=False).size().reset_index(name="n")
                if not po.empty:
                    piv2 = po.pivot(index="phase", columns="outcome_bucket", values="n").fillna(0)
                    fig2 = px.imshow(piv2, aspect="auto", title="Phase × Outcome",
                        color_continuous_scale="Greens", text_auto=True)
                    st.plotly_chart(figure_layout(fig2, 380, "Phase × Outcome"), use_container_width=True)
                else: empty_state()
            # corner side × delivery zone
            cs_dz = team_ev.groupby(["corner_side","delivery_zone"], dropna=False).size().reset_index(name="n")
            if not cs_dz.empty:
                fig_cs = px.bar(cs_dz, x="corner_side", y="n", color="delivery_zone", barmode="group",
                    title="Corner Side × Delivery Zone", color_discrete_sequence=QUAL_PALETTE,
                    labels={"corner_side":"","n":"Corners","delivery_zone":""})
                st.plotly_chart(figure_layout(fig_cs, 360), use_container_width=True)
            # sankey for this team
            st.plotly_chart(sankey_outcome_flow(team_ev, f"{sel_team} — Corner Flow"), use_container_width=True)
            st.plotly_chart(delivery_length_hist(team_ev, f"Delivery Length — {sel_team}"), use_container_width=True)

        with team_tabs[7]:
            st.dataframe(team_ev.reset_index(drop=True), use_container_width=True, height=560)
            c1,c2,c3 = st.columns(3)
            with c1:
                st.download_button("⬇ Events CSV", team_ev.to_csv(index=False).encode(),
                    f"{sel_team.replace(' ','_')}_events.csv", "text/csv", use_container_width=True)
            with c2:
                st.download_button("⬇ Taker CSV", taker_df.to_csv(index=False).encode(),
                    f"{sel_team.replace(' ','_')}_takers.csv", "text/csv", use_container_width=True)
            with c3:
                mp_exp = match_pattern_table(team_ev)
                st.download_button("⬇ Match CSV", mp_exp.to_csv(index=False).encode(),
                    f"{sel_team.replace(' ','_')}_matches.csv", "text/csv", use_container_width=True)


# =========================================================
# PAGE: MATCH EXPLORER
# =========================================================
elif page == "🔍 Match Explorer":
    section_header("Match Explorer", "Deep dive into individual matches")
    avail_matches = sorted(league_match_df["Match"].dropna().unique().tolist()) if not league_match_df.empty else []
    ca, cb = st.columns([2,1])
    with ca: sel_match = st.selectbox("Select Match", ["All Matches"] + avail_matches)
    with cb:
        if sel_match != "All Matches":
            st.markdown(f'<div class="sub-card sub-card-accent" style="text-align:center;font-weight:800;margin-top:8px">{sel_match}</div>', unsafe_allow_html=True)

    m_ev  = league_event_df.copy()
    m_mch = league_match_df.copy()
    if sel_match != "All Matches":
        m_mch = m_mch[m_mch["Match"] == sel_match]
        m_ev  = m_ev[m_ev["match_id"].isin(m_mch["match_id"].unique())]

    m_tabs = st.tabs(["📋 Summary","⏱ Timeline","🎯 Shotmap","🏹 Delivery","📊 Phase",
                        "⚔️ Head to Head","🔴 Event Feed","🗂 Full Data"])

    with m_tabs[0]:
        render_kpis(m_ev, m_mch)
        bc = [c for c in ["Match","home_team","away_team","home_corners","away_corners","total_corners",
             "shots_from_corners","fast_shots","total_xg","shot_rate","xg_per_corner","unique_takers",
             "six_yard_deliveries","corner_diff"] if c in m_mch.columns]
        if not m_mch.empty:
            st.dataframe(m_mch[bc].sort_values(["total_xg","total_corners"],ascending=False).reset_index(drop=True),
                use_container_width=True, height=320)
        if not m_ev.empty:
            section_header("Team Breakdown in Match")
            tb = build_team_summary(m_ev)
            tbc = [c for c in ["team","corners_taken","shots_from_corners","shot_rate","total_xg",
                "fast_shots","six_yard_deliveries","short_corners","inswingers"] if c in tb.columns]
            st.dataframe(tb[tbc].reset_index(drop=True), use_container_width=True, height=260)

    with m_tabs[1]:
        if m_ev.empty: empty_state()
        else:
            c1,c2 = st.columns(2)
            with c1:
                md = m_ev.groupby(["Minute","corner_team"], dropna=False).size().reset_index(name="n")
                fig = px.bar(md, x="Minute", y="n", color="corner_team", barmode="stack",
                    title="Corner Timeline", color_discrete_sequence=QUAL_PALETTE,
                    labels={"n":"Events","Minute":"Match Minute","corner_team":""})
                st.plotly_chart(figure_layout(fig, 400), use_container_width=True)
            with c2:
                st.plotly_chart(cumulative_line(m_ev, title="Cumulative Corners"), use_container_width=True)
            st.plotly_chart(minute_histogram(m_ev, color_col="corner_team", title="Minute Distribution"), use_container_width=True)

    with m_tabs[2]:
        ma, mb = st.columns([1,3])
        with ma: sc2 = st.selectbox("Color by",["corner_team","Shooter","Taker","pass_technique","shot_outcome"],key="mshot")
        s5 = m_ev.dropna(subset=["shot_location_x","shot_location_y"])
        if s5.empty: empty_state("No shot location data.", "🎯")
        else:
            st.plotly_chart(shotmap_figure(s5, sc2, f"Shotmap — {sel_match}"), use_container_width=True)
            st.plotly_chart(xg_heatmap_pitch(s5, f"xG Heatmap — {sel_match}"), use_container_width=True)

    with m_tabs[3]:
        mc, md2 = st.columns([1,3])
        with mc: dc2 = st.selectbox("Color by",["corner_team","delivery_zone","end_zone","pass_technique","Taker"],key="mdel")
        hd3 = not m_ev.dropna(subset=["pass_location_x","pass_location_y","pass_end_location_x","pass_end_location_y"]).empty
        if hd3:
            st.plotly_chart(delivery_map_figure(m_ev, dc2, f"Delivery Map — {sel_match}"), use_container_width=True)
            has_e3 = not m_ev.dropna(subset=["pass_end_location_x","pass_end_location_y"]).empty
            if has_e3:
                st.plotly_chart(heatmap_density(m_ev,"pass_end_location_x","pass_end_location_y",
                    f"Delivery End-Location — {sel_match}"), use_container_width=True)
        else: empty_state("No delivery data.", "🏹")

    with m_tabs[4]:
        if m_ev.empty: empty_state()
        else:
            c1,c2 = st.columns(2)
            with c1: st.plotly_chart(outcome_pie(m_ev, "Outcome Split"), use_container_width=True)
            with c2: st.plotly_chart(technique_pie(m_ev, "Technique Split"), use_container_width=True)
            ph = phase_heatmap(m_ev, "Phase Heatmap")
            if len(ph.data): st.plotly_chart(ph, use_container_width=True)
            else: empty_state("No phase data.")

    with m_tabs[5]:
        # Head-to-head if a specific match
        if sel_match != "All Matches" and not m_mch.empty:
            mr = m_mch.iloc[0]
            home_t, away_t = mr.get("home_team","?"), mr.get("away_team","?")
            hev = m_ev[m_ev["corner_team"] == home_t]
            aev = m_ev[m_ev["corner_team"] == away_t]
            h_row = build_team_summary(hev).iloc[0] if not build_team_summary(hev).empty else pd.Series()
            a_row = build_team_summary(aev).iloc[0] if not build_team_summary(aev).empty else pd.Series()
            metrics_h2h = [
                ("Corners",     "corners_taken",         "val"),
                ("Shot Rate",   "shot_rate",              "pct"),
                ("xG",          "total_xg",               "xg"),
                ("Fast Shots",  "fast_shots",             "val"),
                ("6Y Delivers", "six_yard_deliveries",    "val"),
                ("Inswingers",  "inswingers",             "val"),
                ("Short Corners","short_corners",         "val"),
            ]
            rows_html = ""
            for label, col, fmt in metrics_h2h:
                va = h_row.get(col,np.nan); vb = a_row.get(col,np.nan)
                def _f(v):
                    if pd.isna(v): return "—"
                    if fmt=="pct": return f"{v*100:.1f}%"
                    if fmt=="xg":  return f"{v:.3f}"
                    return str(int(v))
                ca_better = not pd.isna(va) and not pd.isna(vb) and va >= vb
                cb_better = not pd.isna(va) and not pd.isna(vb) and vb > va
                ca_col = SUCCESS if ca_better else TEXT
                cb_col = SUCCESS if cb_better else TEXT
                rows_html += f"""
                <div style="display:flex;align-items:center;padding:10px 0;border-bottom:1px solid {BORDER}">
                    <div style="flex:1;font-weight:800;color:{ca_col};text-align:center;font-size:1.05rem">{_f(va)}</div>
                    <div style="flex:1.4;text-align:center;color:{MUTED};font-size:0.86rem">{label}</div>
                    <div style="flex:1;font-weight:800;color:{cb_col};text-align:center;font-size:1.05rem">{_f(vb)}</div>
                </div>"""
            st.markdown(f"""
            <div style="background:linear-gradient(160deg,{CARD},{CARD_2});border:1px solid {BORDER};
                 border-radius:22px;padding:20px 24px;">
                <div style="display:flex;padding-bottom:14px;border-bottom:1px solid {BORDER_2}">
                    <div style="flex:1;font-size:1.2rem;font-weight:900;color:{ACCENT};text-align:center">{home_t}</div>
                    <div style="flex:1.4;text-align:center;color:{MUTED};font-size:0.9rem;align-self:center">Head to Head</div>
                    <div style="flex:1;font-size:1.2rem;font-weight:900;color:{SUCCESS};text-align:center">{away_t}</div>
                </div>
                {rows_html}
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Select a specific match to see head-to-head comparison.")

    with m_tabs[6]:
        sc = [c for c in ["Match","corner_team","Taker","Shooter","Minute","Second","SP_outcome","shot_xg",
            "Defensive_setup","pass_technique","pass_height","pass_body_part","delivery_zone","end_zone",
            "corner_side","delivery_length_band","outcome_bucket","xg_category"] if c in m_ev.columns]
        if not m_ev.empty:
            st.dataframe(m_ev[sc].sort_values(["Minute","Second"]).reset_index(drop=True),
                use_container_width=True, height=600)
        else: empty_state()

    with m_tabs[7]:
        st.dataframe(m_ev.reset_index(drop=True), use_container_width=True, height=600)
        if not m_ev.empty:
            st.download_button("⬇ Match Events CSV",
                m_ev.to_csv(index=False).encode(),
                f"match_{sel_match.replace(' ','_').replace('/','_')}.csv",
                "text/csv")

# =========================================================
# PAGE: SCOUTING CENTER
# =========================================================
elif page == "👤 Scouting Center":
    section_header("Scouting Center", "Rank, compare, and profile takers & teams across the dataset.")
    sc_tabs = st.tabs(["🏅 Team Rankings","👤 Taker Rankings","🔀 Team Comparison",
                        "📍 Zone Intel","🛡 Defensive Setups","🔭 Advanced Search","⬇ Export"])

    with sc_tabs[0]:
        section_header("Team Rankings", "Sorted by xG/match")
        if league_team_df.empty: empty_state()
        else:
            rd = league_team_df.sort_values(["xg_per_match","shot_rate","corners_taken"], ascending=False).reset_index(drop=True)
            rd.index += 1; rd.insert(0,"#",rd.index)
            st.dataframe(rd, use_container_width=True, height=560)
            c1,c2,c3 = st.columns(3)
            with c1:
                fig = px.bar(rd.head(12), x="team", y="xg_per_match", color="xg_per_match",
                    color_continuous_scale="Blues", title="xG/Match — Top Teams",
                    labels={"team":"","xg_per_match":"xG/Match"})
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(figure_layout(fig, 360), use_container_width=True)
            with c2:
                fig = px.bar(rd.sort_values("shot_rate",ascending=False).head(12), x="team", y="shot_rate",
                    color="shot_rate", color_continuous_scale="Greens", title="Shot Rate — Top Teams",
                    labels={"team":"","shot_rate":"Shot Rate"})
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(figure_layout(fig, 360), use_container_width=True)
            with c3:
                fig = px.bar(rd.sort_values("six_yard_delivery_rate",ascending=False).head(12),
                    x="team", y="six_yard_delivery_rate", color="six_yard_delivery_rate",
                    color_continuous_scale="Oranges", title="6Y Rate — Top Teams",
                    labels={"team":"","six_yard_delivery_rate":"6Y Delivery Rate"})
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(figure_layout(fig, 360), use_container_width=True)

    with sc_tabs[1]:
        section_header("Taker Rankings", "All takers — ranked by xG/corner")
        tl_df = taker_summary_table(league_event_df)
        min_c = st.number_input("Min corners (filter noise)", 1, 50, 5, key="sc_min_c")
        tl_filt = tl_df[tl_df["corners"] >= min_c].reset_index(drop=True) if not tl_df.empty else tl_df
        if tl_filt.empty: empty_state("No taker data.", "👤")
        else:
            st.dataframe(tl_filt, use_container_width=True, height=480)
            c1,c2 = st.columns(2)
            with c1:
                st.plotly_chart(taker_bar_chart(tl_filt, "xg_per_corner", "Top Takers — xG/Corner",
                    h=380, min_corners=int(min_c)), use_container_width=True)
            with c2:
                st.plotly_chart(taker_bar_chart(tl_filt, "shot_rate", "Top Takers — Shot Rate",
                    h=380, min_corners=int(min_c)), use_container_width=True)
            # Radar
            section_header("Taker Radar Profile")
            tlist = [str(t) for t in tl_filt["Taker"].dropna().unique()]
            if tlist:
                sel_tk = st.selectbox("Select Taker", tlist, key="sc_taker")
                c1,c2 = st.columns([1.2,1])
                with c1: st.plotly_chart(taker_radar(tl_df, sel_tk), use_container_width=True)
                with c2:
                    tr = tl_filt[tl_filt["Taker"].astype(str)==sel_tk]
                    if not tr.empty:
                        r = tr.iloc[0]
                        for lbl,col,fmt in [
                            ("Corners",f"{int(r.get('corners',0)):,}",""),
                            ("Shot Rate",human_pct(r.get("shot_rate")),""),
                            ("xG/Corner",human_val(r.get("xg_per_corner"),4),""),
                            ("Goals",f"{int(r.get('goals',0))}",""),
                            ("Fast Shots",f"{int(r.get('fast_shots',0))}",""),
                            ("Inswingers",f"{int(r.get('inswingers',0))}",""),
                        ]:
                            metric_card(lbl, fmt+col, foot="")

    with sc_tabs[2]:
        section_header("Team Comparison", "Side-by-side metrics")
        all_t = [t for t in league_team_df["team"].dropna().unique() if str(t).strip()] if not league_team_df.empty else []
        if len(all_t) < 2: st.info("Need at least 2 teams.")
        else:
            ca, cb = st.columns(2)
            with ca: t_a = st.selectbox("Team A", all_t, 0, key="cmp_a")
            with cb: t_b = st.selectbox("Team B", all_t, min(1,len(all_t)-1), key="cmp_b")
            r_a = league_team_df[league_team_df["team"]==t_a].iloc[0] if not league_team_df[league_team_df["team"]==t_a].empty else pd.Series()
            r_b = league_team_df[league_team_df["team"]==t_b].iloc[0] if not league_team_df[league_team_df["team"]==t_b].empty else pd.Series()
            cmp_metrics = [
                ("Corners/Match","corners_per_match","{:.2f}"),("Shot Rate","shot_rate","{:.1%}"),
                ("xG/Match","xg_per_match","{:.3f}"),("Fast Shot Rate","fast_shot_rate","{:.1%}"),
                ("6Y Delivery","six_yard_delivery_rate","{:.1%}"),("Short Corner %","short_corner_rate","{:.1%}"),
                ("Inswinger %","inswinger_rate","{:.1%}"),("Taker Variety","taker_variety","{:.0f}"),
                ("Total xG","total_xg","{:.2f}"),("Corners Taken","corners_taken","{:.0f}"),
            ]
            rows_html = ""
            for label, col, fmt in cmp_metrics:
                va = r_a.get(col,np.nan); vb = r_b.get(col,np.nan)
                def _fv(v): return fmt.format(v) if not pd.isna(v) else "—"
                ca_c = SUCCESS if (not pd.isna(va) and not pd.isna(vb) and va >= vb) else TEXT
                cb_c = SUCCESS if (not pd.isna(va) and not pd.isna(vb) and vb > va) else TEXT
                rows_html += f"""
                <div style="display:flex;align-items:center;padding:10px 0;border-bottom:1px solid {BORDER}">
                    <div style="flex:1;font-weight:700;color:{ca_c};text-align:center;font-family:'JetBrains Mono',monospace">{_fv(va)}</div>
                    <div style="flex:1.5;text-align:center;color:{MUTED};font-size:0.84rem">{label}</div>
                    <div style="flex:1;font-weight:700;color:{cb_c};text-align:center;font-family:'JetBrains Mono',monospace">{_fv(vb)}</div>
                </div>"""
            st.markdown(f"""
            <div style="background:linear-gradient(160deg,{CARD},{CARD_2});border:1px solid {BORDER};
                border-radius:22px;padding:20px 24px;">
                <div style="display:flex;padding-bottom:14px;border-bottom:1px solid {BORDER_2}">
                    <div style="flex:1;font-size:1.15rem;font-weight:900;color:{ACCENT};text-align:center">{t_a}</div>
                    <div style="flex:1.5;text-align:center;color:{MUTED_2};font-size:0.88rem;align-self:center">vs</div>
                    <div style="flex:1;font-size:1.15rem;font-weight:900;color:{SUCCESS};text-align:center">{t_b}</div>
                </div>
                {rows_html}
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            # Side by side shotmaps
            ev_a = league_event_df[league_event_df["corner_team"]==t_a]
            ev_b = league_event_df[league_event_df["corner_team"]==t_b]
            c1,c2 = st.columns(2)
            with c1:
                sa = ev_a.dropna(subset=["shot_location_x","shot_location_y"])
                if not sa.empty: st.plotly_chart(shotmap_figure(sa,"Taker",f"Shotmap — {t_a}"), use_container_width=True)
                else: empty_state(f"No shots — {t_a}")
            with c2:
                sb = ev_b.dropna(subset=["shot_location_x","shot_location_y"])
                if not sb.empty: st.plotly_chart(shotmap_figure(sb,"Taker",f"Shotmap — {t_b}"), use_container_width=True)
                else: empty_state(f"No shots — {t_b}")
            # Dual radar
            all_takers_sc = taker_summary_table(league_event_df)
            takers_a = [str(t) for t in taker_summary_table(ev_a)["Taker"].dropna().unique()]
            takers_b = [str(t) for t in taker_summary_table(ev_b)["Taker"].dropna().unique()]
            if takers_a and takers_b:
                section_header("Taker vs Taker Radar")
                cx,cy = st.columns(2)
                with cx: tk_a = st.selectbox("Taker (Team A)", takers_a, key="cmp_tka")
                with cy: tk_b = st.selectbox("Taker (Team B)", takers_b, key="cmp_tkb")
                c1,c2 = st.columns(2)
                with c1: st.plotly_chart(taker_radar(all_takers_sc, tk_a), use_container_width=True)
                with c2: st.plotly_chart(taker_radar(all_takers_sc, tk_b), use_container_width=True)

    with sc_tabs[3]:
        section_header("Zone Intelligence", "Team × delivery zone × end zone breakdown")
        zi = team_insight_table(league_event_df)
        if zi.empty: empty_state()
        else:
            st.dataframe(zi.reset_index(drop=True), use_container_width=True, height=500)
            pivot_z = zi.groupby(["corner_team","end_zone"])["corners"].sum().unstack(fill_value=0)
            if not pivot_z.empty:
                fig = px.imshow(pivot_z, aspect="auto", title="Team Zone Targeting Heatmap",
                    color_continuous_scale="Blues", text_auto=True,
                    labels=dict(x="End Zone",y="Team",color="Corners"))
                st.plotly_chart(figure_layout(fig, max(400,45*len(pivot_z))), use_container_width=True)
            # xg_per_corner by zone
            zi_xg = zi.groupby("end_zone")[["corners","shots","total_xg"]].sum().reset_index()
            zi_xg["xg_per_corner"] = zi_xg["total_xg"] / zi_xg["corners"].replace(0,np.nan)
            zi_xg["shot_rate"]     = zi_xg["shots"] / zi_xg["corners"].replace(0,np.nan)
            c1,c2 = st.columns(2)
            with c1:
                fig = px.bar(zi_xg.sort_values("xg_per_corner",ascending=False), x="end_zone", y="xg_per_corner",
                    color="xg_per_corner", color_continuous_scale="Reds",
                    title="xG/Corner by End Zone", labels={"end_zone":"","xg_per_corner":"xG/Corner"})
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(figure_layout(fig, 360), use_container_width=True)
            with c2:
                fig = px.bar(zi_xg.sort_values("shot_rate",ascending=False), x="end_zone", y="shot_rate",
                    color="shot_rate", color_continuous_scale="Greens",
                    title="Shot Rate by End Zone", labels={"end_zone":"","shot_rate":"Shot Rate"})
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(figure_layout(fig, 360), use_container_width=True)

    with sc_tabs[4]:
        section_header("Defensive Setup Analysis", "How each shape affects attacking efficiency")
        def_df = defensive_setup_table(league_event_df)
        if def_df.empty: empty_state("No defensive setup data.", "🛡")
        else:
            st.dataframe(def_df.reset_index(drop=True), use_container_width=True, height=380)
            c1,c2 = st.columns(2)
            with c1:
                fig = px.bar(def_df.sort_values("corners",ascending=False).head(15),
                    x="Defensive_setup", y="corners", title="Most Common Defensive Setups",
                    hover_data=["shots","total_xg","shot_rate"], color_discrete_sequence=[ACCENT],
                    labels={"Defensive_setup":"","corners":"Corners"})
                st.plotly_chart(figure_layout(fig, 380), use_container_width=True)
            with c2:
                fig = px.bar(def_df.sort_values("shot_rate",ascending=False).head(15),
                    x="Defensive_setup", y="shot_rate", title="Shot Rate Allowed by Setup",
                    color_discrete_sequence=[DANGER], labels={"Defensive_setup":"","shot_rate":"Shot Rate Allowed"})
                st.plotly_chart(figure_layout(fig, 380), use_container_width=True)
            c3,c4 = st.columns(2)
            with c3:
                st.plotly_chart(defensive_setup_scatter(def_df, "Setup: Sample Size vs Shot Rate Allowed"), use_container_width=True)
            with c4:
                fig = px.bar(def_df.sort_values("xg_per_corner",ascending=False).head(12),
                    x="Defensive_setup", y="xg_per_corner", color_discrete_sequence=[WARNING],
                    title="xG/Corner Allowed by Setup",
                    labels={"Defensive_setup":"","xg_per_corner":"xG/Corner"})
                st.plotly_chart(figure_layout(fig, 380), use_container_width=True)

    with sc_tabs[5]:
        section_header("Advanced Search", "Multi-filter event lookup")
        ca,cb,cc,cd = st.columns(4)
        teams_all = ["All"] + sorted(league_event_df["corner_team"].dropna().unique().tolist()) if not league_event_df.empty else ["All"]
        with ca: adv_team    = st.selectbox("Team",    teams_all, key="adv_t")
        with cb: adv_outcome = st.selectbox("Outcome", ["Any","Shot ≤3s","First Contact Shot","Shot"], key="adv_oc")
        with cc: adv_zone    = st.selectbox("End Zone", ["Any","6-yard box","Penalty area","Deep box","Outside danger zone"], key="adv_z")
        with cd: adv_tech    = st.selectbox("Technique", ["Any"] + sorted([str(t) for t in league_event_df["pass_technique"].dropna().unique()]) if not league_event_df.empty else ["Any"], key="adv_tech")
        s = league_event_df.copy()
        if adv_team != "All":    s = s[s["corner_team"]==adv_team]
        if adv_outcome != "Any": s = s[s["outcome_bucket"]==adv_outcome]
        if adv_zone != "Any":    s = s[s["end_zone"]==adv_zone]
        if adv_tech != "Any":    s = s[s["pass_technique"].astype(str)==adv_tech]
        st.markdown(f'<div style="color:{MUTED};font-size:0.88rem;margin-bottom:8px">{len(s)} events match.</div>', unsafe_allow_html=True)
        sc_show = [c for c in ["Match","corner_team","Taker","Shooter","Minute","SP_outcome","shot_xg",
            "pass_technique","delivery_zone","end_zone","outcome_bucket","xg_category","Defensive_setup",
            "corner_side","delivery_length_band"] if c in s.columns]
        st.dataframe(s[sc_show].sort_values("shot_xg", ascending=False).reset_index(drop=True),
            use_container_width=True, height=480)
        if not s.empty:
            st.download_button("⬇ Search Results CSV", s.to_csv(index=False).encode(),
                "advanced_search.csv", "text/csv")

    with sc_tabs[6]:
        section_header("Export Centre")
        st.markdown(f'<div class="sub-card">{len(league_event_df):,} events · {league_event_df["match_id"].nunique() if not league_event_df.empty else 0} matches · {league_event_df["corner_team"].nunique() if not league_event_df.empty else 0} teams</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        tl_exp = taker_summary_table(league_event_df)
        zi_exp = team_insight_table(league_event_df)
        add_download_buttons(league_event_df, league_team_df, league_match_df)
        wb = download_excel_workbook(league_event_df, league_team_df, league_match_df, tl_exp, zi_exp)
        if wb:
            st.download_button("⬇ Full Workbook (Excel)", wb,
                "allsvenskan_corners.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# =========================================================
# PAGE: TREND LAB
# =========================================================
elif page == "📈 Trend Lab":
    section_header("Trend Lab", "Rolling trends, match-to-match evolution, and longitudinal patterns.")
    tl_tabs = st.tabs(["📉 Rolling Rates","📊 Match-by-Match","🕐 Phase Trends","🧮 Statistical Tests"])

    with tl_tabs[0]:
        win_size = st.slider("Rolling window (matches)", 2, 10, 5, key="roll_win")
        st.plotly_chart(rolling_shot_rate(league_event_df, window=win_size,
            title=f"Rolling Shot Rate ({win_size}-match window)"), use_container_width=True)
        # Rolling xG
        if not league_event_df.empty:
            mxg = (league_event_df.groupby(["Match","corner_team"], dropna=False)
                .agg(total_xg=("shot_xg","sum"), corners=("match_id","size")).reset_index())
            mxg["xg_per_corner"] = mxg["total_xg"] / mxg["corners"].replace(0,np.nan)
            mxg = mxg.sort_values(["corner_team","Match"])
            fig_rxg = go.Figure()
            for team, grp in mxg.groupby("corner_team"):
                grp = grp.copy().reset_index(drop=True)
                grp["rolling"] = grp["xg_per_corner"].rolling(min(win_size,len(grp)), min_periods=1).mean()
                fig_rxg.add_trace(go.Scatter(x=grp["Match"], y=grp["rolling"],
                    mode="lines+markers", name=str(team), line=dict(width=2.2), marker=dict(size=5)))
            fig_rxg.update_layout(
                title=f"Rolling xG/Corner ({win_size}-match)", height=400,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=TEXT, family="DM Sans, sans-serif"),
                margin=dict(l=8,r=8,t=48,b=8),
                xaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.05)"),
                yaxis=dict(showgrid=True,gridcolor="rgba(255,255,255,0.05)"),
                hoverlabel=dict(bgcolor="#0d1c31",font_color=TEXT),
            )
            st.plotly_chart(fig_rxg, use_container_width=True)

    with tl_tabs[1]:
        if league_event_df.empty: empty_state()
        else:
            # Per-match corner volume trend
            mv = (league_event_df.groupby(["Match","corner_team"], dropna=False)
                .size().reset_index(name="corners").sort_values("Match"))
            fig = px.line(mv, x="Match", y="corners", color="corner_team", markers=True,
                title="Corners per Match by Team", color_discrete_sequence=QUAL_PALETTE,
                labels={"Match":"","corners":"Corners","corner_team":""})
            fig.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(figure_layout(fig, 400), use_container_width=True)
            # Shot rate per match
            ms = (league_event_df.groupby(["Match","corner_team"], dropna=False)
                .agg(corners=("match_id","size"), shots=("led_to_shot","sum")).reset_index())
            ms["shot_rate"] = ms["shots"] / ms["corners"].replace(0,np.nan)
            fig2 = px.line(ms, x="Match", y="shot_rate", color="corner_team", markers=True,
                title="Shot Rate per Match by Team", color_discrete_sequence=QUAL_PALETTE,
                labels={"Match":"","shot_rate":"Shot Rate","corner_team":""})
            fig2.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-30)
            st.plotly_chart(figure_layout(fig2, 400), use_container_width=True)

    with tl_tabs[2]:
        # Phase × match trend
        if league_event_df.empty: empty_state()
        else:
            ph_match = (league_event_df.groupby(["phase","corner_team"], dropna=False)
                .size().reset_index(name="corners"))
            fig = px.bar(ph_match, x="phase", y="corners", color="corner_team", barmode="group",
                title="Corners by Phase and Team", color_discrete_sequence=QUAL_PALETTE,
                labels={"phase":"","corners":"Corners","corner_team":""})
            st.plotly_chart(figure_layout(fig, 380), use_container_width=True)
            # Phase shot rate
            ph_sr = (league_event_df.groupby(["phase","corner_team"], dropna=False)
                .agg(corners=("match_id","size"), shots=("led_to_shot","sum")).reset_index())
            ph_sr["shot_rate"] = ph_sr["shots"] / ph_sr["corners"].replace(0,np.nan)
            fig2 = px.line(ph_sr, x="phase", y="shot_rate", color="corner_team", markers=True,
                title="Shot Rate Across Match Phases", color_discrete_sequence=QUAL_PALETTE,
                labels={"phase":"Phase","shot_rate":"Shot Rate","corner_team":""})
            fig2.update_layout(yaxis_tickformat=".0%")
            st.plotly_chart(figure_layout(fig2, 380), use_container_width=True)

    with tl_tabs[3]:
        section_header("Statistical Tests", "Hypothesis testing on corner outcome differences")
        if league_event_df.empty or league_team_df.empty or len(league_team_df) < 2:
            empty_state("Need at least 2 teams for statistical tests.")
        else:
            st.markdown("""
            <div class="sub-card">
            <b>Chi-square test: Inswinger vs Outswinger — outcome differences</b><br>
            Tests whether delivery type is associated with shot outcome.
            </div>""", unsafe_allow_html=True)
            ct_df = league_event_df[league_event_df["pass_technique"].astype(str).str.contains("swing|short",case=False,na=False)].copy()
            if len(ct_df) >= 10:
                ct_df["is_swing"] = np.where(ct_df["is_inswinger"],"Inswinger","Outswinger/Other")
                ct_df["shot_yn"]  = np.where(ct_df["led_to_shot"],"Shot","No Shot")
                cont = pd.crosstab(ct_df["is_swing"], ct_df["shot_yn"])
                st.dataframe(cont, use_container_width=False)
                try:
                    chi2, p, dof, exp = scipy_stats.chi2_contingency(cont)
                    sig = "✅ Statistically significant (p<0.05)" if p < 0.05 else "❌ Not significant (p≥0.05)"
                    st.markdown(f"""
                    <div class="insight-box {'insight-box-green' if p<0.05 else ''}">
                    <b>χ²={chi2:.3f} | p={p:.4f} | df={dof}</b><br>
                    {sig}
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Test failed: {e}")
            else:
                st.info("Not enough data for chi-square test.")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="sub-card">
            <b>Mann-Whitney U: Home vs Away xG differences</b><br>
            Tests whether corners at home generate more xG than away.
            </div>""", unsafe_allow_html=True)
            home_xg = league_event_df[league_event_df["venue_split"]=="Home"]["shot_xg"].dropna()
            away_xg = league_event_df[league_event_df["venue_split"]=="Away"]["shot_xg"].dropna()
            if len(home_xg) >= 5 and len(away_xg) >= 5:
                try:
                    u_stat, u_p = scipy_stats.mannwhitneyu(home_xg, away_xg, alternative="two-sided")
                    sig_u = "✅ Significant difference (p<0.05)" if u_p < 0.05 else "❌ No significant difference"
                    st.markdown(f"""
                    <div class="insight-box {'insight-box-green' if u_p<0.05 else ''}">
                    <b>Home mean xG: {home_xg.mean():.4f} | Away mean xG: {away_xg.mean():.4f}</b><br>
                    U={u_stat:.1f} | p={u_p:.4f}<br>{sig_u}
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Test failed: {e}")
            else:
                st.info("Not enough home/away xG data.")


# =========================================================
# PAGE: OPPOSITION INTEL
# =========================================================
elif page == "🛡 Opposition Intel":
    section_header("Opposition Intel", "Prepare against a specific opponent — their corner patterns, takers, and vulnerabilities.")
    opp_teams = sorted(league_event_df["corner_team"].dropna().unique().tolist()) if not league_event_df.empty else []
    if not opp_teams: empty_state("No team data available.", "🛡")
    else:
        opp = st.selectbox("Select Opposition", opp_teams, key="opp_select")
        opp_ev = league_event_df[league_event_df["corner_team"] == opp].copy()
        opp_row = league_team_df[league_team_df["team"] == opp].iloc[0] if not league_team_df[league_team_df["team"]==opp].empty else pd.Series()
        opp_takers = taker_summary_table(opp_ev)

        oi_tabs = st.tabs(["📋 Summary","🎯 Attack Map","👤 Key Takers","📐 Tendencies",
                            "⏱ When They Attack","📄 Scout Report"])

        with oi_tabs[0]:
            section_header(f"Scouting — {opp}", "Overview of their corner threat profile")
            if opp_ev.empty: empty_state("No data for this team.")
            else:
                render_kpis(opp_ev, league_match_df[league_match_df["match_id"].isin(opp_ev["match_id"].unique())])
                c1,c2,c3,c4 = st.columns(4)
                with c1: metric_card("Primary Taker",
                    str(opp_takers.iloc[0]["Taker"]) if not opp_takers.empty else "—",
                    foot="Most corners taken", accent_color=WARNING)
                with c2:
                    dom_tech = opp_ev["pass_technique"].value_counts()
                    metric_card("Dominant Technique",
                        str(dom_tech.index[0]) if not dom_tech.empty else "—",
                        foot="Most frequent delivery", accent_color=ORANGE)
                with c3:
                    dom_zone = opp_ev["end_zone"].value_counts()
                    metric_card("Target Zone",
                        str(dom_zone.index[0]) if not dom_zone.empty else "—",
                        foot="Most targeted area", accent_color=DANGER)
                with c4:
                    dom_side = opp_ev["corner_side"].value_counts()
                    metric_card("Favoured Corner Side",
                        str(dom_side.index[0]) if not dom_side.empty else "—",
                        foot="Left or right corner", accent_color=PURPLE)

        with oi_tabs[1]:
            if opp_ev.empty: empty_state()
            else:
                c1,c2 = st.columns(2)
                with c1:
                    s_opp = opp_ev.dropna(subset=["shot_location_x","shot_location_y"])
                    if not s_opp.empty:
                        st.plotly_chart(shotmap_figure(s_opp, "Taker", f"Shot Origins — {opp}"), use_container_width=True)
                    else: empty_state("No shot data.")
                with c2:
                    hd_opp = not opp_ev.dropna(subset=["pass_location_x","pass_location_y","pass_end_location_x","pass_end_location_y"]).empty
                    if hd_opp:
                        st.plotly_chart(delivery_map_figure(opp_ev, "delivery_zone", f"Delivery Patterns — {opp}"), use_container_width=True)
                    else: empty_state("No delivery data.")
                has_e_opp = not opp_ev.dropna(subset=["pass_end_location_x","pass_end_location_y"]).empty
                if has_e_opp:
                    c3,c4 = st.columns(2)
                    with c3:
                        st.plotly_chart(heatmap_density(opp_ev,"pass_end_location_x","pass_end_location_y",
                            f"Delivery End-Location — {opp}"), use_container_width=True)
                    with c4:
                        s2 = opp_ev.dropna(subset=["shot_location_x","shot_location_y"])
                        if not s2.empty: st.plotly_chart(xg_heatmap_pitch(s2, f"xG Heatmap — {opp}"), use_container_width=True)
                        else: empty_state()

        with oi_tabs[2]:
            section_header(f"Key Corner Takers — {opp}")
            if opp_takers.empty: empty_state("No taker data.")
            else:
                st.dataframe(opp_takers.reset_index(drop=True), use_container_width=True, height=400)
                if len(opp_takers) >= 1:
                    top_taker = str(opp_takers.iloc[0]["Taker"])
                    c1,c2 = st.columns([1.2,1])
                    with c1:
                        all_tl = taker_summary_table(league_event_df)
                        st.plotly_chart(taker_radar(all_tl, top_taker), use_container_width=True)
                    with c2:
                        tr = opp_takers[opp_takers["Taker"].astype(str)==top_taker].iloc[0]
                        metric_card("Corners Taken", f"{int(tr.get('corners',0)):,}", foot="Workload")
                        metric_card("Shot Rate",      human_pct(tr.get("shot_rate")),  foot="Danger rate")
                        metric_card("xG/Corner",      human_val(tr.get("xg_per_corner"),4), foot="Quality")
                        metric_card("Inswinger %",    human_pct(tr.get("inswinger_rate")), foot="Delivery preference")

        with oi_tabs[3]:
            if opp_ev.empty: empty_state()
            else:
                c1,c2 = st.columns(2)
                with c1:
                    td = opp_ev.groupby("pass_technique", dropna=False).size().reset_index(name="n")
                    fig = px.bar(td, x="pass_technique", y="n", color="pass_technique",
                        color_discrete_sequence=QUAL_PALETTE, title="Technique Preference",
                        labels={"pass_technique":"","n":"Corners"})
                    st.plotly_chart(figure_layout(fig, 360), use_container_width=True)
                with c2:
                    zd = opp_ev.groupby("end_zone", dropna=False).size().reset_index(name="n")
                    fig = px.bar(zd, x="end_zone", y="n", color="end_zone",
                        color_discrete_sequence=QUAL_PALETTE[2:], title="Zone Targeting",
                        labels={"end_zone":"","n":"Corners"})
                    st.plotly_chart(figure_layout(fig, 360), use_container_width=True)
                c3,c4 = st.columns(2)
                with c3:
                    side_d = opp_ev.groupby("corner_side", dropna=False).size().reset_index(name="n")
                    fig = px.pie(side_d, names="corner_side", values="n", title="Corner Side Preference", hole=0.54,
                                 color_discrete_sequence=QUAL_PALETTE)
                    st.plotly_chart(figure_layout(fig, 340, "Corner Side"), use_container_width=True)
                with c4:
                    ht_d = opp_ev.groupby("pass_height", dropna=False).size().reset_index(name="n")
                    fig = px.pie(ht_d, names="pass_height", values="n", title="Pass Height", hole=0.54,
                                 color_discrete_sequence=QUAL_PALETTE[3:])
                    st.plotly_chart(figure_layout(fig, 340, "Pass Height"), use_container_width=True)
                # technique × outcome
                to_df = opp_ev.groupby(["pass_technique","outcome_bucket"], dropna=False).size().reset_index(name="n")
                if not to_df.empty:
                    piv_to = to_df.pivot(index="pass_technique", columns="outcome_bucket", values="n").fillna(0)
                    fig = px.imshow(piv_to, aspect="auto", title="Technique × Outcome Heatmap",
                        color_continuous_scale="Blues", text_auto=True)
                    st.plotly_chart(figure_layout(fig, 360), use_container_width=True)

        with oi_tabs[4]:
            if opp_ev.empty: empty_state()
            else:
                st.plotly_chart(phase_heatmap(opp_ev, f"{opp} — Corner Phase Pattern"), use_container_width=True)
                c1,c2 = st.columns(2)
                with c1:
                    st.plotly_chart(minute_histogram(opp_ev, title=f"{opp} — Minute Distribution"), use_container_width=True)
                with c2:
                    st.plotly_chart(home_away_comparison(opp_ev, f"{opp} — Home vs Away"), use_container_width=True)

        with oi_tabs[5]:
            # Text scout report
            if not opp_row.empty and not opp_ev.empty:
                dom_tech = opp_ev["pass_technique"].value_counts()
                dom_zone = opp_ev["end_zone"].value_counts()
                dom_side = opp_ev["corner_side"].value_counts()
                top_tk   = opp_takers.iloc[0]["Taker"] if not opp_takers.empty else "Unknown"
                top_xg_tk = opp_takers.iloc[0]["xg_per_corner"] if not opp_takers.empty else 0
                inswing_pct = opp_row.get("inswinger_rate", 0) or 0
                short_pct   = opp_row.get("short_corner_rate", 0) or 0
                sr          = opp_row.get("shot_rate", 0) or 0
                xgpm        = opp_row.get("xg_per_match", 0) or 0

                report_html = f"""
                <div style="background:linear-gradient(160deg,{CARD},{CARD_2});border:1px solid {BORDER};
                     border-radius:22px;padding:24px 28px;line-height:1.8">
                    <div style="font-size:1.4rem;font-weight:900;margin-bottom:16px">
                        🔭 Scout Report — {opp}
                    </div>
                    <div style="color:{MUTED};font-size:0.8rem;margin-bottom:20px;font-style:italic">
                        Auto-generated from {len(opp_ev)} corner events across {opp_ev['match_id'].nunique()} matches.
                    </div>
                    <div style="margin-bottom:14px">
                        <b style="color:{ACCENT_2}">Corner Threat Level</b><br>
                        {opp} generates a shot rate of <b>{sr*100:.1f}%</b> and an average of 
                        <b>{xgpm:.3f} xG per match</b> from corners, making them 
                        {'a significant set piece threat' if sr > 0.15 else 'a moderate set piece threat' if sr > 0.08 else 'a lower set piece threat'}.
                    </div>
                    <div style="margin-bottom:14px">
                        <b style="color:{SUCCESS_2}">Delivery Profile</b><br>
                        Their dominant delivery technique is <b>{dom_tech.index[0] if not dom_tech.empty else 'Unknown'}</b>.
                        {'They favour inswinging deliveries (' + human_pct(inswing_pct) + ' of corners).' if inswing_pct > 0.3 else 'Inswingers are used ' + human_pct(inswing_pct) + ' of the time.'}
                        {'Short corners are a notable weapon at ' + human_pct(short_pct) + '.' if short_pct > 0.15 else ''}
                    </div>
                    <div style="margin-bottom:14px">
                        <b style="color:{WARNING_2}">Target Zones</b><br>
                        Primary delivery target: <b>{dom_zone.index[0] if not dom_zone.empty else 'Unknown'}</b>.
                        They predominantly take corners from the <b>{dom_side.index[0] if not dom_side.empty else 'Unknown'}</b> position.
                    </div>
                    <div style="margin-bottom:14px">
                        <b style="color:{DANGER_2}">Key Personnel</b><br>
                        Primary taker: <b>{top_tk}</b> (xG/corner: {top_xg_tk:.4f}).
                        They have used <b>{int(opp_row.get('taker_variety', 0))}</b> different takers across this dataset.
                    </div>
                    <div>
                        <b style="color:{PURPLE}">Defensive Recommendations</b><br>
                        {'Set a zonal defensive shape to cover the 6-yard box.' if (not dom_zone.empty and dom_zone.index[0]=='6-yard box') else ''}
                        {'Consider pressing the corner taker to reduce delivery quality.' if sr > 0.15 else 'Standard defensive shape should suffice given lower shot rate.'}
                        {'Prepare specifically for short corner routines.' if short_pct > 0.15 else ''}
                    </div>
                </div>"""
                st.markdown(report_html, unsafe_allow_html=True)
            else:
                empty_state("Not enough data to generate scout report.", "📄")


# =========================================================
# PAGE: ADVANCED ANALYTICS
# =========================================================
elif page == "📐 Advanced Analytics":
    section_header("Advanced Analytics", "Correlation analysis, regression, cross-dimensional breakdowns, and experimental metrics.")
    aa_tabs = st.tabs(["📊 Correlation Matrix","📐 Regression Analysis","🧩 Cross-Dimensional",
                        "🏹 Delivery Efficiency","⚡ High-Danger Events","📊 Expected vs Actual"])

    with aa_tabs[0]:
        section_header("Team Metric Correlation Matrix")
        st.plotly_chart(correlation_heatmap_teams(league_team_df,
            title="Team Metric Correlation Matrix (Pearson r)", h=560), use_container_width=True)
        if not league_team_df.empty:
            section_header("Top Correlations", "Which metrics move together?")
            rcols = [c for c in ["corners_per_match","shot_rate","xg_per_match","fast_shot_rate",
                "six_yard_delivery_rate","short_corner_rate","inswinger_rate","penalty_area_delivery_rate","taker_variety"]
                if c in league_team_df.columns]
            if len(rcols) >= 2:
                corr = league_team_df[rcols].corr()
                pairs = []
                for i in range(len(rcols)):
                    for j in range(i+1, len(rcols)):
                        pairs.append({"Metric A": rcols[i], "Metric B": rcols[j], "Pearson r": round(corr.iloc[i,j], 4)})
                pairs_df = pd.DataFrame(pairs).sort_values("Pearson r", key=abs, ascending=False)
                st.dataframe(pairs_df.reset_index(drop=True), use_container_width=True, height=380)

    with aa_tabs[1]:
        section_header("Regression Analysis", "Predict shot outcomes from delivery features")
        if league_event_df.empty: empty_state()
        else:
            x_col = st.selectbox("X variable", ["delivery_length","Minute","pass_location_x","pass_location_y",
                "pass_end_location_x","pass_end_location_y"], key="reg_x")
            y_col = st.selectbox("Y variable", ["shot_xg","led_to_shot","is_fast_shot","is_six_yard_delivery"], key="reg_y")
            color_c = st.selectbox("Color by", ["corner_team","pass_technique","end_zone","corner_side"], key="reg_c")
            plot_df = league_event_df.dropna(subset=[x_col]).copy()
            plot_df["_y"] = pd.to_numeric(plot_df[y_col].astype(str).str.replace("True","1").str.replace("False","0"), errors="coerce")
            plot_df = plot_df.dropna(subset=["_y"])
            if not plot_df.empty:
                fig = px.scatter(plot_df, x=x_col, y="_y", color=color_c,
                    trendline="lowess", opacity=0.55,
                    color_discrete_sequence=QUAL_PALETTE,
                    title=f"{y_col} vs {x_col}",
                    labels={x_col:x_col,"_y":y_col})
                st.plotly_chart(figure_layout(fig, 450), use_container_width=True)
            else: empty_state("Not enough numeric data for this combination.")

    with aa_tabs[2]:
        section_header("Cross-Dimensional Heatmaps")
        dim1 = st.selectbox("Row dimension", ["corner_team","pass_technique","end_zone","corner_side","phase","delivery_type"], key="cd1")
        dim2 = st.selectbox("Col dimension", ["outcome_bucket","end_zone","pass_technique","phase","delivery_length_band"], key="cd2")
        if not league_event_df.empty:
            cross = league_event_df.groupby([dim1, dim2], dropna=False).size().reset_index(name="n")
            if not cross.empty:
                piv = cross.pivot(index=dim1, columns=dim2, values="n").fillna(0)
                fig = px.imshow(piv, aspect="auto", title=f"{dim1} × {dim2} Heatmap",
                    color_continuous_scale="Blues", text_auto=True,
                    labels=dict(x=dim2, y=dim1, color="Count"))
                st.plotly_chart(figure_layout(fig, max(420,45*len(piv))), use_container_width=True)
            else: empty_state()
        else: empty_state()

    with aa_tabs[3]:
        section_header("Delivery Efficiency Matrix", "Which delivery types to which zones produce best outcomes")
        if not league_event_df.empty:
            dm = (league_event_df.groupby(["delivery_type","end_zone"], dropna=False)
                .agg(corners=("match_id","size"), shots=("led_to_shot","sum"), total_xg=("shot_xg","sum")).reset_index())
            dm["xg_per_corner"] = dm["total_xg"] / dm["corners"].replace(0,np.nan)
            dm["shot_rate"]     = dm["shots"] / dm["corners"].replace(0,np.nan)
            st.dataframe(dm.sort_values("xg_per_corner", ascending=False).reset_index(drop=True),
                use_container_width=True, height=380)
            piv_de = dm.pivot_table(index="delivery_type", columns="end_zone", values="xg_per_corner").fillna(0)
            if not piv_de.empty:
                fig = px.imshow(piv_de, aspect="auto", title="xG/Corner: Delivery Type × End Zone",
                    color_continuous_scale="YlOrRd", text_auto=".3f",
                    labels=dict(x="End Zone",y="Delivery Type",color="xG/Corner"))
                st.plotly_chart(figure_layout(fig, max(380, 50*len(piv_de))), use_container_width=True)
        else: empty_state()

    with aa_tabs[4]:
        section_header("High-Danger Events", "Corners resulting in ≥0.10 xG shots")
        if league_event_df.empty: empty_state()
        else:
            hd = league_event_df[league_event_df["shot_xg"].fillna(0) >= 0.10].copy()
            st.markdown(f'<div style="color:{MUTED};font-size:0.9rem;margin-bottom:8px">{len(hd)} high-danger events (xG≥0.10)</div>', unsafe_allow_html=True)
            if hd.empty: empty_state("No high-danger events in current filter.")
            else:
                c1,c2 = st.columns(2)
                with c1: st.plotly_chart(shotmap_figure(hd.dropna(subset=["shot_location_x","shot_location_y"]),
                    "corner_team", "High-Danger Shot Locations (xG≥0.10)"), use_container_width=True)
                with c2:
                    team_hd = hd.groupby("corner_team", dropna=False).agg(
                        hd_shots=("match_id","size"), total_xg=("shot_xg","sum")).reset_index()
                    fig = px.bar(team_hd.sort_values("total_xg",ascending=False),
                        x="corner_team", y="total_xg", color_discrete_sequence=[DANGER],
                        title="High-Danger xG by Team",
                        labels={"corner_team":"","total_xg":"xG"})
                    st.plotly_chart(figure_layout(fig, 400), use_container_width=True)
                show_cols = [c for c in ["Match","corner_team","Taker","Shooter","Minute","shot_xg",
                    "pass_technique","end_zone","outcome_bucket","shot_outcome"] if c in hd.columns]
                st.dataframe(hd[show_cols].sort_values("shot_xg",ascending=False).reset_index(drop=True),
                    use_container_width=True, height=400)
                st.download_button("⬇ High-Danger Events CSV", hd.to_csv(index=False).encode(),
                    "high_danger_events.csv","text/csv")

    with aa_tabs[5]:
        section_header("Expected vs Actual Shots", "Per team: expected shot count vs actual shots taken")
        if league_team_df.empty or league_event_df.empty: empty_state()
        else:
            overall_sr = league_event_df["led_to_shot"].mean()
            ev_summary = (league_event_df.groupby("corner_team", dropna=False).agg(
                corners=("match_id","size"), actual_shots=("led_to_shot","sum")).reset_index())
            ev_summary["expected_shots"] = ev_summary["corners"] * overall_sr
            ev_summary["delta"] = ev_summary["actual_shots"] - ev_summary["expected_shots"]
            ev_summary["overperform"] = ev_summary["delta"] > 0
            fig = px.bar(ev_summary.sort_values("delta",ascending=False),
                x="corner_team", y="delta",
                color="overperform", color_discrete_map={True:SUCCESS, False:DANGER},
                title=f"Actual Shots minus Expected (league avg SR={overall_sr:.2%})",
                labels={"corner_team":"","delta":"Shots above/below expected"})
            fig.add_hline(y=0, line_color="rgba(255,255,255,0.4)", line_dash="dot")
            st.plotly_chart(figure_layout(fig, 400), use_container_width=True)
            st.dataframe(ev_summary.sort_values("delta",ascending=False).reset_index(drop=True),
                use_container_width=True, height=380)


# =========================================================
# PAGE: DATA HUB
# =========================================================
elif page == "🗂 Data Hub":
    section_header("Data Hub", "Clean tables, summaries, and downloadable outputs. Use Visualisation Studio for charts.")
    dh_tabs = st.tabs(["📄 Raw Events","🏟 Team Table","📋 Match Table","🎯 Shot Events",
                         "🏹 Delivery Events","👤 Taker Summary","📍 Zone Table","🛡 Defensive Table",
                         "🔍 Custom Query","⬇ Downloads"])

    with dh_tabs[0]:
        section_header("Raw Events", f"{len(league_event_df):,} events in current filter")
        st.dataframe(league_event_df.reset_index(drop=True), use_container_width=True, height=600)
    with dh_tabs[1]:
        section_header("Team Summary")
        st.dataframe(league_team_df.reset_index(drop=True), use_container_width=True, height=600)
    with dh_tabs[2]:
        section_header("Match Table")
        st.dataframe(league_match_df.reset_index(drop=True), use_container_width=True, height=600)
    with dh_tabs[3]:
        shot_only = league_event_df[league_event_df["led_to_shot"]].reset_index(drop=True) if not league_event_df.empty else pd.DataFrame()
        section_header("Shot Events", f"{len(shot_only)} events")
        st.dataframe(shot_only, use_container_width=True, height=600)
    with dh_tabs[4]:
        del_cols = [c for c in ["Match","corner_team","Taker","Minute","pass_technique","pass_height",
            "pass_body_part","pass_location_x","pass_location_y","pass_end_location_x","pass_end_location_y",
            "delivery_zone","end_zone","corner_side","delivery_length","delivery_length_band",
            "SP_outcome","outcome_bucket"] if c in league_event_df.columns]
        del_only = league_event_df.dropna(subset=["pass_location_x","pass_location_y"], how="all") if not league_event_df.empty else pd.DataFrame()
        section_header("Delivery Events", f"{len(del_only)} events with coordinate data")
        st.dataframe(del_only[del_cols].reset_index(drop=True), use_container_width=True, height=600)
    with dh_tabs[5]:
        tl2 = taker_summary_table(league_event_df)
        section_header("Taker Summary", f"{len(tl2)} takers")
        st.dataframe(tl2.reset_index(drop=True), use_container_width=True, height=600)
    with dh_tabs[6]:
        zi2 = team_insight_table(league_event_df)
        section_header("Zone Table")
        st.dataframe(zi2.reset_index(drop=True), use_container_width=True, height=600)
    with dh_tabs[7]:
        def2 = defensive_setup_table(league_event_df)
        section_header("Defensive Setup Table")
        if def2.empty: empty_state("No defensive setup data.")
        else: st.dataframe(def2.reset_index(drop=True), use_container_width=True, height=600)
    with dh_tabs[8]:
        section_header("Custom Query", "Filter any column by value")
        if not league_event_df.empty:
            query_col = st.selectbox("Filter column", sorted(league_event_df.columns.tolist()), key="qcol")
            unique_vals = sorted([str(v) for v in league_event_df[query_col].dropna().unique()[:100]])
            query_val  = st.selectbox("Value", ["All"] + unique_vals, key="qval")
            if query_val != "All":
                qdf = league_event_df[league_event_df[query_col].astype(str) == query_val]
            else:
                qdf = league_event_df.copy()
            st.markdown(f'<div style="color:{MUTED};font-size:0.88rem;margin-bottom:8px">{len(qdf)} events.</div>', unsafe_allow_html=True)
            st.dataframe(qdf.reset_index(drop=True), use_container_width=True, height=480)
            if not qdf.empty:
                st.download_button("⬇ Custom Query CSV", qdf.to_csv(index=False).encode(),
                    f"custom_query_{query_col}_{query_val}.csv","text/csv")
        else: empty_state()
    with dh_tabs[9]:
        section_header("Downloads")
        add_download_buttons(league_event_df, league_team_df, league_match_df)
        tl3 = taker_summary_table(league_event_df)
        zi3 = team_insight_table(league_event_df)
        wb2 = download_excel_workbook(league_event_df, league_team_df, league_match_df, tl3, zi3)
        if wb2:
            st.download_button("⬇ Full Workbook (Excel — 5 sheets)", wb2,
                "allsvenskan_corners_full.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        else:
            st.warning("Excel export requires openpyxl. Run: pip install openpyxl")
        st.markdown(f"""
        <div class="sub-card" style="margin-top:16px">
            <b>Current filter scope</b><br>
            <span style="color:{MUTED};font-size:0.9rem">
                {len(league_event_df):,} events &middot;
                {league_event_df['match_id'].nunique() if not league_event_df.empty else 0} matches &middot;
                {league_event_df['corner_team'].nunique() if not league_event_df.empty else 0} teams &middot;
                {league_event_df['Taker'].nunique() if not league_event_df.empty else 0} takers
            </span>
        </div>""", unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown(f"""
<div class="footer-note">
    ⚽ Allsvenskan Set Piece Studio Pro · Corner Analytics Platform · 2025 Season ·
    Built with Streamlit + Plotly · scipy {scipy_stats.__module__}
</div>""", unsafe_allow_html=True)
