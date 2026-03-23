import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from io import BytesIO

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
BG = "#07111f"
BG_2 = "#0b1730"
CARD = "#101a2b"
CARD_2 = "#16243a"
BORDER = "rgba(255,255,255,0.08)"
TEXT = "#f3f7fc"
MUTED = "#99adc7"
ACCENT = "#5da8ff"
ACCENT_2 = "#8ad6ff"
SUCCESS = "#34d399"
WARNING = "#fbbf24"
DANGER = "#fb7185"
PITCH = "#133d24"
PITCH_LINE = "rgba(255,255,255,0.60)"

px.defaults.template = "plotly_dark"

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    * {{ font-family: 'Inter', sans-serif; }}

    .stApp {{
        background:
            radial-gradient(circle at top right, rgba(93,168,255,0.12), transparent 24%),
            radial-gradient(circle at left top, rgba(52,211,153,0.08), transparent 20%),
            linear-gradient(180deg, {BG} 0%, {BG_2} 100%);
        color: {TEXT};
    }}

    .block-container {{
        max-width: 1680px;
        padding-top: 1rem;
        padding-bottom: 1.2rem;
    }}

    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0b1526 0%, #08111e 100%);
        border-right: 1px solid {BORDER};
    }}

    [data-testid="stSidebarNav"] {{
        display: none;
    }}

    .hero-wrap {{
        background: linear-gradient(135deg, rgba(93,168,255,0.18), rgba(93,168,255,0.06));
        border: 1px solid rgba(93,168,255,0.20);
        border-radius: 26px;
        padding: 22px 24px 18px 24px;
        margin-bottom: 14px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.18);
    }}

    .hero-title {{
        font-size: 2.15rem;
        font-weight: 850;
        line-height: 1.02;
        margin-bottom: 0.3rem;
        color: {TEXT};
    }}

    .hero-sub {{
        color: {MUTED};
        font-size: 0.98rem;
        max-width: 980px;
    }}

    .pill {{
        display: inline-block;
        padding: 0.32rem 0.72rem;
        border-radius: 999px;
        background: rgba(93,168,255,0.14);
        color: #dcecff;
        border: 1px solid rgba(93,168,255,0.18);
        font-size: 0.78rem;
        margin-right: 0.45rem;
        margin-top: 0.55rem;
    }}

    .kpi-card {{
        background: linear-gradient(180deg, {CARD} 0%, {CARD_2} 100%);
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 16px 16px 14px 16px;
        min-height: 108px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.16);
    }}

    .kpi-label {{
        color: {MUTED};
        text-transform: uppercase;
        font-size: 0.73rem;
        letter-spacing: 0.09em;
        margin-bottom: 7px;
    }}

    .kpi-value {{
        color: {TEXT};
        font-weight: 850;
        font-size: 1.82rem;
        line-height: 1.02;
    }}

    .kpi-foot {{
        margin-top: 8px;
        color: {MUTED};
        font-size: 0.82rem;
    }}

    .kpi-delta-pos {{
        color: {SUCCESS};
        font-size: 0.8rem;
        margin-top: 4px;
    }}

    .kpi-delta-neg {{
        color: {DANGER};
        font-size: 0.8rem;
        margin-top: 4px;
    }}

    .section-title {{
        font-size: 1.12rem;
        font-weight: 820;
        margin: 0.15rem 0 0.20rem 0;
    }}

    .section-sub {{
        color: {MUTED};
        font-size: 0.92rem;
        margin-bottom: 0.8rem;
    }}

    .insight-box {{
        background: linear-gradient(180deg, rgba(93,168,255,0.10), rgba(93,168,255,0.05));
        border: 1px solid rgba(93,168,255,0.16);
        border-radius: 18px;
        padding: 14px 16px;
        margin-bottom: 8px;
        min-height: 72px;
    }}

    .insight-box-warn {{
        background: linear-gradient(180deg, rgba(251,191,36,0.10), rgba(251,191,36,0.05));
        border: 1px solid rgba(251,191,36,0.22);
        border-radius: 18px;
        padding: 14px 16px;
        margin-bottom: 8px;
        min-height: 72px;
    }}

    .insight-box-success {{
        background: linear-gradient(180deg, rgba(52,211,153,0.10), rgba(52,211,153,0.05));
        border: 1px solid rgba(52,211,153,0.22);
        border-radius: 18px;
        padding: 14px 16px;
        margin-bottom: 8px;
        min-height: 72px;
    }}

    .sub-card {{
        background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 12px 14px;
    }}

    .report-card-row {{
        display: flex;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid {BORDER};
    }}

    .report-card-metric {{
        flex: 1;
        color: {MUTED};
        font-size: 0.88rem;
    }}

    .report-card-value {{
        flex: 0.5;
        font-weight: 700;
        font-size: 0.95rem;
        color: {TEXT};
        text-align: right;
    }}

    .percentile-bar-wrap {{
        flex: 2;
        padding: 0 14px;
    }}

    .percentile-bar-bg {{
        height: 7px;
        background: rgba(255,255,255,0.08);
        border-radius: 99px;
        overflow: hidden;
    }}

    .pct-label {{
        font-size: 0.78rem;
        color: {MUTED};
        text-align: right;
    }}

    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 18px;
        overflow: hidden;
        box-shadow: 0 8px 28px rgba(0,0,0,0.14);
    }}

    .footer-note {{
        color: {MUTED};
        font-size: 0.84rem;
        margin-top: 0.6rem;
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        background: rgba(255,255,255,0.03);
        border-radius: 14px;
        padding: 4px;
    }}

    .stTabs [data-baseweb="tab"] {{
        border-radius: 10px;
        padding: 6px 16px;
        font-size: 0.88rem;
    }}

    .stTabs [aria-selected="true"] {{
        background: rgba(93,168,255,0.18) !important;
    }}

    div[data-testid="stSelectbox"] label,
    div[data-testid="stMultiSelect"] label,
    div[data-testid="stSlider"] label {{
        color: {MUTED};
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    .stDownloadButton button {{
        background: linear-gradient(135deg, rgba(93,168,255,0.20), rgba(93,168,255,0.10)) !important;
        border: 1px solid rgba(93,168,255,0.28) !important;
        color: {TEXT} !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
    }}

    .stDownloadButton button:hover {{
        background: linear-gradient(135deg, rgba(93,168,255,0.30), rgba(93,168,255,0.18)) !important;
    }}

    .empty-state {{
        text-align: center;
        padding: 48px 24px;
        color: {MUTED};
        font-size: 0.95rem;
    }}

    .compare-header {{
        background: rgba(255,255,255,0.03);
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 12px 16px;
        margin-bottom: 12px;
        font-weight: 700;
        font-size: 1.05rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOGIN
# =========================================================
def login_screen():
    st.markdown(
        '<div class="hero-wrap">'
        '<div class="hero-title">⚽ Allsvenskan Set Piece Studio Pro</div>'
        '<div class="hero-sub">A premium corner analytics workspace with executive summaries, scouting tools, visual diagnostics, and exportable data views.</div>'
        '<div><span class="pill">Studio UX</span><span class="pill">Data + Visual Split</span><span class="pill">Scouting Flow</span><span class="pill">Export Ready</span></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        st.markdown('<div class="sub-card" style="padding:28px 32px; margin-top: 24px;">', unsafe_allow_html=True)
        st.markdown("#### Sign in to Studio")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Name", placeholder="Enter your name")
            password = st.text_input("Password", type="password", placeholder="Enter password")
            submitted = st.form_submit_button("Sign In →", use_container_width=True)
            if submitted:
                if username == LOGIN_NAME and password == LOGIN_PASSWORD:
                    st.session_state["authenticated"] = True
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please try again.")
        st.markdown("</div>", unsafe_allow_html=True)


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_screen()
    st.stop()

# =========================================================
# HELPERS
# =========================================================
def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def pct(numerator, denominator):
    if denominator in [0, None] or pd.isna(denominator):
        return np.nan
    return numerator / denominator


def metric_card(label, value, suffix="", foot="", delta=None):
    delta_html = ""
    if delta is not None:
        cls = "kpi-delta-pos" if delta >= 0 else "kpi-delta-neg"
        sign = "▲" if delta >= 0 else "▼"
        delta_html = f'<div class="{cls}">{sign} {abs(delta):.2f} vs league avg</div>'
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}{suffix}</div>
            <div class="kpi-foot">{foot}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(title, subtitle=""):
    st.markdown(
        f'<div class="section-title">{title}</div>'
        + (f'<div class="section-sub">{subtitle}</div>' if subtitle else ""),
        unsafe_allow_html=True,
    )


def insight_box(title, body, variant="default"):
    css_class = {"default": "insight-box", "warn": "insight-box-warn", "success": "insight-box-success"}.get(variant, "insight-box")
    st.markdown(
        f'<div class="{css_class}"><b>{title}</b><br><span style="font-size:0.9rem">{body}</span></div>',
        unsafe_allow_html=True,
    )


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


def split_match_name(match_value):
    if not isinstance(match_value, str):
        return None, None
    for sep in [" - ", " vs ", " v "]:
        if sep in match_value:
            left, right = match_value.split(sep, 1)
            return left.strip(), right.strip()
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


def percentile_rank(series, value):
    s = series.dropna()
    if len(s) == 0 or pd.isna(value):
        return np.nan
    return float((s <= value).mean() * 100)


def left_right_from_y(y):
    if pd.isna(y):
        return "Unknown"
    if y < 30:
        return "Near Post Zone"
    if y <= 50:
        return "Central Zone"
    return "Far Post Zone"


def corner_side_from_start_y(y):
    if pd.isna(y):
        return "Unknown"
    return "Right Corner" if y < 40 else "Left Corner"


def delivery_length(x_start, y_start, x_end, y_end):
    if pd.isna(x_start) or pd.isna(y_start) or pd.isna(x_end) or pd.isna(y_end):
        return np.nan
    return float(np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2))


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


def human_pct(v):
    return "-" if pd.isna(v) else f"{v * 100:.1f}%"


def human_val(v, decimals=2):
    return "-" if pd.isna(v) else f"{v:.{decimals}f}"


def color_for_percentile(p):
    if pd.isna(p):
        return MUTED
    if p >= 80:
        return SUCCESS
    if p >= 50:
        return ACCENT
    if p >= 25:
        return WARNING
    return DANGER


def render_kpis(events, matches):
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        metric_card("Corner Events", f"{len(events):,}", foot="Filtered corner actions")
    with c2:
        n_matches = events["match_id"].nunique() if not events.empty else 0
        metric_card("Matches", f"{n_matches:,}", foot="Unique matches in view")
    with c3:
        avg_c = matches["total_corners"].mean() if not matches.empty else 0
        metric_card("Avg Corners / Match", f"{avg_c:.2f}", foot="Volume benchmark")
    with c4:
        shots = int(events["led_to_shot"].sum()) if not events.empty else 0
        metric_card("Shot Outcomes", f"{shots:,}", foot="Corners leading to shots")
    with c5:
        total_xg = events["shot_xg"].fillna(0).sum() if not events.empty else 0
        metric_card("Total xG", f"{total_xg:.2f}", foot="Shot xG generated")
    with c6:
        shot_rate = (events["led_to_shot"].mean() * 100) if len(events) > 0 else 0
        metric_card("Shot Rate", f"{shot_rate:.1f}", "%", foot="Shots per corner")


def figure_layout(fig, height=420, title=None):
    fig.update_layout(
        height=height,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=48 if title else 10, b=8),
        legend_title_text="",
        font=dict(color=TEXT, family="Inter, sans-serif"),
        hoverlabel=dict(bgcolor="#0f172a", font_color=TEXT, font_family="Inter, sans-serif"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


def draw_pitch(fig, title=None, height=560):
    fig.update_xaxes(range=[0, 120], visible=False)
    fig.update_yaxes(range=[0, 80], visible=False, scaleanchor="x", scaleratio=1)
    fig.update_layout(
        title=title,
        paper_bgcolor=PITCH,
        plot_bgcolor=PITCH,
        margin=dict(l=10, r=10, t=45 if title else 10, b=10),
        height=height,
        shapes=[
            # Outer boundary
            dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color=PITCH_LINE, width=2)),
            # Halfway line
            dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color=PITCH_LINE, width=2)),
            # Centre circle
            dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color=PITCH_LINE, width=2)),
            # Left penalty area
            dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color=PITCH_LINE, width=2)),
            # Left 6-yard box
            dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color=PITCH_LINE, width=2)),
            # Right penalty area
            dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color=PITCH_LINE, width=2)),
            # Right 6-yard box
            dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color=PITCH_LINE, width=2)),
            # Left penalty spot
            dict(type="circle", x0=10, y0=38, x1=14, y1=42, line=dict(color=PITCH_LINE, width=2)),
            # Right penalty spot
            dict(type="circle", x0=106, y0=38, x1=110, y1=42, line=dict(color=PITCH_LINE, width=2)),
            # Left penalty arc
            dict(type="path", path="M 18 33 Q 28 40 18 47", line=dict(color=PITCH_LINE, width=2)),
            # Right penalty arc
            dict(type="path", path="M 102 33 Q 92 40 102 47", line=dict(color=PITCH_LINE, width=2)),
            # Corner arcs
            dict(type="path", path="M 0 3 Q 3 0 3 0", line=dict(color=PITCH_LINE, width=1.5)),
            dict(type="path", path="M 0 77 Q 3 80 3 80", line=dict(color=PITCH_LINE, width=1.5)),
            dict(type="path", path="M 120 3 Q 117 0 117 0", line=dict(color=PITCH_LINE, width=1.5)),
            dict(type="path", path="M 120 77 Q 117 80 117 80", line=dict(color=PITCH_LINE, width=1.5)),
        ],
        legend_title_text="",
        font=dict(color="white", family="Inter, sans-serif"),
    )
    return fig


def shotmap_figure(df_shots, color_col="corner_team", title="Shotmap"):
    fig = draw_pitch(go.Figure(), title=title)
    if df_shots.empty:
        return fig
    plot_df = df_shots.copy()
    plot_df["shot_xg_plot"] = plot_df["shot_xg"].fillna(0)
    plot_df["marker_size"] = np.clip(plot_df["shot_xg_plot"] * 95 + 10, 10, 36)
    categories = plot_df[color_col].fillna("Unknown").astype(str).unique().tolist()
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Bold + px.colors.qualitative.Safe
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}
    for cat in categories:
        sub = plot_df[plot_df[color_col].fillna("Unknown").astype(str) == cat]
        fig.add_trace(
            go.Scatter(
                x=sub["shot_location_x"],
                y=sub["shot_location_y"],
                mode="markers",
                name=str(cat),
                marker=dict(
                    size=sub["marker_size"],
                    color=color_map[cat],
                    opacity=0.9,
                    line=dict(color="white", width=1),
                ),
                text=[
                    f"<b>{row.get('Match', '')}</b><br>"
                    f"Team: {row.get('corner_team', '')}<br>"
                    f"Taker: {row.get('Taker', '')}<br>"
                    f"Shooter: {row.get('Shooter', '')}<br>"
                    f"Body Part: {row.get('shot_body_part', '')}<br>"
                    f"Outcome: {row.get('SP_outcome', '')}<br>"
                    f"Shot Result: {row.get('shot_outcome', '')}<br>"
                    f"xG: {0 if pd.isna(row.get('shot_xg')) else row.get('shot_xg', 0):.3f}<br>"
                    f"Minute: {int(row['Minute']) if pd.notna(row.get('Minute')) else ''}"
                    for _, row in sub.iterrows()
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    return fig


def delivery_map_figure(df_events, color_col="delivery_zone", title="Delivery Map"):
    fig = draw_pitch(go.Figure(), title=title, height=620)
    if df_events.empty:
        return fig
    plot_df = df_events.dropna(
        subset=["pass_location_x", "pass_location_y", "pass_end_location_x", "pass_end_location_y"]
    ).copy()
    if plot_df.empty:
        return fig
    categories = plot_df[color_col].fillna("Unknown").astype(str).unique().tolist()
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Safe
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}
    legend_added = set()
    for _, row in plot_df.iterrows():
        cat = str(row[color_col]) if pd.notna(row[color_col]) else "Unknown"
        show_legend = cat not in legend_added
        if show_legend:
            legend_added.add(cat)
        fig.add_trace(
            go.Scatter(
                x=[row["pass_location_x"], row["pass_end_location_x"]],
                y=[row["pass_location_y"], row["pass_end_location_y"]],
                mode="lines+markers",
                line=dict(color=color_map[cat], width=2.3),
                marker=dict(size=[6, 9], color=[color_map[cat], "white"], opacity=[0.8, 1.0]),
                name=str(cat),
                legendgroup=str(cat),
                showlegend=show_legend,
                text=(
                    f"<b>{row.get('Match', '')}</b><br>"
                    f"Team: {row.get('corner_team', '')}<br>"
                    f"Taker: {row.get('Taker', '')}<br>"
                    f"Technique: {row.get('pass_technique', '')}<br>"
                    f"Height: {row.get('pass_height', '')}<br>"
                    f"Body Part: {row.get('pass_body_part', '')}<br>"
                    f"Delivery Zone: {row.get('delivery_zone', '')}<br>"
                    f"End Zone: {row.get('end_zone', '')}<br>"
                    f"Corner Side: {row.get('corner_side', '')}<br>"
                    f"Outcome: {row.get('SP_outcome', '')}<br>"
                    f"Minute: {int(row['Minute']) if pd.notna(row.get('Minute')) else ''}"
                ),
                hovertemplate="%{text}<extra></extra>",
            )
        )
    return fig


def delivery_end_heatmap(df_events, title="Delivery End Locations"):
    plot_df = df_events.dropna(subset=["pass_end_location_x", "pass_end_location_y"]).copy()
    if plot_df.empty:
        return go.Figure()
    fig = px.density_heatmap(
        plot_df,
        x="pass_end_location_x",
        y="pass_end_location_y",
        nbinsx=18,
        nbinsy=14,
        title=title,
        color_continuous_scale="Blues",
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[0, 80])
    fig.update_xaxes(range=[0, 120])
    return figure_layout(fig, height=520, title=title)


def outcome_pie_figure(df_events, title="Outcome Split"):
    if df_events.empty:
        return go.Figure()
    summary = df_events.groupby("outcome_bucket", dropna=False).size().reset_index(name="corners")
    fig = px.pie(
        summary,
        names="outcome_bucket",
        values="corners",
        title=title,
        hole=0.58,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return figure_layout(fig, height=380, title=title)


def technique_pie_figure(df_events, title="Delivery Technique Split"):
    if df_events.empty:
        return go.Figure()
    summary = df_events.groupby("pass_technique", dropna=False).size().reset_index(name="corners")
    fig = px.pie(
        summary,
        names="pass_technique",
        values="corners",
        title=title,
        hole=0.58,
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig.update_traces(textposition="outside", textinfo="percent+label")
    return figure_layout(fig, height=380, title=title)


def cumulative_timeline_figure(df_events, color_col="corner_team", title="Cumulative Corners Over Time"):
    if df_events.empty:
        return go.Figure()
    base = (
        df_events.groupby(["Minute", color_col], dropna=False)
        .size()
        .reset_index(name="corners")
        .sort_values([color_col, "Minute"])
    )
    base["cumulative_corners"] = base.groupby(color_col)["corners"].cumsum()
    fig = px.line(base, x="Minute", y="cumulative_corners", color=color_col, markers=True, title=title)
    return figure_layout(fig, height=400, title=title)


def team_scatter_figure(team_df, x_col, y_col, size_col, title):
    if team_df.empty:
        return go.Figure()
    hover_cols = [c for c in ["corners_taken", "matches", "total_xg", "fast_shot_rate", "box_delivery_rate"] if c in team_df.columns]
    fig = px.scatter(
        team_df,
        x=x_col,
        y=y_col,
        size=size_col,
        hover_name="team",
        hover_data=hover_cols,
        title=title,
        text="team",
        color_discrete_sequence=[ACCENT],
    )
    fig.update_traces(textposition="top center", textfont=dict(size=9, color=TEXT))
    # Add reference lines at medians
    if not team_df[x_col].dropna().empty:
        fig.add_vline(x=team_df[x_col].median(), line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text="Median", annotation_position="top right")
    if not team_df[y_col].dropna().empty:
        fig.add_hline(y=team_df[y_col].median(), line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text="Median", annotation_position="top right")
    return figure_layout(fig, height=420, title=title)


def phase_heatmap_figure(df_events, title="Corner Timing Heatmap"):
    if df_events.empty:
        return go.Figure()
    tmp = df_events.groupby(["corner_team", "phase"], dropna=False).size().reset_index(name="corners")
    pivot = tmp.pivot(index="corner_team", columns="phase", values="corners").fillna(0)
    phase_order = [p for p in ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"] if p in pivot.columns]
    pivot = pivot.reindex(columns=phase_order)
    fig = px.imshow(
        pivot,
        aspect="auto",
        title=title,
        labels=dict(x="Phase", y="Team", color="Corners"),
        text_auto=True,
        color_continuous_scale="Blues",
    )
    return figure_layout(fig, height=max(380, 40 * max(6, len(pivot.index))), title=title)


def end_zone_bar_figure(df_events, group_col="corner_team", title="End Zone Volume"):
    if df_events.empty:
        return go.Figure()
    zone_order = ["6-yard box", "Penalty area", "Deep box", "Outside danger zone", "Unknown"]
    summary = df_events.groupby([group_col, "end_zone"], dropna=False).size().reset_index(name="corners")
    fig = px.bar(
        summary,
        x=group_col,
        y="corners",
        color="end_zone",
        title=title,
        category_orders={"end_zone": zone_order},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    return figure_layout(fig, height=420, title=title)


def minute_histogram_figure(df_events, title="Corner Minute Distribution"):
    if df_events.empty:
        return go.Figure()
    fig = px.histogram(df_events, x="Minute", nbins=24, title=title, color_discrete_sequence=[ACCENT])
    fig.update_traces(opacity=0.85)
    return figure_layout(fig, height=380, title=title)


def xg_over_time_figure(df_events, title="xG Accumulation Over Time"):
    if df_events.empty:
        return go.Figure()
    df_sorted = df_events.dropna(subset=["shot_xg", "Minute"]).sort_values("Minute")
    if df_sorted.empty:
        return go.Figure()
    df_sorted["cumulative_xg"] = df_sorted["shot_xg"].cumsum()
    fig = px.line(df_sorted, x="Minute", y="cumulative_xg", title=title, color_discrete_sequence=[SUCCESS])
    fig.update_traces(fill="tozeroy", fillcolor="rgba(52,211,153,0.12)")
    return figure_layout(fig, height=360, title=title)


def taker_radar_figure(taker_df, taker_name):
    """Radar chart for a single taker's profile."""
    if taker_df.empty:
        return go.Figure()
    row = taker_df[taker_df["Taker"] == taker_name]
    if row.empty:
        return go.Figure()
    row = row.iloc[0]
    categories = ["Shot Rate", "xG/Corner", "Fast Shot Rate", "6Y Delivery Rate", "Inswinger Rate"]
    # Normalize each to 0-1 using taker dataset
    def norm(val, col):
        s = taker_df[col].dropna()
        if s.max() == s.min() or pd.isna(val):
            return 0
        return float((val - s.min()) / (s.max() - s.min()))
    values = [
        norm(row.get("shot_rate", 0), "shot_rate"),
        norm(row.get("xg_per_corner", 0), "xg_per_corner"),
        norm(row.get("fast_shots", 0) / max(row.get("corners", 1), 1), "shot_rate"),
        norm(row.get("six_yard_deliveries", 0) / max(row.get("corners", 1), 1), "shot_rate"),
        norm(row.get("inswingers", 0) / max(row.get("corners", 1), 1), "shot_rate"),
    ]
    values += [values[0]]
    categories += [categories[0]]
    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(93,168,255,0.25)",
        line=dict(color=ACCENT, width=2),
        name=taker_name,
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.2)"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=20, r=20, t=40, b=20),
        title=f"Profile — {taker_name}",
        font=dict(color=TEXT, family="Inter, sans-serif"),
        showlegend=False,
    )
    return fig


def build_team_summary(source_df):
    if source_df.empty:
        return pd.DataFrame(columns=[
            "team", "corners_taken", "matches", "shots_from_corners", "first_contact_shots",
            "fast_shots", "total_xg", "avg_xg_per_corner", "taker_variety", "inswingers",
            "outswingers", "short_corners", "target_box_deliveries", "six_yard_deliveries",
            "penalty_area_deliveries", "corners_per_match", "shot_rate", "first_contact_rate",
            "fast_shot_rate", "xg_per_match", "box_delivery_rate", "six_yard_delivery_rate",
            "penalty_area_delivery_rate", "short_corner_rate"
        ])
    ts = (
        source_df.groupby("corner_team", dropna=False)
        .agg(
            corners_taken=("match_id", "size"),
            matches=("match_id", pd.Series.nunique),
            shots_from_corners=("led_to_shot", "sum"),
            first_contact_shots=("is_first_contact_shot", "sum"),
            fast_shots=("is_fast_shot", "sum"),
            total_xg=("shot_xg", "sum"),
            avg_xg_per_corner=("shot_xg", "mean"),
            taker_variety=("Taker", pd.Series.nunique),
            inswingers=("is_inswinger", "sum"),
            outswingers=("is_outswinger", "sum"),
            short_corners=("is_short_corner", "sum"),
            target_box_deliveries=("is_goal_kick_zone_delivery", "sum"),
            six_yard_deliveries=("is_six_yard_delivery", "sum"),
            penalty_area_deliveries=("is_penalty_area_delivery", "sum"),
        )
        .reset_index()
        .rename(columns={"corner_team": "team"})
    )
    ts["corners_per_match"] = ts["corners_taken"] / ts["matches"].replace(0, np.nan)
    ts["shot_rate"] = ts["shots_from_corners"] / ts["corners_taken"].replace(0, np.nan)
    ts["first_contact_rate"] = ts["first_contact_shots"] / ts["corners_taken"].replace(0, np.nan)
    ts["fast_shot_rate"] = ts["fast_shots"] / ts["corners_taken"].replace(0, np.nan)
    ts["xg_per_match"] = ts["total_xg"] / ts["matches"].replace(0, np.nan)
    ts["box_delivery_rate"] = ts["target_box_deliveries"] / ts["corners_taken"].replace(0, np.nan)
    ts["six_yard_delivery_rate"] = ts["six_yard_deliveries"] / ts["corners_taken"].replace(0, np.nan)
    ts["penalty_area_delivery_rate"] = ts["penalty_area_deliveries"] / ts["corners_taken"].replace(0, np.nan)
    ts["short_corner_rate"] = ts["short_corners"] / ts["corners_taken"].replace(0, np.nan)
    ts["inswinger_rate"] = ts["inswingers"] / ts["corners_taken"].replace(0, np.nan)
    ts["outswinger_rate"] = ts["outswingers"] / ts["corners_taken"].replace(0, np.nan)
    return ts


def add_advanced_features(source_df):
    df2 = source_df.copy()
    if df2.empty:
        for col in ["venue_split", "delivery_length_band", "xg_created", "goal_from_corner", "delivery_success_proxy"]:
            if col not in df2.columns:
                df2[col] = np.nan
        return df2
    df2["venue_split"] = np.where(
        df2["is_home_corner"], "Home", np.where(df2["is_away_corner"], "Away", "Unknown")
    )
    df2["delivery_length_band"] = pd.cut(
        df2["delivery_length"],
        bins=[-0.1, 8, 16, 28, 200],
        labels=["Short", "Medium", "Long", "Very Long"],
        right=True,
    ).astype(str)
    df2["xg_created"] = df2["shot_xg"].fillna(0)
    df2["goal_from_corner"] = df2["shot_outcome"].astype(str).str.contains("goal", case=False, na=False)
    df2["delivery_success_proxy"] = (
        df2["led_to_shot"].fillna(False)
        | df2["is_first_contact_shot"].fillna(False)
        | df2["is_goal_kick_zone_delivery"].fillna(False)
    )
    return df2


def team_insight_table(source_df):
    if source_df.empty:
        return pd.DataFrame()
    out = (
        source_df.groupby(["corner_team", "delivery_zone", "end_zone"], dropna=False)
        .agg(
            corners=("match_id", "size"),
            shots=("led_to_shot", "sum"),
            total_xg=("shot_xg", "sum"),
            fast_shots=("is_fast_shot", "sum"),
        )
        .reset_index()
    )
    out["shot_rate"] = out["shots"] / out["corners"].replace(0, np.nan)
    out["xg_per_corner"] = out["total_xg"] / out["corners"].replace(0, np.nan)
    return out.sort_values(["corner_team", "corners"], ascending=[True, False])


def taker_summary_table(source_df):
    if source_df.empty:
        return pd.DataFrame()
    out = (
        source_df.groupby(["corner_team", "Taker"], dropna=False)
        .agg(
            corners=("match_id", "size"),
            shots=("led_to_shot", "sum"),
            fast_shots=("is_fast_shot", "sum"),
            first_contact_shots=("is_first_contact_shot", "sum"),
            goals=("goal_from_corner", "sum"),
            total_xg=("xg_created", "sum"),
            inswingers=("is_inswinger", "sum"),
            outswingers=("is_outswinger", "sum"),
            short_corners=("is_short_corner", "sum"),
            six_yard_deliveries=("is_six_yard_delivery", "sum"),
            penalty_area_deliveries=("is_penalty_area_delivery", "sum"),
        )
        .reset_index()
    )
    out["shot_rate"] = out["shots"] / out["corners"].replace(0, np.nan)
    out["xg_per_corner"] = out["total_xg"] / out["corners"].replace(0, np.nan)
    out["goal_rate"] = out["goals"] / out["corners"].replace(0, np.nan)
    out["fast_shot_rate"] = out["fast_shots"] / out["corners"].replace(0, np.nan)
    out["inswinger_rate"] = out["inswingers"] / out["corners"].replace(0, np.nan)
    return out.sort_values(["corners", "total_xg"], ascending=False)


def match_pattern_table(source_df):
    if source_df.empty:
        return pd.DataFrame()
    out = (
        source_df.groupby(["Match", "corner_team", "venue_split"], dropna=False)
        .agg(
            corners=("match_id", "size"),
            shots=("led_to_shot", "sum"),
            fast_shots=("is_fast_shot", "sum"),
            goals=("goal_from_corner", "sum"),
            total_xg=("xg_created", "sum"),
            six_yard_deliveries=("is_six_yard_delivery", "sum"),
            short_corners=("is_short_corner", "sum"),
        )
        .reset_index()
    )
    out["shot_rate"] = out["shots"] / out["corners"].replace(0, np.nan)
    out["xg_per_corner"] = out["total_xg"] / out["corners"].replace(0, np.nan)
    return out.sort_values(["total_xg", "corners"], ascending=False)


def percentile_rank(series, value):
    s = series.dropna()
    if len(s) == 0 or pd.isna(value):
        return np.nan
    return float((s <= value).mean() * 100)


def team_report_card_html(source_df, league_team_df, selected_team_name):
    """Render a rich visual report card with percentile bars."""
    if source_df.empty or selected_team_name == "All Teams":
        st.info("Select a specific team for the report card.")
        return
    team_row = league_team_df[league_team_df["team"] == selected_team_name]
    if team_row.empty:
        st.warning("No data for selected team.")
        return
    row = team_row.iloc[0]

    metrics = [
        ("Corners / Match", row.get("corners_per_match", np.nan), league_team_df["corners_per_match"], "Volume", "High = takes many corners"),
        ("Shot Rate", row.get("shot_rate", np.nan), league_team_df["shot_rate"], "%", "Shots per corner taken"),
        ("xG / Match", row.get("xg_per_match", np.nan), league_team_df["xg_per_match"], "xG", "Attacking threat"),
        ("Fast Shot Rate", row.get("fast_shot_rate", np.nan), league_team_df["fast_shot_rate"], "%", "Shots within 3s"),
        ("6Y Delivery Rate", row.get("six_yard_delivery_rate", np.nan), league_team_df["six_yard_delivery_rate"], "%", "Balls into 6-yard box"),
        ("Short Corner Rate", row.get("short_corner_rate", np.nan), league_team_df["short_corner_rate"], "%", "Use of short corners"),
        ("Inswinger Rate", row.get("inswinger_rate", np.nan), league_team_df.get("inswinger_rate", pd.Series(dtype=float)), "%", "Inswinging deliveries"),
        ("Taker Variety", row.get("taker_variety", np.nan), league_team_df["taker_variety"], "", "Unique takers used"),
    ]

    html_rows = ""
    for label, val, league_series, unit, tip in metrics:
        p = percentile_rank(league_series, val)
        if pd.isna(val):
            display_val = "—"
        elif unit == "%":
            display_val = f"{val*100:.1f}%"
        elif unit == "xG":
            display_val = f"{val:.3f}"
        else:
            display_val = f"{val:.2f}"

        if pd.isna(p):
            bar_w = 0
            pct_label = "—"
            bar_color = MUTED
        else:
            bar_w = int(p)
            pct_label = f"{p:.0f}th"
            bar_color = color_for_percentile(p)

        html_rows += f"""
        <div style="display:flex;align-items:center;padding:10px 0;border-bottom:1px solid {BORDER};">
            <div style="flex:1.2;color:{MUTED};font-size:0.87rem;">{label}<br><span style="font-size:0.76rem;color:rgba(153,173,199,0.6)">{tip}</span></div>
            <div style="flex:0.6;font-weight:700;font-size:0.95rem;color:{TEXT};text-align:right;padding-right:14px">{display_val}</div>
            <div style="flex:2;padding:0 4px;">
                <div style="height:8px;background:rgba(255,255,255,0.07);border-radius:99px;overflow:hidden;">
                    <div style="height:100%;width:{bar_w}%;background:{bar_color};border-radius:99px;transition:width 0.6s ease;"></div>
                </div>
            </div>
            <div style="flex:0.4;font-size:0.78rem;color:{bar_color};text-align:right;">{pct_label}</div>
        </div>
        """

    st.markdown(
        f"""
        <div style="background:linear-gradient(180deg,{CARD},{CARD_2});border:1px solid {BORDER};border-radius:20px;padding:20px 22px;">
            <div style="font-size:1.1rem;font-weight:800;margin-bottom:14px;">📊 {selected_team_name} — Report Card</div>
            {html_rows}
            <div style="margin-top:12px;font-size:0.78rem;color:{MUTED}">Percentile ranks relative to all teams in current filtered dataset.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def top_insights(events, teams_df):
    insights = []
    variants = []
    if events.empty or teams_df.empty:
        return [("No data available for current filter selection.", "default")]

    best_shot = teams_df.sort_values("shot_rate", ascending=False).head(1)
    best_xg = teams_df.sort_values("xg_per_match", ascending=False).head(1)
    best_6y = teams_df.sort_values("six_yard_delivery_rate", ascending=False).head(1)
    most_short = teams_df.sort_values("short_corner_rate", ascending=False).head(1)
    fastest = teams_df.sort_values("fast_shot_rate", ascending=False).head(1)

    if not best_shot.empty:
        r = best_shot.iloc[0]
        insights.append(f"Best shot-rate: <b>{r['team']}</b> at <b>{human_pct(r['shot_rate'])}</b> of corners leading to shots.")
        variants.append("success")
    if not best_xg.empty:
        r = best_xg.iloc[0]
        insights.append(f"Highest xG/match: <b>{r['team']}</b> generating <b>{r['xg_per_match']:.3f}</b> xG per match from corners.")
        variants.append("default")
    if not best_6y.empty:
        r = best_6y.iloc[0]
        insights.append(f"Best 6-yard targeting: <b>{r['team']}</b> delivering <b>{human_pct(r['six_yard_delivery_rate'])}</b> into the 6-yard box.")
        variants.append("warn")
    if not fastest.empty and len(insights) < 4:
        r = fastest.iloc[0]
        insights.append(f"Fastest transitions: <b>{r['team']}</b> with <b>{human_pct(r.get('fast_shot_rate', np.nan))}</b> fast shot rate (≤3s).")
        variants.append("success")
    if not most_short.empty and len(insights) < 5:
        r = most_short.iloc[0]
        insights.append(f"Most short corners: <b>{r['team']}</b> using short routines <b>{human_pct(r['short_corner_rate'])}</b> of the time.")
        variants.append("warn")
    return list(zip(insights, variants))[:3]


def add_download_buttons(events_df, team_df, match_df):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "⬇ Download Events CSV",
            data=events_df.to_csv(index=False).encode("utf-8"),
            file_name="filtered_corner_events.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c2:
        st.download_button(
            "⬇ Download Team Summary CSV",
            data=team_df.to_csv(index=False).encode("utf-8"),
            file_name="team_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with c3:
        st.download_button(
            "⬇ Download Match Summary CSV",
            data=match_df.to_csv(index=False).encode("utf-8"),
            file_name="match_summary.csv",
            mime="text/csv",
            use_container_width=True,
        )


def download_excel(events_df, team_df, match_df):
    """Download all sheets as a single Excel workbook."""
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        events_df.to_excel(writer, sheet_name="Events", index=False)
        team_df.to_excel(writer, sheet_name="Team Summary", index=False)
        match_df.to_excel(writer, sheet_name="Match Summary", index=False)
    return buf.getvalue()


# =========================================================
# DATA LOAD / PREP
# =========================================================
@st.cache_data
def load_data():
    if not os.path.exists(FILE_NAME):
        raise FileNotFoundError(
            f"{FILE_NAME} not found. Place it in the same directory as this app."
        )
    return pd.read_excel(FILE_NAME)


@st.cache_data
def prepare_data(raw_df):
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    match_id_col = find_col(df, ["match_id", "match id"])
    match_col = find_col(df, ["match"])
    team_col = find_col(df, ["pass_team_name", "team", "team_name"])
    minute_col = find_col(df, ["minute"])
    second_col = find_col(df, ["second"])
    outcome_col = find_col(df, ["sp_outcome", "outcome"])
    xg_col = find_col(df, ["shot.statsbomb_xg", "shot_xg", "xg"])
    taker_col = find_col(df, ["taker"])
    shooter_col = find_col(df, ["shooter"])
    defensive_setup_col = find_col(df, ["defensive_setup"])
    shot_team_col = find_col(df, ["shot_team_name"])
    pass_x_col = find_col(df, ["pass_location_x"])
    pass_y_col = find_col(df, ["pass_location_y"])
    pass_end_x_col = find_col(df, ["pass_end_location_x"])
    pass_end_y_col = find_col(df, ["pass_end_location_y"])
    shot_x_col = find_col(df, ["shot_location_x"])
    shot_y_col = find_col(df, ["shot_location_y"])
    shot_z_col = find_col(df, ["shot_location_z"])
    pass_technique_col = find_col(df, ["pass.technique.name", "pass_technique"])
    pass_height_col = find_col(df, ["pass.height.name", "pass_height"])
    pass_body_col = find_col(df, ["pass.body_part.name", "pass_body_part"])
    shot_body_col = find_col(df, ["shot.body_part.name", "shot_body_part"])
    shot_outcome_col = find_col(df, ["shot.outcome.name", "shot_outcome"])
    pass_outcome_col = find_col(df, ["pass.outcome.name", "pass_outcome"])
    pass_position_col = find_col(df, ["pass_position"])
    shot_position_col = find_col(df, ["shot_position"])

    required = {
        "match_id": match_id_col,
        "match": match_col,
        "pass_team_name": team_col,
        "minute": minute_col,
        "second": second_col,
    }
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )

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
        defensive_setup_col: "Defensive_setup",
        shot_team_col: "shot_team_name",
        pass_x_col: "pass_location_x",
        pass_y_col: "pass_location_y",
        pass_end_x_col: "pass_end_location_x",
        pass_end_y_col: "pass_end_location_y",
        shot_x_col: "shot_location_x",
        shot_y_col: "shot_location_y",
        shot_z_col: "shot_location_z",
        pass_technique_col: "pass_technique",
        pass_height_col: "pass_height",
        pass_body_col: "pass_body_part",
        shot_body_col: "shot_body_part",
        shot_outcome_col: "shot_outcome",
        pass_outcome_col: "pass_outcome",
        pass_position_col: "pass_position",
        shot_position_col: "shot_position",
    }
    for k, v in optional_map.items():
        if k is not None and k != v:
            rename_map[k] = v

    df = df.rename(columns=rename_map)

    defaults = {
        "SP_outcome": "",
        "shot_xg": np.nan,
        "Taker": np.nan,
        "Shooter": np.nan,
        "Defensive_setup": np.nan,
        "shot_team_name": np.nan,
        "pass_location_x": np.nan,
        "pass_location_y": np.nan,
        "pass_end_location_x": np.nan,
        "pass_end_location_y": np.nan,
        "shot_location_x": np.nan,
        "shot_location_y": np.nan,
        "shot_location_z": np.nan,
        "pass_technique": np.nan,
        "pass_height": np.nan,
        "pass_body_part": np.nan,
        "shot_body_part": np.nan,
        "shot_outcome": np.nan,
        "pass_outcome": np.nan,
        "pass_position": np.nan,
        "shot_position": np.nan,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    numeric_cols = [
        "Minute", "Second", "shot_xg", "pass_location_x", "pass_location_y",
        "pass_end_location_x", "pass_end_location_y", "shot_location_x",
        "shot_location_y", "shot_location_z",
    ]
    for col in numeric_cols:
        df[col] = safe_numeric(df[col])

    df["corner_team"] = df["corner_team"].astype(str).str.strip()
    df["event_minute"] = df["Minute"].fillna(0) + df["Second"].fillna(0) / 60.0

    homes, aways = [], []
    for m in df["Match"]:
        h, a = split_match_name(m)
        homes.append(h)
        aways.append(a)
    df["home_team"] = homes
    df["away_team"] = aways
    df["is_home_corner"] = df["corner_team"] == df["home_team"]
    df["is_away_corner"] = df["corner_team"] == df["away_team"]

    df["led_to_shot"] = df["SP_outcome"].astype(str).str.contains("shot", case=False, na=False)
    df["is_first_contact_shot"] = df["SP_outcome"].astype(str).str.contains("first contact", case=False, na=False)
    df["is_fast_shot"] = df["SP_outcome"].astype(str).str.contains("within 3 seconds", case=False, na=False)
    df["outcome_bucket"] = df["SP_outcome"].apply(classify_outcome)
    df["is_inswinger"] = df["pass_technique"].astype(str).str.contains("inswing", case=False, na=False)
    df["is_outswinger"] = df["pass_technique"].astype(str).str.contains("outswing", case=False, na=False)
    df["is_short_corner"] = df["pass_technique"].astype(str).str.contains("short", case=False, na=False)

    df["delivery_zone"] = df["pass_end_location_y"].apply(left_right_from_y)
    df["corner_side"] = df["pass_location_y"].apply(corner_side_from_start_y)
    df["delivery_length"] = df.apply(
        lambda r: delivery_length(
            r["pass_location_x"], r["pass_location_y"],
            r["pass_end_location_x"], r["pass_end_location_y"]
        ),
        axis=1,
    )
    df["end_zone"] = df.apply(
        lambda r: zone_from_end_location(r["pass_end_location_x"], r["pass_end_location_y"]), axis=1
    )
    df["is_goal_kick_zone_delivery"] = (
        df["pass_end_location_x"].between(114, 120, inclusive="both") &
        df["pass_end_location_y"].between(30, 50, inclusive="both")
    )
    df["is_six_yard_delivery"] = df["end_zone"].eq("6-yard box")
    df["is_penalty_area_delivery"] = df["end_zone"].eq("Penalty area")
    df["phase"] = pd.cut(
        df["event_minute"],
        bins=[-0.1, 15, 30, 45, 60, 75, 120],
        labels=["0-15", "16-30", "31-45", "46-60", "61-75", "76+"],
        right=True,
    ).astype(str)

    # Match-level summary
    match_summary = (
        df.groupby(["match_id", "Match", "home_team", "away_team"], dropna=False)
        .agg(
            total_corners=("match_id", "size"),
            shots_from_corners=("led_to_shot", "sum"),
            first_contact_shots=("is_first_contact_shot", "sum"),
            fast_shots=("is_fast_shot", "sum"),
            total_xg=("shot_xg", "sum"),
            avg_xg=("shot_xg", "mean"),
            unique_takers=("Taker", pd.Series.nunique),
            inswingers=("is_inswinger", "sum"),
            outswingers=("is_outswinger", "sum"),
            short_corners=("is_short_corner", "sum"),
            six_yard_deliveries=("is_six_yard_delivery", "sum"),
            penalty_area_deliveries=("is_penalty_area_delivery", "sum"),
        )
        .reset_index()
    )

    def count_team_corners(match_id, team_name):
        if pd.isna(team_name):
            return np.nan
        return int(((df["match_id"] == match_id) & (df["corner_team"] == team_name)).sum())

    match_summary["home_corners"] = match_summary.apply(
        lambda r: count_team_corners(r["match_id"], r["home_team"]), axis=1
    )
    match_summary["away_corners"] = match_summary.apply(
        lambda r: count_team_corners(r["match_id"], r["away_team"]), axis=1
    )
    match_summary["shot_rate"] = match_summary["shots_from_corners"] / match_summary["total_corners"].replace(0, np.nan)
    match_summary["xg_per_corner"] = match_summary["total_xg"] / match_summary["total_corners"].replace(0, np.nan)

    team_summary = build_team_summary(df)
    return df, match_summary, team_summary


# =========================================================
# LOAD DATA
# =========================================================
try:
    raw_df = load_data()
    df, match_summary, team_summary = prepare_data(raw_df)
except Exception as e:
    st.error("Failed to load or prepare the Excel file.")
    st.exception(e)
    st.stop()

# =========================================================
# HEADER
# =========================================================
st.markdown(
    '<div class="hero-wrap">'
    '<div class="hero-title">⚽ Allsvenskan Set Piece Studio Pro</div>'
    '<div class="hero-sub">Premium corner analytics — executive summaries, pitch visualisations, scouting flows, taker profiling, and exportable data workbooks.</div>'
    '<div>'
    '<span class="pill">Executive Dashboard</span>'
    '<span class="pill">Visualisation Studio</span>'
    '<span class="pill">Team Analysis</span>'
    '<span class="pill">Match Explorer</span>'
    '<span class="pill">Scouting Center</span>'
    '<span class="pill">Data Hub</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("🎛 Studio Controls")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Workspace",
    [
        "Executive Dashboard",
        "Visualisation Studio",
        "Team Analysis",
        "Match Explorer",
        "Scouting Center",
        "Data Hub",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Filters**")

all_teams = sorted([t for t in team_summary["team"].dropna().unique().tolist() if str(t).strip()])
selected_team = st.sidebar.selectbox("Team", ["All Teams"] + all_teams)
all_takers = sorted([str(t) for t in df["Taker"].dropna().astype(str).unique().tolist() if str(t).strip()])
selected_takers = st.sidebar.multiselect("Taker", all_takers)
all_matches = sorted(df["Match"].dropna().astype(str).unique().tolist())
selected_matches = st.sidebar.multiselect("Matches", all_matches)

minute_min = int(df["Minute"].min()) if not df["Minute"].dropna().empty else 0
minute_max = int(df["Minute"].max()) if not df["Minute"].dropna().empty else 120
if minute_max <= minute_min:
    minute_range = (minute_min, minute_max)
    st.sidebar.caption(f"Minute Range: {minute_min}")
else:
    minute_range = st.sidebar.slider(
        "Minute Range", min_value=minute_min, max_value=minute_max, value=(minute_min, minute_max)
    )

if len(match_summary) > 0:
    min_corners = int(match_summary["total_corners"].min())
    max_corners = int(match_summary["total_corners"].max())
else:
    min_corners, max_corners = 0, 0

if max_corners <= min_corners:
    corner_range = (min_corners, max_corners)
    st.sidebar.caption(f"Match Corner Range: {min_corners}")
else:
    corner_range = st.sidebar.slider(
        "Match Corner Range", min_value=min_corners, max_value=max_corners, value=(min_corners, max_corners)
    )

st.sidebar.markdown("**Delivery Type**")
show_shot_only = st.sidebar.checkbox("Shot outcomes only", value=False)
show_inswing_only = st.sidebar.checkbox("Inswingers only", value=False)
show_outswing_only = st.sidebar.checkbox("Outswingers only", value=False)
show_short_only = st.sidebar.checkbox("Short corners only", value=False)

all_delivery_zones = [
    z for z in ["Near Post Zone", "Central Zone", "Far Post Zone", "Unknown"]
    if z in df["delivery_zone"].astype(str).unique().tolist()
]
selected_delivery_zones = st.sidebar.multiselect("Delivery Zone", all_delivery_zones)
all_end_zones = [
    z for z in ["6-yard box", "Penalty area", "Deep box", "Outside danger zone", "Unknown"]
    if z in df["end_zone"].astype(str).unique().tolist()
]
selected_end_zones = st.sidebar.multiselect("End Zone", all_end_zones)
all_setups = sorted([
    str(x) for x in df["Defensive_setup"].dropna().astype(str).unique().tolist() if str(x).strip()
])
selected_setups = st.sidebar.multiselect("Defensive Setup", all_setups)
venue_filter = st.sidebar.multiselect(
    "Home / Away", ["Home", "Away", "Unknown"], default=["Home", "Away", "Unknown"]
)
phase_filter = st.sidebar.multiselect(
    "Phase",
    ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"],
    default=["0-15", "16-30", "31-45", "46-60", "61-75", "76+"],
)
outcome_filter = st.sidebar.multiselect(
    "Outcome Bucket",
    ["Shot ≤3s", "First Contact Shot", "Shot", "No First Contact", "Other", "Unknown"],
    default=["Shot ≤3s", "First Contact Shot", "Shot", "No First Contact", "Other", "Unknown"],
)

st.sidebar.markdown("---")
quick_mode = st.sidebar.selectbox(
    "Quick Mode",
    ["Balanced", "Attacking Patterns", "Shot Creation", "Delivery Zones", "Short Corners"],
)

# Quick mode overrides
if quick_mode == "Shot Creation":
    show_shot_only = True
elif quick_mode == "Short Corners":
    show_short_only = True

# =========================================================
# GLOBAL FILTERS
# =========================================================
league_match_df = match_summary[
    (match_summary["total_corners"] >= corner_range[0]) &
    (match_summary["total_corners"] <= corner_range[1])
].copy()

league_event_df = df[df["match_id"].isin(league_match_df["match_id"].unique())].copy()
league_event_df = add_advanced_features(league_event_df)
league_event_df = league_event_df[
    (league_event_df["Minute"].fillna(0) >= minute_range[0]) &
    (league_event_df["Minute"].fillna(0) <= minute_range[1])
]

if selected_team != "All Teams":
    league_event_df = league_event_df[league_event_df["corner_team"] == selected_team]
if selected_takers:
    league_event_df = league_event_df[league_event_df["Taker"].astype(str).isin(selected_takers)]
if selected_matches:
    league_event_df = league_event_df[league_event_df["Match"].astype(str).isin(selected_matches)]
if show_shot_only:
    league_event_df = league_event_df[league_event_df["led_to_shot"]]
if show_inswing_only and not show_outswing_only:
    league_event_df = league_event_df[league_event_df["is_inswinger"]]
if show_outswing_only and not show_inswing_only:
    league_event_df = league_event_df[league_event_df["is_outswinger"]]
if show_short_only:
    league_event_df = league_event_df[league_event_df["is_short_corner"]]
if selected_delivery_zones:
    league_event_df = league_event_df[league_event_df["delivery_zone"].isin(selected_delivery_zones)]
if selected_end_zones:
    league_event_df = league_event_df[league_event_df["end_zone"].isin(selected_end_zones)]
if selected_setups:
    league_event_df = league_event_df[league_event_df["Defensive_setup"].astype(str).isin(selected_setups)]
if venue_filter:
    league_event_df = league_event_df[league_event_df["venue_split"].isin(venue_filter)]
if phase_filter:
    league_event_df = league_event_df[league_event_df["phase"].isin(phase_filter)]
if outcome_filter:
    league_event_df = league_event_df[league_event_df["outcome_bucket"].isin(outcome_filter)]

league_match_df = league_match_df[
    league_match_df["match_id"].isin(league_event_df["match_id"].unique())
]
league_team_df = build_team_summary(league_event_df) if not league_event_df.empty else build_team_summary(pd.DataFrame())

# =========================================================
# EXECUTIVE DASHBOARD
# =========================================================
if page == "Executive Dashboard":
    render_kpis(league_event_df, league_match_df)

    insights = top_insights(league_event_df, league_team_df)
    ic1, ic2, ic3 = st.columns(3)
    cards = [ic1, ic2, ic3]
    for i, (txt, variant) in enumerate(insights):
        with cards[i]:
            insight_box("Key Insight", txt, variant=variant)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns([1.15, 1])
    with c1:
        section_header("Corner Volume by Team", "Total corners taken across the filtered sample")
        if not league_team_df.empty:
            fig = px.bar(
                league_team_df.sort_values("corners_taken", ascending=False),
                x="team",
                y="corners_taken",
                color="corners_per_match",
                color_continuous_scale="Blues",
                hover_data=["matches", "corners_per_match", "shots_from_corners", "shot_rate", "xg_per_match"],
                labels={"corners_taken": "Corners", "team": ""},
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(figure_layout(fig, 430), use_container_width=True)
        else:
            st.markdown('<div class="empty-state">No team data for current filters.</div>', unsafe_allow_html=True)
    with c2:
        section_header("Efficiency Map", "Shot rate vs xG per match — bubble size = volume")
        if not league_team_df.empty:
            fig = team_scatter_figure(
                league_team_df, "shot_rate", "xg_per_match", "corners_taken",
                "Shot Rate vs xG/Match"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="empty-state">No data.</div>', unsafe_allow_html=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(outcome_pie_figure(league_event_df, title="Outcome Split"), use_container_width=True)
    with c4:
        st.plotly_chart(technique_pie_figure(league_event_df, title="Delivery Technique Split"), use_container_width=True)

    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(
            cumulative_timeline_figure(league_event_df, color_col="corner_team", title="Cumulative Corner Volume"),
            use_container_width=True,
        )
    with c6:
        st.plotly_chart(minute_histogram_figure(league_event_df, title="Minute Distribution"), use_container_width=True)

    section_header("Executive Match Board", "Per-match breakdown — sortable, filterable")
    board_cols = [
        c for c in [
            "Match", "home_team", "away_team", "home_corners", "away_corners",
            "total_corners", "shots_from_corners", "fast_shots", "total_xg",
            "shot_rate", "xg_per_corner", "unique_takers",
        ]
        if c in league_match_df.columns
    ]
    if not league_match_df.empty:
        st.dataframe(
            league_match_df[board_cols]
            .sort_values(["total_corners", "shots_from_corners"], ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            height=430,
        )
    else:
        st.markdown('<div class="empty-state">No match data for current filters.</div>', unsafe_allow_html=True)

# =========================================================
# VISUALISATION STUDIO
# =========================================================
elif page == "Visualisation Studio":
    section_header(
        "Visualisation Studio",
        "Chart-first workspace. Use the Data Hub for raw tables and exports.",
    )
    viz_tabs = st.tabs([
        "🎯 Shotmaps",
        "🏹 Delivery Maps",
        "🔥 Heatmaps",
        "📊 Team Comparison",
        "⏱ Timing Patterns",
        "📍 Zone Profiles",
        "📈 xG Trends",
        "🧩 Visual Summary",
    ])

    with viz_tabs[0]:
        shot_df = league_event_df.dropna(subset=["shot_location_x", "shot_location_y"]).copy()
        col_a, col_b = st.columns([1, 3])
        with col_a:
            shot_color = st.selectbox(
                "Color by", ["corner_team", "Shooter", "Taker", "pass_technique", "shot_body_part", "shot_outcome"],
                index=0, key="shot_color_main"
            )
        if shot_df.empty:
            st.markdown('<div class="empty-state">No shots found for current filters. Try removing the "Shot outcomes only" filter or broadening the selection.</div>', unsafe_allow_html=True)
        else:
            st.plotly_chart(
                shotmap_figure(shot_df, color_col=shot_color, title=f"League Shotmap — {len(shot_df)} shots from corners"),
                use_container_width=True,
            )
            st.markdown(
                f'<div class="footer-note">Bubble size represents shot xG. Hover over markers for full event detail. Total xG shown: {shot_df["shot_xg"].fillna(0).sum():.3f}</div>',
                unsafe_allow_html=True,
            )

    with viz_tabs[1]:
        col_a, col_b = st.columns([1, 3])
        with col_a:
            delivery_color = st.selectbox(
                "Color by", ["delivery_zone", "end_zone", "pass_technique", "corner_team", "Taker", "corner_side"],
                index=0, key="delivery_color_main"
            )
        has_delivery = not league_event_df.dropna(
            subset=["pass_location_x", "pass_location_y", "pass_end_location_x", "pass_end_location_y"]
        ).empty
        if has_delivery:
            st.plotly_chart(
                delivery_map_figure(league_event_df, color_col=delivery_color, title="Delivery Map — All Corners"),
                use_container_width=True,
            )
        else:
            st.markdown('<div class="empty-state">No delivery coordinate data available.</div>', unsafe_allow_html=True)

    with viz_tabs[2]:
        has_end_loc = not league_event_df.dropna(subset=["pass_end_location_x", "pass_end_location_y"]).empty
        if has_end_loc:
            heatmap_fig = delivery_end_heatmap(league_event_df, title="Delivery End-Location Density")
            st.plotly_chart(heatmap_fig, use_container_width=True)
            st.markdown('<div class="footer-note">Darker = more deliveries targeted to that area of the pitch.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="empty-state">No end-location data available.</div>', unsafe_allow_html=True)

    with viz_tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                team_scatter_figure(
                    league_team_df, "corners_per_match", "shot_rate", "corners_taken",
                    "Corners/Match vs Shot Rate"
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                team_scatter_figure(
                    league_team_df, "six_yard_delivery_rate", "xg_per_match", "shots_from_corners",
                    "6Y Delivery Rate vs xG/Match"
                ),
                use_container_width=True,
            )
        c3, c4 = st.columns(2)
        with c3:
            st.plotly_chart(
                team_scatter_figure(
                    league_team_df, "short_corner_rate", "fast_shot_rate", "corners_taken",
                    "Short Corner Rate vs Fast Shot Rate"
                ),
                use_container_width=True,
            )
        with c4:
            st.plotly_chart(
                team_scatter_figure(
                    league_team_df, "inswinger_rate", "shot_rate", "corners_taken",
                    "Inswinger Rate vs Shot Rate"
                ),
                use_container_width=True,
            )

    with viz_tabs[4]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                cumulative_timeline_figure(
                    league_event_df, color_col="corner_team", title="Cumulative Corners by Team"
                ),
                use_container_width=True,
            )
        with c2:
            st.plotly_chart(
                minute_histogram_figure(league_event_df, title="Corner Minute Distribution"),
                use_container_width=True,
            )
        fig_phase = phase_heatmap_figure(league_event_df, title="League Phase Heatmap")
        if len(fig_phase.data) > 0:
            st.plotly_chart(fig_phase, use_container_width=True)
        else:
            st.markdown('<div class="empty-state">No phase data to display.</div>', unsafe_allow_html=True)

    with viz_tabs[5]:
        fig_zone = end_zone_bar_figure(league_event_df, group_col="corner_team", title="Team End-Zone Volume")
        if len(fig_zone.data) > 0:
            st.plotly_chart(fig_zone, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            # Corner side split
            side_df = league_event_df.groupby(["corner_team", "corner_side"], dropna=False).size().reset_index(name="corners")
            fig_side = px.bar(side_df, x="corner_team", y="corners", color="corner_side", title="Corner Side (Left vs Right)", barmode="stack")
            st.plotly_chart(figure_layout(fig_side, 400), use_container_width=True)
        with c2:
            # Delivery length band
            band_df = league_event_df.groupby("delivery_length_band", dropna=False).size().reset_index(name="corners")
            fig_band = px.bar(band_df, x="delivery_length_band", y="corners", title="Delivery Length Band Distribution", color="delivery_length_band", color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(figure_layout(fig_band, 400), use_container_width=True)

    with viz_tabs[6]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(xg_over_time_figure(league_event_df, title="Cumulative xG Over Match Minutes"), use_container_width=True)
        with c2:
            # xG by team bar
            xg_team = league_event_df.groupby("corner_team", dropna=False).agg(
                total_xg=("shot_xg", "sum"), corners=("match_id", "size")
            ).reset_index()
            xg_team["xg_per_corner"] = xg_team["total_xg"] / xg_team["corners"].replace(0, np.nan)
            fig_xg = px.bar(
                xg_team.sort_values("total_xg", ascending=False),
                x="corner_team", y="total_xg", title="Total xG from Corners by Team",
                hover_data=["xg_per_corner", "corners"], color_discrete_sequence=[SUCCESS],
                labels={"corner_team": "", "total_xg": "Total xG"}
            )
            st.plotly_chart(figure_layout(fig_xg, 360), use_container_width=True)
        # xG by outcome
        xg_outcome = league_event_df.groupby("outcome_bucket", dropna=False).agg(
            total_xg=("shot_xg", "sum"), corners=("match_id", "size")
        ).reset_index()
        fig_xg_oc = px.bar(
            xg_outcome.sort_values("total_xg", ascending=False),
            x="outcome_bucket", y="total_xg", color="outcome_bucket",
            title="xG by Outcome Bucket",
            color_discrete_sequence=px.colors.qualitative.Set2,
            labels={"outcome_bucket": "", "total_xg": "Total xG"}
        )
        st.plotly_chart(figure_layout(fig_xg_oc, 360), use_container_width=True)

    with viz_tabs[7]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(outcome_pie_figure(league_event_df, title="Outcome Split"), use_container_width=True)
        with c2:
            st.plotly_chart(technique_pie_figure(league_event_df, title="Technique Split"), use_container_width=True)
        c3, c4 = st.columns(2)
        with c3:
            # Body part split for passes
            body_df = league_event_df.groupby("pass_body_part", dropna=False).size().reset_index(name="corners")
            fig_body = px.pie(body_df, names="pass_body_part", values="corners", title="Pass Body Part", hole=0.5)
            st.plotly_chart(figure_layout(fig_body, 340), use_container_width=True)
        with c4:
            # Pass height split
            height_df = league_event_df.groupby("pass_height", dropna=False).size().reset_index(name="corners")
            fig_height = px.pie(height_df, names="pass_height", values="corners", title="Pass Height", hole=0.5)
            st.plotly_chart(figure_layout(fig_height, 340), use_container_width=True)
        fig_phase2 = phase_heatmap_figure(league_event_df, title="Timing Heatmap by Team")
        if len(fig_phase2.data) > 0:
            st.plotly_chart(fig_phase2, use_container_width=True)

# =========================================================
# TEAM ANALYSIS
# =========================================================
elif page == "Team Analysis":
    if selected_team == "All Teams":
        st.info("👈 Select a specific team in the sidebar to access full team intelligence.")

        section_header("League-Wide Team Overview", "Compare all teams across core corner metrics")
        if not league_team_df.empty:
            display_df = league_team_df[[
                "team", "corners_taken", "matches", "corners_per_match",
                "shots_from_corners", "shot_rate", "total_xg", "xg_per_match",
                "fast_shot_rate", "six_yard_delivery_rate", "short_corner_rate",
                "inswinger_rate", "taker_variety"
            ]].copy()
            display_df["shot_rate"] = (display_df["shot_rate"] * 100).round(1).astype(str) + "%"
            display_df["xg_per_match"] = display_df["xg_per_match"].round(3)
            display_df["fast_shot_rate"] = (display_df["fast_shot_rate"] * 100).round(1).astype(str) + "%"
            display_df["six_yard_delivery_rate"] = (display_df["six_yard_delivery_rate"] * 100).round(1).astype(str) + "%"
            display_df["short_corner_rate"] = (display_df["short_corner_rate"] * 100).round(1).astype(str) + "%"
            display_df["inswinger_rate"] = (display_df["inswinger_rate"] * 100).round(1).astype(str) + "%"
            st.dataframe(display_df.reset_index(drop=True), use_container_width=True, height=560)
    else:
        team_event_df = league_event_df[league_event_df["corner_team"] == selected_team].copy()
        team_match_df = league_match_df[
            league_match_df["match_id"].isin(team_event_df["match_id"].unique())
        ].copy()
        team_row_df = league_team_df[league_team_df["team"] == selected_team].copy()
        taker_df = taker_summary_table(team_event_df)

        team_tabs = st.tabs([
            "📊 Overview",
            "🎯 Visuals",
            "👤 Taker Intelligence",
            "📋 Match Review",
            "🏆 Report Card",
            "🗂 Raw Data",
        ])

        with team_tabs[0]:
            section_header(f"{selected_team} — Team Overview", "High-level snapshot")
            render_kpis(team_event_df, team_match_df)

            c1, c2 = st.columns(2)
            with c1:
                outcome_df = (
                    team_event_df.groupby("outcome_bucket", dropna=False)
                    .size()
                    .reset_index(name="events")
                    .sort_values("events", ascending=False)
                )
                fig = px.bar(
                    outcome_df, x="outcome_bucket", y="events", title="Outcome Profile",
                    color="outcome_bucket", color_discrete_sequence=px.colors.qualitative.Set2,
                    labels={"outcome_bucket": "", "events": "Corners"}
                )
                st.plotly_chart(figure_layout(fig, 400), use_container_width=True)
            with c2:
                zone_df = (
                    team_event_df.groupby("end_zone", dropna=False)
                    .size()
                    .reset_index(name="corners")
                    .sort_values("corners", ascending=False)
                )
                fig = px.bar(
                    zone_df, x="end_zone", y="corners", title="End-Zone Profile",
                    color="end_zone", color_discrete_sequence=px.colors.qualitative.Bold,
                    labels={"end_zone": "", "corners": "Corners"}
                )
                st.plotly_chart(figure_layout(fig, 400), use_container_width=True)

            c3, c4 = st.columns(2)
            with c3:
                technique_df = team_event_df.groupby("pass_technique", dropna=False).size().reset_index(name="corners")
                fig = px.pie(technique_df, names="pass_technique", values="corners", title="Delivery Technique", hole=0.55)
                st.plotly_chart(figure_layout(fig, 360), use_container_width=True)
            with c4:
                phase_df = team_event_df.groupby("phase", dropna=False).size().reset_index(name="corners")
                fig = px.bar(phase_df, x="phase", y="corners", title="Corner Phase Distribution",
                             color_discrete_sequence=[ACCENT], labels={"phase": "Phase", "corners": "Corners"})
                st.plotly_chart(figure_layout(fig, 360), use_container_width=True)

        with team_tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                team_shots = team_event_df.dropna(subset=["shot_location_x", "shot_location_y"]).copy()
                if team_shots.empty:
                    st.markdown('<div class="empty-state">No shot location data for this team.</div>', unsafe_allow_html=True)
                else:
                    st.plotly_chart(
                        shotmap_figure(team_shots, color_col="Shooter", title=f"Shotmap — {selected_team}"),
                        use_container_width=True,
                    )
            with c2:
                has_del = not team_event_df.dropna(
                    subset=["pass_location_x", "pass_location_y", "pass_end_location_x", "pass_end_location_y"]
                ).empty
                if has_del:
                    st.plotly_chart(
                        delivery_map_figure(team_event_df, color_col="delivery_zone", title=f"Delivery Map — {selected_team}"),
                        use_container_width=True,
                    )
                else:
                    st.markdown('<div class="empty-state">No delivery coordinate data for this team.</div>', unsafe_allow_html=True)

            # Heatmap
            has_end = not team_event_df.dropna(subset=["pass_end_location_x", "pass_end_location_y"]).empty
            if has_end:
                st.plotly_chart(
                    delivery_end_heatmap(team_event_df, title=f"End-Location Heatmap — {selected_team}"),
                    use_container_width=True,
                )

        with team_tabs[2]:
            section_header("Taker Intelligence", "Production, profile, and efficiency by corner taker")
            if taker_df.empty:
                st.markdown('<div class="empty-state">No taker data available.</div>', unsafe_allow_html=True)
            else:
                st.dataframe(taker_df.reset_index(drop=True), use_container_width=True, height=420)
                st.markdown("---")
                section_header("Taker Radar Profile", "Normalized radar relative to all takers in filtered dataset")
                all_takers_for_team = [str(t) for t in taker_df["Taker"].dropna().unique().tolist()]
                if all_takers_for_team:
                    all_takers_league = taker_summary_table(league_event_df)
                    selected_taker_profile = st.selectbox(
                        "Select Taker for Radar", all_takers_for_team, key="taker_radar"
                    )
                    c1, c2 = st.columns(2)
                    with c1:
                        st.plotly_chart(
                            taker_radar_figure(all_takers_league, selected_taker_profile),
                            use_container_width=True,
                        )
                    with c2:
                        taker_row = taker_df[taker_df["Taker"].astype(str) == selected_taker_profile]
                        if not taker_row.empty:
                            tr = taker_row.iloc[0]
                            metric_card("Corners Taken", f"{int(tr.get('corners', 0))}", foot="Sample size")
                            metric_card("Shot Rate", f"{human_pct(tr.get('shot_rate', np.nan))}", foot="Corners → shots")
                            metric_card("xG / Corner", f"{human_val(tr.get('xg_per_corner', np.nan), 4)}", foot="Attacking quality")
                            metric_card("Goals Directly", f"{int(tr.get('goals', 0))}", foot="Goals from corners")

        with team_tabs[3]:
            section_header("Match Review", "Per-match breakdown for this team")
            match_tbl = match_pattern_table(team_event_df)
            if match_tbl.empty:
                st.markdown('<div class="empty-state">No match data.</div>', unsafe_allow_html=True)
            else:
                st.dataframe(match_tbl.reset_index(drop=True), use_container_width=True, height=480)

            section_header("Match xG Timeline", "xG generated across matches")
            match_xg = team_event_df.groupby("Match", dropna=False).agg(
                total_xg=("shot_xg", "sum"), corners=("match_id", "size"), shots=("led_to_shot", "sum")
            ).reset_index().sort_values("total_xg", ascending=False)
            if not match_xg.empty:
                fig = px.bar(
                    match_xg, x="Match", y="total_xg", title="xG by Match",
                    hover_data=["corners", "shots"],
                    color="total_xg", color_continuous_scale="Blues",
                    labels={"total_xg": "Total xG", "Match": ""}
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(figure_layout(fig, 380), use_container_width=True)

        with team_tabs[4]:
            team_report_card_html(team_event_df, league_team_df, selected_team)
            if not team_row_df.empty:
                row = team_row_df.iloc[0]
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    p = percentile_rank(league_team_df["corners_per_match"], row.get("corners_per_match", np.nan))
                    metric_card("Volume Percentile", f"{p:.0f}" if not pd.isna(p) else "—", "th", "League standing")
                with c2:
                    p = percentile_rank(league_team_df["shot_rate"], row.get("shot_rate", np.nan))
                    metric_card("Shot Rate Percentile", f"{p:.0f}" if not pd.isna(p) else "—", "th", "League standing")
                with c3:
                    p = percentile_rank(league_team_df["xg_per_match"], row.get("xg_per_match", np.nan))
                    metric_card("xG/Match Percentile", f"{p:.0f}" if not pd.isna(p) else "—", "th", "League standing")
                with c4:
                    metric_card(
                        "6Y Delivery Rate",
                        f"{row.get('six_yard_delivery_rate', 0) * 100:.1f}" if not pd.isna(row.get("six_yard_delivery_rate")) else "—",
                        "%",
                        "High-danger targeting",
                    )

        with team_tabs[5]:
            st.dataframe(team_event_df.reset_index(drop=True), use_container_width=True, height=560)
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "⬇ Download Team Events CSV",
                    data=team_event_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_team.replace(' ', '_')}_events.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    "⬇ Download Taker Table CSV",
                    data=taker_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_team.replace(' ', '_')}_takers.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

# =========================================================
# MATCH EXPLORER
# =========================================================
elif page == "Match Explorer":
    section_header("Match Explorer", "Drill down into individual matches")
    available_matches = sorted(league_match_df["Match"].dropna().unique().tolist()) if not league_match_df.empty else []

    col_a, col_b = st.columns([2, 1])
    with col_a:
        selected_match = st.selectbox("Select Match", ["All Matches"] + available_matches)
    with col_b:
        st.markdown("<br>", unsafe_allow_html=True)
        if selected_match != "All Matches":
            st.markdown(
                f'<div class="sub-card" style="text-align:center;font-weight:700">{selected_match}</div>',
                unsafe_allow_html=True
            )

    match_event_df = league_event_df.copy()
    match_board_df = league_match_df.copy()
    if selected_match != "All Matches":
        match_board_df = match_board_df[match_board_df["Match"] == selected_match]
        match_event_df = match_event_df[match_event_df["match_id"].isin(match_board_df["match_id"].unique())]

    tabs = st.tabs(["📋 Summary", "⏱ Timeline", "🎯 Shotmap", "🏹 Delivery Map", "📊 Phase Analysis", "🔴 Event Feed", "🗂 Full Data"])

    with tabs[0]:
        section_header("Match Summary", "Overview of selected match(es)")
        render_kpis(match_event_df, match_board_df)
        board_cols = [
            c for c in [
                "Match", "home_team", "away_team", "home_corners", "away_corners",
                "total_corners", "shots_from_corners", "fast_shots", "total_xg",
                "shot_rate", "xg_per_corner", "unique_takers", "six_yard_deliveries",
            ]
            if c in match_board_df.columns
        ]
        if not match_board_df.empty:
            st.dataframe(
                match_board_df[board_cols].sort_values(["total_corners", "shots_from_corners"], ascending=False).reset_index(drop=True),
                use_container_width=True,
                height=360,
            )

        # Team-by-team summary within match
        if not match_event_df.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            section_header("Team Breakdown within Match")
            team_in_match = build_team_summary(match_event_df)
            cols_show = [c for c in [
                "team", "corners_taken", "shots_from_corners", "shot_rate", "total_xg",
                "fast_shots", "six_yard_deliveries", "short_corners", "inswingers"
            ] if c in team_in_match.columns]
            st.dataframe(team_in_match[cols_show].reset_index(drop=True), use_container_width=True, height=280)

    with tabs[1]:
        if not match_event_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                minute_df = (
                    match_event_df.groupby(["Minute", "corner_team"], dropna=False)
                    .size()
                    .reset_index(name="corner_events")
                    .sort_values("Minute")
                )
                fig = px.bar(minute_df, x="Minute", y="corner_events", color="corner_team",
                             title="Corner Timeline by Team", barmode="stack",
                             labels={"corner_events": "Events", "Minute": "Match Minute"})
                st.plotly_chart(figure_layout(fig, 420), use_container_width=True)
            with c2:
                st.plotly_chart(
                    cumulative_timeline_figure(match_event_df, color_col="corner_team", title="Cumulative Corners"),
                    use_container_width=True
                )
        else:
            st.markdown('<div class="empty-state">No event data.</div>', unsafe_allow_html=True)

    with tabs[2]:
        col_a, _ = st.columns([1, 3])
        with col_a:
            shot_color = st.selectbox(
                "Color by", ["corner_team", "Shooter", "Taker", "pass_technique", "shot_outcome"],
                index=0, key="match_shot_color"
            )
        shot_match_df = match_event_df.dropna(subset=["shot_location_x", "shot_location_y"])
        if shot_match_df.empty:
            st.markdown('<div class="empty-state">No shot location data for this match.</div>', unsafe_allow_html=True)
        else:
            st.plotly_chart(
                shotmap_figure(shot_match_df, color_col=shot_color, title=f"Shotmap — {selected_match}"),
                use_container_width=True,
            )

    with tabs[3]:
        col_a, _ = st.columns([1, 3])
        with col_a:
            delivery_color = st.selectbox(
                "Color by", ["corner_team", "delivery_zone", "end_zone", "pass_technique", "Taker"],
                index=0, key="match_delivery_color"
            )
        has_del = not match_event_df.dropna(
            subset=["pass_location_x", "pass_location_y", "pass_end_location_x", "pass_end_location_y"]
        ).empty
        if has_del:
            st.plotly_chart(
                delivery_map_figure(match_event_df, color_col=delivery_color, title=f"Delivery Map — {selected_match}"),
                use_container_width=True,
            )
        else:
            st.markdown('<div class="empty-state">No delivery coordinate data for this match.</div>', unsafe_allow_html=True)

    with tabs[4]:
        if not match_event_df.empty:
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(outcome_pie_figure(match_event_df, title="Outcome Split"), use_container_width=True)
            with c2:
                st.plotly_chart(technique_pie_figure(match_event_df, title="Technique Split"), use_container_width=True)
            fig_phase = phase_heatmap_figure(match_event_df, title="Phase Heatmap")
            if len(fig_phase.data) > 0:
                st.plotly_chart(fig_phase, use_container_width=True)
        else:
            st.markdown('<div class="empty-state">No phase data.</div>', unsafe_allow_html=True)

    with tabs[5]:
        show_cols = [
            c for c in [
                "Match", "corner_team", "Taker", "Shooter", "Minute", "Second",
                "SP_outcome", "shot_xg", "Defensive_setup", "pass_technique",
                "pass_height", "pass_body_part", "delivery_zone", "end_zone",
                "corner_side", "delivery_length_band", "outcome_bucket",
            ]
            if c in match_event_df.columns
        ]
        st.dataframe(
            match_event_df[show_cols].sort_values(["Minute", "Second"]).reset_index(drop=True),
            use_container_width=True,
            height=620,
        )

    with tabs[6]:
        st.dataframe(match_event_df.reset_index(drop=True), use_container_width=True, height=620)
        if not match_event_df.empty:
            st.download_button(
                "⬇ Download Match Events CSV",
                data=match_event_df.to_csv(index=False).encode("utf-8"),
                file_name=f"match_events_{selected_match.replace(' ', '_').replace('/', '-')}.csv",
                mime="text/csv",
            )

# =========================================================
# SCOUTING CENTER
# =========================================================
elif page == "Scouting Center":
    section_header(
        "Scouting Center",
        "Dedicated workflow for ranking, comparing, and identifying corner patterns",
    )
    scout_tabs = st.tabs([
        "🏅 Team Rankings",
        "👤 Taker Rankings",
        "🔀 Team Comparison",
        "📍 Zone Intelligence",
        "🛡 Defensive Setups",
        "🔍 Advanced Search",
        "⬇ Export",
    ])

    with scout_tabs[0]:
        section_header("Team Rankings", "Sorted by xG/match — adjust using sidebar filters")
        if league_team_df.empty:
            st.markdown('<div class="empty-state">No team data.</div>', unsafe_allow_html=True)
        else:
            rank_df = league_team_df.sort_values(["xg_per_match", "shot_rate", "corners_taken"], ascending=False).reset_index(drop=True)
            rank_df.index = rank_df.index + 1  # 1-based rank
            # Add rank column
            rank_df.insert(0, "Rank", rank_df.index)
            display_rank = rank_df[[
                "Rank", "team", "corners_taken", "matches", "corners_per_match",
                "shots_from_corners", "shot_rate", "total_xg", "xg_per_match",
                "fast_shot_rate", "six_yard_delivery_rate", "short_corner_rate",
                "inswinger_rate", "taker_variety"
            ]].copy()
            st.dataframe(display_rank, use_container_width=True, height=600)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(
                    rank_df.head(10),
                    x="team", y="xg_per_match",
                    title="Top 10 Teams — xG/Match",
                    color="xg_per_match", color_continuous_scale="Blues",
                    labels={"team": "", "xg_per_match": "xG/Match"}
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(figure_layout(fig, 380), use_container_width=True)
            with c2:
                fig = px.bar(
                    rank_df.sort_values("shot_rate", ascending=False).head(10),
                    x="team", y="shot_rate",
                    title="Top 10 Teams — Shot Rate",
                    color="shot_rate", color_continuous_scale="Greens",
                    labels={"team": "", "shot_rate": "Shot Rate"}
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(figure_layout(fig, 380), use_container_width=True)

    with scout_tabs[1]:
        section_header("Taker Rankings", "All takers in filtered dataset — ranked by xG/corner")
        taker_league_df = taker_summary_table(league_event_df)
        if taker_league_df.empty:
            st.markdown('<div class="empty-state">No taker data.</div>', unsafe_allow_html=True)
        else:
            min_corners_filter = st.number_input("Min corners (filter noise)", min_value=1, max_value=50, value=5, step=1)
            filtered_takers = taker_league_df[taker_league_df["corners"] >= min_corners_filter].reset_index(drop=True)
            st.dataframe(filtered_takers, use_container_width=True, height=520)

            if not filtered_takers.empty:
                top_by_xg = filtered_takers.sort_values("xg_per_corner", ascending=False).head(15)
                fig = px.bar(
                    top_by_xg, x="Taker", y="xg_per_corner",
                    title=f"Top Takers by xG/Corner (min {min_corners_filter} corners)",
                    color="xg_per_corner", color_continuous_scale="Blues",
                    hover_data=["corner_team", "corners", "shots", "total_xg"],
                    labels={"Taker": "", "xg_per_corner": "xG/Corner"}
                )
                fig.update_layout(coloraxis_showscale=False)
                st.plotly_chart(figure_layout(fig, 400), use_container_width=True)

    with scout_tabs[2]:
        section_header("Team Comparison", "Side-by-side profile for up to 2 teams")
        all_teams_scout = [t for t in league_team_df["team"].dropna().unique().tolist() if str(t).strip()]
        if len(all_teams_scout) < 2:
            st.info("At least 2 teams required for comparison.")
        else:
            c_left, c_right = st.columns(2)
            with c_left:
                team_a = st.selectbox("Team A", all_teams_scout, index=0, key="compare_a")
            with c_right:
                team_b = st.selectbox("Team B", all_teams_scout, index=min(1, len(all_teams_scout)-1), key="compare_b")

            events_a = league_event_df[league_event_df["corner_team"] == team_a]
            events_b = league_event_df[league_event_df["corner_team"] == team_b]
            row_a = league_team_df[league_team_df["team"] == team_a].iloc[0] if not league_team_df[league_team_df["team"] == team_a].empty else pd.Series()
            row_b = league_team_df[league_team_df["team"] == team_b].iloc[0] if not league_team_df[league_team_df["team"] == team_b].empty else pd.Series()

            compare_metrics = [
                ("Corners/Match", "corners_per_match", "{:.2f}"),
                ("Shot Rate", "shot_rate", "{:.1%}"),
                ("xG/Match", "xg_per_match", "{:.3f}"),
                ("Fast Shot Rate", "fast_shot_rate", "{:.1%}"),
                ("6Y Delivery Rate", "six_yard_delivery_rate", "{:.1%}"),
                ("Short Corner Rate", "short_corner_rate", "{:.1%}"),
                ("Inswinger Rate", "inswinger_rate", "{:.1%}"),
                ("Taker Variety", "taker_variety", "{:.0f}"),
            ]

            comparison_rows = ""
            for label, col, fmt in compare_metrics:
                val_a = row_a.get(col, np.nan) if not row_a.empty else np.nan
                val_b = row_b.get(col, np.nan) if not row_b.empty else np.nan
                str_a = fmt.format(val_a) if not pd.isna(val_a) else "—"
                str_b = fmt.format(val_b) if not pd.isna(val_b) else "—"
                # Highlight better
                if not pd.isna(val_a) and not pd.isna(val_b):
                    color_a = SUCCESS if val_a >= val_b else TEXT
                    color_b = SUCCESS if val_b > val_a else TEXT
                else:
                    color_a = color_b = TEXT
                comparison_rows += f"""
                <div style="display:flex;padding:9px 0;border-bottom:1px solid {BORDER};align-items:center">
                    <div style="flex:1;font-weight:700;color:{color_a};text-align:center">{str_a}</div>
                    <div style="flex:1.5;text-align:center;color:{MUTED};font-size:0.85rem">{label}</div>
                    <div style="flex:1;font-weight:700;color:{color_b};text-align:center">{str_b}</div>
                </div>
                """
            st.markdown(
                f"""
                <div style="background:linear-gradient(180deg,{CARD},{CARD_2});border:1px solid {BORDER};border-radius:20px;padding:16px 20px;margin-bottom:16px">
                    <div style="display:flex;padding:0 0 12px 0;border-bottom:1px solid {BORDER};">
                        <div style="flex:1;font-size:1.05rem;font-weight:800;color:{ACCENT};text-align:center">{team_a}</div>
                        <div style="flex:1.5;text-align:center;color:{MUTED};">vs</div>
                        <div style="flex:1;font-size:1.05rem;font-weight:800;color:{SUCCESS};text-align:center">{team_b}</div>
                    </div>
                    {comparison_rows}
                </div>
                """,
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                shot_a = events_a.dropna(subset=["shot_location_x", "shot_location_y"])
                if not shot_a.empty:
                    st.plotly_chart(
                        shotmap_figure(shot_a, color_col="Taker", title=f"Shotmap — {team_a}"),
                        use_container_width=True
                    )
            with c2:
                shot_b = events_b.dropna(subset=["shot_location_x", "shot_location_y"])
                if not shot_b.empty:
                    st.plotly_chart(
                        shotmap_figure(shot_b, color_col="Taker", title=f"Shotmap — {team_b}"),
                        use_container_width=True
                    )

    with scout_tabs[3]:
        section_header("Zone Intelligence", "Team-by-zone breakdown for delivery targeting")
        zone_tbl = team_insight_table(league_event_df)
        if zone_tbl.empty:
            st.markdown('<div class="empty-state">No zone data.</div>', unsafe_allow_html=True)
        else:
            st.dataframe(zone_tbl.reset_index(drop=True), use_container_width=True, height=520)
            # Pivot for heatmap
            zone_pivot = zone_tbl.groupby(["corner_team", "end_zone"])["corners"].sum().unstack(fill_value=0)
            if not zone_pivot.empty:
                fig = px.imshow(
                    zone_pivot,
                    aspect="auto",
                    title="Team Zone Targeting Heatmap",
                    labels=dict(x="End Zone", y="Team", color="Corners"),
                    text_auto=True,
                    color_continuous_scale="Blues",
                )
                st.plotly_chart(figure_layout(fig, max(380, 40 * len(zone_pivot))), use_container_width=True)

    with scout_tabs[4]:
        section_header("Defensive Setups", "How different defensive shapes affect attacking outcomes")
        if "Defensive_setup" in league_event_df.columns:
            defensive_df = (
                league_event_df.groupby("Defensive_setup", dropna=False)
                .agg(
                    corners=("match_id", "size"),
                    shots=("led_to_shot", "sum"),
                    total_xg=("shot_xg", "sum"),
                    fast_shots=("is_fast_shot", "sum"),
                    six_yard_deliveries=("is_six_yard_delivery", "sum"),
                )
                .reset_index()
            )
            defensive_df["shot_rate"] = defensive_df["shots"] / defensive_df["corners"].replace(0, np.nan)
            defensive_df["xg_per_corner"] = defensive_df["total_xg"] / defensive_df["corners"].replace(0, np.nan)
            defensive_df["fast_shot_rate"] = defensive_df["fast_shots"] / defensive_df["corners"].replace(0, np.nan)
            defensive_df = defensive_df[defensive_df["Defensive_setup"].astype(str) != "nan"].reset_index(drop=True)

            if defensive_df.empty:
                st.markdown('<div class="empty-state">No defensive setup data available.</div>', unsafe_allow_html=True)
            else:
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.bar(
                        defensive_df.sort_values("corners", ascending=False).head(15),
                        x="Defensive_setup", y="corners",
                        title="Most Common Defensive Setups",
                        hover_data=["shots", "total_xg", "shot_rate"],
                        color_discrete_sequence=[ACCENT],
                        labels={"Defensive_setup": "", "corners": "Corners"}
                    )
                    st.plotly_chart(figure_layout(fig, 430), use_container_width=True)
                with c2:
                    fig = px.bar(
                        defensive_df.sort_values("shot_rate", ascending=False).head(15),
                        x="Defensive_setup", y="shot_rate",
                        title="Shot Rate Allowed by Defensive Setup",
                        hover_data=["corners", "shots", "total_xg"],
                        color_discrete_sequence=[DANGER],
                        labels={"Defensive_setup": "", "shot_rate": "Shot Rate"}
                    )
                    st.plotly_chart(figure_layout(fig, 430), use_container_width=True)

                c3, c4 = st.columns(2)
                with c3:
                    fig = px.scatter(
                        defensive_df,
                        x="corners", y="shot_rate",
                        size="total_xg",
                        hover_name="Defensive_setup",
                        title="Sample Size vs Shot Rate Allowed",
                        color="xg_per_corner",
                        color_continuous_scale="Reds",
                        labels={"corners": "Sample Size", "shot_rate": "Shot Rate"}
                    )
                    st.plotly_chart(figure_layout(fig, 400), use_container_width=True)
                with c4:
                    fig = px.bar(
                        defensive_df.sort_values("xg_per_corner", ascending=False).head(12),
                        x="Defensive_setup", y="xg_per_corner",
                        title="xG/Corner Allowed by Setup",
                        color_discrete_sequence=[WARNING],
                        labels={"Defensive_setup": "", "xg_per_corner": "xG/Corner"}
                    )
                    st.plotly_chart(figure_layout(fig, 400), use_container_width=True)
                st.dataframe(defensive_df.sort_values("corners", ascending=False).reset_index(drop=True), use_container_width=True, height=360)
        else:
            st.markdown('<div class="empty-state">No defensive setup column found in data.</div>', unsafe_allow_html=True)

    with scout_tabs[5]:
        section_header("Advanced Search", "Filter corners by any combination of attributes")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            search_team = st.selectbox("Team", ["All"] + sorted(league_event_df["corner_team"].dropna().unique().tolist()), key="adv_team")
        with col_b:
            search_outcome = st.selectbox(
                "Min Outcome",
                ["Any", "Shot ≤3s", "First Contact Shot", "Shot"],
                key="adv_outcome"
            )
        with col_c:
            search_zone = st.selectbox(
                "End Zone",
                ["Any", "6-yard box", "Penalty area", "Deep box", "Outside danger zone"],
                key="adv_zone"
            )

        search_df = league_event_df.copy()
        if search_team != "All":
            search_df = search_df[search_df["corner_team"] == search_team]
        if search_outcome != "Any":
            search_df = search_df[search_df["outcome_bucket"] == search_outcome]
        if search_zone != "Any":
            search_df = search_df[search_df["end_zone"] == search_zone]

        st.markdown(f'<div style="color:{MUTED};font-size:0.88rem;margin-bottom:8px">{len(search_df)} events match your search criteria.</div>', unsafe_allow_html=True)

        show_cols = [c for c in [
            "Match", "corner_team", "Taker", "Shooter", "Minute", "SP_outcome",
            "shot_xg", "pass_technique", "delivery_zone", "end_zone", "outcome_bucket",
            "Defensive_setup", "corner_side", "delivery_length_band",
        ] if c in search_df.columns]
        st.dataframe(search_df[show_cols].sort_values(["shot_xg"], ascending=False).reset_index(drop=True), use_container_width=True, height=500)

        if not search_df.empty:
            st.download_button(
                "⬇ Download Search Results CSV",
                data=search_df.to_csv(index=False).encode("utf-8"),
                file_name="advanced_search_results.csv",
                mime="text/csv",
            )

    with scout_tabs[6]:
        section_header("Export Centre", "Download filtered data in multiple formats")
        st.markdown('<div class="sub-card">', unsafe_allow_html=True)
        st.markdown(
            f"**Current filter summary:** {len(league_event_df):,} events | "
            f"{league_event_df['match_id'].nunique()} matches | "
            f"{league_event_df['corner_team'].nunique()} teams"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        add_download_buttons(league_event_df, league_team_df, league_match_df)
        st.markdown("<br>", unsafe_allow_html=True)
        # Excel workbook export
        try:
            excel_bytes = download_excel(league_event_df, league_team_df, league_match_df)
            st.download_button(
                "⬇ Download Full Workbook (Excel)",
                data=excel_bytes,
                file_name="allsvenskan_corners_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=False,
            )
        except Exception as ex:
            st.warning(f"Excel export unavailable: {ex}. Install openpyxl.")

        st.markdown(
            '<div class="footer-note">All exports reflect the current filter state. Adjust sidebar filters to change the export scope.</div>',
            unsafe_allow_html=True,
        )

# =========================================================
# DATA HUB
# =========================================================
elif page == "Data Hub":
    section_header(
        "Data Hub",
        "Clean tables, summaries, and downloadable outputs. Use Visualisation Studio for charts.",
    )
    tabs = st.tabs([
        "📄 Raw Events",
        "🏟 Team Table",
        "📋 Match Table",
        "🎯 Shot Events",
        "🏹 Delivery Events",
        "👤 Taker Summary",
        "📍 Zone Table",
        "🛡 Defensive Table",
        "⬇ Downloads",
    ])

    with tabs[0]:
        section_header("Raw Events", f"{len(league_event_df):,} corner events in current filter")
        st.dataframe(league_event_df.reset_index(drop=True), use_container_width=True, height=620)

    with tabs[1]:
        section_header("Team Summary Table", "One row per team — all key metrics")
        st.dataframe(league_team_df.reset_index(drop=True), use_container_width=True, height=620)

    with tabs[2]:
        section_header("Match Table", "One row per match")
        st.dataframe(league_match_df.reset_index(drop=True), use_container_width=True, height=620)

    with tabs[3]:
        section_header("Shot Events", "All events where a shot was recorded")
        shot_only = league_event_df[league_event_df["led_to_shot"]].reset_index(drop=True)
        st.markdown(f'<div style="color:{MUTED};font-size:0.88rem;margin-bottom:6px">{len(shot_only)} shot events</div>', unsafe_allow_html=True)
        st.dataframe(shot_only, use_container_width=True, height=620)

    with tabs[4]:
        section_header("Delivery Events", "Corners with coordinate data")
        delivery_cols = [
            c for c in [
                "Match", "corner_team", "Taker", "Minute", "pass_technique",
                "pass_height", "pass_body_part", "pass_location_x", "pass_location_y",
                "pass_end_location_x", "pass_end_location_y", "delivery_zone", "end_zone",
                "corner_side", "delivery_length", "delivery_length_band", "SP_outcome",
                "outcome_bucket",
            ]
            if c in league_event_df.columns
        ]
        delivery_only = league_event_df.dropna(
            subset=["pass_location_x", "pass_location_y"], how="all"
        )
        st.markdown(f'<div style="color:{MUTED};font-size:0.88rem;margin-bottom:6px">{len(delivery_only)} events with delivery coords</div>', unsafe_allow_html=True)
        st.dataframe(delivery_only[delivery_cols].reset_index(drop=True), use_container_width=True, height=620)

    with tabs[5]:
        section_header("Taker Summary", "Aggregated per-taker statistics")
        taker_tbl = taker_summary_table(league_event_df)
        st.dataframe(taker_tbl.reset_index(drop=True), use_container_width=True, height=620)

    with tabs[6]:
        section_header("Zone Table", "Delivery zone × end zone breakdown per team")
        zone_tbl = team_insight_table(league_event_df)
        st.dataframe(zone_tbl.reset_index(drop=True), use_container_width=True, height=620)

    with tabs[7]:
        section_header("Defensive Setup Table", "Outcome aggregated by defensive shape")
        if "Defensive_setup" in league_event_df.columns:
            def_tbl = (
                league_event_df.groupby(["Defensive_setup", "corner_team"], dropna=False)
                .agg(
                    corners=("match_id", "size"),
                    shots=("led_to_shot", "sum"),
                    total_xg=("shot_xg", "sum"),
                )
                .reset_index()
            )
            def_tbl["shot_rate"] = def_tbl["shots"] / def_tbl["corners"].replace(0, np.nan)
            def_tbl["xg_per_corner"] = def_tbl["total_xg"] / def_tbl["corners"].replace(0, np.nan)
            st.dataframe(def_tbl.sort_values("corners", ascending=False).reset_index(drop=True), use_container_width=True, height=620)
        else:
            st.markdown('<div class="empty-state">No Defensive_setup column in data.</div>', unsafe_allow_html=True)

    with tabs[8]:
        section_header("Downloads", "Export filtered data")
        add_download_buttons(league_event_df, league_team_df, league_match_df)
        try:
            excel_bytes = download_excel(league_event_df, league_team_df, league_match_df)
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "⬇ Download Full Workbook (Excel)",
                data=excel_bytes,
                file_name="allsvenskan_corners_export.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )
        except Exception as ex:
            st.warning(f"Excel export unavailable: {ex}")

        st.markdown(
            f"""
            <div class="sub-card" style="margin-top:16px">
                <b>Filter scope</b><br>
                <span style="color:{MUTED};font-size:0.9rem">
                    {len(league_event_df):,} events &middot;
                    {league_event_df['match_id'].nunique()} matches &middot;
                    {league_event_df['corner_team'].nunique()} teams &middot;
                    {league_event_df['Taker'].nunique()} unique takers
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    f'<div class="footer-note" style="margin-top:24px;padding-top:12px;border-top:1px solid {BORDER};">'
    "Allsvenskan Set Piece Studio Pro · Corner Analytics · 2025 Season · "
    "Built with Streamlit + Plotly"
    "</div>",
    unsafe_allow_html=True,
)
