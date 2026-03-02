import os
import pandas as pd
import numpy as np
import streamlit as st

DEFAULT_FILE = "Allsvenskan - Corners 2025.xlsx"
DEFAULT_SHEET = "Sheet 1"

ACCENT_COLORS = [
    "#6366f1", "#a855f7", "#22d3a0", "#f97316",
    "#f43f5e", "#facc15", "#38bdf8", "#fb7185",
    "#34d399", "#818cf8",
]

def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([np.nan] * len(df), index=df.index, dtype="object")

def _safe_unique(series) -> list[str]:
    return sorted(pd.Series(series).dropna().astype(str).unique().tolist())

def _to_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _contains(s: pd.Series, q: str) -> pd.Series:
    return s.fillna("").astype(str).str.lower().str.contains(q, na=False)

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    """Load + normalize the Excel into a consistent schema used by pages."""
    if not os.path.exists(DEFAULT_FILE):
        st.error(
            f"Data file not found.\n\n"
            f"Place `{DEFAULT_FILE}` next to `Home.py` and restart."
        )
        st.stop()

    df = pd.read_excel(DEFAULT_FILE, sheet_name=DEFAULT_SHEET)
    df.columns = [str(c).strip() for c in df.columns]

    # Numeric time
    df["Minute_num"] = _to_num(_col(df, "Minute"))
    df["Second_num"] = _to_num(_col(df, "Second"))

    # Shot flags
    shot_ts = _col(df, "shot_timestamp").notna()
    shot_out = _col(df, "shot.outcome.name").notna()
    df["is_shot"] = (shot_ts | shot_out).fillna(False)

    # xG
    if "shot.statsbomb_xg" in df.columns:
        df["xg"] = _to_num(_col(df, "shot.statsbomb_xg")).fillna(0.0)
    else:
        df["xg"] = 0.0

    # Standard semantic fields used by pages
    df["team"] = _col(df, "pass_team_name")
    df["match"] = _col(df, "Match")
    df["taker"] = _col(df, "Taker")
    df["technique"] = _col(df, "pass.technique.name")
    df["height"] = _col(df, "pass.height.name")
    df["shot_outcome"] = _col(df, "shot.outcome.name")
    df["sp_outcome"] = _col(df, "SP_outcome")

    return df

# ── Plotly chart config ──
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'DM Sans', sans-serif", color="#b7b7d8", size=12),
    margin=dict(l=8, r=8, t=8, b=8),
    xaxis=dict(showgrid=False, zeroline=False, showline=False, tickfont=dict(size=11)),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,.06)",
        zeroline=False,
        showline=False,
        tickfont=dict(size=11),
    ),
    hoverlabel=dict(bgcolor="#14141b", font_size=12, font_family="'DM Sans', sans-serif"),
    showlegend=False,
)

def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&display=swap');

        :root{
          --bg: #0b0b10;
          --panel: #10101a;
          --panel2: #141427;
          --text: #e9e9ff;
          --muted: #b7b7d8;
          --muted2: #8f8fb3;
          --border: rgba(255,255,255,.08);
          --accent: #6366f1;
          --accent2: #a855f7;
        }

        html, body, [class*="css"]{
          font-family: "DM Sans", system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
        }

        /* App background */
        .stApp{
          background: radial-gradient(1200px 700px at 20% -10%, rgba(99,102,241,.18), transparent 60%),
                      radial-gradient(900px 600px at 90% 0%, rgba(168,85,247,.14), transparent 55%),
                      var(--bg);
          color: var(--text);
        }

        /* Sidebar branding */
        .sidebar-brand{
          display:flex; gap:12px; align-items:center;
          padding: 10px 10px 14px 10px;
          border-bottom: 1px solid var(--border);
          margin-bottom: 10px;
        }
        .sidebar-dot{
          width:34px; height:34px; border-radius:10px;
          background: linear-gradient(135deg, var(--accent), var(--accent2));
          box-shadow: 0 12px 30px rgba(99,102,241,.18);
          flex-shrink:0;
        }
        .sidebar-title{ font-weight: 800; font-size: 14px; color: var(--text); line-height: 1; }
        .sidebar-sub{ font-size: 12px; color: var(--muted2); margin-top: 2px; }

        /* Hero */
        .hero{
          background: linear-gradient(180deg, rgba(16,16,26,.85), rgba(16,16,26,.55));
          border: 1px solid var(--border);
          border-radius: 18px;
          padding: 18px 18px 16px 18px;
          margin: 6px 0 14px 0;
          box-shadow: 0 18px 50px rgba(0,0,0,.35);
        }
        .hero-eyebrow{ color: var(--muted2); font-size: 12px; letter-spacing: .2px; }
        .hero-title{ font-size: 44px; font-weight: 900; line-height: 1.0; margin: 6px 0 10px 0; }
        .hero-sub{ color: var(--muted); font-size: 14px; max-width: 920px; }
        .hero-badges{ margin-top: 12px; display:flex; gap:8px; flex-wrap: wrap; }
        .badge{
          display:inline-flex; align-items:center; gap:6px;
          padding: 6px 10px;
          border: 1px solid var(--border);
          border-radius: 999px;
          color: var(--muted);
          background: rgba(20,20,39,.55);
          font-size: 12px;
        }

        /* KPI grid */
        .kpi-grid{
          display:grid;
          grid-template-columns: repeat(6, minmax(0, 1fr));
          gap: 10px;
          margin: 10px 0 16px 0;
        }
        .kpi{
          background: rgba(16,16,26,.75);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 12px 12px 10px 12px;
        }
        .kpi-value{ font-size: 22px; font-weight: 900; color: var(--text); }
        .kpi-label{ font-size: 12px; color: var(--muted); margin-top: 2px; }
        .kpi-hint{ font-size: 11px; color: var(--muted2); margin-top: 2px; }

        @media (max-width: 1200px){
          .kpi-grid{ grid-template-columns: repeat(3, minmax(0, 1fr)); }
          .hero-title{ font-size: 36px; }
        }

        /* Section + cards */
        .section-title{
          font-size: 16px;
          font-weight: 800;
          margin: 6px 0 10px 0;
        }
        .card-grid{
          display:grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          gap: 10px;
        }
        @media (max-width: 1100px){
          .card-grid{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
        }
        @media (max-width: 700px){
          .card-grid{ grid-template-columns: 1fr; }
        }

        .navcard{
          display:block;
          text-decoration:none !important;
          background: rgba(16,16,26,.75);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 14px 14px 12px 14px;
          position: relative;
          overflow: hidden;
        }
        .navcard:hover{
          border-color: rgba(99,102,241,.35);
          box-shadow: 0 16px 55px rgba(0,0,0,.35);
          transform: translateY(-1px);
          transition: .18s ease;
        }
        .navcard-title{
          color: var(--text);
          font-weight: 900;
          font-size: 14px;
          margin-bottom: 6px;
        }
        .navcard-sub{
          color: var(--muted);
          font-size: 12px;
          line-height: 1.35;
          max-width: 95%;
        }
        .navcard-cta{
          position:absolute;
          right: 12px;
          top: 12px;
          color: var(--muted2);
          font-weight: 900;
        }

        /* Footer */
        .footer{
          color: var(--muted2);
          font-size: 12px;
          margin-top: 16px;
          padding-top: 10px;
          border-top: 1px solid var(--border);
        }

        /* Streamlit tweaks */
        .stDataFrame, .stPlotlyChart{
          background: transparent !important;
          border-radius: 14px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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
        st.dataframe(df_, use_container_width=True, hide_index=True)
        return

    if color_col and color_col in df_.columns:
        fig = px.bar(
            df_, x=x, y=y, color=color_col,
            color_discrete_sequence=ACCENT_COLORS,
            orientation=orientation,
        )
    else:
        fig = px.bar(
            df_, x=x, y=y,
            orientation=orientation,
            color_discrete_sequence=[ACCENT_COLORS[0]],
        )
        fig.update_traces(marker_line_width=0, marker_color=ACCENT_COLORS[0])

    fig.update_traces(
        hovertemplate=("%{x}<br>%{y}" if orientation == "v" else "%{y}<br>%{x}")
    )
    fig.update_layout(height=height, **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def styled_donut(df_, names, values, height=320):
    if df_ is None or len(df_) == 0:
        st.info("No data.")
        return
    if not PLOTLY_OK:
        st.dataframe(df_, use_container_width=True, hide_index=True)
        return

    fig = go.Figure(
        go.Pie(
            labels=df_[names],
            values=df_[values],
            hole=0.65,
            textinfo="percent",
            textfont=dict(size=11, color="#b7b7d8"),
            marker=dict(colors=ACCENT_COLORS, line=dict(color="#0b0b10", width=2)),
            hovertemplate="%{label}<br>%{value} (%{percent})",
        )
    )
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'DM Sans',sans-serif", color="#b7b7d8"),
        margin=dict(l=8, r=8, t=8, b=8),
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def styled_histogram(series, nbins=24, height=280):
    if not PLOTLY_OK or len(series.dropna()) == 0:
        st.info("No data.")
        return
    fig = px.histogram(
        series.dropna().to_frame("x"),
        x="x",
        nbins=nbins,
        color_discrete_sequence=[ACCENT_COLORS[0]],
    )
    fig.update_traces(marker_line_width=0, hovertemplate="Minute: %{x}<br>Count: %{y}")
    fig.update_layout(height=height, **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def styled_scatter(df_, x, y, text=None, height=340):
    if not PLOTLY_OK or len(df_) == 0:
        st.info("No data.")
        return
    fig = px.scatter(df_, x=x, y=y, text=text, color_discrete_sequence=ACCENT_COLORS)
    fig.update_traces(
        marker=dict(size=10, line=dict(width=0), color=ACCENT_COLORS[0], opacity=0.85),
        textfont=dict(size=10, color="#b7b7d8"),
        textposition="top center",
    )
    fig.update_layout(height=height, **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

def page_header(eyebrow: str, title: str, sub: str = ""):
    sub_html = f"<div class='hero-sub'>{sub}</div>" if sub else ""
    st.markdown(
        f"""
        <div class="hero" style="padding:14px 16px 12px 16px;margin-top:0;">
          <div class="hero-eyebrow">{eyebrow}</div>
          <div style="font-size:26px;font-weight:900;line-height:1.15;margin-top:6px;">{title}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

def kpi_strip(items: list[tuple[str, str, str]]):
    """items = list of (label, value, hint)"""
    cards = []
    for lbl, val, hint in items:
        cards.append(
            f"""
            <div class="kpi">
              <div class="kpi-value">{val}</div>
              <div class="kpi-label">{lbl}</div>
              <div class="kpi-hint">{hint}</div>
            </div>
            """
        )
    st.markdown(
        f"<div class='kpi-grid' style='margin-top:-6px;'>{''.join(cards)}</div>",
        unsafe_allow_html=True,
    )
