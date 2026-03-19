import os
import numpy as np
import pandas as pd
import streamlit as st

DEFAULT_FILE = "Allsvenskan - Corners 2025.xlsx"
DEFAULT_SHEET = "Sheet 1"
ACCENT_COLORS = [
    "#6366f1", "#a855f7", "#22d3a0", "#f97316",
    "#f43f5e", "#facc15", "#38bdf8", "#fb7185",
    "#34d399", "#818cf8",
]


# ----------------------------
# Helpers
# ----------------------------
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([np.nan] * len(df), index=df.index, dtype="object")


def _to_num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    if not os.path.exists(DEFAULT_FILE):
        st.error(
            f"Data file not found.\n\nPlace `{DEFAULT_FILE}` next to this app file and restart."
        )
        st.stop()

    df = pd.read_excel(DEFAULT_FILE, sheet_name=DEFAULT_SHEET)
    df.columns = [str(c).strip() for c in df.columns]

    df["Minute_num"] = _to_num(_col(df, "Minute"))
    df["Second_num"] = _to_num(_col(df, "Second"))

    shot_ts = _col(df, "shot_timestamp").notna()
    shot_out = _col(df, "shot.outcome.name").notna()
    df["is_shot"] = (shot_ts | shot_out).fillna(False)

    if "shot.statsbomb_xg" in df.columns:
        df["xg"] = _to_num(_col(df, "shot.statsbomb_xg")).fillna(0.0)
    else:
        df["xg"] = 0.0

    df["team"] = _col(df, "pass_team_name").fillna("Unknown")
    df["match"] = _col(df, "Match").fillna("Unknown")
    df["taker"] = _col(df, "Taker").fillna("Unknown")
    df["technique"] = _col(df, "pass.technique.name").fillna("Unknown")
    df["height"] = _col(df, "pass.height.name").fillna("Unknown")
    df["shot_outcome"] = _col(df, "shot.outcome.name").fillna("")
    df["sp_outcome"] = _col(df, "SP_outcome").fillna("Unknown")

    return df


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
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700;800&display=swap');

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

        .stApp{
          background: radial-gradient(1200px 700px at 20% -10%, rgba(99,102,241,.18), transparent 60%),
                      radial-gradient(900px 600px at 90% 0%, rgba(168,85,247,.14), transparent 55%),
                      var(--bg);
          color: var(--text);
        }

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

        .hero{
          background: linear-gradient(180deg, rgba(16,16,26,.85), rgba(16,16,26,.55));
          border: 1px solid var(--border);
          border-radius: 18px;
          padding: 18px 18px 16px 18px;
          margin: 6px 0 14px 0;
          box-shadow: 0 18px 50px rgba(0,0,0,.35);
        }
        .hero-eyebrow{ color: var(--muted2); font-size: 12px; letter-spacing: .2px; }
        .hero-title{ font-size: 40px; font-weight: 900; line-height: 1.0; margin: 6px 0 10px 0; }
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

        .section-title{
          font-size: 16px;
          font-weight: 800;
          margin: 6px 0 10px 0;
        }

        .insight-grid{
          display:grid;
          grid-template-columns: repeat(4, minmax(0,1fr));
          gap:10px;
          margin: 0 0 16px 0;
        }
        .insight-card{
          background: rgba(16,16,26,.75);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 12px 12px 10px 12px;
        }
        .insight-label{ font-size: 11px; color: var(--muted2); text-transform: uppercase; letter-spacing:.4px; }
        .insight-value{ font-size: 16px; font-weight: 800; color: var(--text); margin-top:4px; }
        .insight-sub{ font-size: 12px; color: var(--muted); margin-top:4px; }

        .footer{
          color: var(--muted2);
          font-size: 12px;
          margin-top: 16px;
          padding-top: 10px;
          border-top: 1px solid var(--border);
        }

        .stDataFrame, .stPlotlyChart{
          background: transparent !important;
          border-radius: 14px;
        }

        @media (max-width: 1200px){
          .kpi-grid{ grid-template-columns: repeat(3, minmax(0, 1fr)); }
          .insight-grid{ grid-template-columns: repeat(2, minmax(0, 1fr)); }
          .hero-title{ font-size: 34px; }
        }
        @media (max-width: 700px){
          .kpi-grid, .insight-grid{ grid-template-columns: 1fr; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


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
        fig = px.bar(df_, x=x, y=y, color=color_col, orientation=orientation, color_discrete_sequence=ACCENT_COLORS)
    else:
        fig = px.bar(df_, x=x, y=y, orientation=orientation, color_discrete_sequence=[ACCENT_COLORS[0]])
        fig.update_traces(marker_line_width=0, marker_color=ACCENT_COLORS[0])

    fig.update_traces(hovertemplate=("%{x}<br>%{y}" if orientation == "v" else "%{y}<br>%{x}"))
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
    fig = px.histogram(series.dropna().to_frame("x"), x="x", nbins=nbins, color_discrete_sequence=[ACCENT_COLORS[0]])
    fig.update_traces(marker_line_width=0, hovertemplate="Minute: %{x}<br>Count: %{y}")
    fig.update_layout(height=height, **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def styled_scatter(df_, x, y, text=None, height=340):
    if not PLOTLY_OK or len(df_) == 0:
        st.info("No data.")
        return
    fig = px.scatter(df_, x=x, y=y, text=text, color_discrete_sequence=[ACCENT_COLORS[0]])
    fig.update_traces(
        marker=dict(size=10, line=dict(width=0), color=ACCENT_COLORS[0], opacity=0.85),
        textfont=dict(size=10, color="#b7b7d8"),
        textposition="top center",
        hovertemplate="%{text}<br>%{x}<br>%{y}<extra></extra>" if text else None,
    )
    fig.update_layout(height=height, **CHART_LAYOUT)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ----------------------------
# App
# ----------------------------
st.set_page_config(
    page_title="Corners · Allsvenskan 2025",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
df = load_data()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
          <div class="sidebar-dot"></div>
          <div>
            <div class="sidebar-title">Corner Analytics</div>
            <div class="sidebar-sub">Allsvenskan 2025</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Quick Filters")

    teams_all = sorted(df["team"].dropna().astype(str).unique().tolist())
    techniques_all = sorted(df["technique"].dropna().astype(str).unique().tolist())

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Reset"):
            for key in ["team_sel", "tech_sel", "top_n", "only_shots"]:
                if key in st.session_state:
                    del st.session_state[key]
    with col_b:
        st.toggle("Only shots", value=False, key="only_shots")

    team_sel = st.multiselect("Teams", teams_all, default=teams_all, key="team_sel")
    tech_sel = st.multiselect("Technique", techniques_all, default=techniques_all, key="tech_sel")
    top_n = st.slider("Top teams shown", 5, min(20, max(5, len(teams_all))), min(10, max(5, len(teams_all))), key="top_n")

    mins = _to_num(df["Minute_num"]).dropna()
    minute_range = None
    if len(mins) > 0:
        min_val = int(mins.min())
        max_val = int(mins.max())
        if min_val < max_val:
            minute_range = st.slider("Minute range", min_val, max_val, (min_val, max_val))
        else:
            st.caption(f"Minute range: all events at minute {min_val}")

f = df.copy()
if team_sel:
    f = f[f["team"].isin(team_sel)]
if tech_sel:
    f = f[f["technique"].isin(tech_sel)]
if minute_range:
    f = f[_to_num(f["Minute_num"]).between(minute_range[0], minute_range[1])]
if st.session_state.get("only_shots"):
    f = f[f["is_shot"].fillna(False)]

if len(f) == 0:
    st.markdown(
        """
        <div class="hero">
          <div class="hero-eyebrow">Allsvenskan 2025</div>
          <div class="hero-title">No results</div>
          <div class="hero-sub">Try widening your filters to bring events back into the view.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


total = int(len(f))
teams = int(f["team"].nunique())
matches = int(f["match"].astype(str).replace("nan", np.nan).dropna().nunique())
shots = int(f["is_shot"].fillna(False).sum())
shot_rate = shots / total if total else 0.0
xg = float(f["xg"].fillna(0).sum())
cpm = total / matches if matches else 0.0
goals = int(f["shot_outcome"].fillna("").astype(str).str.contains("goal", case=False, na=False).sum())

by_team = (
    f.groupby("team", dropna=False)
    .agg(corners=("team", "size"), xg=("xg", "sum"), shots=("is_shot", "sum"))
    .reset_index()
)
by_team["xg_per_corner"] = by_team["xg"] / by_team["corners"].replace(0, np.nan)
by_team["shot_rate"] = by_team["shots"] / by_team["corners"].replace(0, np.nan)
by_team = by_team.sort_values(["corners", "xg"], ascending=[False, False])

tech = (
    f.assign(technique=f["technique"].fillna("Unknown"))
    .groupby("technique", dropna=False)
    .size()
    .reset_index(name="n")
    .sort_values("n", ascending=False)
)

best_xg_team_row = by_team.dropna(subset=["xg_per_corner"]).sort_values("xg_per_corner", ascending=False)
most_active_team = by_team.iloc[0]["team"] if len(by_team) else "—"
best_xg_team = best_xg_team_row.iloc[0]["team"] if len(best_xg_team_row) else "—"
most_common_tech = tech.iloc[0]["technique"] if len(tech) else "—"

filtered_match_scores = (
    f.groupby("match", dropna=False)
    .agg(corners=("match", "size"), xg=("xg", "sum"), shots=("is_shot", "sum"))
    .reset_index()
)
filtered_match_scores["score"] = filtered_match_scores["xg"] * 3 + filtered_match_scores["shots"] + filtered_match_scores["corners"] * 0.1
featured_match = filtered_match_scores.sort_values("score", ascending=False).iloc[0]["match"] if len(filtered_match_scores) else "—"

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-eyebrow">Allsvenskan 2025 · Corner Kick Events</div>
      <div class="hero-title">Corner Kick Analytics</div>
      <div class="hero-sub">
        A cleaner, sharper dashboard for exploring corner volume, delivery style, shot conversion and xG generation.
        <br/><span style="color: var(--muted2);">Filters applied:</span>
        <span class="badge">Teams: {len(team_sel)}</span>
        <span class="badge">Techniques: {len(tech_sel)}</span>
        <span class="badge">Only shots: {'Yes' if st.session_state.get('only_shots') else 'No'}</span>
      </div>
      <div class="hero-badges">
        <span class="badge">Top team: {most_active_team}</span>
        <span class="badge">Best xG/corner: {best_xg_team}</span>
        <span class="badge">Top technique: {most_common_tech}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="kpi-grid">
      <div class="kpi"><div class="kpi-value">{total:,}</div><div class="kpi-label">Corners</div><div class="kpi-hint">Current selection</div></div>
      <div class="kpi"><div class="kpi-value">{matches:,}</div><div class="kpi-label">Matches</div><div class="kpi-hint">Unique fixtures</div></div>
      <div class="kpi"><div class="kpi-value">{teams:,}</div><div class="kpi-label">Teams</div><div class="kpi-hint">Unique clubs</div></div>
      <div class="kpi"><div class="kpi-value">{cpm:.1f}</div><div class="kpi-label">Corners / match</div><div class="kpi-hint">Average pace</div></div>
      <div class="kpi"><div class="kpi-value">{shot_rate*100:.1f}%</div><div class="kpi-label">Shot rate</div><div class="kpi-hint">Corner → shot</div></div>
      <div class="kpi"><div class="kpi-value">{xg:.2f}</div><div class="kpi-label">Total xG</div><div class="kpi-hint">Goals: {goals}</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="insight-grid">
      <div class="insight-card"><div class="insight-label">Most active team</div><div class="insight-value">{most_active_team}</div><div class="insight-sub">Highest corner volume in current filters</div></div>
      <div class="insight-card"><div class="insight-label">Best xG per corner</div><div class="insight-value">{best_xg_team}</div><div class="insight-sub">Most efficient corner creator</div></div>
      <div class="insight-card"><div class="insight-label">Most common technique</div><div class="insight-value">{most_common_tech}</div><div class="insight-sub">Dominant delivery profile</div></div>
      <div class="insight-card"><div class="insight-label">Featured match</div><div class="insight-value">{featured_match}</div><div class="insight-sub">Chosen from the filtered dataset</div></div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab1, tab2, tab3 = st.tabs(["Volume", "Quality", "Timing"])

with tab1:
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("<div class='section-title'>Top Teams</div><div class='hero-sub'>Corner volume by team</div>", unsafe_allow_html=True)
        top_corners = by_team.head(top_n).sort_values("corners", ascending=True)
        styled_bar(top_corners, x="corners", y="team", orientation="h", height=420)
    with right:
        st.markdown("<div class='section-title'>Delivery Profile</div><div class='hero-sub'>Technique distribution</div>", unsafe_allow_html=True)
        styled_donut(tech, "technique", "n", height=420)

with tab2:
    left, right = st.columns([1.15, 1], gap="large")
    with left:
        st.markdown("<div class='section-title'>Volume vs Quality</div><div class='hero-sub'>Which teams turn corners into xG?</div>", unsafe_allow_html=True)
        scatter_df = by_team.sort_values("corners", ascending=False).head(20)
        styled_scatter(scatter_df, x="corners", y="xg", text="team", height=420)
    with right:
        st.markdown("<div class='section-title'>Top xG Teams</div><div class='hero-sub'>Expected goals created from corners</div>", unsafe_allow_html=True)
        top_xg = by_team.head(top_n).sort_values("xg", ascending=True)
        styled_bar(top_xg, x="xg", y="team", orientation="h", height=420)

with tab3:
    left, right = st.columns([1.2, 1], gap="large")
    with left:
        st.markdown("<div class='section-title'>Timing</div><div class='hero-sub'>When corners happen</div>", unsafe_allow_html=True)
        styled_histogram(_to_num(f["Minute_num"]), nbins=28, height=360)
    with right:
        st.markdown("<div class='section-title'>Featured Match</div><div class='hero-sub'>Based on current filters</div>", unsafe_allow_html=True)
        filtered_matches = filtered_match_scores.sort_values(["score", "corners"], ascending=[False, False])["match"].astype(str).tolist()
        match_pick = st.selectbox("Match", filtered_matches, index=0)
        mf = f[f["match"].astype(str) == str(match_pick)].copy()
        m_total = len(mf)
        m_shots = int(mf["is_shot"].fillna(False).sum())
        m_xg = float(mf["xg"].fillna(0).sum())
        m_sr = (m_shots / m_total) if m_total else 0.0

        st.markdown(
            f"""
            <div class="hero" style="padding:12px 14px 10px 14px;">
              <div class="hero-eyebrow">Match snapshot</div>
              <div style="font-weight:900;font-size:16px;line-height:1.15;margin-top:4px;">{match_pick}</div>
              <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;margin-top:12px;">
                <div class="kpi" style="padding:10px 10px 8px 10px;"><div class="kpi-value" style="font-size:18px;">{m_total:,}</div><div class="kpi-label">Corners</div><div class="kpi-hint">This match</div></div>
                <div class="kpi" style="padding:10px 10px 8px 10px;"><div class="kpi-value" style="font-size:18px;">{m_sr*100:.1f}%</div><div class="kpi-label">Shot rate</div><div class="kpi-hint">Corner → shot</div></div>
                <div class="kpi" style="padding:10px 10px 8px 10px;"><div class="kpi-value" style="font-size:18px;">{m_xg:.2f}</div><div class="kpi-label">xG</div><div class="kpi-hint">From shots</div></div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        mt = mf.groupby("team", dropna=False).size().reset_index(name="corners").sort_values("corners", ascending=True)
        styled_bar(mt, x="corners", y="team", orientation="h", height=240)

st.markdown(
    f"""
    <div class="footer">
      Showing <b>{total:,}</b> filtered corner events. Expected Excel file: <code>{DEFAULT_FILE}</code>.
    </div>
    """,
    unsafe_allow_html=True,
)
