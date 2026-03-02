import streamlit as st
import pandas as pd
import numpy as np

from utils import load_data, inject_css

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

    st.markdown("### Pages")

    # ✅ Streamlit Cloud-safe: entrypoint is streamlit_app.py
    try:
        st.page_link("streamlit_app.py", label="🏠 Home", icon="🏠")
        st.page_link("pages/1_League_Overview.py", label="🏟️ League Overview", icon="🏟️")
        st.page_link("pages/2_Team_Analysis.py", label="🧭 Team Analysis", icon="🧭")
        st.page_link("pages/3_Match_View.py", label="🗓️ Match View", icon="🗓️")
        st.page_link("pages/4_Player_Profiles.py", label="🧑‍💼 Player Profiles", icon="🧑‍💼")
        st.page_link("pages/5_Data_Explorer.py", label="📋 Data Explorer", icon="📋")
    except Exception:
        st.markdown(
            """
            - [🏟️ League Overview](./League_Overview)
            - [🧭 Team Analysis](./Team_Analysis)
            - [🗓️ Match View](./Match_View)
            - [🧑‍💼 Player Profiles](./Player_Profiles)
            - [📋 Data Explorer](./Data_Explorer)
            """
        )

st.markdown(
    """
    <div class="hero">
      <div class="hero-eyebrow">Allsvenskan 2025 · StatsBomb Data</div>
      <div class="hero-title">Corner Kick<br/>Analytics</div>
      <div class="hero-sub">
        Explore corner kick deliveries, taker profiles, outcomes, and xG generated from corners.
      </div>
      <div class="hero-badges">
        <span class="badge">⚽ Filters</span>
        <span class="badge">xG powered</span>
        <span class="badge">All matches</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

total = int(len(df))
teams = int(df.get("team", pd.Series(dtype=object)).nunique())

match_series = df.get("match", pd.Series(dtype=object)).astype(str).replace("nan", np.nan).dropna()
matches = int(match_series.nunique())

is_shot = df.get("is_shot", pd.Series([False] * len(df))).fillna(False).astype(bool)
shots = int(is_shot.sum())
shot_rate = shots / total if total else 0.0

xg = float(df.get("xg", pd.Series([0.0] * len(df))).fillna(0).sum())
cpm = total / matches if matches else 0.0

st.markdown(
    f"""
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-value">{total:,}</div>
        <div class="kpi-label">Total corners</div>
        <div class="kpi-hint">Full season</div>
      </div>
      <div class="kpi">
        <div class="kpi-value">{matches:,}</div>
        <div class="kpi-label">Matches</div>
        <div class="kpi-hint">Unique fixtures</div>
      </div>
      <div class="kpi">
        <div class="kpi-value">{teams:,}</div>
        <div class="kpi-label">Teams</div>
        <div class="kpi-hint">Unique clubs</div>
      </div>
      <div class="kpi">
        <div class="kpi-value">{cpm:.1f}</div>
        <div class="kpi-label">Corners / match</div>
        <div class="kpi-hint">Average</div>
      </div>
      <div class="kpi">
        <div class="kpi-value">{shot_rate*100:.1f}%</div>
        <div class="kpi-label">Shot rate</div>
        <div class="kpi-hint">Corner → shot</div>
      </div>
      <div class="kpi">
        <div class="kpi-value">{xg:.2f}</div>
        <div class="kpi-label">Total xG</div>
        <div class="kpi-hint">From corner shots</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="section-title">Explore</div>
    <div class="card-grid">
      <a class="navcard" href="./League_Overview">
        <div class="navcard-title">League Overview</div>
        <div class="navcard-sub">Team volumes, technique mix, xG rankings, outcomes, timing.</div>
        <div class="navcard-cta">→</div>
      </a>
      <a class="navcard" href="./Team_Analysis">
        <div class="navcard-title">Team Analysis</div>
        <div class="navcard-sub">Pick a club and break down takers, styles, outcomes and efficiency.</div>
        <div class="navcard-cta">→</div>
      </a>
      <a class="navcard" href="./Match_View">
        <div class="navcard-title">Match View</div>
        <div class="navcard-sub">Select a fixture and inspect corners by team, timing and results.</div>
        <div class="navcard-cta">→</div>
      </a>
      <a class="navcard" href="./Player_Profiles">
        <div class="navcard-title">Player Profiles</div>
        <div class="navcard-sub">Individual taker profiles: volume, xG, style preferences.</div>
        <div class="navcard-cta">→</div>
      </a>
      <a class="navcard" href="./Data_Explorer">
        <div class="navcard-title">Data Explorer</div>
        <div class="navcard-sub">Filter, search, inspect and export the full corner event table.</div>
        <div class="navcard-cta">→</div>
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="footer">
      Data rows loaded: {total:,}. Excel expected: <code>{'Allsvenskan - Corners 2025.xlsx'}</code>.
    </div>
    """,
    unsafe_allow_html=True,
)