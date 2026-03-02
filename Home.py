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

# ── Hero ──
st.markdown(
    """
    <div class="hero">
      <div class="hero-eyebrow">Allsvenskan 2025 · StatsBomb Data</div>
      <div class="hero-title">Corner Kick<br/>Analytics</div>
      <div class="hero-sub">
        Deep-dive into every corner kick taken in Allsvenskan 2025 — delivery technique,
        taker profiles, xG generation, and set-piece outcomes.
      </div>
      <div class="hero-badges">
        <span class="badge">⚽ Live filters</span>
        <span class="badge">xG powered</span>
        <span class="badge">Full season</span>
        <span class="badge">All matches</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPIs ──
total = len(df)
n_teams = int(df["team"].nunique())
n_mat = df["match"].astype(str).replace("nan", np.nan).dropna().nunique()
shots = int(df["is_shot"].fillna(False).sum())
sr = shots / total if total else 0.0
xg_tot = float(df["xg"].fillna(0).sum()) if "xg" in df.columns else 0.0
goals = int(
    df.get("shot_outcome", pd.Series(dtype=str))
    .fillna("")
    .astype(str)
    .str.contains("Goal", case=False, na=False)
    .sum()
)
cpm = total / n_mat if n_mat else 0.0

st.markdown(
    f"""
    <div class="kpi-grid">
      <div class="kpi">
        <div class="kpi-value">{total:,}</div>
        <div class="kpi-label">Total corners</div>
        <div class="kpi-hint">Full season</div>
      </div>
      <div class="kpi">
        <div class="kpi-value">{n_mat:,}</div>
        <div class="kpi-label">Matches</div>
        <div class="kpi-hint">Allsvenskan 2025</div>
      </div>
      <div class="kpi">
        <div class="kpi-value">{cpm:.1f}</div>
        <div class="kpi-label">Corners / match</div>
        <div class="kpi-hint">League average</div>
      </div>
      <div class="kpi">
        <div class="kpi-value">{sr*100:.1f}%</div>
        <div class="kpi-label">Shot rate</div>
        <div class="kpi-hint">Corner → shot</div>
      </div>
      <div class="kpi">
        <div class="kpi-value">{xg_tot:.2f}</div>
        <div class="kpi-label">Total xG</div>
        <div class="kpi-hint">From corner shots</div>
      </div>
      <div class="kpi">
        <div class="kpi-value">{goals}</div>
        <div class="kpi-label">Goals</div>
        <div class="kpi-hint">Direct from corners</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Nav cards ──
st.markdown(
    """
    <div class="section-title">Explore the data</div>
    <div class="card-grid">
      <a class="navcard" href="./League_Overview">
        <div class="navcard-title">League Overview</div>
        <div class="navcard-sub">Volume by team, technique mix, xG rankings, shot outcomes and timing distribution.</div>
        <div class="navcard-cta">→</div>
      </a>
      <a class="navcard" href="./Team_Analysis">
        <div class="navcard-title">Team Analysis</div>
        <div class="navcard-sub">Drill into any club — taker profiles, delivery style, outcomes and efficiency.</div>
        <div class="navcard-cta">→</div>
      </a>
      <a class="navcard" href="./Match_View">
        <div class="navcard-title">Match View</div>
        <div class="navcard-sub">Select any fixture and inspect every corner — timing, technique, height and outcomes.</div>
        <div class="navcard-cta">→</div>
      </a>
      <a class="navcard" href="./Player_Profiles">
        <div class="navcard-title">Player Profiles</div>
        <div class="navcard-sub">Individual taker cards — volume, shot rate, xG per corner and preferences.</div>
        <div class="navcard-cta">→</div>
      </a>
      <a class="navcard" href="./Data_Explorer">
        <div class="navcard-title">Data Explorer</div>
        <div class="navcard-sub">Full filterable table with every corner event. Search, filter and export to CSV.</div>
        <div class="navcard-cta">→</div>
      </a>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="footer">
      Built with StatsBomb open data · Allsvenskan 2025 · Corner kick events
    </div>
    """,
    unsafe_allow_html=True,
)
