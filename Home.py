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
    st.markdown("""
<div style="padding:4px 0 16px 0;">
  <div style="display:flex;gap:10px;align-items:center;">
    <div style="width:30px;height:30px;border-radius:8px;
         background:linear-gradient(135deg,#6366f1,#a855f7);
         box-shadow:0 6px 18px rgba(99,102,241,.4);flex-shrink:0;"></div>
    <div>
      <div style="font-size:13px;font-weight:800;color:#f0f0f8;letter-spacing:-.02em;">Corner Analytics</div>
      <div style="font-size:10px;color:#5a5a7a;">Allsvenskan 2025</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──
st.markdown("""
<div class="hero">
  <div class="hero-bg"></div>
  <div class="hero-orb hero-orb1"></div>
  <div class="hero-orb hero-orb2"></div>
  <div class="hero-content">
    <div class="hero-eyebrow">Allsvenskan 2025 · StatsBomb Data</div>
    <h1 class="hero-title">Corner Kick<br><span class="hero-accent">Analytics</span></h1>
    <p class="hero-desc">
      Deep-dive into every corner kick taken in Allsvenskan 2025 —
      delivery technique, taker profiles, xG generation, and set-piece outcomes.
    </p>
    <div class="hero-pills">
      <span class="pill">⚽ Live filters</span>
      <span class="pill">📊 xG powered</span>
      <span class="pill">🏆 Full season</span>
      <span class="pill">📅 All matches</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── KPIs ──
total   = len(df)
n_teams = df["team"].nunique()
n_mat   = df["match"].astype(str).replace("nan", np.nan).dropna().nunique()
shots   = int(df["is_shot"].fillna(False).sum())
sr      = shots / total if total else 0
xg_tot  = float(df["xg"].fillna(0).sum()) if "xg" in df.columns else 0.0
goals   = int(df.get("shot_outcome", pd.Series(dtype=str)).fillna("").astype(str)
             .str.contains("Goal", case=False, na=False).sum())
cpm     = total / n_mat if n_mat else 0

st.markdown(f"""
<div class="stat-grid">
  <div class="stat-card">
    <div class="stat-icon">📐</div>
    <div class="stat-val">{total:,}</div>
    <div class="stat-label">Total corners</div>
    <div class="stat-sub">Full season</div>
  </div>
  <div class="stat-card">
    <div class="stat-icon">🏟️</div>
    <div class="stat-val">{n_mat:,}</div>
    <div class="stat-label">Matches</div>
    <div class="stat-sub">Allsvenskan 2025</div>
  </div>
  <div class="stat-card">
    <div class="stat-icon">⚡</div>
    <div class="stat-val">{cpm:.1f}</div>
    <div class="stat-label">Corners / match</div>
    <div class="stat-sub">League average</div>
  </div>
  <div class="stat-card">
    <div class="stat-icon">🎯</div>
    <div class="stat-val">{sr*100:.1f}%</div>
    <div class="stat-label">Shot rate</div>
    <div class="stat-sub">Corner → shot</div>
  </div>
  <div class="stat-card">
    <div class="stat-icon">💡</div>
    <div class="stat-val">{xg_tot:.2f}</div>
    <div class="stat-label">Total xG</div>
    <div class="stat-sub">From corner shots</div>
  </div>
  <div class="stat-card">
    <div class="stat-icon">🥅</div>
    <div class="stat-val">{goals}</div>
    <div class="stat-label">Goals</div>
    <div class="stat-sub">Direct from corners</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Nav cards ──
st.markdown("""
<div class="nav-section-title">Explore the data</div>
<div class="nav-grid">
  <a href="/League_Overview" target="_self" class="nav-card">
    <div class="nav-card-icon">🌍</div>
    <div class="nav-card-body">
      <div class="nav-card-title">League Overview</div>
      <div class="nav-card-desc">Volume by team, technique mix, xG rankings, shot outcomes and timing distribution across the full league.</div>
    </div>
    <div class="nav-card-arrow">→</div>
  </a>
  <a href="/Team_Analysis" target="_self" class="nav-card">
    <div class="nav-card-icon">🛡️</div>
    <div class="nav-card-body">
      <div class="nav-card-title">Team Analysis</div>
      <div class="nav-card-desc">Drill into any club — taker profiles, delivery style, set-piece outcomes and efficiency scatter.</div>
    </div>
    <div class="nav-card-arrow">→</div>
  </a>
  <a href="/Match_View" target="_self" class="nav-card">
    <div class="nav-card-icon">📅</div>
    <div class="nav-card-body">
      <div class="nav-card-title">Match View</div>
      <div class="nav-card-desc">Select any fixture and inspect every corner — timing, technique, height and shot outcomes for that game.</div>
    </div>
    <div class="nav-card-arrow">→</div>
  </a>
  <a href="/Player_Profiles" target="_self" class="nav-card">
    <div class="nav-card-icon">👤</div>
    <div class="nav-card-body">
      <div class="nav-card-title">Player Profiles</div>
      <div class="nav-card-desc">Individual taker cards — volume, shot rate, xG per corner, technique and height preferences.</div>
    </div>
    <div class="nav-card-arrow">→</div>
  </a>
  <a href="/Data_Explorer" target="_self" class="nav-card">
    <div class="nav-card-icon">🔍</div>
    <div class="nav-card-body">
      <div class="nav-card-title">Data Explorer</div>
      <div class="nav-card-desc">Full filterable table with every corner event. Search, filter and export to CSV.</div>
    </div>
    <div class="nav-card-arrow">→</div>
  </a>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
  Built with StatsBomb open data · Allsvenskan 2025 · Corner kick events
</div>
""", unsafe_allow_html=True)
