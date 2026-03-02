import streamlit as st
import numpy as np
import pandas as pd

from utils import (
    load_data, inject_css, _to_num, _safe_unique,
    styled_donut, styled_histogram, styled_bar,
    page_header, kpi_strip
)

st.set_page_config(page_title="Player Profiles · Corners", page_icon="🧑‍💼", layout="wide")
inject_css()
df = load_data()

takers_all = sorted(df["taker"].dropna().astype(str).unique().tolist())

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

    st.markdown("#### Select Player")
    focus = st.selectbox("Taker", takers_all)

    st.markdown("#### Filters")
    teams_for_player = _safe_unique(df[df["taker"].astype(str) == str(focus)]["team"])
    sel_teams = st.multiselect("Team context", teams_for_player, default=teams_for_player)

f = df[df["taker"].astype(str) == str(focus)].copy()
if sel_teams:
    f = f[f["team"].isin(sel_teams)]

total = len(f)
shots = int(f["is_shot"].fillna(False).sum())
sr = shots / total if total else 0.0
xg = float(f["xg"].fillna(0).sum()) if "xg" in f.columns else 0.0
xg_c = xg / total if total else 0.0
goals = int(
    f.get("shot_outcome", pd.Series(dtype=str))
    .fillna("")
    .astype(str)
    .str.contains("Goal", case=False, na=False)
    .sum()
)

top_ht = "—"
if total and "height" in f.columns:
    vc = f["height"].fillna("Unknown").astype(str).value_counts()
    if len(vc):
        top_ht = vc.index[0]

top_tech = "—"
if total and "technique" in f.columns:
    vc2 = f["technique"].fillna("Unknown").astype(str).value_counts()
    if len(vc2):
        top_tech = vc2.index[0]

team_label = ", ".join(sel_teams) if sel_teams else "All teams"

page_header("Player Profiles", focus, f"Corners taker · {team_label}")
kpi_strip(
    [
        ("Corners", f"{total:,}", "Volume"),
        ("Shot rate", f"{sr*100:.1f}%", "→ shot"),
        ("Total xG", f"{xg:.3f}", "From shots"),
        ("xG / corner", f"{xg_c:.4f}", "Efficiency"),
        ("Top height", top_ht, "Preferred"),
        ("Top technique", top_tech, "Preferred"),
    ]
)

c1, c2 = st.columns(2)

with c1:
    st.markdown("<div class='section-title'>Technique Mix</div>", unsafe_allow_html=True)
    tech = f.groupby("technique", dropna=False).size().reset_index(name="n")
    styled_donut(tech, "technique", "n", height=280)

with c2:
    st.markdown("<div class='section-title'>Delivery Height Mix</div>", unsafe_allow_html=True)
    ht = f.groupby("height", dropna=False).size().reset_index(name="n")
    styled_donut(ht, "height", "n", height=280)

st.markdown("<div class='section-title'>Corner Timing</div><div class='hero-sub'>Minute distribution</div>", unsafe_allow_html=True)
styled_histogram(_to_num(f["Minute_num"]), nbins=18, height=240)

st.markdown("<div class='section-title'>Set Piece Outcomes</div>", unsafe_allow_html=True)
sp = f.groupby("sp_outcome", dropna=False).size().sort_values(ascending=True).reset_index(name="count")
styled_bar(sp, x="count", y="sp_outcome", orientation="h", height=260)

league_sr = df["is_shot"].fillna(False).sum() / len(df) if len(df) else 0.0
league_xgc = df["xg"].fillna(0).sum() / len(df) if len(df) else 0.0

st.markdown(
    f"""
    <div class="hero" style="padding:14px 16px 12px 16px;">
      <div class="section-title" style="margin:0 0 6px 0;">vs League Average</div>
      <div class="hero-sub">How this taker compares to the full-season baseline</div>
      <div style="display:flex;gap:18px;flex-wrap:wrap;margin-top:10px;">
        <div>
          <div class="kpi-label">Shot rate — player</div>
          <div class="kpi-value" style="font-size:20px;">{sr*100:.1f}%</div>
          <div class="kpi-hint">League avg: {league_sr*100:.1f}%</div>
        </div>
        <div>
          <div class="kpi-label">xG/corner — player</div>
          <div class="kpi-value" style="font-size:20px;">{xg_c:.4f}</div>
          <div class="kpi-hint">League avg: {league_xgc:.4f}</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
