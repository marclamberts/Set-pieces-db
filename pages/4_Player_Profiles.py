import streamlit as st
import numpy as np
from utils import load_data, inject_css, _to_num, styled_donut, styled_histogram, styled_bar, page_header, kpi_strip, _safe_unique

st.set_page_config(page_title="Player Profiles · Corners", page_icon="👤", layout="wide")
inject_css()
df = load_data()

takers_all = sorted(df["taker"].dropna().astype(str).unique().tolist())

with st.sidebar:
    st.markdown("""
<div style="padding:4px 0 16px 0;">
  <div style="display:flex;gap:10px;align-items:center;">
    <div style="width:30px;height:30px;border-radius:8px;
         background:linear-gradient(135deg,#6366f1,#a855f7);flex-shrink:0;"></div>
    <div><div style="font-size:13px;font-weight:800;color:#f0f0f8;">Corner Analytics</div>
    <div style="font-size:10px;color:#5a5a7a;">Allsvenskan 2025</div></div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("#### Select Player")
    focus = st.selectbox("Taker", takers_all)

    st.markdown("#### Filters")
    teams_for_player = _safe_unique(df[df["taker"].astype(str)==focus]["team"])
    sel_teams = st.multiselect("Team context", teams_for_player, default=teams_for_player)

f = df[df["taker"].astype(str) == focus].copy()
if sel_teams: f = f[f["team"].isin(sel_teams)]

total  = len(f)
shots  = int(f["is_shot"].fillna(False).sum())
sr     = shots / total if total else 0
xg     = float(f["xg"].fillna(0).sum()) if "xg" in f.columns else 0.0
xg_c   = xg / total if total else 0
goals  = int(f.get("shot_outcome", __import__("pandas").Series(dtype=str))
             .fillna("").astype(str).str.contains("Goal", case=False, na=False).sum())
top_ht = "—"
if total and "height" in f.columns:
    vc = f["height"].fillna("Unknown").astype(str).value_counts()
    if len(vc): top_ht = vc.index[0]
top_tech = "—"
if total and "technique" in f.columns:
    vc2 = f["technique"].fillna("Unknown").astype(str).value_counts()
    if len(vc2): top_tech = vc2.index[0]
team_label = ", ".join(sel_teams) if sel_teams else "All teams"

page_header("Player Profiles", focus, f"Corners taker · {team_label}")

kpi_strip([
    ("Corners", f"{total:,}", "Volume"),
    ("Shot rate", f"{sr*100:.1f}%", "→ shot"),
    ("Total xG", f"{xg:.3f}", "From shots"),
    ("xG / corner", f"{xg_c:.4f}", "Efficiency"),
    ("Top height", top_ht, "Preferred"),
    ("Top technique", top_tech, "Preferred"),
])

c1, c2 = st.columns(2)
with c1:
    st.markdown("<div class='card'><div class='card-title'>Technique Mix</div>", unsafe_allow_html=True)
    tech = f.groupby("technique", dropna=False).size().reset_index(name="n")
    styled_donut(tech, "technique", "n", height=280)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'><div class='card-title'>Delivery Height Mix</div>", unsafe_allow_html=True)
    ht = f.groupby("height", dropna=False).size().reset_index(name="n")
    styled_donut(ht, "height", "n", height=280)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'><div class='card-title'>Corner Timing</div><div class='card-sub'>Minute distribution</div>", unsafe_allow_html=True)
styled_histogram(_to_num(f["Minute_num"]), nbins=18, height=240)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'><div class='card-title'>Set Piece Outcomes</div>", unsafe_allow_html=True)
sp = f.groupby("sp_outcome", dropna=False).size().sort_values(ascending=True).reset_index(name="count")
styled_bar(sp, x="count", y="sp_outcome", orientation="h", height=260)
st.markdown("</div>", unsafe_allow_html=True)

# Compare vs league average
import pandas as pd
league_sr  = df["is_shot"].fillna(False).sum() / len(df) if len(df) else 0
league_xgc = df["xg"].fillna(0).sum() / len(df) if len(df) else 0

st.markdown(f"""
<div class="card">
  <div class="card-title">vs League Average</div>
  <div class="card-sub">How this taker compares to the full-season baseline</div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px;">
    <div class="kpi">
      <div class="kpi-label">Shot rate — player</div>
      <div class="kpi-value" style="color:{'#22d3a0' if sr > league_sr else '#f43f5e'}">
        {sr*100:.1f}%
      </div>
      <div class="kpi-hint">League avg: {league_sr*100:.1f}%</div>
    </div>
    <div class="kpi">
      <div class="kpi-label">xG/corner — player</div>
      <div class="kpi-value" style="color:{'#22d3a0' if xg_c > league_xgc else '#f43f5e'}">
        {xg_c:.4f}
      </div>
      <div class="kpi-hint">League avg: {league_xgc:.4f}</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)
