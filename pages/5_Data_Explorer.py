import streamlit as st
import numpy as np
from utils import load_data, inject_css, _to_num, _contains, _safe_unique, page_header

st.set_page_config(page_title="Data Explorer · Corners", page_icon="🔍", layout="wide")
inject_css()
df = load_data()

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

    st.markdown("#### Filters")
    q = st.text_input("Search", placeholder="Team / match / player…")
    teams_all = _safe_unique(df["team"])
    sel_teams = st.multiselect("Teams", teams_all, default=teams_all)
    matches_all = _safe_unique(df["match"])
    sel_matches = st.multiselect("Matches", matches_all, default=matches_all)
    takers_all = _safe_unique(df["taker"])
    sel_takers = st.multiselect("Takers", takers_all, default=takers_all)
    tech_all = _safe_unique(df["technique"])
    sel_tech = st.multiselect("Technique", tech_all, default=tech_all)
    ht_all = _safe_unique(df["height"])
    sel_ht = st.multiselect("Height", ht_all, default=ht_all)

    min_s = _to_num(df["Minute_num"]).dropna()
    if len(min_s) > 1:
        minute_range = st.slider("Minutes",
                                 int(min_s.min()), int(min_s.max()),
                                 (int(min_s.min()), int(min_s.max())))
    else:
        minute_range = None

    only_shots = st.toggle("Only corners → shot")

f = df.copy()
if sel_teams:   f = f[f["team"].isin(sel_teams)]
if sel_matches: f = f[f["match"].isin(sel_matches)]
if sel_takers:  f = f[f["taker"].isin(sel_takers)]
if sel_tech:    f = f[f["technique"].isin(sel_tech)]
if sel_ht:      f = f[f["height"].isin(sel_ht)]
if q.strip():
    qq = q.strip().lower()
    mask = _contains(f["team"],qq)|_contains(f["match"],qq)|_contains(f["taker"],qq)
    f = f[mask]
if minute_range:
    f = f[_to_num(f["Minute_num"]).between(minute_range[0], minute_range[1])]
if only_shots:
    f = f[f["is_shot"] == True]

page_header("Data Explorer", "Data Explorer", f"{len(f):,} events match current filters")

preferred = ["match","team","taker","Minute_num","Second_num",
             "technique","height","sp_outcome","is_shot","shot_outcome","xg"]
cols = [c for c in preferred if c in f.columns] + [c for c in f.columns if c not in preferred]

st.markdown("<div class='card'>", unsafe_allow_html=True)
st.dataframe(f[cols].reset_index(drop=True), use_container_width=True, hide_index=True, height=560)

csv = f[cols].to_csv(index=False).encode("utf-8")
st.download_button(
    "⬇ Download CSV (current view)",
    data=csv, file_name="corners_export.csv", mime="text/csv",
)
st.markdown("</div>", unsafe_allow_html=True)
