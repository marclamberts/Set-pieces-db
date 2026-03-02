import streamlit as st
import numpy as np
from utils import load_data, inject_css, _to_num, styled_bar, styled_donut, styled_histogram, styled_scatter, page_header, kpi_strip

st.set_page_config(page_title="League Overview · Corners", page_icon="🌍", layout="wide")
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
    teams_all = sorted(df["team"].dropna().astype(str).unique().tolist())
    sel_teams = st.multiselect("Teams", teams_all, default=teams_all)
    tech_all  = sorted(df["technique"].dropna().astype(str).unique().tolist())
    sel_tech  = st.multiselect("Technique", tech_all, default=tech_all)
    ht_all    = sorted(df["height"].dropna().astype(str).unique().tolist())
    sel_ht    = st.multiselect("Height", ht_all, default=ht_all)

    min_s = _to_num(df["Minute_num"]).dropna()
    if len(min_s) > 1:
        minute_range = st.slider("Minute range",
                                 int(min_s.min()), int(min_s.max()),
                                 (int(min_s.min()), int(min_s.max())))
    else:
        minute_range = None

# ── filter ──
f = df.copy()
if sel_teams: f = f[f["team"].isin(sel_teams)]
if sel_tech:  f = f[f["technique"].isin(sel_tech)]
if sel_ht:    f = f[f["height"].isin(sel_ht)]
if minute_range:
    f = f[_to_num(f["Minute_num"]).between(minute_range[0], minute_range[1])]

total  = len(f)
n_mat  = f["match"].astype(str).replace("nan", np.nan).dropna().nunique()
shots  = int(f["is_shot"].fillna(False).sum())
sr     = shots / total if total else 0
xg     = float(f["xg"].fillna(0).sum()) if "xg" in f.columns else 0.0
goals  = int(f.get("shot_outcome", __import__("pandas").Series(dtype=str))
             .fillna("").astype(str).str.contains("Goal", case=False, na=False).sum())
cpm    = total / n_mat if n_mat else 0

page_header("League Overview", "League Overview", f"{total:,} corners · {n_mat} matches")

kpi_strip([
    ("Corners", f"{total:,}", "Filtered"),
    ("Matches", f"{n_mat}", "Unique"),
    ("Corners / match", f"{cpm:.1f}", "Average"),
    ("Shot rate", f"{sr*100:.1f}%", "→ shot"),
    ("Total xG", f"{xg:.3f}", "From shots"),
    ("Goals", f"{goals}", "Direct"),
])

# Row 1
c1, c2 = st.columns(2)
with c1:
    st.markdown("<div class='card'><div class='card-title'>Corner Volume by Team</div><div class='card-sub'>Sorted by total corners taken</div>", unsafe_allow_html=True)
    tc = f.groupby("team", dropna=False).size().sort_values(ascending=True).reset_index(name="corners")
    styled_bar(tc, x="corners", y="team", orientation="h", height=380)
    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown("<div class='card'><div class='card-title'>Delivery Technique Mix</div><div class='card-sub'>League-wide distribution</div>", unsafe_allow_html=True)
    tech = f.groupby("technique", dropna=False).size().reset_index(name="n")
    styled_donut(tech, "technique", "n", height=380)
    st.markdown("</div>", unsafe_allow_html=True)

# Row 2
c3, c4 = st.columns(2)
with c3:
    st.markdown("<div class='card'><div class='card-title'>xG from Corners by Team</div><div class='card-sub'>Total expected goals generated</div>", unsafe_allow_html=True)
    xg_t = f.groupby("team", dropna=False)["xg"].sum().sort_values(ascending=True).reset_index()
    styled_bar(xg_t, x="xg", y="team", orientation="h", height=380)
    st.markdown("</div>", unsafe_allow_html=True)

with c4:
    st.markdown("<div class='card'><div class='card-title'>Shot Outcomes</div><div class='card-sub'>All shots from corners</div>", unsafe_allow_html=True)
    sout = f[f["is_shot"]==True].groupby("shot_outcome", dropna=False).size().reset_index(name="shots")
    styled_donut(sout, "shot_outcome", "shots", height=380)
    st.markdown("</div>", unsafe_allow_html=True)

# Scatter
st.markdown("<div class='card'><div class='card-title'>Team Efficiency: Shot Rate vs xG / Shot</div><div class='card-sub'>Top-right = most dangerous · size = corner volume</div>", unsafe_allow_html=True)
eff = f.groupby("team", dropna=False).agg(
    corners=("is_shot","count"), shot_count=("is_shot","sum"), total_xg=("xg","sum")
).reset_index()
eff["shot_rate"]   = eff["shot_count"] / eff["corners"].replace(0, np.nan)
eff["xg_per_shot"] = eff["total_xg"]   / eff["shot_count"].replace(0, np.nan)
eff = eff.dropna(subset=["shot_rate","xg_per_shot"])
styled_scatter(eff, x="shot_rate", y="xg_per_shot", text="team", height=340)
st.markdown("</div>", unsafe_allow_html=True)

# Timing
st.markdown("<div class='card'><div class='card-title'>Corner Timing Distribution</div><div class='card-sub'>When in the match are corners awarded?</div>", unsafe_allow_html=True)
styled_histogram(_to_num(f["Minute_num"]), nbins=30, height=260)
st.markdown("</div>", unsafe_allow_html=True)

# Delivery height
c5, c6 = st.columns(2)
with c5:
    st.markdown("<div class='card'><div class='card-title'>Delivery Height Mix</div><div class='card-sub'>League-wide</div>", unsafe_allow_html=True)
    ht = f.groupby("height", dropna=False).size().reset_index(name="n")
    styled_donut(ht, "height", "n", height=300)
    st.markdown("</div>", unsafe_allow_html=True)

with c6:
    st.markdown("<div class='card'><div class='card-title'>SP Outcome Breakdown</div><div class='card-sub'>Set piece outcome tags</div>", unsafe_allow_html=True)
    sp = f.groupby("sp_outcome", dropna=False).size().sort_values(ascending=True).reset_index(name="count")
    styled_bar(sp, x="count", y="sp_outcome", orientation="h", height=300)
    st.markdown("</div>", unsafe_allow_html=True)
