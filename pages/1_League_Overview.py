import streamlit as st
import numpy as np
import pandas as pd

from utils import (
    load_data, inject_css, _to_num,
    styled_bar, styled_donut, styled_histogram, styled_scatter,
    page_header, kpi_strip
)

st.set_page_config(page_title="League Overview · Corners", page_icon="🏟️", layout="wide")
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

    st.markdown("#### Filters")
    teams_all = sorted(df["team"].dropna().astype(str).unique().tolist())
    sel_teams = st.multiselect("Teams", teams_all, default=teams_all)

    tech_all = sorted(df["technique"].dropna().astype(str).unique().tolist())
    sel_tech = st.multiselect("Technique", tech_all, default=tech_all)

    ht_all = sorted(df["height"].dropna().astype(str).unique().tolist())
    sel_ht = st.multiselect("Height", ht_all, default=ht_all)

    min_s = _to_num(df["Minute_num"]).dropna()
    minute_range = None
    if len(min_s) > 1:
        minute_range = st.slider(
            "Minute range",
            int(min_s.min()),
            int(min_s.max()),
            (int(min_s.min()), int(min_s.max())),
        )

# ── Filter ──
f = df.copy()
if sel_teams:
    f = f[f["team"].isin(sel_teams)]
if sel_tech:
    f = f[f["technique"].isin(sel_tech)]
if sel_ht:
    f = f[f["height"].isin(sel_ht)]
if minute_range:
    f = f[_to_num(f["Minute_num"]).between(minute_range[0], minute_range[1])]

total = len(f)
n_mat = f["match"].astype(str).replace("nan", np.nan).dropna().nunique()
shots = int(f["is_shot"].fillna(False).sum())
sr = shots / total if total else 0.0
xg = float(f["xg"].fillna(0).sum()) if "xg" in f.columns else 0.0
goals = int(
    f.get("shot_outcome", pd.Series(dtype=str))
    .fillna("")
    .astype(str)
    .str.contains("Goal", case=False, na=False)
    .sum()
)
cpm = total / n_mat if n_mat else 0.0

page_header("League Overview", "League Overview", f"{total:,} corners · {n_mat} matches")
kpi_strip(
    [
        ("Corners", f"{total:,}", "Filtered"),
        ("Matches", f"{n_mat}", "Unique"),
        ("Corners / match", f"{cpm:.1f}", "Average"),
        ("Shot rate", f"{sr*100:.1f}%", "→ shot"),
        ("Total xG", f"{xg:.3f}", "From shots"),
        ("Goals", f"{goals}", "Direct"),
    ]
)

# Row 1
c1, c2 = st.columns(2)

with c1:
    st.markdown(
        "<div class='section-title'>Corner Volume by Team</div>"
        "<div class='hero-sub'>Sorted by total corners taken</div>",
        unsafe_allow_html=True,
    )
    tc = (
        f.groupby("team", dropna=False)
        .size()
        .sort_values(ascending=True)
        .reset_index(name="corners")
    )
    styled_bar(tc, x="corners", y="team", orientation="h", height=380)

with c2:
    st.markdown(
        "<div class='section-title'>Delivery Technique Mix</div>"
        "<div class='hero-sub'>League-wide distribution</div>",
        unsafe_allow_html=True,
    )
    tech = f.groupby("technique", dropna=False).size().reset_index(name="n")
    styled_donut(tech, "technique", "n", height=380)

# Row 2
c3, c4 = st.columns(2)

with c3:
    st.markdown(
        "<div class='section-title'>xG from Corners by Team</div>"
        "<div class='hero-sub'>Total expected goals generated</div>",
        unsafe_allow_html=True,
    )
    xg_t = (
        f.groupby("team", dropna=False)["xg"]
        .sum()
        .sort_values(ascending=True)
        .reset_index()
    )
    styled_bar(xg_t, x="xg", y="team", orientation="h", height=380)

with c4:
    st.markdown(
        "<div class='section-title'>Shot Outcomes</div>"
        "<div class='hero-sub'>All shots from corners</div>",
        unsafe_allow_html=True,
    )
    sout = (
        f[f["is_shot"] == True]
        .groupby("shot_outcome", dropna=False)
        .size()
        .reset_index(name="shots")
    )
    styled_donut(sout, "shot_outcome", "shots", height=380)

# Scatter
st.markdown(
    "<div class='section-title'>Team Efficiency: Shot Rate vs xG / Shot</div>"
    "<div class='hero-sub'>Top-right = most dangerous</div>",
    unsafe_allow_html=True,
)
eff = (
    f.groupby("team", dropna=False)
    .agg(
        corners=("is_shot", "count"),
        shot_count=("is_shot", "sum"),
        total_xg=("xg", "sum"),
    )
    .reset_index()
)
eff["shot_rate"] = eff["shot_count"] / eff["corners"].replace(0, np.nan)
eff["xg_per_shot"] = eff["total_xg"] / eff["shot_count"].replace(0, np.nan)
eff = eff.dropna(subset=["shot_rate", "xg_per_shot"])
styled_scatter(eff, x="shot_rate", y="xg_per_shot", text="team", height=340)

# Timing
st.markdown(
    "<div class='section-title'>Corner Timing Distribution</div>"
    "<div class='hero-sub'>When in the match are corners awarded?</div>",
    unsafe_allow_html=True,
)
styled_histogram(_to_num(f["Minute_num"]), nbins=30, height=260)

# Height + SP outcome
c5, c6 = st.columns(2)

with c5:
    st.markdown(
        "<div class='section-title'>Delivery Height Mix</div>"
        "<div class='hero-sub'>League-wide</div>",
        unsafe_allow_html=True,
    )
    ht = f.groupby("height", dropna=False).size().reset_index(name="n")
    styled_donut(ht, "height", "n", height=300)

with c6:
    st.markdown(
        "<div class='section-title'>SP Outcome Breakdown</div>"
        "<div class='hero-sub'>Set piece outcome tags</div>",
        unsafe_allow_html=True,
    )
    sp = (
        f.groupby("sp_outcome", dropna=False)
        .size()
        .sort_values(ascending=True)
        .reset_index(name="count")
    )
    styled_bar(sp, x="count", y="sp_outcome", orientation="h", height=300)
