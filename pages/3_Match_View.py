import streamlit as st
import numpy as np
import pandas as pd

from utils import (
    load_data, inject_css, _to_num,
    styled_bar, styled_donut, styled_histogram,
    page_header, kpi_strip
)

st.set_page_config(page_title="Match View · Corners", page_icon="🗓️", layout="wide")
inject_css()
df = load_data()

matches_all = sorted(df["match"].dropna().astype(str).unique().tolist())

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

    st.markdown("#### Select Match")
    focus = st.selectbox("Match", matches_all)

    st.markdown("#### Filters")
    teams_in_match = sorted(
        df[df["match"].astype(str) == focus]["team"].dropna().astype(str).unique().tolist()
    )
    sel_teams = st.multiselect("Teams", teams_in_match, default=teams_in_match)
    only_shots = st.toggle("Only corners → shot", value=False)

f = df[df["match"].astype(str) == focus].copy()
if sel_teams:
    f = f[f["team"].isin(sel_teams)]
if only_shots:
    f = f[f["is_shot"] == True]

total = len(f)
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

page_header("Match View", focus, f"{total:,} corners in this fixture")
kpi_strip(
    [
        ("Corners", f"{total:,}", "This match"),
        ("Teams", f"{f['team'].nunique()}", "Unique"),
        ("Shots from corners", f"{shots}", "Total"),
        ("Shot rate", f"{sr*100:.1f}%", "→ shot"),
        ("xG", f"{xg:.3f}", "From shots"),
        ("Goals", f"{goals}", "Direct"),
    ]
)

c1, c2 = st.columns(2)

with c1:
    st.markdown("<div class='section-title'>Corners by Team</div>", unsafe_allow_html=True)
    by_team = f.groupby("team", dropna=False).size().sort_values(ascending=False).reset_index(name="corners")
    styled_bar(by_team, x="team", y="corners", height=300)

with c2:
    st.markdown(
        "<div class='section-title'>Corner Timing</div><div class='hero-sub'>When were corners taken?</div>",
        unsafe_allow_html=True,
    )
    styled_histogram(_to_num(f["Minute_num"]), nbins=20, height=300)

c3, c4 = st.columns(2)

with c3:
    st.markdown("<div class='section-title'>Technique Distribution</div>", unsafe_allow_html=True)
    tech = f.groupby("technique", dropna=False).size().reset_index(name="n")
    styled_donut(tech, "technique", "n", height=280)

with c4:
    st.markdown("<div class='section-title'>Shot Outcomes</div>", unsafe_allow_html=True)
    sout = f[f["is_shot"] == True].groupby("shot_outcome", dropna=False).size().reset_index(name="shots")
    styled_donut(sout, "shot_outcome", "shots", height=280)

st.markdown("<div class='section-title'>Set Piece Outcomes by Team</div>", unsafe_allow_html=True)
sp = f.groupby(["team", "sp_outcome"], dropna=False).size().reset_index(name="count")
styled_bar(sp, x="count", y="sp_outcome", orientation="h", color_col="team", height=280)

st.markdown("<div class='section-title'>All Corner Events</div>", unsafe_allow_html=True)
preferred = ["team", "taker", "Minute_num", "Second_num", "technique", "height", "sp_outcome", "is_shot", "shot_outcome", "xg"]
cols = [c for c in preferred if c in f.columns] + [c for c in f.columns if c not in preferred]
st.dataframe(f[cols].reset_index(drop=True), use_container_width=True, hide_index=True, height=380)
