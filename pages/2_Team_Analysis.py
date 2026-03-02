import streamlit as st
import numpy as np
import pandas as pd

from utils import (
    load_data, inject_css, _to_num,
    styled_bar, styled_donut, styled_histogram,
    page_header, kpi_strip
)

st.set_page_config(page_title="Team Analysis · Corners", page_icon="🧭", layout="wide")
inject_css()
df = load_data()

teams_all = sorted(df["team"].dropna().astype(str).unique().tolist())

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

    st.markdown("#### Select Team")
    focus = st.selectbox("Team", teams_all)

    st.markdown("#### Filters")
    tech_all = sorted(df["technique"].dropna().astype(str).unique().tolist())
    sel_tech = st.multiselect("Technique", tech_all, default=tech_all)

    ht_all = sorted(df["height"].dropna().astype(str).unique().tolist())
    sel_ht = st.multiselect("Height", ht_all, default=ht_all)

f = df[df["team"].astype(str) == str(focus)].copy()
if sel_tech:
    f = f[f["technique"].isin(sel_tech)]
if sel_ht:
    f = f[f["height"].isin(sel_ht)]

total = len(f)
n_mat = f["match"].astype(str).replace("nan", np.nan).dropna().nunique()
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

page_header("Team Analysis", focus, f"{total:,} corners in {n_mat} matches")
kpi_strip(
    [
        ("Corners", f"{total:,}", "Total"),
        ("Matches", f"{n_mat}", "Unique"),
        ("Shot rate", f"{sr*100:.1f}%", "→ shot"),
        ("Total xG", f"{xg:.3f}", "From shots"),
        ("xG / corner", f"{xg_c:.4f}", "Efficiency"),
        ("Goals", f"{goals}", "Direct"),
    ]
)

c1, c2 = st.columns(2)

with c1:
    st.markdown(
        "<div class='section-title'>Primary Takers — Volume</div>"
        "<div class='hero-sub'>Corners taken per player (top 15)</div>",
        unsafe_allow_html=True,
    )
    tk = (
        f.groupby("taker", dropna=False)
        .size()
        .sort_values(ascending=True)
        .head(15)
        .reset_index(name="corners")
    )
    styled_bar(tk, x="corners", y="taker", orientation="h", height=340)

with c2:
    st.markdown(
        "<div class='section-title'>Primary Takers — xG Created</div>"
        "<div class='hero-sub'>Expected goals per taker (top 15)</div>",
        unsafe_allow_html=True,
    )
    xg_tk = (
        f.groupby("taker", dropna=False)["xg"]
        .sum()
        .sort_values(ascending=True)
        .head(15)
        .reset_index()
    )
    styled_bar(xg_tk, x="xg", y="taker", orientation="h", height=340)

c3, c4 = st.columns(2)

with c3:
    st.markdown("<div class='section-title'>Technique Profile</div>", unsafe_allow_html=True)
    tech = f.groupby("technique", dropna=False).size().reset_index(name="n")
    styled_donut(tech, "technique", "n", height=300)

with c4:
    st.markdown("<div class='section-title'>Delivery Height Profile</div>", unsafe_allow_html=True)
    ht = f.groupby("height", dropna=False).size().reset_index(name="n")
    styled_donut(ht, "height", "n", height=300)

st.markdown(
    "<div class='section-title'>Set Piece Outcomes</div>"
    "<div class='hero-sub'>SP_outcome breakdown for this team</div>",
    unsafe_allow_html=True,
)
sp = (
    f.groupby("sp_outcome", dropna=False)
    .size()
    .sort_values(ascending=True)
    .reset_index(name="count")
)
styled_bar(sp, x="count", y="sp_outcome", orientation="h", height=280)

st.markdown(
    "<div class='section-title'>Corner Timing</div>"
    "<div class='hero-sub'>Minute distribution for this team</div>",
    unsafe_allow_html=True,
)
styled_histogram(_to_num(f["Minute_num"]), nbins=20, height=240)
