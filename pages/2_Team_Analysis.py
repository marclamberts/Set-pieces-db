import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import VerticalPitch

from utils import (
    load_data, inject_css, _to_num,
    styled_bar, styled_donut, styled_histogram,
    page_header, kpi_strip
)

st.set_page_config(page_title="Team Analysis · Corners", page_icon="🧭", layout="wide")
inject_css()
df = load_data().copy()

# -----------------------------
# Fit to your actual dataset
# -----------------------------
df["team"] = df["pass_team_name"].astype(str)
df["match"] = df["Match"].astype(str)
df["taker"] = df["Taker"].astype(str)
df["technique"] = df["pass.technique.name"].astype(str)
df["height"] = df["pass.height.name"].astype(str)
df["minute_num"] = pd.to_numeric(df["Minute"], errors="coerce")
df["xg"] = pd.to_numeric(df["shot.statsbomb_xg"], errors="coerce").fillna(0)
df["shot_outcome"] = df["shot.outcome.name"].astype(str)
df["sp_outcome"] = df["SP_outcome"].astype(str)

# Shot flag
df["is_shot"] = (
    df["shot_timestamp"].notna()
    | df["Shooter"].notna()
    | (df["xg"] > 0)
)

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

    st.markdown("#### Shot / xG")
    shots_only = st.checkbox("Only corners that led to a shot", value=False)

    xg_max = float(df["xg"].max()) if not df["xg"].empty else 0.0
    xg_slider_max = max(round(xg_max, 2), 0.01)

    sel_xg = st.slider(
        "shot.xG range",
        min_value=0.0,
        max_value=xg_slider_max,
        value=(0.0, xg_slider_max),
        step=0.01,
    )

# -----------------------------
# Filtering
# -----------------------------
f = df[df["team"] == str(focus)].copy()

if sel_tech:
    f = f[f["technique"].isin(sel_tech)]

if sel_ht:
    f = f[f["height"].isin(sel_ht)]

if shots_only:
    f = f[f["is_shot"]]

f = f[(f["xg"] >= sel_xg[0]) & (f["xg"] <= sel_xg[1])]

# -----------------------------
# KPIs
# -----------------------------
total = len(f)
n_mat = f["match"].replace("nan", np.nan).dropna().nunique()
shots = int(f["is_shot"].fillna(False).sum())
sr = shots / total if total else 0.0
xg = float(f["xg"].sum()) if total else 0.0
xg_c = xg / total if total else 0.0
goals = int(
    f["shot_outcome"]
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

# -----------------------------
# Top section: pitch map replaces code box
# -----------------------------
st.markdown(
    "<div class='section-title'>Total Pitch Map</div>"
    "<div class='hero-sub'>All filtered corner deliveries and shot locations</div>",
    unsafe_allow_html=True,
)

# Match the app dark background
PANEL_BG = "#111827"
LINE_COL = "#2a3342"
TEXT_COL = "#e5e7eb"
DELIVERY_COL = "#94a3b8"
SHOT_FACE = "#38bdf8"
SHOT_EDGE = "#e2e8f0"

# Delivery end locations
deliveries = f[["pass_end_location_x", "pass_end_location_y"]].copy()
deliveries["pass_end_location_x"] = pd.to_numeric(deliveries["pass_end_location_x"], errors="coerce")
deliveries["pass_end_location_y"] = pd.to_numeric(deliveries["pass_end_location_y"], errors="coerce")
deliveries = deliveries.dropna()

# Shot locations
shots_df = f.loc[f["is_shot"], ["shot_location_x", "shot_location_y", "xg"]].copy()
shots_df["shot_location_x"] = pd.to_numeric(shots_df["shot_location_x"], errors="coerce")
shots_df["shot_location_y"] = pd.to_numeric(shots_df["shot_location_y"], errors="coerce")
shots_df["xg"] = pd.to_numeric(shots_df["xg"], errors="coerce").fillna(0)
shots_df = shots_df.dropna(subset=["shot_location_x", "shot_location_y"])

if deliveries.empty and shots_df.empty:
    st.info("No valid pitch coordinates available after filtering.")
else:
    pitch = VerticalPitch(
        pitch_type="statsbomb",
        half=True,
        pitch_color=PANEL_BG,
        line_color=LINE_COL,
        linewidth=1.2,
        line_zorder=2,
    )

    fig, ax = pitch.draw(figsize=(7, 9))
    fig.patch.set_facecolor(PANEL_BG)
    ax.set_facecolor(PANEL_BG)

    if not deliveries.empty:
        pitch.scatter(
            deliveries["pass_end_location_x"],
            deliveries["pass_end_location_y"],
            ax=ax,
            s=42,
            color=DELIVERY_COL,
            alpha=0.35,
            zorder=3,
        )

    if not shots_df.empty:
        shot_sizes = 70 + (shots_df["xg"].clip(lower=0, upper=1) * 260)
        pitch.scatter(
            shots_df["shot_location_x"],
            shots_df["shot_location_y"],
            ax=ax,
            s=shot_sizes,
            color=SHOT_FACE,
            edgecolors=SHOT_EDGE,
            linewidths=0.9,
            alpha=0.95,
            zorder=4,
        )

    ax.set_title(
        f"{focus} · Corner deliveries and shot locations",
        color=TEXT_COL,
        fontsize=14,
        pad=12,
    )

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

# -----------------------------
# Main charts
# -----------------------------
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
        .tail(15)
        .reset_index(name="corners")
    )
    styled_bar(tk, x="corners", y="taker", orientation="h", height=340)

with c2:
    st.markdown(
        "<div class='section-title'>Primary Takers — xG Created</div>"
        "<div class='hero-sub'>Expected goals generated from corners (top 15)</div>",
        unsafe_allow_html=True,
    )
    xg_tk = (
        f.groupby("taker", dropna=False)["xg"]
        .sum()
        .sort_values(ascending=True)
        .tail(15)
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
styled_bar(sp, x="count", y="sp_outcome", orientation="h", height=300)

st.markdown(
    "<div class='section-title'>Corner Timing</div>"
    "<div class='hero-sub'>Minute distribution for this team</div>",
    unsafe_allow_html=True,
)
styled_histogram(_to_num(f["minute_num"]), nbins=20, height=240)