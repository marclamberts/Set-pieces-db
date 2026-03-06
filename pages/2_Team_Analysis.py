import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils import (
    load_data,
    inject_css,
    _to_num,
    styled_bar,
    styled_donut,
    styled_histogram,
    page_header,
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

df["is_shot"] = (
    df["shot_timestamp"].notna()
    | df["Shooter"].notna()
    | (df["xg"] > 0)
)

teams_all = sorted(df["team"].dropna().unique().tolist())

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
    tech_all = sorted(df["technique"].dropna().unique().tolist())
    sel_tech = st.multiselect("Technique", tech_all, default=tech_all)

    ht_all = sorted(df["height"].dropna().unique().tolist())
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
f = df[df["team"] == focus].copy()

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
    .str.contains("Goal", case=False, na=False)
    .sum()
)

page_header("Team Analysis", focus, f"{total:,} corners in {n_mat} matches")

# Native Streamlit metrics instead of kpi_strip()
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Corners", f"{total:,}")
m2.metric("Matches", f"{n_mat}")
m3.metric("Shot rate", f"{sr*100:.1f}%")
m4.metric("Total xG", f"{xg:.3f}")
m5.metric("xG / corner", f"{xg_c:.4f}")
m6.metric("Goals", f"{goals}")

# -----------------------------
# Plotly vertical half pitch
# -----------------------------
def draw_plotly_half_pitch(deliveries, shots_df, title):
    pitch_bg = "#0b1020"
    line_col = "#24324a"
    text_col = "#e5e7eb"
    delivery_col = "rgba(148,163,184,0.35)"
    shot_col = "#38bdf8"
    shot_line = "#e2e8f0"

    fig = go.Figure()

    # StatsBomb pitch dimensions
    pitch_length = 120
    pitch_width = 80

    # Half-pitch attacking end: x from 60 to 120
    x0, x1 = 60, 120
    y0, y1 = 0, 80

    shapes = []

    # Outer boundary
    shapes.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(color=line_col, width=2)))

    # Halfway line
    shapes.append(dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color=line_col, width=2)))

    # Penalty area
    shapes.append(dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color=line_col, width=2)))

    # Six-yard box
    shapes.append(dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color=line_col, width=2)))

    # Goal
    shapes.append(dict(type="rect", x0=120, y0=36, x1=122, y1=44, line=dict(color=line_col, width=2)))

    # Penalty spot
    shapes.append(dict(type="circle", x0=107.5 - 0.4, y0=40 - 0.4, x1=107.5 + 0.4, y1=40 + 0.4,
                       line=dict(color=line_col, width=2), fillcolor=line_col))

    # Centre arc on half-way side
    shapes.append(dict(
        type="path",
        path="""
            M 69.15 31.85
            Q 60 40 69.15 48.15
        """,
        line=dict(color=line_col, width=2)
    ))

    # Penalty arc
    theta = np.linspace(np.deg2rad(130), np.deg2rad(230), 100)
    arc_x = 107.5 + 10 * np.cos(theta)
    arc_y = 40 + 10 * np.sin(theta)
    path = "M " + " L ".join([f"{x} {y}" for x, y in zip(arc_x, arc_y)])
    shapes.append(dict(type="path", path=path, line=dict(color=line_col, width=2)))

    # Corner arcs
    corner_r = 2
    t1 = np.linspace(0, np.pi / 2, 25)
    path_bl = "M " + " L ".join([f"{120 - corner_r*np.cos(t)} {0 + corner_r*np.sin(t)}" for t in t1])
    path_tl = "M " + " L ".join([f"{120 - corner_r*np.cos(t)} {80 - corner_r*np.sin(t)}" for t in t1])
    shapes.append(dict(type="path", path=path_bl, line=dict(color=line_col, width=2)))
    shapes.append(dict(type="path", path=path_tl, line=dict(color=line_col, width=2)))

    fig.update_layout(
        shapes=shapes,
        paper_bgcolor=pitch_bg,
        plot_bgcolor=pitch_bg,
        margin=dict(l=10, r=10, t=50, b=10),
        height=760,
        title=dict(text=title, font=dict(size=18, color=text_col), x=0.02),
        xaxis=dict(
            range=[58, 123],
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            range=[-2, 82],
            showgrid=False,
            zeroline=False,
            visible=False,
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=text_col),
            bgcolor="rgba(0,0,0,0)"
        ),
    )

    if not deliveries.empty:
        fig.add_trace(
            go.Scatter(
                x=deliveries["pass_end_location_x"],
                y=deliveries["pass_end_location_y"],
                mode="markers",
                name="Deliveries",
                marker=dict(size=9, color=delivery_col),
                hovertemplate="Delivery<br>x=%{x}<br>y=%{y}<extra></extra>",
            )
        )

    if not shots_df.empty:
        sizes = 10 + shots_df["xg"].clip(lower=0, upper=1) * 28
        fig.add_trace(
            go.Scatter(
                x=shots_df["shot_location_x"],
                y=shots_df["shot_location_y"],
                mode="markers",
                name="Shots",
                marker=dict(
                    size=sizes,
                    color=shot_col,
                    line=dict(color=shot_line, width=1.2),
                    opacity=0.95,
                ),
                customdata=np.stack([shots_df["xg"]], axis=-1),
                hovertemplate="Shot<br>x=%{x}<br>y=%{y}<br>xG=%{customdata[0]:.3f}<extra></extra>",
            )
        )

    return fig


st.markdown(
    "<div class='section-title'>Total Pitch Map</div>"
    "<div class='hero-sub'>All filtered corner deliveries and shot locations</div>",
    unsafe_allow_html=True,
)

deliveries = f[["pass_end_location_x", "pass_end_location_y"]].copy()
deliveries["pass_end_location_x"] = pd.to_numeric(deliveries["pass_end_location_x"], errors="coerce")
deliveries["pass_end_location_y"] = pd.to_numeric(deliveries["pass_end_location_y"], errors="coerce")
deliveries = deliveries.dropna()

shots_df = f.loc[f["is_shot"], ["shot_location_x", "shot_location_y", "xg"]].copy()
shots_df["shot_location_x"] = pd.to_numeric(shots_df["shot_location_x"], errors="coerce")
shots_df["shot_location_y"] = pd.to_numeric(shots_df["shot_location_y"], errors="coerce")
shots_df["xg"] = pd.to_numeric(shots_df["xg"], errors="coerce").fillna(0)
shots_df = shots_df.dropna(subset=["shot_location_x", "shot_location_y"])

if deliveries.empty and shots_df.empty:
    st.info("No valid pitch coordinates available after filtering.")
else:
    fig = draw_plotly_half_pitch(
        deliveries=deliveries,
        shots_df=shots_df,
        title=f"{focus} · Corner deliveries and shot locations",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

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