import os
import pandas as pd
import numpy as np
import streamlit as st

# Optional (interactive charts). If plotly isn't installed, the app will fall back to Streamlit charts.
try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False


st.set_page_config(page_title="Allsvenskan Corners 2025", layout="wide")

DEFAULT_FILE = "Allsvenskan - Corners 2025.xlsx"
DEFAULT_SHEET = "Sheet 1"

st.title("⚽ Allsvenskan – Corners 2025 Dashboard")
st.caption("Filter, explore, and summarize corner events (and resulting shots/xG).")


@st.cache_data(show_spinner=False)
def load_data_from_excel(file) -> pd.DataFrame:
    df = pd.read_excel(file, sheet_name=DEFAULT_SHEET)

    # Normalize column names just in case
    df.columns = [str(c).strip() for c in df.columns]

    # Derived fields
    df["Minute_num"] = pd.to_numeric(df.get("Minute", np.nan), errors="coerce")
    df["Second_num"] = pd.to_numeric(df.get("Second", np.nan), errors="coerce")

    # A row represents a corner event; "is_shot" if a shot timestamp exists (or shot outcome exists)
    df["is_shot"] = df.get("shot_timestamp").notna() | df.get("shot.outcome.name").notna()

    # xG as numeric
    if "shot.statsbomb_xg" in df.columns:
        df["xg"] = pd.to_numeric(df["shot.statsbomb_xg"], errors="coerce").fillna(0.0)
    else:
        df["xg"] = 0.0

    # Some convenience labels
    df["team"] = df.get("pass_team_name")
    df["match"] = df.get("Match")
    df["taker"] = df.get("Taker")
    df["technique"] = df.get("pass.technique.name")
    df["height"] = df.get("pass.height.name")
    df["pass_outcome"] = df.get("pass.outcome.name")
    df["shot_outcome"] = df.get("shot.outcome.name")
    df["sp_outcome"] = df.get("SP_outcome")
    df["def_setup"] = df.get("Defensive_setup")

    return df


# --- Data input
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

    if uploaded is not None:
        data_file = uploaded
        st.success("Using uploaded file.")
    else:
        # Use default file if present in the same folder
        if os.path.exists(DEFAULT_FILE):
            data_file = DEFAULT_FILE
            st.info(f"Using local file: {DEFAULT_FILE}")
        else:
            st.warning(
                "No file uploaded and default file not found.\n\n"
                f"Place `{DEFAULT_FILE}` next to `app.py`, or upload it here."
            )
            st.stop()

df = load_data_from_excel(data_file)

# --- Sidebar filters
with st.sidebar:
    st.header("Filters")

    # Team filter
    teams = sorted([t for t in df["team"].dropna().unique()])
    sel_teams = st.multiselect("Team (corner taker team)", teams, default=teams)

    # Match filter (dependent)
    df_team = df[df["team"].isin(sel_teams)] if sel_teams else df
    matches = sorted([m for m in df_team["match"].dropna().unique()])
    sel_matches = st.multiselect("Match", matches, default=matches)

    # Taker filter (dependent)
    df_match = df_team[df_team["match"].isin(sel_matches)] if sel_matches else df_team
    takers = sorted([p for p in df_match["taker"].dropna().unique()])
    sel_takers = st.multiselect("Taker", takers, default=takers)

    # Technique / height / outcomes
    techniques = sorted([x for x in df_match["technique"].dropna().unique()])
    sel_techniques = st.multiselect("Technique", techniques, default=techniques)

    heights = sorted([x for x in df_match["height"].dropna().unique()])
    sel_heights = st.multiselect("Pass height", heights, default=heights)

    sp_outcomes = sorted([x for x in df_match["sp_outcome"].dropna().unique()])
    sel_sp_outcomes = st.multiselect("Set-piece outcome (SP_outcome)", sp_outcomes, default=sp_outcomes)

    # Minute range
    min_minute = int(np.nanmin(df["Minute_num"])) if df["Minute_num"].notna().any() else 0
    max_minute = int(np.nanmax(df["Minute_num"])) if df["Minute_num"].notna().any() else 120
    minute_range = st.slider("Minute range", min_value=min_minute, max_value=max_minute, value=(min_minute, max_minute))

    only_shots = st.toggle("Only corners that lead to a shot", value=False)

# --- Apply filters
f = df.copy()

if sel_teams:
    f = f[f["team"].isin(sel_teams)]
if sel_matches:
    f = f[f["match"].isin(sel_matches)]
if sel_takers:
    f = f[f["taker"].isin(sel_takers)]
if sel_techniques:
    f = f[f["technique"].isin(sel_techniques)]
if sel_heights:
    f = f[f["height"].isin(sel_heights)]
if sel_sp_outcomes:
    f = f[f["sp_outcome"].isin(sel_sp_outcomes)]

f = f[(f["Minute_num"].fillna(min_minute) >= minute_range[0]) & (f["Minute_num"].fillna(max_minute) <= minute_range[1])]

if only_shots:
    f = f[f["is_shot"]]

# --- KPIs
total_corners = len(f)
n_matches = f["match_id"].nunique() if "match_id" in f.columns else f["match"].nunique()
corners_per_match = (total_corners / n_matches) if n_matches else 0.0

shot_corners = int(f["is_shot"].sum())
shot_rate = (shot_corners / total_corners) if total_corners else 0.0

total_xg = float(f["xg"].sum())

# "shot within 3 seconds" heuristic: SP_outcome contains that phrase in your file
shot_within_3s = int(f["sp_outcome"].fillna("").str.contains("shot within 3 seconds", case=False).sum())
first_contact = int(f["sp_outcome"].fillna("").str.contains("First contact", case=False).sum())

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6 = st.columns(6)
kpi1.metric("Corners (rows)", f"{total_corners:,}")
kpi2.metric("Matches", f"{n_matches:,}")
kpi3.metric("Corners / match", f"{corners_per_match:.2f}")
kpi4.metric("Shot rate", f"{shot_rate*100:.1f}%")
kpi5.metric("Total xG", f"{total_xg:.3f}")
kpi6.metric("Shots ≤3s after corner", f"{shot_within_3s:,}")

st.divider()

# --- Charts
left, right = st.columns(2)

# Corners by team
team_counts = (
    f.groupby("team", dropna=False)
     .size()
     .sort_values(ascending=False)
     .reset_index(name="corners")
)
with left:
    st.subheader("Corners by team")
    if PLOTLY_OK:
        fig = px.bar(team_counts, x="team", y="corners", hover_data=["corners"])
        fig.update_layout(xaxis_title="", yaxis_title="Corners")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(team_counts.set_index("team")["corners"])

# Technique share
tech_counts = (
    f.groupby("technique", dropna=False)
     .size()
     .sort_values(ascending=False)
     .reset_index(name="corners")
)
with right:
    st.subheader("Technique distribution")
    if PLOTLY_OK:
        fig = px.pie(tech_counts, names="technique", values="corners", hole=0.45)
        fig.update_layout(legend_title_text="Technique")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(tech_counts, use_container_width=True, hide_index=True)

st.divider()

left2, right2 = st.columns(2)

# Top takers
taker_counts = (
    f.groupby("taker", dropna=False)
     .size()
     .sort_values(ascending=False)
     .head(15)
     .reset_index(name="corners")
)
with left2:
    st.subheader("Top 15 corner takers")
    if PLOTLY_OK:
        fig = px.bar(taker_counts, x="corners", y="taker", orientation="h")
        fig.update_layout(xaxis_title="Corners", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(taker_counts, use_container_width=True, hide_index=True)

# Time distribution
with right2:
    st.subheader("Corners over time (minute)")
    time_df = f[["Minute_num"]].dropna().copy()
    if len(time_df) == 0:
        st.info("No minute data available for current filters.")
    else:
        if PLOTLY_OK:
            fig = px.histogram(time_df, x="Minute_num", nbins=30)
            fig.update_layout(xaxis_title="Minute", yaxis_title="Corners")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.bar_chart(time_df["Minute_num"].value_counts().sort_index())

st.divider()

# Shot outcomes + xG by team
c1, c2 = st.columns(2)

with c1:
    st.subheader("Shot outcomes (from corners)")
    shots = f[f["is_shot"]].copy()
    if len(shots) == 0:
        st.info("No shots in the current filtered data.")
    else:
        shot_out = (
            shots.groupby("shot_outcome", dropna=False)
                 .size()
                 .sort_values(ascending=False)
                 .reset_index(name="shots")
        )
        if PLOTLY_OK:
            fig = px.bar(shot_out, x="shot_outcome", y="shots")
            fig.update_layout(xaxis_title="", yaxis_title="Shots")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(shot_out, use_container_width=True, hide_index=True)

with c2:
    st.subheader("xG from corners by team")
    xg_team = (
        f.groupby("team", dropna=False)["xg"]
         .sum()
         .sort_values(ascending=False)
         .reset_index()
    )
    if PLOTLY_OK:
        fig = px.bar(xg_team, x="team", y="xg")
        fig.update_layout(xaxis_title="", yaxis_title="Total xG")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(xg_team, use_container_width=True, hide_index=True)

st.divider()

# --- Data table + download
st.subheader("Filtered data")
st.dataframe(f, use_container_width=True, hide_index=True)

csv = f.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered data as CSV",
    data=csv,
    file_name="allsvenskan_corners_filtered.csv",
    mime="text/csv",
)
