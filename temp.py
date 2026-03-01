import os
import pandas as pd
import numpy as np
import streamlit as st

# Optional interactive charts
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
    df.columns = [str(c).strip() for c in df.columns]

    # Numeric conversions (safe)
    df["Minute_num"] = pd.to_numeric(df["Minute"], errors="coerce") if "Minute" in df.columns else np.nan
    df["Second_num"] = pd.to_numeric(df["Second"], errors="coerce") if "Second" in df.columns else np.nan

    # Shot flag (robust if columns missing)
    shot_ts = df["shot_timestamp"].notna() if "shot_timestamp" in df.columns else pd.Series(False, index=df.index)
    shot_out = df["shot.outcome.name"].notna() if "shot.outcome.name" in df.columns else pd.Series(False, index=df.index)
    df["is_shot"] = shot_ts | shot_out

    # xG numeric
    if "shot.statsbomb_xg" in df.columns:
        df["xg"] = pd.to_numeric(df["shot.statsbomb_xg"], errors="coerce").fillna(0.0)
    else:
        df["xg"] = 0.0

    # Convenience aliases
    df["team"] = df["pass_team_name"] if "pass_team_name" in df.columns else None
    df["match"] = df["Match"] if "Match" in df.columns else None
    df["taker"] = df["Taker"] if "Taker" in df.columns else None
    df["technique"] = df["pass.technique.name"] if "pass.technique.name" in df.columns else None
    df["height"] = df["pass.height.name"] if "pass.height.name" in df.columns else None
    df["pass_outcome"] = df["pass.outcome.name"] if "pass.outcome.name" in df.columns else None
    df["shot_outcome"] = df["shot.outcome.name"] if "shot.outcome.name" in df.columns else None
    df["sp_outcome"] = df["SP_outcome"] if "SP_outcome" in df.columns else None
    df["def_setup"] = df["Defensive_setup"] if "Defensive_setup" in df.columns else None

    return df


# --- Data input
with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload an Excel file (.xlsx)", type=["xlsx"])

    if uploaded is not None:
        data_file = uploaded
        st.success("Using uploaded file.")
    else:
        if os.path.exists(DEFAULT_FILE):
            data_file = DEFAULT_FILE
            st.info(f"Using local file: {DEFAULT_FILE}")
        else:
            st.warning(
                "No file uploaded and default file not found.\n\n"
                f"Place `{DEFAULT_FILE}` next to this script, or upload it here."
            )
            st.stop()

df = load_data_from_excel(data_file)

# --- Sidebar filters
with st.sidebar:
    st.header("Filters")

    # Team filter
    teams = sorted(pd.Series(df["team"]).dropna().unique().tolist())
    sel_teams = st.multiselect("Team (corner taker team)", teams, default=teams)

    df_team = df[df["team"].isin(sel_teams)] if sel_teams else df

    # Match filter (dependent)
    matches = sorted(pd.Series(df_team["match"]).dropna().unique().tolist())
    sel_matches = st.multiselect("Match", matches, default=matches)

    df_match = df_team[df_team["match"].isin(sel_matches)] if sel_matches else df_team

    # Taker filter (dependent)
    takers = sorted(pd.Series(df_match["taker"]).dropna().unique().tolist())
    sel_takers = st.multiselect("Taker", takers, default=takers)

    # Technique / height / outcomes
    techniques = sorted(pd.Series(df_match["technique"]).dropna().unique().tolist())
    sel_techniques = st.multiselect("Technique", techniques, default=techniques)

    heights = sorted(pd.Series(df_match["height"]).dropna().unique().tolist())
    sel_heights = st.multiselect("Pass height", heights, default=heights)

    sp_outcomes = sorted(pd.Series(df_match["sp_outcome"]).dropna().unique().tolist())
    sel_sp_outcomes = st.multiselect("Set-piece outcome (SP_outcome)", sp_outcomes, default=sp_outcomes)

    # Minute filter (SAFE: never create a slider when min==max)
    minute_series = pd.to_numeric(df_match.get("Minute_num", pd.Series(dtype=float)), errors="coerce").dropna()
    minute_unique = np.sort(minute_series.unique()) if len(minute_series) else np.array([])

    minute_mode = "disabled"
    minute_range = None
    minute_single_value = None
    apply_single_minute = False

    if len(minute_unique) == 0:
        st.info("Minute column is empty for current filters. Minute filtering disabled.")
    elif len(minute_unique) == 1:
        minute_single_value = int(minute_unique[0])
        apply_single_minute = st.checkbox(f"Filter to minute {minute_single_value}", value=False)
        minute_mode = "single"
    else:
        min_minute = int(minute_unique.min())
        max_minute = int(minute_unique.max())
        minute_range = st.slider(
            "Minute range",
            min_value=min_minute,
            max_value=max_minute,
            value=(min_minute, max_minute),
        )
        minute_mode = "range"

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

# Minute filter application (SAFE)
if "Minute_num" in f.columns:
    f = f.copy()
    f["Minute_num"] = pd.to_numeric(f["Minute_num"], errors="coerce")

    if minute_mode == "range" and minute_range is not None:
        f = f[f["Minute_num"].between(minute_range[0], minute_range[1], inclusive="both")]
    elif minute_mode == "single" and apply_single_minute and minute_single_value is not None:
        f = f[f["Minute_num"] == minute_single_value]

if only_shots:
    f = f[f["is_shot"]]


# --- KPIs
total_corners = len(f)

if "match_id" in f.columns:
    n_matches = int(f["match_id"].nunique())
else:
    n_matches = int(pd.Series(f["match"]).nunique())

corners_per_match = (total_corners / n_matches) if n_matches else 0.0

shot_corners = int(f["is_shot"].sum()) if "is_shot" in f.columns else 0
shot_rate = (shot_corners / total_corners) if total_corners else 0.0

total_xg = float(pd.Series(f["xg"]).sum()) if "xg" in f.columns else 0.0

sp_txt = pd.Series(f.get("sp_outcome", "")).fillna("").astype(str)
shot_within_3s = int(sp_txt.str.contains("shot within 3 seconds", case=False, na=False).sum())

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

team_counts = (
    f.groupby("team", dropna=False)
     .size()
     .sort_values(ascending=False)
     .reset_index(name="corners")
)

with left:
    st.subheader("Corners by team")
    if len(team_counts) == 0:
        st.info("No data for current filters.")
    elif PLOTLY_OK:
        fig = px.bar(team_counts, x="team", y="corners", hover_data=["corners"])
        fig.update_layout(xaxis_title="", yaxis_title="Corners")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(team_counts.set_index("team")["corners"])

tech_counts = (
    f.groupby("technique", dropna=False)
     .size()
     .sort_values(ascending=False)
     .reset_index(name="corners")
)

with right:
    st.subheader("Technique distribution")
    if len(tech_counts) == 0:
        st.info("No data for current filters.")
    elif PLOTLY_OK:
        fig = px.pie(tech_counts, names="technique", values="corners", hole=0.45)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(tech_counts, use_container_width=True, hide_index=True)

st.divider()

left2, right2 = st.columns(2)

taker_counts = (
    f.groupby("taker", dropna=False)
     .size()
     .sort_values(ascending=False)
     .head(15)
     .reset_index(name="corners")
)

with left2:
    st.subheader("Top 15 corner takers")
    if len(taker_counts) == 0:
        st.info("No data for current filters.")
    elif PLOTLY_OK:
        fig = px.bar(taker_counts, x="corners", y="taker", orientation="h")
        fig.update_layout(xaxis_title="Corners", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(taker_counts, use_container_width=True, hide_index=True)

with right2:
    st.subheader("Corners over time (minute)")
    time_df = f[["Minute_num"]].copy() if "Minute_num" in f.columns else pd.DataFrame()
    time_df["Minute_num"] = pd.to_numeric(time_df.get("Minute_num", np.nan), errors="coerce")
    time_df = time_df.dropna()

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

c1, c2 = st.columns(2)

with c1:
    st.subheader("Shot outcomes (from corners)")
    shots = f[f["is_shot"]].copy() if "is_shot" in f.columns else f.iloc[0:0].copy()
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
    if "xg" not in f.columns:
        st.info("No xG column found.")
    else:
        xg_team = (
            f.groupby("team", dropna=False)["xg"]
             .sum()
             .sort_values(ascending=False)
             .reset_index()
        )
        if len(xg_team) == 0:
            st.info("No data for current filters.")
        elif PLOTLY_OK:
            fig = px.bar(xg_team, x="team", y="xg")
            fig.update_layout(xaxis_title="", yaxis_title="Total xG")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.dataframe(xg_team, use_container_width=True, hide_index=True)

st.divider()

st.subheader("Filtered data")
st.dataframe(f, use_container_width=True, hide_index=True)

csv = f.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download filtered data as CSV",
    data=csv,
    file_name="allsvenskan_corners_filtered.csv",
    mime="text/csv",
)
