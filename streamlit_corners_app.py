import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Allsvenskan Corners 2025", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
REQUIRED_COLUMNS = [
    "match_id",
    "Match",
    "pass_team_name",
    "Minute",
    "Second",
    "SP_outcome",
]

DEFAULT_FILE = "Allsvenskan - Corners 2025.xlsx"


@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        return df

    if os.path.exists(DEFAULT_FILE):
        df = pd.read_excel(DEFAULT_FILE)
        return df

    return None


def validate_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    return missing


def split_match(match_name: str):
    if not isinstance(match_name, str) or " - " not in match_name:
        return None, None
    home, away = match_name.split(" - ", 1)
    return home.strip(), away.strip()


def prepare_data(df: pd.DataFrame):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    # numeric safety
    if "Minute" in df.columns:
        df["Minute"] = pd.to_numeric(df["Minute"], errors="coerce")
    if "Second" in df.columns:
        df["Second"] = pd.to_numeric(df["Second"], errors="coerce")
    if "shot.statsbomb_xg" in df.columns:
        df["shot.statsbomb_xg"] = pd.to_numeric(df["shot.statsbomb_xg"], errors="coerce")

    # derived time column
    df["event_minute"] = df["Minute"].fillna(0) + df["Second"].fillna(0) / 60.0

    # split teams from match title
    homes = []
    aways = []
    for m in df["Match"]:
        h, a = split_match(m)
        homes.append(h)
        aways.append(a)

    df["home_team"] = homes
    df["away_team"] = aways

    # corner team = taker/pass team
    df["corner_team"] = df["pass_team_name"].astype(str).str.strip()

    # whether corner led to a shot
    df["led_to_shot"] = df["SP_outcome"].astype(str).str.contains("shot", case=False, na=False)

    # whether first contact led to shot quickly
    df["first_contact_shot_3s"] = (
        df["SP_outcome"].astype(str).str.strip().eq("First contact - shot within 3 seconds")
    )

    # match-level summary
    match_summary = (
        df.groupby(["match_id", "Match", "home_team", "away_team"], dropna=False)
        .agg(
            total_corners=("match_id", "size"),
            home_corners=("corner_team", lambda s: int((s == s.name).sum()) if False else 0),
            shots_from_corners=("led_to_shot", "sum"),
            first_contact_shots_3s=("first_contact_shot_3s", "sum"),
            avg_xg=("shot.statsbomb_xg", "mean"),
            total_xg=("shot.statsbomb_xg", "sum"),
        )
        .reset_index()
    )

    # correct home/away corner counts
    match_summary["home_corners"] = match_summary.apply(
        lambda r: int(((df["match_id"] == r["match_id"]) & (df["corner_team"] == r["home_team"])).sum()),
        axis=1,
    )
    match_summary["away_corners"] = match_summary.apply(
        lambda r: int(((df["match_id"] == r["match_id"]) & (df["corner_team"] == r["away_team"])).sum()),
        axis=1,
    )

    # team summary
    team_summary = (
        df.groupby("corner_team", dropna=False)
        .agg(
            corners_taken=("match_id", "size"),
            matches=("match_id", pd.Series.nunique),
            shots_from_corners=("led_to_shot", "sum"),
            first_contact_shots_3s=("first_contact_shot_3s", "sum"),
            total_xg=("shot.statsbomb_xg", "sum"),
            avg_xg_per_corner=("shot.statsbomb_xg", "mean"),
        )
        .reset_index()
        .rename(columns={"corner_team": "team"})
    )

    team_summary["corners_per_match"] = team_summary["corners_taken"] / team_summary["matches"].replace(0, np.nan)
    team_summary["shot_rate"] = team_summary["shots_from_corners"] / team_summary["corners_taken"].replace(0, np.nan)

    return df, match_summary, team_summary


# ----------------------------
# UI
# ----------------------------
st.title("⚽ Allsvenskan Corners 2025")
st.caption("Event-level dashboard for corner analysis")

uploaded_file = st.sidebar.file_uploader(
    "Upload Excel file",
    type=["xlsx"],
    help="Upload your corners dataset. If omitted, the app tries to load 'Allsvenskan - Corners 2025.xlsx' from the app folder.",
)

raw_df = load_data(uploaded_file)

if raw_df is None:
    st.error(
        "No Excel file found. Upload the file in the sidebar or place "
        "'Allsvenskan - Corners 2025.xlsx' in the same folder as app.py."
    )
    st.stop()

missing_cols = validate_columns(raw_df)
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.write("Columns found in file:")
    st.write(list(raw_df.columns))
    st.stop()

df, match_summary, team_summary = prepare_data(raw_df)

all_teams = sorted([t for t in team_summary["team"].dropna().unique().tolist() if str(t).strip()])
selected_team = st.sidebar.selectbox("Team", ["All Teams"] + all_teams)

min_total_corners = int(match_summary["total_corners"].min()) if not match_summary.empty else 0
max_total_corners = int(match_summary["total_corners"].max()) if not match_summary.empty else 0

corner_range = st.sidebar.slider(
    "Match total corners range",
    min_value=min_total_corners,
    max_value=max_total_corners,
    value=(min_total_corners, max_total_corners),
)

show_only_shot_corners = st.sidebar.checkbox("Only events that led to a shot", value=False)

# ----------------------------
# Filters
# ----------------------------
event_df = df.copy()
match_df = match_summary.copy()
team_df = team_summary.copy()

if selected_team != "All Teams":
    event_df = event_df[event_df["corner_team"] == selected_team]
    match_ids = event_df["match_id"].unique()
    match_df = match_df[match_df["match_id"].isin(match_ids)]
    team_df = team_df[team_df["team"] == selected_team]

match_df = match_df[
    (match_df["total_corners"] >= corner_range[0]) &
    (match_df["total_corners"] <= corner_range[1])
]

match_ids_after_corner_filter = match_df["match_id"].unique()
event_df = event_df[event_df["match_id"].isin(match_ids_after_corner_filter)]

if show_only_shot_corners:
    event_df = event_df[event_df["led_to_shot"]]
    match_ids_shot = event_df["match_id"].unique()
    match_df = match_df[match_df["match_id"].isin(match_ids_shot)]

# ----------------------------
# KPIs
# ----------------------------
total_events = len(event_df)
total_matches = event_df["match_id"].nunique()
shots_from_corners = int(event_df["led_to_shot"].sum()) if "led_to_shot" in event_df.columns else 0
avg_xg = float(event_df["shot.statsbomb_xg"].fillna(0).mean()) if "shot.statsbomb_xg" in event_df.columns else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Corner Events", f"{total_events}")
c2.metric("Matches", f"{total_matches}")
c3.metric("Shot Outcomes", f"{shots_from_corners}")
c4.metric("Avg xG / Event", f"{avg_xg:.3f}")

# ----------------------------
# Charts
# ----------------------------
left, right = st.columns(2)

with left:
    st.subheader("Corners per Team")
    if not team_df.empty:
        fig_team = px.bar(
            team_df.sort_values("corners_taken", ascending=False),
            x="team",
            y="corners_taken",
            hover_data=["matches", "corners_per_match", "shots_from_corners", "shot_rate", "total_xg"],
        )
        fig_team.update_layout(xaxis_title="", yaxis_title="Corners Taken")
        st.plotly_chart(fig_team, use_container_width=True)
    else:
        st.info("No team data for current filters.")

with right:
    st.subheader("Match Total Corners Distribution")
    if not match_df.empty:
        fig_hist = px.histogram(
            match_df,
            x="total_corners",
            nbins=min(20, max(5, len(match_df))),
        )
        fig_hist.update_layout(xaxis_title="Total Corners", yaxis_title="Matches")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No match data for current filters.")

st.subheader("Corner Timing")
if not event_df.empty:
    bins = list(range(0, 96, 5))
    timing_df = event_df.copy()
    timing_df["minute_band"] = pd.cut(timing_df["event_minute"], bins=bins, right=False)

    timing_summary = (
        timing_df.groupby("minute_band", observed=False)
        .size()
        .reset_index(name="corners")
    )
    timing_summary["minute_band"] = timing_summary["minute_band"].astype(str)

    fig_timing = px.bar(timing_summary, x="minute_band", y="corners")
    fig_timing.update_layout(xaxis_title="Minute Band", yaxis_title="Corner Events")
    st.plotly_chart(fig_timing, use_container_width=True)
else:
    st.info("No event data for current filters.")

st.subheader("Top Matches by Total Corners")
if not match_df.empty:
    top_matches = (
        match_df.sort_values(["total_corners", "shots_from_corners"], ascending=[False, False])
        .head(20)
        .reset_index(drop=True)
    )
    st.dataframe(
        top_matches[
            [
                "Match",
                "home_team",
                "away_team",
                "home_corners",
                "away_corners",
                "total_corners",
                "shots_from_corners",
                "first_contact_shots_3s",
                "total_xg",
            ]
        ],
        use_container_width=True,
    )
else:
    st.info("No matches found.")

st.subheader("Event Data")
event_display_cols = [
    c for c in [
        "match_id",
        "Match",
        "corner_team",
        "Minute",
        "Second",
        "SP_outcome",
        "shot.statsbomb_xg",
        "Taker",
        "Shooter",
        "Defensive_setup",
    ] if c in event_df.columns
]

st.dataframe(
    event_df[event_display_cols].reset_index(drop=True),
    use_container_width=True,
    height=420,
)
