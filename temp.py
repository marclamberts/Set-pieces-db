import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Allsvenskan Corners Studio", layout="wide")

FILE_NAME = "Allsvenskan - Corners 2025.xlsx"


# -----------------------------
# Helpers
# -----------------------------
def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def find_col(df, candidates):
    lower_map = {str(c).strip().lower(): c for c in df.columns}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for c in df.columns:
        c_low = str(c).strip().lower()
        for cand in candidates:
            if cand.lower() in c_low:
                return c
    return None


def split_match_name(match_value):
    if not isinstance(match_value, str):
        return None, None

    for sep in [" - ", " vs ", " v "]:
        if sep in match_value:
            left, right = match_value.split(sep, 1)
            return left.strip(), right.strip()

    return None, None


@st.cache_data
def load_data():
    if not os.path.exists(FILE_NAME):
        raise FileNotFoundError(
            f"{FILE_NAME} not found. Put it in the same folder as temp.py"
        )
    return pd.read_excel(FILE_NAME)


def prepare_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    match_id_col = find_col(df, ["match_id", "match id"])
    match_col = find_col(df, ["match"])
    team_col = find_col(df, ["pass_team_name", "team", "team_name"])
    minute_col = find_col(df, ["minute"])
    second_col = find_col(df, ["second"])
    outcome_col = find_col(df, ["sp_outcome", "outcome"])
    xg_col = find_col(df, ["shot.statsbomb_xg", "shot_xg", "xg"])

    required = {
        "match_id": match_id_col,
        "match": match_col,
        "pass_team_name": team_col,
        "minute": minute_col,
        "second": second_col,
    }

    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )

    df = df.rename(columns={
        match_id_col: "match_id",
        match_col: "Match",
        team_col: "corner_team",
        minute_col: "Minute",
        second_col: "Second",
    })

    if outcome_col is not None:
        df = df.rename(columns={outcome_col: "SP_outcome"})
    else:
        df["SP_outcome"] = ""

    if xg_col is not None:
        df = df.rename(columns={xg_col: "shot_xg"})
    else:
        df["shot_xg"] = np.nan

    df["Minute"] = safe_numeric(df["Minute"])
    df["Second"] = safe_numeric(df["Second"])
    df["shot_xg"] = safe_numeric(df["shot_xg"])

    df["corner_team"] = df["corner_team"].astype(str).str.strip()
    df["event_minute"] = df["Minute"].fillna(0) + df["Second"].fillna(0) / 60.0
    df["led_to_shot"] = df["SP_outcome"].astype(str).str.contains("shot", case=False, na=False)
    df["is_first_contact_shot"] = df["SP_outcome"].astype(str).str.contains(
        "first contact", case=False, na=False
    )

    homes = []
    aways = []
    for m in df["Match"]:
        h, a = split_match_name(m)
        homes.append(h)
        aways.append(a)

    df["home_team"] = homes
    df["away_team"] = aways

    match_summary = (
        df.groupby(["match_id", "Match", "home_team", "away_team"], dropna=False)
        .agg(
            total_corners=("match_id", "size"),
            shots_from_corners=("led_to_shot", "sum"),
            first_contact_shots=("is_first_contact_shot", "sum"),
            total_xg=("shot_xg", "sum"),
            avg_xg=("shot_xg", "mean"),
        )
        .reset_index()
    )

    def count_team_corners(match_id, team_name):
        if pd.isna(team_name):
            return np.nan
        return int(((df["match_id"] == match_id) & (df["corner_team"] == team_name)).sum())

    match_summary["home_corners"] = match_summary.apply(
        lambda r: count_team_corners(r["match_id"], r["home_team"]), axis=1
    )
    match_summary["away_corners"] = match_summary.apply(
        lambda r: count_team_corners(r["match_id"], r["away_team"]), axis=1
    )
    match_summary["corners_per_shot"] = np.where(
        match_summary["shots_from_corners"] > 0,
        match_summary["total_corners"] / match_summary["shots_from_corners"],
        np.nan,
    )

    team_summary = (
        df.groupby("corner_team", dropna=False)
        .agg(
            corners_taken=("match_id", "size"),
            matches=("match_id", pd.Series.nunique),
            shots_from_corners=("led_to_shot", "sum"),
            first_contact_shots=("is_first_contact_shot", "sum"),
            total_xg=("shot_xg", "sum"),
            avg_xg_per_corner=("shot_xg", "mean"),
        )
        .reset_index()
        .rename(columns={"corner_team": "team"})
    )

    team_summary["corners_per_match"] = (
        team_summary["corners_taken"] / team_summary["matches"].replace(0, np.nan)
    )
    team_summary["shot_rate"] = (
        team_summary["shots_from_corners"] / team_summary["corners_taken"].replace(0, np.nan)
    )
    team_summary["xg_per_match"] = (
        team_summary["total_xg"] / team_summary["matches"].replace(0, np.nan)
    )
    team_summary["first_contact_rate"] = (
        team_summary["first_contact_shots"] / team_summary["corners_taken"].replace(0, np.nan)
    )

    return df, match_summary, team_summary


def metric_row(event_df, match_df, title_prefix=""):
    c1, c2, c3, c4, c5 = st.columns(5)

    corners = len(event_df)
    matches = event_df["match_id"].nunique() if not event_df.empty else 0
    avg_corners = match_df["total_corners"].mean() if not match_df.empty else 0
    shot_outcomes = int(event_df["led_to_shot"].sum()) if not event_df.empty else 0
    total_xg = event_df["shot_xg"].fillna(0).sum() if "shot_xg" in event_df.columns and not event_df.empty else 0

    c1.metric(f"{title_prefix}Corner Events", f"{corners}")
    c2.metric(f"{title_prefix}Matches", f"{matches}")
    c3.metric(f"{title_prefix}Avg Corners/Match", f"{avg_corners:.2f}")
    c4.metric(f"{title_prefix}Shot Outcomes", f"{shot_outcomes}")
    c5.metric(f"{title_prefix}Total xG", f"{total_xg:.2f}")


def build_league_team_summary(source_df):
    league_team_summary = (
        source_df.groupby("corner_team", dropna=False)
        .agg(
            corners_taken=("match_id", "size"),
            matches=("match_id", pd.Series.nunique),
            shots_from_corners=("led_to_shot", "sum"),
            first_contact_shots=("is_first_contact_shot", "sum"),
            total_xg=("shot_xg", "sum"),
            avg_xg_per_corner=("shot_xg", "mean"),
        )
        .reset_index()
        .rename(columns={"corner_team": "team"})
    )

    league_team_summary["corners_per_match"] = (
        league_team_summary["corners_taken"] / league_team_summary["matches"].replace(0, np.nan)
    )
    league_team_summary["shot_rate"] = (
        league_team_summary["shots_from_corners"] / league_team_summary["corners_taken"].replace(0, np.nan)
    )
    league_team_summary["xg_per_match"] = (
        league_team_summary["total_xg"] / league_team_summary["matches"].replace(0, np.nan)
    )
    league_team_summary["first_contact_rate"] = (
        league_team_summary["first_contact_shots"] / league_team_summary["corners_taken"].replace(0, np.nan)
    )
    return league_team_summary


# -----------------------------
# Load
# -----------------------------
st.title("⚽ Allsvenskan Corners Studio")
st.caption("American-style soccer analysis dashboard")

try:
    raw_df = load_data()
except Exception as e:
    st.error("Failed to load the Excel file.")
    st.exception(e)
    st.stop()

try:
    df, match_summary, team_summary = prepare_data(raw_df)
except Exception as e:
    st.error("Failed to prepare the data.")
    st.exception(e)
    st.write("Detected columns:")
    st.write(list(raw_df.columns))
    st.stop()


# -----------------------------
# Sidebar menu
# -----------------------------
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Go to",
    ["League Overview", "Team Analysis", "Match Explorer", "Data Center"],
)

teams = sorted([t for t in team_summary["team"].dropna().unique().tolist() if str(t).strip()])
selected_team = st.sidebar.selectbox("Team Filter", ["All Teams"] + teams)

if len(match_summary) > 0:
    min_c = int(match_summary["total_corners"].min())
    max_c = int(match_summary["total_corners"].max())
else:
    min_c, max_c = 0, 0

corner_range = st.sidebar.slider(
    "Match Total Corners",
    min_value=min_c,
    max_value=max_c,
    value=(min_c, max_c),
)

st.sidebar.markdown("---")
st.sidebar.caption("Dashboard style: American soccer analysis")


# -----------------------------
# Common filtered frames
# -----------------------------
event_df = df.copy()
match_df = match_summary.copy()
team_df = team_summary.copy()

if selected_team != "All Teams":
    event_df = event_df[event_df["corner_team"] == selected_team]
    selected_match_ids = event_df["match_id"].unique()
    match_df = match_df[match_df["match_id"].isin(selected_match_ids)]
    team_df = team_df[team_df["team"] == selected_team]

match_df = match_df[
    (match_df["total_corners"] >= corner_range[0]) &
    (match_df["total_corners"] <= corner_range[1])
]

filtered_match_ids = match_df["match_id"].unique()
event_df = event_df[event_df["match_id"].isin(filtered_match_ids)]

league_match_df = match_summary[
    (match_summary["total_corners"] >= corner_range[0]) &
    (match_summary["total_corners"] <= corner_range[1])
].copy()

league_match_ids = league_match_df["match_id"].unique()
league_event_df = df[df["match_id"].isin(league_match_ids)].copy()
league_team_df = build_league_team_summary(league_event_df)


# -----------------------------
# Page: League Overview
# -----------------------------
if page == "League Overview":
    top_tabs = st.tabs(["Snapshot", "Team Rankings", "League Trends"])

    with top_tabs[0]:
        st.subheader("League Snapshot")
        metric_row(league_event_df, league_match_df, "")

        a, b = st.columns(2)

        with a:
            st.subheader("Corners per Team")
            if not league_team_df.empty:
                fig = px.bar(
                    league_team_df.sort_values("corners_taken", ascending=False),
                    x="team",
                    y="corners_taken",
                    hover_data=["matches", "corners_per_match", "shots_from_corners", "shot_rate", "total_xg"],
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available.")

        with b:
            st.subheader("Average Corners per Match")
            if not league_team_df.empty:
                fig = px.bar(
                    league_team_df.sort_values("corners_per_match", ascending=False),
                    x="team",
                    y="corners_per_match",
                    hover_data=["corners_taken", "matches", "shot_rate", "xg_per_match"],
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available.")

    with top_tabs[1]:
        st.subheader("Team Rankings")

        rank_tabs = st.tabs(["Volume", "Efficiency", "Chance Creation"])

        with rank_tabs[0]:
            st.dataframe(
                league_team_df.sort_values("corners_taken", ascending=False).reset_index(drop=True),
                use_container_width=True,
                height=520,
            )

        with rank_tabs[1]:
            efficiency_table = league_team_df.copy()
            efficiency_table = efficiency_table.sort_values(
                ["shot_rate", "first_contact_rate"], ascending=False
            )
            st.dataframe(
                efficiency_table.reset_index(drop=True),
                use_container_width=True,
                height=520,
            )

        with rank_tabs[2]:
            chance_table = league_team_df.copy()
            chance_table = chance_table.sort_values(
                ["total_xg", "xg_per_match", "avg_xg_per_corner"], ascending=False
            )
            st.dataframe(
                chance_table.reset_index(drop=True),
                use_container_width=True,
                height=520,
            )

    with top_tabs[2]:
        st.subheader("League Trends")

        x, y = st.columns(2)

        with x:
            st.subheader("Match Total Corners Distribution")
            if not league_match_df.empty:
                fig = px.histogram(
                    league_match_df,
                    x="total_corners",
                    nbins=min(20, max(5, len(league_match_df))),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No data available.")

        with y:
            st.subheader("Corner Timing")
            if not league_event_df.empty:
                timing = league_event_df.copy()
                bins = list(range(0, 101, 5))
                timing["minute_band"] = pd.cut(timing["event_minute"], bins=bins, right=False)
                timing_summary = timing.groupby("minute_band", observed=False).size().reset_index(name="corners")
                timing_summary["minute_band"] = timing_summary["minute_band"].astype(str)
                fig = px.bar(timing_summary, x="minute_band", y="corners")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No timing data available.")

        st.subheader("League Match Board")
        board_cols = [
            c for c in [
                "Match",
                "home_team",
                "away_team",
                "home_corners",
                "away_corners",
                "total_corners",
                "shots_from_corners",
                "first_contact_shots",
                "total_xg",
                "avg_xg",
            ] if c in league_match_df.columns
        ]
        st.dataframe(
            league_match_df[board_cols]
            .sort_values(["total_corners", "shots_from_corners"], ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            height=500,
        )


# -----------------------------
# Page: Team Analysis
# -----------------------------
elif page == "Team Analysis":
    team_tabs = st.tabs(["Team Snapshot", "Production Profile", "Event Log"])

    with team_tabs[0]:
        title = selected_team if selected_team != "All Teams" else "All Teams"
        st.subheader(f"Team Snapshot — {title}")
        metric_row(event_df, match_df, "")

        left, right = st.columns(2)

        with left:
            st.subheader("Team Corner Volume")
            if not team_df.empty:
                fig = px.bar(
                    team_df.sort_values("corners_taken", ascending=False),
                    x="team",
                    y="corners_taken",
                    hover_data=["matches", "corners_per_match", "shots_from_corners", "shot_rate", "total_xg"],
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No team data available.")

        with right:
            st.subheader("Shot Outcomes from Corners")
            if not team_df.empty:
                fig = px.bar(
                    team_df.sort_values("shots_from_corners", ascending=False),
                    x="team",
                    y="shots_from_corners",
                    hover_data=["corners_taken", "shot_rate", "first_contact_shots", "first_contact_rate"],
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No team data available.")

    with team_tabs[1]:
        st.subheader("Production Profile")

        left, right = st.columns(2)

        with left:
            if not team_df.empty:
                fig = px.scatter(
                    team_df,
                    x="corners_per_match",
                    y="shot_rate",
                    size="corners_taken",
                    hover_name="team",
                    hover_data=["matches", "total_xg", "xg_per_match"],
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No production profile available.")

        with right:
            if not team_df.empty:
                fig = px.scatter(
                    team_df,
                    x="corners_per_match",
                    y="xg_per_match",
                    size="shots_from_corners",
                    hover_name="team",
                    hover_data=["shot_rate", "avg_xg_per_corner", "first_contact_rate"],
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No production profile available.")

        st.subheader("Team Table")
        st.dataframe(
            team_df.sort_values(
                ["corners_taken", "shots_from_corners", "total_xg"], ascending=False
            ).reset_index(drop=True),
            use_container_width=True,
            height=500,
        )

    with team_tabs[2]:
        st.subheader("Event Log")
        show_cols = [
            c for c in [
                "match_id",
                "Match",
                "corner_team",
                "Minute",
                "Second",
                "SP_outcome",
                "shot_xg",
                "home_team",
                "away_team",
            ] if c in event_df.columns
        ]
        st.dataframe(
            event_df[show_cols].sort_values(["match_id", "Minute", "Second"]).reset_index(drop=True),
            use_container_width=True,
            height=550,
        )


# -----------------------------
# Page: Match Explorer
# -----------------------------
elif page == "Match Explorer":
    available_matches = sorted(match_df["Match"].dropna().unique().tolist()) if not match_df.empty else []
    selected_match = st.selectbox("Select Match", ["All Matches"] + available_matches)

    match_tabs = st.tabs(["Match Board", "Timeline", "Corner Breakdown"])

    match_event_df = event_df.copy()
    match_board_df = match_df.copy()

    if selected_match != "All Matches":
        match_board_df = match_board_df[match_board_df["Match"] == selected_match]
        selected_ids = match_board_df["match_id"].unique()
        match_event_df = match_event_df[match_event_df["match_id"].isin(selected_ids)]

    with match_tabs[0]:
        st.subheader("Match Board")
        metric_row(match_event_df, match_board_df, "")

        board_cols = [
            c for c in [
                "Match",
                "home_team",
                "away_team",
                "home_corners",
                "away_corners",
                "total_corners",
                "shots_from_corners",
                "first_contact_shots",
                "total_xg",
                "avg_xg",
            ] if c in match_board_df.columns
        ]

        st.dataframe(
            match_board_df[board_cols]
            .sort_values(["total_corners", "shots_from_corners"], ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            height=500,
        )

    with match_tabs[1]:
        st.subheader("Timeline")
        if not match_event_df.empty:
            timing = (
                match_event_df.groupby(["Minute"], dropna=False)
                .size()
                .reset_index(name="corner_events")
                .sort_values("Minute")
            )
            fig = px.bar(timing, x="Minute", y="corner_events")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No timeline data available.")

    with match_tabs[2]:
        st.subheader("Corner Breakdown")
        if not match_event_df.empty:
            left, right = st.columns(2)

            with left:
                by_team = (
                    match_event_df.groupby("corner_team", dropna=False)
                    .agg(
                        corners=("match_id", "size"),
                        shots=("led_to_shot", "sum"),
                        total_xg=("shot_xg", "sum"),
                    )
                    .reset_index()
                )
                fig = px.bar(by_team, x="corner_team", y="corners", hover_data=["shots", "total_xg"])
                st.plotly_chart(fig, use_container_width=True)

            with right:
                outcome_summary = (
                    match_event_df.groupby("SP_outcome", dropna=False)
                    .size()
                    .reset_index(name="events")
                    .sort_values("events", ascending=False)
                    .head(15)
                )
                fig = px.bar(outcome_summary, x="SP_outcome", y="events")
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                match_event_df.sort_values(["Minute", "Second"]).reset_index(drop=True),
                use_container_width=True,
                height=420,
            )
        else:
            st.info("No match event data available.")


# -----------------------------
# Page: Data Center
# -----------------------------
elif page == "Data Center":
    data_tabs = st.tabs(["Raw Events", "Team Table", "Match Table"])

    with data_tabs[0]:
        st.subheader("Raw Event Data")
        st.dataframe(
            event_df.reset_index(drop=True),
            use_container_width=True,
            height=600,
        )

    with data_tabs[1]:
        st.subheader("Team Table")
        st.dataframe(
            team_df.sort_values("corners_taken", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=600,
        )

    with data_tabs[2]:
        st.subheader("Match Table")
        st.dataframe(
            match_df.sort_values("total_corners", ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=600,
        )
