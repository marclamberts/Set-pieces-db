import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Corners App", layout="wide")


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

    separators = [" - ", " vs ", " v "]
    for sep in separators:
        if sep in match_value:
            left, right = match_value.split(sep, 1)
            return left.strip(), right.strip()

    return None, None


@st.cache_data
def load_excel(uploaded_file):
    return pd.read_excel(uploaded_file)


def prepare_data(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    match_id_col = find_col(df, ["match_id", "match id"])
    match_col = find_col(df, ["Match", "match"])
    team_col = find_col(df, ["pass_team_name", "team", "team_name"])
    minute_col = find_col(df, ["Minute", "minute"])
    second_col = find_col(df, ["Second", "second"])
    outcome_col = find_col(df, ["SP_outcome", "outcome"])
    xg_col = find_col(df, ["shot.statsbomb_xg", "xg", "shot_xg"])

    required_missing = []
    for name, col in {
        "match_id": match_id_col,
        "Match": match_col,
        "pass_team_name": team_col,
        "Minute": minute_col,
        "Second": second_col,
    }.items():
        if col is None:
            required_missing.append(name)

    if required_missing:
        raise ValueError(
            f"Missing required columns: {required_missing}. "
            f"Available columns: {list(df.columns)}"
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

    team_summary = (
        df.groupby("corner_team", dropna=False)
        .agg(
            corners_taken=("match_id", "size"),
            matches=("match_id", pd.Series.nunique),
            shots_from_corners=("led_to_shot", "sum"),
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

    return df, match_summary, team_summary


st.title("⚽ Corners Dashboard")
st.caption("Upload your Excel file and explore corner-event data.")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file is None:
    st.info("Upload your Excel file to start.")
    st.stop()

try:
    raw_df = load_excel(uploaded_file)
except Exception as e:
    st.error("Could not read the Excel file.")
    st.exception(e)
    st.stop()

try:
    df, match_summary, team_summary = prepare_data(raw_df)
except Exception as e:
    st.error("Could not prepare the data.")
    st.exception(e)
    st.write("Detected columns:")
    st.write(list(raw_df.columns))
    st.stop()

st.success("File loaded successfully.")

teams = sorted([t for t in team_summary["team"].dropna().unique().tolist() if str(t).strip()])
selected_team = st.sidebar.selectbox("Team", ["All Teams"] + teams)

if len(match_summary) > 0:
    min_c = int(match_summary["total_corners"].min())
    max_c = int(match_summary["total_corners"].max())
else:
    min_c, max_c = 0, 0

corner_range = st.sidebar.slider(
    "Match total corners",
    min_value=min_c,
    max_value=max_c,
    value=(min_c, max_c),
)

event_df = df.copy()
match_df = match_summary.copy()
team_df = team_summary.copy()

if selected_team != "All Teams":
    event_df = event_df[event_df["corner_team"] == selected_team]
    valid_match_ids = event_df["match_id"].unique()
    match_df = match_df[match_df["match_id"].isin(valid_match_ids)]
    team_df = team_df[team_df["team"] == selected_team]

match_df = match_df[
    (match_df["total_corners"] >= corner_range[0]) &
    (match_df["total_corners"] <= corner_range[1])
]

valid_match_ids = match_df["match_id"].unique()
event_df = event_df[event_df["match_id"].isin(valid_match_ids)]

c1, c2, c3, c4 = st.columns(4)
c1.metric("Corner Events", len(event_df))
c2.metric("Matches", event_df["match_id"].nunique())
c3.metric("Teams", event_df["corner_team"].nunique())
c4.metric("Shot Outcomes", int(event_df["led_to_shot"].sum()))

tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Teams", "Matches", "Raw Data"])

with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Corners by Team")
        if not team_df.empty:
            fig = px.bar(
                team_df.sort_values("corners_taken", ascending=False),
                x="team",
                y="corners_taken",
                hover_data=["matches", "corners_per_match", "shots_from_corners", "shot_rate", "total_xg"],
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for current filters.")

    with col_b:
        st.subheader("Match Total Corners")
        if not match_df.empty:
            fig = px.histogram(match_df, x="total_corners", nbins=min(20, max(5, len(match_df))))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for current filters.")

    st.subheader("Corner Timing")
    if not event_df.empty:
        timing = event_df.copy()
        bins = list(range(0, 101, 5))
        timing["minute_band"] = pd.cut(timing["event_minute"], bins=bins, right=False)
        timing_summary = timing.groupby("minute_band", observed=False).size().reset_index(name="corners")
        timing_summary["minute_band"] = timing_summary["minute_band"].astype(str)
        fig = px.bar(timing_summary, x="minute_band", y="corners")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No event data available.")

with tab2:
    st.subheader("Team Summary")
    st.dataframe(
        team_df.sort_values("corners_taken", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=500,
    )

with tab3:
    st.subheader("Match Summary")
    show_cols = [
        c for c in [
            "Match",
            "home_team",
            "away_team",
            "home_corners",
            "away_corners",
            "total_corners",
            "shots_from_corners",
            "total_xg",
            "avg_xg",
        ] if c in match_df.columns
    ]
    st.dataframe(
        match_df[show_cols].sort_values("total_corners", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=500,
    )

with tab4:
    st.subheader("Raw Event Data")
    st.dataframe(event_df.reset_index(drop=True), use_container_width=True, height=500)
