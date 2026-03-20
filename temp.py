import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Allsvenskan Set Piece Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

FILE_NAME = "Allsvenskan - Corners 2025.xlsx"


# =========================================================
# STYLING
# =========================================================
st.markdown("""
<style>
.block-container {
    padding-top: 1.0rem;
    padding-bottom: 1rem;
    max-width: 1550px;
}
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(120,120,120,0.18);
}
.kpi-card {
    background: linear-gradient(180deg, rgba(20,24,33,1) 0%, rgba(31,36,47,1) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px;
    padding: 14px 16px;
    margin-bottom: 8px;
}
.kpi-label {
    font-size: 0.78rem;
    color: #9fb0c7;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.kpi-value {
    font-size: 1.85rem;
    font-weight: 800;
    color: white;
    line-height: 1.08;
    margin-top: 4px;
}
.section-title {
    font-size: 1.12rem;
    font-weight: 800;
    margin-top: 0.3rem;
    margin-bottom: 0.7rem;
}
.subtle {
    color: #97a6ba;
    font-size: 0.92rem;
}
.badge {
    display: inline-block;
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    background: rgba(70,130,180,0.18);
    color: #d8e8ff;
    font-size: 0.82rem;
    margin-right: 0.35rem;
}
.insight-box {
    background: rgba(65, 90, 119, 0.12);
    border: 1px solid rgba(120, 160, 220, 0.18);
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
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


def classify_outcome(text):
    s = str(text).lower().strip()

    if "first contact - shot within 3 seconds" in s:
        return "Shot ≤3s"
    if "first contact" in s and "shot" in s:
        return "First Contact Shot"
    if "shot" in s:
        return "Shot"
    if "no first contact" in s:
        return "No First Contact"
    if s in ["", "nan"]:
        return "Unknown"
    return "Other"


def percentile_rank(series, value):
    s = series.dropna()
    if len(s) == 0 or pd.isna(value):
        return np.nan
    return float((s <= value).mean() * 100)


def metric_card(label, value, suffix=""):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}{suffix}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpis(events, matches):
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        metric_card("Corner Events", f"{len(events):,}")

    with c2:
        metric_card("Matches", f"{events['match_id'].nunique() if not events.empty else 0:,}")

    with c3:
        avg_corners = matches["total_corners"].mean() if not matches.empty else 0
        metric_card("Avg Corners / Match", f"{avg_corners:.2f}")

    with c4:
        shot_outcomes = int(events["led_to_shot"].sum()) if not events.empty else 0
        metric_card("Shot Outcomes", f"{shot_outcomes:,}")

    with c5:
        total_xg = events["shot_xg"].fillna(0).sum() if not events.empty else 0
        metric_card("Total xG", f"{total_xg:.2f}")

    with c6:
        shot_rate = events["led_to_shot"].mean() * 100 if len(events) > 0 else 0
        metric_card("Shot Rate", f"{shot_rate:.1f}", suffix="%")


def draw_pitch(fig):
    line_color = "rgba(255,255,255,0.55)"
    pitch_color = "rgba(18,60,34,1)"

    fig.update_xaxes(range=[0, 120], visible=False)
    fig.update_yaxes(range=[0, 80], visible=False, scaleanchor="x", scaleratio=1)

    fig.update_layout(
        paper_bgcolor=pitch_color,
        plot_bgcolor=pitch_color,
        margin=dict(l=10, r=10, t=10, b=10),
        height=560,
        shapes=[
            dict(type="rect", x0=0, y0=0, x1=120, y1=80, line=dict(color=line_color, width=2)),
            dict(type="line", x0=60, y0=0, x1=60, y1=80, line=dict(color=line_color, width=2)),
            dict(type="circle", x0=50, y0=30, x1=70, y1=50, line=dict(color=line_color, width=2)),
            dict(type="rect", x0=0, y0=18, x1=18, y1=62, line=dict(color=line_color, width=2)),
            dict(type="rect", x0=0, y0=30, x1=6, y1=50, line=dict(color=line_color, width=2)),
            dict(type="rect", x0=102, y0=18, x1=120, y1=62, line=dict(color=line_color, width=2)),
            dict(type="rect", x0=114, y0=30, x1=120, y1=50, line=dict(color=line_color, width=2)),
            dict(type="circle", x0=10, y0=38, x1=14, y1=42, line=dict(color=line_color, width=2)),
            dict(type="circle", x0=106, y0=38, x1=110, y1=42, line=dict(color=line_color, width=2)),
        ],
    )
    return fig


def shotmap_figure(df_shots, color_col="corner_team", title="Shotmap"):
    fig = go.Figure()
    fig = draw_pitch(fig)

    if df_shots.empty:
        fig.update_layout(title=title)
        return fig

    plot_df = df_shots.copy()
    plot_df["shot_xg_plot"] = plot_df["shot_xg"].fillna(0)
    plot_df["marker_size"] = np.clip(plot_df["shot_xg_plot"] * 90 + 10, 10, 38)

    categories = plot_df[color_col].fillna("Unknown").astype(str).unique().tolist()
    palette = px.colors.qualitative.Set2 + px.colors.qualitative.Bold + px.colors.qualitative.Safe
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

    for cat in categories:
        sub = plot_df[plot_df[color_col].fillna("Unknown").astype(str) == cat]
        fig.add_trace(
            go.Scatter(
                x=sub["shot_location_x"],
                y=sub["shot_location_y"],
                mode="markers",
                name=str(cat),
                marker=dict(
                    size=sub["marker_size"],
                    color=color_map[cat],
                    opacity=0.82,
                    line=dict(color="white", width=1),
                ),
                text=[
                    f"{row['Match']}<br>"
                    f"Team: {row['corner_team']}<br>"
                    f"Shooter: {row['Shooter']}<br>"
                    f"xG: {0 if pd.isna(row['shot_xg']) else row['shot_xg']:.3f}<br>"
                    f"Minute: {int(row['Minute']) if pd.notna(row['Minute']) else ''}"
                    for _, row in sub.iterrows()
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    fig.update_layout(title=title, legend_title_text="")
    return fig


# =========================================================
# DATA LOAD
# =========================================================
@st.cache_data
def load_data():
    if not os.path.exists(FILE_NAME):
        raise FileNotFoundError(
            f"{FILE_NAME} not found. Put it in the same folder as temp.py"
        )
    return pd.read_excel(FILE_NAME)


@st.cache_data
def prepare_data(raw_df):
    df = raw_df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    match_id_col = find_col(df, ["match_id", "match id"])
    match_col = find_col(df, ["match"])
    team_col = find_col(df, ["pass_team_name", "team", "team_name"])
    minute_col = find_col(df, ["minute"])
    second_col = find_col(df, ["second"])
    outcome_col = find_col(df, ["sp_outcome", "outcome"])
    xg_col = find_col(df, ["shot.statsbomb_xg", "shot_xg", "xg"])
    taker_col = find_col(df, ["taker"])
    shooter_col = find_col(df, ["shooter"])
    defensive_setup_col = find_col(df, ["defensive_setup"])
    shot_team_col = find_col(df, ["shot_team_name"])
    pass_x_col = find_col(df, ["pass_location_x"])
    pass_y_col = find_col(df, ["pass_location_y"])
    pass_end_x_col = find_col(df, ["pass_end_location_x"])
    pass_end_y_col = find_col(df, ["pass_end_location_y"])
    shot_x_col = find_col(df, ["shot_location_x"])
    shot_y_col = find_col(df, ["shot_location_y"])
    shot_z_col = find_col(df, ["shot_location_z"])
    pass_technique_col = find_col(df, ["pass.technique.name"])
    pass_height_col = find_col(df, ["pass.height.name"])
    pass_body_col = find_col(df, ["pass.body_part.name"])
    shot_body_col = find_col(df, ["shot.body_part.name"])
    shot_outcome_col = find_col(df, ["shot.outcome.name"])
    pass_outcome_col = find_col(df, ["pass.outcome.name"])
    pass_position_col = find_col(df, ["pass_position"])
    shot_position_col = find_col(df, ["shot_position"])

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

    rename_map = {
        match_id_col: "match_id",
        match_col: "Match",
        team_col: "corner_team",
        minute_col: "Minute",
        second_col: "Second",
    }

    optional_map = {
        outcome_col: "SP_outcome",
        xg_col: "shot_xg",
        taker_col: "Taker",
        shooter_col: "Shooter",
        defensive_setup_col: "Defensive_setup",
        shot_team_col: "shot_team_name",
        pass_x_col: "pass_location_x",
        pass_y_col: "pass_location_y",
        pass_end_x_col: "pass_end_location_x",
        pass_end_y_col: "pass_end_location_y",
        shot_x_col: "shot_location_x",
        shot_y_col: "shot_location_y",
        shot_z_col: "shot_location_z",
        pass_technique_col: "pass_technique",
        pass_height_col: "pass_height",
        pass_body_col: "pass_body_part",
        shot_body_col: "shot_body_part",
        shot_outcome_col: "shot_outcome",
        pass_outcome_col: "pass_outcome",
        pass_position_col: "pass_position",
        shot_position_col: "shot_position",
    }

    for k, v in optional_map.items():
        if k is not None:
            rename_map[k] = v

    df = df.rename(columns=rename_map)

    defaults = {
        "SP_outcome": "",
        "shot_xg": np.nan,
        "Taker": np.nan,
        "Shooter": np.nan,
        "Defensive_setup": np.nan,
        "shot_team_name": np.nan,
        "pass_location_x": np.nan,
        "pass_location_y": np.nan,
        "pass_end_location_x": np.nan,
        "pass_end_location_y": np.nan,
        "shot_location_x": np.nan,
        "shot_location_y": np.nan,
        "shot_location_z": np.nan,
        "pass_technique": np.nan,
        "pass_height": np.nan,
        "pass_body_part": np.nan,
        "shot_body_part": np.nan,
        "shot_outcome": np.nan,
        "pass_outcome": np.nan,
        "pass_position": np.nan,
        "shot_position": np.nan,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    numeric_cols = [
        "Minute", "Second", "shot_xg", "pass_location_x", "pass_location_y",
        "pass_end_location_x", "pass_end_location_y", "shot_location_x",
        "shot_location_y", "shot_location_z"
    ]
    for col in numeric_cols:
        df[col] = safe_numeric(df[col])

    df["corner_team"] = df["corner_team"].astype(str).str.strip()
    df["event_minute"] = df["Minute"].fillna(0) + df["Second"].fillna(0) / 60.0

    homes, aways = [], []
    for m in df["Match"]:
        h, a = split_match_name(m)
        homes.append(h)
        aways.append(a)

    df["home_team"] = homes
    df["away_team"] = aways
    df["is_home_corner"] = df["corner_team"] == df["home_team"]
    df["is_away_corner"] = df["corner_team"] == df["away_team"]

    df["led_to_shot"] = df["SP_outcome"].astype(str).str.contains("shot", case=False, na=False)
    df["is_first_contact_shot"] = df["SP_outcome"].astype(str).str.contains("first contact", case=False, na=False)
    df["is_fast_shot"] = df["SP_outcome"].astype(str).str.contains("within 3 seconds", case=False, na=False)
    df["outcome_bucket"] = df["SP_outcome"].apply(classify_outcome)
    df["is_inswinger"] = df["pass_technique"].astype(str).str.contains("inswing", case=False, na=False)
    df["is_outswinger"] = df["pass_technique"].astype(str).str.contains("outswing", case=False, na=False)
    df["is_goal_kick_zone_delivery"] = (
        (df["pass_end_location_x"].between(114, 120, inclusive="both")) &
        (df["pass_end_location_y"].between(30, 50, inclusive="both"))
    )

    df["phase"] = pd.cut(
        df["event_minute"],
        bins=[-0.1, 15, 30, 45, 60, 75, 120],
        labels=["0-15", "16-30", "31-45", "46-60", "61-75", "76+"],
        right=True,
    ).astype(str)

    match_summary = (
        df.groupby(["match_id", "Match", "home_team", "away_team"], dropna=False)
        .agg(
            total_corners=("match_id", "size"),
            shots_from_corners=("led_to_shot", "sum"),
            first_contact_shots=("is_first_contact_shot", "sum"),
            fast_shots=("is_fast_shot", "sum"),
            total_xg=("shot_xg", "sum"),
            avg_xg=("shot_xg", "mean"),
            unique_takers=("Taker", pd.Series.nunique),
            inswingers=("is_inswinger", "sum"),
            outswingers=("is_outswinger", "sum"),
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
    match_summary["shot_rate"] = match_summary["shots_from_corners"] / match_summary["total_corners"].replace(0, np.nan)
    match_summary["xg_per_corner"] = match_summary["total_xg"] / match_summary["total_corners"].replace(0, np.nan)

    team_summary = (
        df.groupby("corner_team", dropna=False)
        .agg(
            corners_taken=("match_id", "size"),
            matches=("match_id", pd.Series.nunique),
            shots_from_corners=("led_to_shot", "sum"),
            first_contact_shots=("is_first_contact_shot", "sum"),
            fast_shots=("is_fast_shot", "sum"),
            total_xg=("shot_xg", "sum"),
            avg_xg_per_corner=("shot_xg", "mean"),
            taker_variety=("Taker", pd.Series.nunique),
            inswingers=("is_inswinger", "sum"),
            outswingers=("is_outswinger", "sum"),
            target_box_deliveries=("is_goal_kick_zone_delivery", "sum"),
        )
        .reset_index()
        .rename(columns={"corner_team": "team"})
    )

    team_summary["corners_per_match"] = team_summary["corners_taken"] / team_summary["matches"].replace(0, np.nan)
    team_summary["shot_rate"] = team_summary["shots_from_corners"] / team_summary["corners_taken"].replace(0, np.nan)
    team_summary["first_contact_rate"] = team_summary["first_contact_shots"] / team_summary["corners_taken"].replace(0, np.nan)
    team_summary["fast_shot_rate"] = team_summary["fast_shots"] / team_summary["corners_taken"].replace(0, np.nan)
    team_summary["xg_per_match"] = team_summary["total_xg"] / team_summary["matches"].replace(0, np.nan)
    team_summary["box_delivery_rate"] = team_summary["target_box_deliveries"] / team_summary["corners_taken"].replace(0, np.nan)

    return df, match_summary, team_summary


def build_team_summary(source_df):
    ts = (
        source_df.groupby("corner_team", dropna=False)
        .agg(
            corners_taken=("match_id", "size"),
            matches=("match_id", pd.Series.nunique),
            shots_from_corners=("led_to_shot", "sum"),
            first_contact_shots=("is_first_contact_shot", "sum"),
            fast_shots=("is_fast_shot", "sum"),
            total_xg=("shot_xg", "sum"),
            avg_xg_per_corner=("shot_xg", "mean"),
            taker_variety=("Taker", pd.Series.nunique),
            inswingers=("is_inswinger", "sum"),
            outswingers=("is_outswinger", "sum"),
            target_box_deliveries=("is_goal_kick_zone_delivery", "sum"),
        )
        .reset_index()
        .rename(columns={"corner_team": "team"})
    )

    ts["corners_per_match"] = ts["corners_taken"] / ts["matches"].replace(0, np.nan)
    ts["shot_rate"] = ts["shots_from_corners"] / ts["corners_taken"].replace(0, np.nan)
    ts["first_contact_rate"] = ts["first_contact_shots"] / ts["corners_taken"].replace(0, np.nan)
    ts["fast_shot_rate"] = ts["fast_shots"] / ts["corners_taken"].replace(0, np.nan)
    ts["xg_per_match"] = ts["total_xg"] / ts["matches"].replace(0, np.nan)
    ts["box_delivery_rate"] = ts["target_box_deliveries"] / ts["corners_taken"].replace(0, np.nan)
    return ts


# =========================================================
# LOAD
# =========================================================
st.title("⚽ Allsvenskan Set Piece Studio")
st.markdown(
    '<span class="badge">League Dashboard</span>'
    '<span class="badge">Team Intelligence</span>'
    '<span class="badge">Shotmap</span>'
    '<span class="badge">Set Piece Lab</span>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtle">Detailed American-style corner analysis built directly from your event-level Excel.</div>',
    unsafe_allow_html=True,
)

try:
    raw_df = load_data()
    df, match_summary, team_summary = prepare_data(raw_df)
except Exception as e:
    st.error("Failed to load or prepare the Excel file.")
    st.exception(e)
    st.stop()


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.title("Menu")
page = st.sidebar.radio(
    "Select Page",
    ["League Overview", "Team Analysis", "Match Explorer", "Set Piece Lab", "Data Center"],
)

all_teams = sorted([t for t in team_summary["team"].dropna().unique().tolist() if str(t).strip()])
selected_team = st.sidebar.selectbox("Team", ["All Teams"] + all_teams)

if len(match_summary) > 0:
    min_corners = int(match_summary["total_corners"].min())
    max_corners = int(match_summary["total_corners"].max())
else:
    min_corners, max_corners = 0, 0

corner_range = st.sidebar.slider(
    "Match Corner Range",
    min_value=min_corners,
    max_value=max_corners,
    value=(min_corners, max_corners),
)

show_shot_only = st.sidebar.checkbox("Shot outcomes only", value=False)
show_inswing_only = st.sidebar.checkbox("Inswingers only", value=False)
show_outswing_only = st.sidebar.checkbox("Outswingers only", value=False)

st.sidebar.markdown("---")
st.sidebar.caption("Style: analyst desk / broadcast dashboard")


# =========================================================
# FILTERS
# =========================================================
league_match_df = match_summary[
    (match_summary["total_corners"] >= corner_range[0]) &
    (match_summary["total_corners"] <= corner_range[1])
].copy()

league_match_ids = league_match_df["match_id"].unique()
league_event_df = df[df["match_id"].isin(league_match_ids)].copy()

if show_shot_only:
    league_event_df = league_event_df[league_event_df["led_to_shot"]]

if show_inswing_only and not show_outswing_only:
    league_event_df = league_event_df[league_event_df["is_inswinger"]]

if show_outswing_only and not show_inswing_only:
    league_event_df = league_event_df[league_event_df["is_outswinger"]]

valid_ids = league_event_df["match_id"].unique()
league_match_df = league_match_df[league_match_df["match_id"].isin(valid_ids)]
league_team_df = build_team_summary(league_event_df)

event_df = league_event_df.copy()
match_df = league_match_df.copy()
team_df = league_team_df.copy()

if selected_team != "All Teams":
    event_df = event_df[event_df["corner_team"] == selected_team]
    valid_ids = event_df["match_id"].unique()
    match_df = match_df[match_df["match_id"].isin(valid_ids)]
    team_df = team_df[team_df["team"] == selected_team]


# =========================================================
# PAGE: LEAGUE OVERVIEW
# =========================================================
if page == "League Overview":
    top_tabs = st.tabs(["League Snapshot", "Rankings", "League Trends", "Shotmap"])

    with top_tabs[0]:
        render_kpis(league_event_df, league_match_df)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title">Corner Volume by Team</div>', unsafe_allow_html=True)
            if not league_team_df.empty:
                fig = px.bar(
                    league_team_df.sort_values("corners_taken", ascending=False),
                    x="team",
                    y="corners_taken",
                    hover_data=["matches", "corners_per_match", "shots_from_corners", "shot_rate", "xg_per_match"],
                )
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">Efficiency: Shot Rate by Team</div>', unsafe_allow_html=True)
            if not league_team_df.empty:
                fig = px.scatter(
                    league_team_df,
                    x="corners_per_match",
                    y="shot_rate",
                    size="corners_taken",
                    hover_name="team",
                    hover_data=["matches", "total_xg", "fast_shot_rate", "first_contact_rate", "box_delivery_rate"],
                )
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

    with top_tabs[1]:
        rank_tabs = st.tabs(["Volume", "Chance Creation", "Speed to Shot", "Delivery Profile"])

        with rank_tabs[0]:
            st.dataframe(
                league_team_df.sort_values(["corners_taken", "corners_per_match"], ascending=False).reset_index(drop=True),
                use_container_width=True,
                height=520,
            )

        with rank_tabs[1]:
            st.dataframe(
                league_team_df.sort_values(["total_xg", "xg_per_match", "avg_xg_per_corner"], ascending=False).reset_index(drop=True),
                use_container_width=True,
                height=520,
            )

        with rank_tabs[2]:
            st.dataframe(
                league_team_df.sort_values(["fast_shot_rate", "first_contact_rate", "shot_rate"], ascending=False).reset_index(drop=True),
                use_container_width=True,
                height=520,
            )

        with rank_tabs[3]:
            st.dataframe(
                league_team_df.sort_values(["box_delivery_rate", "inswingers", "outswingers"], ascending=False).reset_index(drop=True),
                use_container_width=True,
                height=520,
            )

    with top_tabs[2]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title">Match Total Corner Distribution</div>', unsafe_allow_html=True)
            if not league_match_df.empty:
                fig = px.histogram(
                    league_match_df,
                    x="total_corners",
                    nbins=min(20, max(5, len(league_match_df))),
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">League Corner Timing</div>', unsafe_allow_html=True)
            if not league_event_df.empty:
                timing_summary = (
                    league_event_df.groupby("phase", dropna=False)
                    .size()
                    .reset_index(name="corners")
                )
                phase_order = ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"]
                timing_summary["phase"] = pd.Categorical(timing_summary["phase"], categories=phase_order, ordered=True)
                timing_summary = timing_summary.sort_values("phase")
                fig = px.bar(timing_summary, x="phase", y="corners")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">League Match Board</div>', unsafe_allow_html=True)
        board_cols = [
            c for c in [
                "Match", "home_team", "away_team", "home_corners", "away_corners",
                "total_corners", "shots_from_corners", "fast_shots", "total_xg", "shot_rate"
            ] if c in league_match_df.columns
        ]
        st.dataframe(
            league_match_df[board_cols]
            .sort_values(["total_corners", "shots_from_corners"], ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            height=460,
        )

    with top_tabs[3]:
        st.markdown('<div class="section-title">League Shotmap</div>', unsafe_allow_html=True)
        shot_df = league_event_df.dropna(subset=["shot_location_x", "shot_location_y"]).copy()

        if st.button("Shotmap", key="league_shotmap_button"):
            fig = shotmap_figure(shot_df, color_col="corner_team", title="League Shotmap — All Corner Shots")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<div class="insight-box">Button added as requested. Bubble size scales with xG. Colors separate teams.</div>',
            unsafe_allow_html=True,
        )


# =========================================================
# PAGE: TEAM ANALYSIS
# =========================================================
elif page == "Team Analysis":
    team_tabs = st.tabs(["Snapshot", "Profile", "Personnel", "Shotmap"])

    with team_tabs[0]:
        st.subheader(f"Team Snapshot — {selected_team}")
        render_kpis(event_df, match_df)

        if selected_team == "All Teams":
            st.info("Select a specific team in the sidebar for deeper team benchmarking.")
        else:
            team_row = league_team_df[league_team_df["team"] == selected_team].copy()

            if not team_row.empty:
                team_row = team_row.iloc[0]
                vol_pct = percentile_rank(league_team_df["corners_per_match"], team_row["corners_per_match"])
                shot_pct = percentile_rank(league_team_df["shot_rate"], team_row["shot_rate"])
                xg_pct = percentile_rank(league_team_df["xg_per_match"], team_row["xg_per_match"])

                a, b, c = st.columns(3)
                with a:
                    metric_card("Volume Percentile", f"{vol_pct:.0f}", suffix="th")
                with b:
                    metric_card("Shot Rate Percentile", f"{shot_pct:.0f}", suffix="th")
                with c:
                    metric_card("xG per Match Percentile", f"{xg_pct:.0f}", suffix="th")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="section-title">Outcome Profile</div>', unsafe_allow_html=True)
                outcome_df = (
                    event_df.groupby("outcome_bucket", dropna=False)
                    .size()
                    .reset_index(name="events")
                    .sort_values("events", ascending=False)
                )
                fig = px.bar(outcome_df, x="outcome_bucket", y="events")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown('<div class="section-title">Corner Timing Profile</div>', unsafe_allow_html=True)
                timing_df = (
                    event_df.groupby("phase", dropna=False)
                    .size()
                    .reset_index(name="corners")
                )
                phase_order = ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"]
                timing_df["phase"] = pd.Categorical(timing_df["phase"], categories=phase_order, ordered=True)
                timing_df = timing_df.sort_values("phase")
                fig = px.bar(timing_df, x="phase", y="corners")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

    with team_tabs[1]:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="section-title">Volume vs Shot Conversion</div>', unsafe_allow_html=True)
            if not league_team_df.empty:
                fig = px.scatter(
                    league_team_df,
                    x="corners_per_match",
                    y="shot_rate",
                    size="corners_taken",
                    hover_name="team",
                    hover_data=["xg_per_match", "fast_shot_rate", "taker_variety", "box_delivery_rate"],
                )
                fig.update_layout(height=430)
                st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">Volume vs xG Output</div>', unsafe_allow_html=True)
            if not league_team_df.empty:
                fig = px.scatter(
                    league_team_df,
                    x="corners_per_match",
                    y="xg_per_match",
                    size="shots_from_corners",
                    hover_name="team",
                    hover_data=["shot_rate", "avg_xg_per_corner", "first_contact_rate", "box_delivery_rate"],
                )
                fig.update_layout(height=430)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">Team Table</div>', unsafe_allow_html=True)
        st.dataframe(
            team_df.sort_values(["corners_taken", "shots_from_corners", "total_xg"], ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=460,
        )

    with team_tabs[2]:
        st.markdown('<div class="section-title">Taker Leaderboard</div>', unsafe_allow_html=True)
        taker_table = (
            event_df.groupby("Taker", dropna=False)
            .agg(
                corners=("match_id", "size"),
                shots=("led_to_shot", "sum"),
                fast_shots=("is_fast_shot", "sum"),
                total_xg=("shot_xg", "sum"),
            )
            .reset_index()
        )
        taker_table["shot_rate"] = taker_table["shots"] / taker_table["corners"].replace(0, np.nan)
        taker_table = taker_table.sort_values(["corners", "shots"], ascending=False)

        st.dataframe(
            taker_table.reset_index(drop=True),
            use_container_width=True,
            height=500,
        )

        st.markdown('<div class="section-title">Shooter Leaderboard</div>', unsafe_allow_html=True)
        shooter_table = (
            event_df.groupby("Shooter", dropna=False)
            .agg(
                shots=("led_to_shot", "sum"),
                total_xg=("shot_xg", "sum"),
            )
            .reset_index()
            .sort_values(["shots", "total_xg"], ascending=False)
        )
        st.dataframe(
            shooter_table.reset_index(drop=True),
            use_container_width=True,
            height=380,
        )

    with team_tabs[3]:
        st.markdown('<div class="section-title">Team Shotmap</div>', unsafe_allow_html=True)
        shot_df = event_df.dropna(subset=["shot_location_x", "shot_location_y"]).copy()

        if st.button("Shotmap", key="team_shotmap_button"):
            fig = shotmap_figure(shot_df, color_col="Shooter", title=f"Shotmap — {selected_team}")
            st.plotly_chart(fig, use_container_width=True)

        if not shot_df.empty:
            st.dataframe(
                shot_df[["Match", "corner_team", "Shooter", "shot_xg", "shot_location_x", "shot_location_y", "Minute"]]
                .sort_values(["shot_xg", "Minute"], ascending=[False, True])
                .reset_index(drop=True),
                use_container_width=True,
                height=300,
            )


# =========================================================
# PAGE: MATCH EXPLORER
# =========================================================
elif page == "Match Explorer":
    available_matches = sorted(match_df["Match"].dropna().unique().tolist()) if not match_df.empty else []
    selected_match = st.selectbox("Select Match", ["All Matches"] + available_matches)

    match_event_df = event_df.copy()
    match_board_df = match_df.copy()

    if selected_match != "All Matches":
        match_board_df = match_board_df[match_board_df["Match"] == selected_match]
        selected_ids = match_board_df["match_id"].unique()
        match_event_df = match_event_df[match_event_df["match_id"].isin(selected_ids)]

    match_tabs = st.tabs(["Match Board", "Timeline", "Event Feed", "Shotmap"])

    with match_tabs[0]:
        render_kpis(match_event_df, match_board_df)

        board_cols = [
            c for c in [
                "Match", "home_team", "away_team", "home_corners", "away_corners",
                "total_corners", "shots_from_corners", "fast_shots", "total_xg", "shot_rate"
            ] if c in match_board_df.columns
        ]
        st.dataframe(
            match_board_df[board_cols]
            .sort_values(["total_corners", "shots_from_corners"], ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
            height=460,
        )

    with match_tabs[1]:
        st.markdown('<div class="section-title">Corner Timeline</div>', unsafe_allow_html=True)
        if not match_event_df.empty:
            minute_df = (
                match_event_df.groupby("Minute", dropna=False)
                .size()
                .reset_index(name="corner_events")
                .sort_values("Minute")
            )
            fig = px.bar(minute_df, x="Minute", y="corner_events")
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)

            by_team_df = (
                match_event_df.groupby(["Minute", "corner_team"], dropna=False)
                .size()
                .reset_index(name="events")
                .sort_values("Minute")
            )
            fig = px.line(by_team_df, x="Minute", y="events", color="corner_team", markers=True)
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)

    with match_tabs[2]:
        show_cols = [
            c for c in [
                "Match", "corner_team", "Taker", "Shooter", "Minute", "Second",
                "SP_outcome", "shot_xg", "Defensive_setup", "pass_technique", "pass_height"
            ] if c in match_event_df.columns
        ]
        st.dataframe(
            match_event_df[show_cols]
            .sort_values(["Minute", "Second"])
            .reset_index(drop=True),
            use_container_width=True,
            height=600,
        )

    with match_tabs[3]:
        st.markdown('<div class="section-title">Match Shotmap</div>', unsafe_allow_html=True)
        shot_df = match_event_df.dropna(subset=["shot_location_x", "shot_location_y"]).copy()

        if st.button("Shotmap", key="match_shotmap_button"):
            fig = shotmap_figure(shot_df, color_col="corner_team", title=f"Shotmap — {selected_match}")
            st.plotly_chart(fig, use_container_width=True)

        if not shot_df.empty:
            st.dataframe(
                shot_df[["Match", "corner_team", "Shooter", "shot_xg", "shot_location_x", "shot_location_y", "Minute", "SP_outcome"]]
                .sort_values(["Minute", "shot_xg"], ascending=[True, False])
                .reset_index(drop=True),
                use_container_width=True,
                height=300,
            )


# =========================================================
# PAGE: SET PIECE LAB
# =========================================================
elif page == "Set Piece Lab":
    lab_tabs = st.tabs(["Efficiency", "Timing", "Defensive Looks", "Delivery Lab", "Shotmap"])

    with lab_tabs[0]:
        st.markdown('<div class="section-title">League Efficiency Model</div>', unsafe_allow_html=True)
        efficiency_table = league_team_df.copy()
        efficiency_table["shot_rate_pct"] = efficiency_table["shot_rate"] * 100
        efficiency_table["fast_shot_rate_pct"] = efficiency_table["fast_shot_rate"] * 100

        col1, col2 = st.columns(2)

        with col1:
            fig = px.scatter(
                efficiency_table,
                x="corners_per_match",
                y="shot_rate_pct",
                size="total_xg",
                hover_name="team",
                hover_data=["xg_per_match", "fast_shot_rate_pct", "taker_variety", "box_delivery_rate"],
            )
            fig.update_layout(height=430, xaxis_title="Corners per Match", yaxis_title="Shot Rate %")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.scatter(
                efficiency_table,
                x="fast_shot_rate_pct",
                y="xg_per_match",
                size="shots_from_corners",
                hover_name="team",
                hover_data=["corners_per_match", "shot_rate_pct", "avg_xg_per_corner", "box_delivery_rate"],
            )
            fig.update_layout(height=430, xaxis_title="Fast Shot Rate %", yaxis_title="xG per Match")
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            efficiency_table.sort_values(["xg_per_match", "shot_rate"], ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=420,
        )

    with lab_tabs[1]:
        st.markdown('<div class="section-title">Time-Phase Analysis</div>', unsafe_allow_html=True)

        phase_team = (
            league_event_df.groupby(["corner_team", "phase"], dropna=False)
            .size()
            .reset_index(name="corners")
        )
        phase_order = ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"]
        phase_team["phase"] = pd.Categorical(phase_team["phase"], categories=phase_order, ordered=True)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                phase_team.sort_values(["phase", "corners"]),
                x="phase",
                y="corners",
                color="corner_team",
            )
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            phase_xg = (
                league_event_df.groupby("phase", dropna=False)
                .agg(total_xg=("shot_xg", "sum"), shots=("led_to_shot", "sum"))
                .reset_index()
            )
            phase_xg["phase"] = pd.Categorical(phase_xg["phase"], categories=phase_order, ordered=True)
            phase_xg = phase_xg.sort_values("phase")
            fig = px.bar(phase_xg, x="phase", y="total_xg", hover_data=["shots"])
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)

    with lab_tabs[2]:
        st.markdown('<div class="section-title">Defensive Setup Breakdown</div>', unsafe_allow_html=True)

        defensive_df = (
            league_event_df.groupby("Defensive_setup", dropna=False)
            .agg(
                corners=("match_id", "size"),
                shots=("led_to_shot", "sum"),
                total_xg=("shot_xg", "sum"),
            )
            .reset_index()
        )
        defensive_df["shot_rate"] = defensive_df["shots"] / defensive_df["corners"].replace(0, np.nan)
        defensive_df = defensive_df.sort_values("corners", ascending=False)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(defensive_df.head(15), x="Defensive_setup", y="corners", hover_data=["shots", "total_xg", "shot_rate"])
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(defensive_df.head(15), x="Defensive_setup", y="shot_rate", hover_data=["corners", "shots", "total_xg"])
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(defensive_df.reset_index(drop=True), use_container_width=True, height=420)

    with lab_tabs[3]:
        st.markdown('<div class="section-title">Delivery Lab</div>', unsafe_allow_html=True)

        delivery_df = (
            league_event_df.groupby("pass_technique", dropna=False)
            .agg(
                corners=("match_id", "size"),
                shots=("led_to_shot", "sum"),
                total_xg=("shot_xg", "sum"),
            )
            .reset_index()
        )
        delivery_df["shot_rate"] = delivery_df["shots"] / delivery_df["corners"].replace(0, np.nan)

        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(delivery_df.sort_values("corners", ascending=False), x="pass_technique", y="corners", hover_data=["shots", "shot_rate", "total_xg"])
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = px.bar(delivery_df.sort_values("shot_rate", ascending=False), x="pass_technique", y="shot_rate", hover_data=["corners", "shots", "total_xg"])
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

        zone_df = league_event_df.dropna(subset=["pass_end_location_x", "pass_end_location_y"]).copy()
        if not zone_df.empty:
            fig = px.density_heatmap(
                zone_df,
                x="pass_end_location_x",
                y="pass_end_location_y",
                nbinsx=15,
                nbinsy=12,
                title="Corner Delivery End Locations",
            )
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)

    with lab_tabs[4]:
        st.markdown('<div class="section-title">Set Piece Lab Shotmap</div>', unsafe_allow_html=True)
        shot_df = league_event_df.dropna(subset=["shot_location_x", "shot_location_y"]).copy()

        if st.button("Shotmap", key="lab_shotmap_button"):
            fig = shotmap_figure(shot_df, color_col="pass_technique", title="Shotmap by Delivery Type")
            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# PAGE: DATA CENTER
# =========================================================
elif page == "Data Center":
    data_tabs = st.tabs(["Raw Events", "Team Table", "Match Table", "Shot Events"])

    with data_tabs[0]:
        st.dataframe(event_df.reset_index(drop=True), use_container_width=True, height=620)

    with data_tabs[1]:
        st.dataframe(team_df.reset_index(drop=True), use_container_width=True, height=620)

    with data_tabs[2]:
        st.dataframe(match_df.reset_index(drop=True), use_container_width=True, height=620)

    with data_tabs[3]:
        shot_events = event_df[event_df["led_to_shot"]].copy()
        st.dataframe(
            shot_events.reset_index(drop=True),
            use_container_width=True,
            height=620,
        )
