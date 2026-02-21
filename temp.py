# -------------------- Imports & Config --------------------
import streamlit as st
import pandas as pd
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Set Piece Dashboard", layout="wide", page_icon="⚽")

# -------------------- Style (clean + professional) --------------------
fivethirtyeight_style = """
<style>
    :root {
        --fte-blue: #1a73e8;
        --fte-red: #ed3b3b;
        --fte-green: #4caf50;
        --fte-purple: #9c27b0;
        --fte-dark: #202020;
        --fte-light: #f8f8f8;
        --fte-gray: #757575;
        --border: #e6e6e6;
    }

    .main { 
        background-color: white; 
        font-family: 'Decima Mono', 'Helvetica Neue', Arial, sans-serif;
    }

    /* Make sidebar feel like a nav */
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--border);
    }

    h1, h2, h3, h4, h5, h6 { 
        color: var(--fte-dark);
        font-family: 'Decima Mono', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 800;
        letter-spacing: -0.3px;
    }

    /* Buttons */
    .stButton>button { 
        background-color: var(--fte-blue); 
        color: white; 
        border-radius: 8px;
        font-weight: 700;
        border: 0px;
        padding: 0.55rem 0.9rem;
    }
    .stButton>button:hover { 
        background-color: #0d5bba; 
        transform: none; 
        box-shadow: none;
    }

    /* Metrics */
    [data-testid="metric-container"] { 
        background-color: white; 
        border-radius: 12px; 
        padding: 16px; 
        border: 1px solid var(--border);
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }

    /* Dataframe */
    .stDataFrame { font-family: 'Decima Mono', 'Helvetica Neue', Arial, sans-serif; }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { 
        gap: 0px;
        border-bottom: 2px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] { 
        padding: 10px 18px; 
        background-color: transparent;
        border: none;
        font-weight: 700;
        color: var(--fte-gray);
        font-family: 'Decima Mono', 'Helvetica Neue', Arial, sans-serif;
    }
    .stTabs [aria-selected="true"] { 
        background-color: transparent;
        color: var(--fte-blue);
        border-bottom: 3px solid var(--fte-blue);
    }

    /* Leaderboard */
    .leaderboard-table { 
        width: 100%; 
        border-collapse: collapse;
        font-family: 'Decima Mono', 'Helvetica Neue', Arial, sans-serif;
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }
    .leaderboard-table th { 
        background-color: var(--fte-dark); 
        color: white; 
        padding: 12px; 
        text-align: left;
        font-weight: 800;
        font-size: 0.95rem;
    }
    .leaderboard-table td { 
        padding: 10px 12px; 
        border-bottom: 1px solid var(--border);
        font-size: 0.92rem;
    }
    .leaderboard-table tr:nth-child(even) { background-color: #fbfbfb; }

    .annotation {
        font-size: 0.86em;
        color: var(--fte-gray);
        font-style: italic;
        margin-top: 6px;
    }

    .footer {
        font-size: 0.85em;
        color: var(--fte-gray);
        border-top: 1px solid var(--border);
        padding-top: 14px;
        margin-top: 26px;
        font-family: 'Decima Mono', 'Helvetica Neue', Arial, sans-serif;
    }
</style>
"""
st.markdown(fivethirtyeight_style, unsafe_allow_html=True)

# -------------------- Helpers (fast parsing + caching) --------------------
BASE_PATH = os.path.dirname(__file__)

FILTER_COLS = [
    "competition.country_name",
    "competition.competition_name",
    "season.season_name",
]

@st.cache_data(ttl=3600)
def load_shots_data() -> pd.DataFrame:
    df = pd.read_excel(os.path.join(BASE_PATH, "db.xlsx"))

    # clean filter columns once
    for col in FILTER_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("").str.strip()

    # fast vectorized parse for location: "[x, y, z]" -> columns
    if "location" in df.columns:
        loc = df["location"].astype("string").fillna("").str.strip()
        parts = loc.str.strip("[]").str.split(",", expand=True)

        df["location_x"] = pd.to_numeric(parts[0], errors="coerce") if parts.shape[1] > 0 else np.nan
        df["location_y"] = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else np.nan
        df["location_z"] = pd.to_numeric(parts[2], errors="coerce") if parts.shape[1] > 2 else np.nan
    else:
        df["location_x"] = np.nan
        df["location_y"] = np.nan
        df["location_z"] = np.nan

    # dedupe and basic validity filters once
    subset_cols = [c for c in [
        "location_x", "location_y", "shot.statsbomb_xg",
        "team.name", "player.name", "Match", "shot.body_part.name"
    ] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)

    if "shot.statsbomb_xg" in df.columns:
        df["shot.statsbomb_xg"] = pd.to_numeric(df["shot.statsbomb_xg"], errors="coerce")

    df = df[df["location_x"].notna() & df["shot.statsbomb_xg"].notna()].copy()
    return df

@st.cache_data(ttl=3600)
def load_corner_data() -> pd.DataFrame:
    df = pd.read_excel(os.path.join(BASE_PATH, "corner_passes_and_shots_with_metadata.xlsx"))
    df.columns = df.columns.astype(str).str.strip()

    # Clean some common string columns (optional)
    for col in FILTER_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("").str.strip()

    # location_x/y from "location" (often list-like string)
    if "location" in df.columns:
        loc = df["location"].astype("string").fillna("").str.strip()
        parts = loc.str.strip("[]()").str.split(",", expand=True)
        df["location_x"] = pd.to_numeric(parts[0], errors="coerce") if parts.shape[1] > 0 else np.nan
        df["location_y"] = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else np.nan
    else:
        df["location_x"] = np.nan
        df["location_y"] = np.nan

    # pass_end_x/y from "pass.end_location" typically "x, y"
    if "pass.end_location" in df.columns:
        pel = df["pass.end_location"].astype("string").fillna("").str.split(",", expand=True)
        df["pass_end_x"] = pd.to_numeric(pel[0], errors="coerce") if pel.shape[1] > 0 else np.nan
        df["pass_end_y"] = pd.to_numeric(pel[1], errors="coerce") if pel.shape[1] > 1 else np.nan
    else:
        df["pass_end_x"] = np.nan
        df["pass_end_y"] = np.nan

    # Ensure xG numeric if present
    if "shot.statsbomb_xg" in df.columns:
        df["shot.statsbomb_xg"] = pd.to_numeric(df["shot.statsbomb_xg"], errors="coerce").fillna(0)

    return df

@st.cache_data(ttl=3600)
def build_corner_summary(df_corner: pd.DataFrame) -> pd.DataFrame:
    """
    Fast corner classification using possession group ops (avoids iterrows loop).
    Requires: event_type, possession, and stable ordering (index/event_id).
    """
    df = df_corner.copy()

    # determine sort column
    sort_col = "index" if "index" in df.columns else ("event_id" if "event_id" in df.columns else None)
    if sort_col:
        df = df.sort_values(["possession", sort_col]).reset_index(drop=True)
    else:
        df = df.sort_values(["possession"]).reset_index(drop=True)

    if "event_type" not in df.columns or "possession" not in df.columns:
        return pd.DataFrame()

    df["is_shot"] = df["event_type"].eq("Shot")
    df["xg"] = df.get("shot.statsbomb_xg", 0)
    pos_xg = df.groupby("possession")["xg"].sum()

    # next event inside possession
    df["next_event_type"] = df.groupby("possession")["event_type"].shift(-1)

    # shot within next 3 events (excluding current row)
    df["shot_within_3"] = (
        df.groupby("possession")["is_shot"]
          .transform(lambda s: s.shift(-1).rolling(3, min_periods=1).max())
          .fillna(False)
          .astype(bool)
    )

    corner_passes = df[df["event_type"].eq("CornerPass")].copy()
    if corner_passes.empty:
        return pd.DataFrame()

    # side detection from corner location_y
    # Your data uses y=0.1 for left and y=80 for right
    y = corner_passes["location_y"]
    corner_passes["side"] = np.select(
        [y.eq(0.1), y.eq(80)],
        ["Left", "Right"],
        default="Unknown"
    )

    corner_passes["classification"] = np.select(
        [
            corner_passes["next_event_type"].eq("Shot"),
            corner_passes["shot_within_3"].eq(True),
            corner_passes["possession"].map(pos_xg).gt(0),
        ],
        [
            "First contact - direct shot",
            "First contact - shot within 3 seconds",
            "No first contact - shot",
        ],
        default="First contact - no shot"
    )

    corner_passes["xG"] = corner_passes["possession"].map(pos_xg).fillna(0)

    # Build summary columns expected by your UI
    out = pd.DataFrame({
        "corner_index": corner_passes.index,  # row index after sorting
        "classification": corner_passes["classification"],
        "side": corner_passes["side"],
        "pass_height": corner_passes.get("pass.height.name", "Unknown"),
        "pass_body_part": corner_passes.get("pass.body_part.name", "Unknown"),
        "pass_outcome": corner_passes.get("pass.outcome.name", "Unknown"),
        "pass_technique": corner_passes.get("pass.technique.name", "Unknown"),
        "pass_end_x": corner_passes.get("pass_end_x", np.nan),
        "pass_end_y": corner_passes.get("pass_end_y", np.nan),
        "team.name": corner_passes.get("team.name", "Unknown"),
        "player.name": corner_passes.get("player.name", "Unknown"),
        "Match": corner_passes.get("Match", "Unknown"),
        "competition.competition_name": corner_passes.get("competition.competition_name", "Unknown"),
        "season.season_name": corner_passes.get("season.season_name", "Unknown"),
        "possession": corner_passes.get("possession", np.nan),
        "xG": corner_passes["xG"],
    })
    return out

def safe_unique(series: pd.Series):
    return sorted([x for x in series.dropna().astype(str).unique().tolist() if str(x).strip() != "" and str(x) != "nan"])

# -------------------- Sidebar Navigation --------------------
with st.sidebar:
    st.markdown("## ⚽ Set Piece Dashboard")
    st.caption("Fast, clean, and filterable set piece insights.")
    section = st.radio("Navigation", ["Shots Analysis", "Corner Routines", "Penalty Analysis"], index=0)

    st.markdown("---")
    with st.expander("App Controls", expanded=False):
        if st.button("Reset all inputs"):
            for k in list(st.session_state.keys()):
                if k not in ("section_nav",):
                    del st.session_state[k]
            st.rerun()

# -------------------- SHOTS ANALYSIS SECTION --------------------
def render_shots():
    df = load_shots_data()

    # Pre-filter goals in final third once
    df_goals = df[(df.get("shot.outcome.name") == "Goal") & (df["location_x"] >= 60)].copy()
    if df_goals.empty:
        st.warning("No goals found in the dataset with location_x >= 60.")
        return

    # Sidebar Filters (in a form to avoid constant reruns)
    with st.sidebar:
        with st.form("shots_filters"):
            st.markdown("### Filters")
            set_piece_vals = ["All"] + safe_unique(df_goals.get("play_pattern.name", pd.Series(dtype=str)))
            team_vals = ["All"] + safe_unique(df_goals.get("team.name", pd.Series(dtype=str)))
            pos_vals = ["All"] + safe_unique(df_goals.get("position.name", pd.Series(dtype=str)))
            nation_vals = ["All"] + safe_unique(df_goals.get("competition.country_name", pd.Series(dtype=str)))
            league_vals = ["All"] + safe_unique(df_goals.get("competition.competition_name", pd.Series(dtype=str)))
            season_vals = ["All"] + safe_unique(df_goals.get("season.season_name", pd.Series(dtype=str)))
            match_vals = ["All"] + safe_unique(df_goals.get("Match", pd.Series(dtype=str)))
            body_vals = ["All"] + safe_unique(df_goals.get("shot.body_part.name", pd.Series(dtype=str)))

            filters = {}
            filters["Set Piece Type"] = st.selectbox("Set Piece", set_piece_vals, key="shots_set_piece_filter")
            filters["Team"] = st.selectbox("Team", team_vals, key="shots_team_filter")
            filters["Position"] = st.selectbox("Position", pos_vals, key="shots_position_filter")
            filters["Nation"] = st.selectbox("Nation", nation_vals, key="shots_nation_filter")
            filters["League"] = st.selectbox("League", league_vals, key="shots_league_filter")
            filters["Season"] = st.selectbox("Season", season_vals, key="shots_season_filter")
            filters["Match"] = st.selectbox("Match", match_vals, key="shots_match_filter")
            filters["Body Part"] = st.selectbox("Body Part", body_vals, key="shots_body_part_filter")

            first_time = st.selectbox("First-Time Shot", ["All", "Yes", "No"], key="shots_first_time_filter")

            xg_min = float(np.nanmin(df_goals["shot.statsbomb_xg"].values))
            xg_max = float(np.nanmax(df_goals["shot.statsbomb_xg"].values))
            xg_range = st.slider("xG Range", xg_min, xg_max, (max(xg_min, 0.0), min(xg_max, 1.0)), 0.01, key="shots_xg_slider")

            apply_btn = st.form_submit_button("Apply")

    # Apply filters (cheap)
    filtered = df_goals.copy()
    mapping = [
        ("Set Piece Type", "play_pattern.name"),
        ("Team", "team.name"),
        ("Match", "Match"),
        ("Position", "position.name"),
        ("Body Part", "shot.body_part.name"),
        ("Nation", "competition.country_name"),
        ("League", "competition.competition_name"),
        ("Season", "season.season_name"),
    ]
    for key, col in mapping:
        if key in filters and filters[key] != "All" and col in filtered.columns:
            filtered = filtered[filtered[col] == filters[key]]

    if first_time != "All" and "shot.first_time" in filtered.columns:
        filtered = filtered[filtered["shot.first_time"] == (first_time == "Yes")]

    filtered = filtered[filtered["shot.statsbomb_xg"].between(*xg_range)]

    # UI
    st.title("Set Piece Goals Analysis")
    st.caption(
        f"Showing **{len(filtered)}** goals | xG range **{xg_range[0]:.2f}–{xg_range[1]:.2f}**"
        + (f" | Team: **{filters['Team']}**" if filters.get("Team") and filters["Team"] != "All" else "")
    )

    if filtered.empty:
        st.warning("No goals found matching these filters.")
        return

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Goals", int(len(filtered)))
    c2.metric("Avg. xG", f"{filtered['shot.statsbomb_xg'].mean():.3f}")
    c3.metric("Top Team", filtered["team.name"].mode().iat[0] if "team.name" in filtered.columns else "—")
    c4.metric("Most Common Type", filtered["play_pattern.name"].mode().iat[0] if "play_pattern.name" in filtered.columns else "—")

    tab0, tab1, tab4, tab_lb = st.tabs(["General Dashboard", "Goal Map", "Goal Placement", "Leaderboard"])

    # -------- General Dashboard --------
    with tab0:
        colA, colB = st.columns(2)

        with colA:
            team_counts = filtered["team.name"].value_counts().reset_index()
            team_counts.columns = ["Team", "Goals"]
            fig_team = px.bar(team_counts, x="Team", y="Goals", template="plotly_white")
            fig_team.update_layout(title="Goals by Team", height=420, margin=dict(t=50, b=20))
            fig_team.update_traces(marker_line_width=0)
            st.plotly_chart(fig_team, use_container_width=True)
            st.markdown('<div class="annotation">Number of set piece goals by team</div>', unsafe_allow_html=True)

        with colB:
            type_counts = filtered["play_pattern.name"].value_counts().reset_index()
            type_counts.columns = ["Set Piece Type", "Goals"]
            fig_type = px.bar(type_counts, x="Set Piece Type", y="Goals", template="plotly_white")
            fig_type.update_layout(title="Goals by Set Piece Type", height=420, margin=dict(t=50, b=20))
            fig_type.update_traces(marker_line_width=0)
            st.plotly_chart(fig_type, use_container_width=True)
            st.markdown('<div class="annotation">Distribution across set piece types</div>', unsafe_allow_html=True)

        with st.expander("Show filtered rows"):
            show_cols = [c for c in [
                "player.name", "team.name", "play_pattern.name", "shot.statsbomb_xg", "shot.body_part.name",
                "Match", "competition.competition_name", "season.season_name"
            ] if c in filtered.columns]
            st.dataframe(filtered[show_cols], use_container_width=True)

    # -------- Goal Map (attacking half) --------
    with tab1:
        st.markdown("### Goal Locations (Attacking Half)")

        filtered_half = filtered[filtered["location_x"] >= 60].copy()
        if filtered_half.empty:
            st.info("No goals in the attacking half for these filters.")
        else:
            # Convert StatsBomb pitch coordinates to a 80x60 attacking-half view (your existing mapping)
            filtered_half["plot_x"] = filtered_half["location_y"]
            filtered_half["plot_y"] = 120 - filtered_half["location_x"]

            hover_text = (
                "Player: " + filtered_half.get("player.name", "").astype(str) +
                "<br>Team: " + filtered_half.get("team.name", "").astype(str) +
                "<br>xG: " + filtered_half["shot.statsbomb_xg"].round(2).astype(str) +
                "<br>Body Part: " + filtered_half.get("shot.body_part.name", "").astype(str) +
                "<br>Match: " + filtered_half.get("Match", "").astype(str)
            )

            fig = go.Figure()

            # Pitch (80 x 60)
            fig.update_layout(
                xaxis=dict(range=[0, 80], showgrid=False, zeroline=False, visible=False, scaleanchor="y"),
                yaxis=dict(range=[0, 60], showgrid=False, zeroline=False, visible=False),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=700,
                margin=dict(t=50, b=20),
                title="Goal Locations from Set Pieces",
                shapes=[
                    dict(type="rect", x0=0, y0=0, x1=80, y1=60, line=dict(color="black", width=2)),
                    dict(type="rect", x0=18, y0=0, x1=62, y1=18, line=dict(color="black", width=2)),
                    dict(type="rect", x0=30, y0=0, x1=50, y1=6, line=dict(color="black", width=2)),
                    dict(type="line", x0=30, y0=0, x1=50, y1=0, line=dict(color="black", width=4)),
                    dict(type="circle", x0=38, y0=7, x1=40, y1=9, line=dict(color="black", width=2)),
                    dict(type="path", path="M 18 0 A 20 22 0 0 1 62 0", line=dict(color="black", width=2)),
                    dict(type="line", x0=0, y0=60, x1=80, y1=60, line=dict(color="black", width=2)),
                    dict(type="path", path="M 30 60 A 20 20 0 0 1 50 60", line=dict(color="black", width=2)),
                ],
            )

            # WebGL scatter for speed
            fig.add_trace(go.Scattergl(
                x=filtered_half["plot_x"],
                y=filtered_half["plot_y"],
                mode="markers",
                marker=dict(
                    size=filtered_half["shot.statsbomb_xg"] * 40 + 6,
                    color=filtered_half["shot.statsbomb_xg"],
                    colorscale="Bluered",
                    colorbar=dict(title="xG"),
                    line=dict(width=0.5, color="black"),
                    opacity=0.85
                ),
                text=hover_text,
                hoverinfo="text",
                name="Goals"
            ))

            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<div class="annotation">Marker size & color represent xG</div>', unsafe_allow_html=True)

            players = safe_unique(filtered_half["player.name"])
            if players:
                selected_player = st.selectbox("Select Player", players, key="shots_player_selector")
                cols = [c for c in [
                    "player.name", "team.name", "shot.statsbomb_xg", "shot.body_part.name", "Match", "competition.competition_name"
                ] if c in filtered_half.columns]
                st.dataframe(filtered_half[filtered_half["player.name"] == selected_player][cols], use_container_width=True)

    # -------- Goal Placement (end_location) --------
    with tab4:
        st.markdown("### Goal Placement from shot.end_location")

        if "shot.end_location" not in filtered.columns:
            st.info("shot.end_location column not found.")
        else:
            # Expecting "x, y, z"
            endloc = filtered["shot.end_location"].astype("string").fillna("").str.split(",", expand=True)
            filtered["shot.end_location_x"] = pd.to_numeric(endloc[0], errors="coerce") if endloc.shape[1] > 0 else np.nan
            filtered["shot.end_location_y"] = pd.to_numeric(endloc[1], errors="coerce") if endloc.shape[1] > 1 else np.nan
            filtered["shot.end_location_z"] = pd.to_numeric(endloc[2], errors="coerce") if endloc.shape[1] > 2 else np.nan

            GOAL_WIDTH = 7.32
            GOAL_HEIGHT = 2.44
            LEFT_POST_Y = 36.8
            RIGHT_POST_Y = 43.2

            goals = filtered.dropna(subset=["shot.end_location_y"]).copy()
            goals = goals[(goals["shot.end_location_y"] >= LEFT_POST_Y) & (goals["shot.end_location_y"] <= RIGHT_POST_Y)]

            if goals.empty:
                st.info("No goals with shot.end_location_y inside goalposts found.")
            else:
                goals["goal_x_m"] = (goals["shot.end_location_y"] - LEFT_POST_Y) * (GOAL_WIDTH / (RIGHT_POST_Y - LEFT_POST_Y))
                goals["goal_z_m"] = goals["shot.end_location_z"].fillna(0)

                xg = goals["shot.statsbomb_xg"].fillna(0)
                marker_size = np.interp(xg, (xg.min(), xg.max()), (6, 20)) if len(goals) > 1 else np.full(len(goals), 12)

                fig = go.Figure()
                fig.add_shape(type="rect", x0=0, y0=0, x1=GOAL_WIDTH, y1=GOAL_HEIGHT, line=dict(color="black", width=3))
                fig.add_shape(type="line", x0=0, y0=GOAL_HEIGHT/2, x1=GOAL_WIDTH, y1=GOAL_HEIGHT/2, line=dict(color="#757575", dash="dash"))
                fig.add_shape(type="line", x0=GOAL_WIDTH/3, y0=0, x1=GOAL_WIDTH/3, y1=GOAL_HEIGHT, line=dict(color="#757575", dash="dash"))
                fig.add_shape(type="line", x0=2*GOAL_WIDTH/3, y0=0, x1=2*GOAL_WIDTH/3, y1=GOAL_HEIGHT, line=dict(color="#757575", dash="dash"))

                fig.add_trace(go.Scattergl(
                    x=goals["goal_x_m"],
                    y=goals["goal_z_m"],
                    mode="markers",
                    marker=dict(
                        size=marker_size,
                        color=goals["shot.statsbomb_xg"],
                        colorscale="Bluered",
                        showscale=True,
                        colorbar=dict(title="xG"),
                        line=dict(width=0.5, color="black"),
                        opacity=0.85
                    ),
                    text=goals.get("player.name", "").astype(str),
                    hovertemplate="Player: %{text}<br>Width: %{x:.2f} m<br>Height: %{y:.2f} m<br>xG: %{marker.color:.3f}<extra></extra>",
                    name="Goals"
                ))

                fig.update_layout(
                    title="Goal Placement (Size & Color by xG)",
                    xaxis=dict(title="Goal Width (m)", range=[0, GOAL_WIDTH], showgrid=False, zeroline=False),
                    yaxis=dict(title="Goal Height (m)", range=[0, GOAL_HEIGHT], showgrid=False, zeroline=False),
                    height=600,
                    margin=dict(t=50, b=20),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    yaxis_scaleanchor="x"
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('<div class="annotation">Goal mouth split into 6 zones</div>', unsafe_allow_html=True)

                with st.expander("Show goal placement data"):
                    cols = [c for c in [
                        "player.name", "team.name", "shot.end_location",
                        "shot.end_location_x", "shot.end_location_y", "shot.end_location_z",
                        "shot.statsbomb_xg"
                    ] if c in goals.columns]
                    st.dataframe(goals[cols], use_container_width=True)

    # -------- Leaderboard --------
    with tab_lb:
        st.markdown("### Performance Leaderboard")

        metric = st.selectbox(
            "Rank players by:",
            ["Total Goals", "Total xG", "Average xG per Goal", "Goals per Match"],
            key="shots_leaderboard_metric"
        )

        grp = filtered.groupby(["player.name", "team.name"], dropna=False).agg(
            Total_Goals=("player.name", "count"),
            Total_xG=("shot.statsbomb_xg", "sum"),
            Matches=("Match", "nunique")
        ).reset_index()

        grp["Avg_xG_per_Goal"] = grp["Total_xG"] / grp["Total_Goals"]
        grp["Goals_per_Match"] = grp["Total_Goals"] / grp["Matches"].replace(0, np.nan)

        if metric == "Total Goals":
            grp = grp.sort_values("Total_Goals", ascending=False)
            metric_col = "Total_Goals"
        elif metric == "Total xG":
            grp = grp.sort_values("Total_xG", ascending=False)
            metric_col = "Total_xG"
        elif metric == "Average xG per Goal":
            grp = grp.sort_values("Avg_xG_per_Goal", ascending=False)
            metric_col = "Avg_xG_per_Goal"
        else:
            grp = grp.sort_values("Goals_per_Match", ascending=False)
            metric_col = "Goals_per_Match"

        grp[metric_col] = grp[metric_col].round(3)

        st.markdown(f"#### Top 20 Players by {metric}")

        st.write(
            f"""
            <table class="leaderboard-table">
                <thead>
                    <tr>
                        <th>Rank</th>
                        <th>Player</th>
                        <th>Team</th>
                        <th>{metric}</th>
                        <th>Total Goals</th>
                        <th>Total xG</th>
                        <th>Matches</th>
                    </tr>
                </thead>
                <tbody>
            """,
            unsafe_allow_html=True
        )

        top20 = grp.head(20).reset_index(drop=True)
        for i, row in top20.iterrows():
            st.write(
                f"""
                <tr>
                    <td>{i+1}</td>
                    <td>{row['player.name']}</td>
                    <td>{row['team.name']}</td>
                    <td><strong>{row[metric_col]}</strong></td>
                    <td>{int(row['Total_Goals'])}</td>
                    <td>{row['Total_xG']:.2f}</td>
                    <td>{int(row['Matches'])}</td>
                </tr>
                """,
                unsafe_allow_html=True
            )

        st.write("</tbody></table>", unsafe_allow_html=True)
        st.markdown('<div class="annotation">Set piece specialists ranked by selected metric</div>', unsafe_allow_html=True)

        with st.expander("Show full leaderboard data"):
            st.dataframe(grp, use_container_width=True)

    st.markdown('<div class="footer">© Football Analytics Team</div>', unsafe_allow_html=True)

# -------------------- CORNER ROUTINES SECTION --------------------
def render_corners():
    df_corner = load_corner_data()
    if df_corner.empty:
        st.error("No data loaded for corner routines analysis.")
        return
    if "event_type" not in df_corner.columns or "possession" not in df_corner.columns:
        st.error("Corner file must include 'event_type' and 'possession' columns.")
        return

    corner_summary = build_corner_summary(df_corner)
    if corner_summary.empty:
        st.info("No corner passes found in the data (event_type == 'CornerPass').")
        return

    st.title("Corner Kick Analysis")
    st.caption("Corner outcomes + end locations, optimized for speed.")

    # Sidebar Filters in a form
    with st.sidebar:
        with st.form("corner_filters"):
            st.markdown("### Corner Filters")

            team_vals = ["All"] + safe_unique(corner_summary["team.name"])
            technique_vals = ["All"] + safe_unique(corner_summary["pass_technique"])
            side_vals = ["All", "Left", "Right", "Unknown"]
            height_vals = ["All"] + safe_unique(corner_summary["pass_height"])
            body_vals = ["All"] + safe_unique(corner_summary["pass_body_part"])
            outcome_vals = ["All"] + safe_unique(corner_summary["pass_outcome"])
            class_vals = ["All"] + safe_unique(corner_summary["classification"])
            league_vals = ["All"] + safe_unique(corner_summary["competition.competition_name"])

            f_team = st.selectbox("Team", team_vals, key="corner_team_filter")
            f_tech = st.selectbox("Corner Technique", technique_vals, key="corner_technique_filter")
            f_side = st.selectbox("Corner Side", side_vals, key="corner_side_filter")
            f_height = st.selectbox("Pass Height", height_vals, key="pass_height_filter")
            f_body = st.selectbox("Pass Body Part", body_vals, key="pass_body_part_filter")
            f_outcome = st.selectbox("Pass Outcome", outcome_vals, key="pass_outcome_filter")
            f_class = st.selectbox("Outcome Classification", class_vals, key="classification_filter")
            f_league = st.selectbox("League", league_vals, key="corner_league_filter")

            apply_btn = st.form_submit_button("Apply")

    filtered_corners = corner_summary.copy()

    if f_team != "All":
        filtered_corners = filtered_corners[filtered_corners["team.name"] == f_team]
    if f_tech != "All":
        filtered_corners = filtered_corners[filtered_corners["pass_technique"] == f_tech]
    if f_side != "All":
        filtered_corners = filtered_corners[filtered_corners["side"] == f_side]
    if f_height != "All":
        filtered_corners = filtered_corners[filtered_corners["pass_height"] == f_height]
    if f_body != "All":
        filtered_corners = filtered_corners[filtered_corners["pass_body_part"] == f_body]
    if f_outcome != "All":
        filtered_corners = filtered_corners[filtered_corners["pass_outcome"] == f_outcome]
    if f_class != "All":
        filtered_corners = filtered_corners[filtered_corners["classification"] == f_class]
    if f_league != "All":
        filtered_corners = filtered_corners[filtered_corners["competition.competition_name"] == f_league]

    if filtered_corners.empty:
        st.info("No corners found for the selected filters.")
        return

    # Metrics based on unique possession
    unique_pos = filtered_corners["possession"].dropna().unique().tolist()
    total_corners = len(filtered_corners)
    total_xg = float(filtered_corners.drop_duplicates("possession")["xG"].sum()) if unique_pos else 0.0

    # Shot counts from underlying df_corner within those possessions
    df_shots = df_corner[df_corner["possession"].isin(unique_pos) & df_corner["event_type"].eq("Shot")]
    total_shots = int(len(df_shots))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Corners", total_corners)
    c2.metric("Total Shots from Corners", total_shots)
    c3.metric("Total xG Generated", f"{total_xg:.2f}")
    c4.metric("Avg xG per Shot", f"{(total_xg / total_shots):.3f}" if total_shots > 0 else "N/A")

    # Plot: Corner pass end locations (vertical attacking half)
    valid_locations = filtered_corners.dropna(subset=["pass_end_x", "pass_end_y"])
    if valid_locations.empty:
        st.info("No valid location data found for corner passes.")
    else:
        color_map = {
            "First contact - direct shot": "red",
            "First contact - shot within 3 seconds": "blue",
            "No first contact - shot": "green",
            "First contact - no shot": "orange",
            "No first contact - no shot": "gray"
        }

        pitch_length = 120
        pitch_width = 80

        fig = go.Figure()
        fig.update_layout(
            title="Corner Pass End Locations (Attacking Half - Vertical Pitch)",
            showlegend=True,
            height=980,
            margin=dict(t=60, b=20),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-5, pitch_width + 5]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[55, pitch_length + 5]),
            plot_bgcolor="white",
            paper_bgcolor="white",
            shapes=[
                dict(type="rect", x0=0, y0=60, x1=pitch_width, y1=pitch_length, line=dict(color="black", width=2)),
                dict(type="rect", x0=18, y0=102, x1=pitch_width - 18, y1=pitch_length, line=dict(color="black", width=2)),
                dict(type="rect", x0=(pitch_width/2)-9, y0=114, x1=(pitch_width/2)+9, y1=pitch_length, line=dict(color="black", width=2)),
                dict(type="line", x0=(pitch_width/2)-3.66, y0=pitch_length, x1=(pitch_width/2)+3.66, y1=pitch_length, line=dict(color="black", width=4)),
                dict(type="circle", x0=(pitch_width/2)-0.5, y0=108-0.5, x1=(pitch_width/2)+0.5, y1=108+0.5, line=dict(color="black", width=2)),
                dict(type="path",
                    path=f"M {pitch_width/2 - 10},{102} A 10,10 0 0,1 {pitch_width/2 + 10},{102}",
                    line=dict(color="black", width=2)),
            ]
        )

        for classification, df_group in valid_locations.groupby("classification"):
            fig.add_trace(go.Scattergl(
                x=df_group["pass_end_y"],   # swap axes for vertical view
                y=df_group["pass_end_x"],
                mode="markers",
                name=classification,
                marker=dict(size=10, color=color_map.get(classification, "gray"), opacity=0.85, line=dict(width=1, color="black")),
                hovertemplate=(
                    "Team: %{customdata[0]}<br>"
                    "Player: %{customdata[1]}<br>"
                    "Classification: %{customdata[2]}<br>"
                    "xG (possession): %{customdata[3]:.2f}<extra></extra>"
                ),
                customdata=df_group[["team.name", "player.name", "classification", "xG"]].values
            ))

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="annotation">End locations of corner deliveries, grouped by outcome</div>', unsafe_allow_html=True)

    # Download
    st.download_button(
        "Download Filtered Data as CSV",
        data=filtered_corners.to_csv(index=False),
        file_name="filtered_corner_passes.csv",
        key="download_corners"
    )

    with st.expander("Show filtered corner summary"):
        st.dataframe(filtered_corners, use_container_width=True)

    st.markdown('<div class="footer">© Football Analytics Team</div>', unsafe_allow_html=True)

# -------------------- PENALTY ANALYSIS SECTION (placeholder) --------------------
def render_penalties():
    st.title("Penalty Analysis")
    st.info("Penalty Analysis section is not implemented in your provided code.")
    st.markdown('<div class="footer">© Football Analytics Team</div>', unsafe_allow_html=True)

# -------------------- Router --------------------
if section == "Shots Analysis":
    render_shots()
elif section == "Corner Routines":
    render_corners()
else:
    render_penalties()
