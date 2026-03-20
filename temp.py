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
LOGIN_NAME = "Admin"
LOGIN_PASSWORD = "Football2026"


# =========================================================
# LOGIN
# =========================================================
def login_screen():
    st.title("⚽ Set Piece Studio")
    st.subheader("Login")

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Name")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

        if submitted:
            if username == LOGIN_NAME and password == LOGIN_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Invalid name or password.")


if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    login_screen()
    st.stop()


# =========================================================
# STYLING
# =========================================================
st.markdown(
    """
<style>
.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    max-width: 1600px;
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
""",
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================
def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def pct(numerator, denominator):
    if denominator in [0, None] or pd.isna(denominator):
        return np.nan
    return numerator / denominator


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


def left_right_from_y(y):
    if pd.isna(y):
        return "Unknown"
    if y < 30:
        return "Near Post Zone"
    if y <= 50:
        return "Central Zone"
    return "Far Post Zone"


def corner_side_from_start_y(y):
    if pd.isna(y):
        return "Unknown"
    return "Right Corner" if y < 40 else "Left Corner"


def delivery_length(x_start, y_start, x_end, y_end):
    if pd.isna(x_start) or pd.isna(y_start) or pd.isna(x_end) or pd.isna(y_end):
        return np.nan
    return float(np.sqrt((x_end - x_start) ** 2 + (y_end - y_start) ** 2))


def zone_from_end_location(x, y):
    if pd.isna(x) or pd.isna(y):
        return "Unknown"
    if x >= 114 and 30 <= y <= 50:
        return "6-yard box"
    if x >= 108 and 18 <= y <= 62:
        return "Penalty area"
    if x >= 100 and 18 <= y <= 62:
        return "Deep box"
    return "Outside danger zone"


def render_kpis(events, matches):
    c1, c2, c3, c4, c5, c6 = st.columns(6)

    with c1:
        metric_card("Corner Events", f"{len(events):,}")
    with c2:
        metric_card("Matches", f"{events['match_id'].nunique() if not events.empty else 0:,}")
    with c3:
        metric_card("Avg Corners / Match", f"{matches['total_corners'].mean() if not matches.empty else 0:.2f}")
    with c4:
        metric_card("Shot Outcomes", f"{int(events['led_to_shot'].sum()) if not events.empty else 0:,}")
    with c5:
        metric_card("Total xG", f"{events['shot_xg'].fillna(0).sum() if not events.empty else 0:.2f}")
    with c6:
        metric_card("Shot Rate", f"{(events['led_to_shot'].mean() * 100) if len(events) > 0 else 0:.1f}", "%")


def draw_pitch(fig, title=None, height=560):
    line_color = "rgba(255,255,255,0.60)"
    pitch_color = "rgba(18,60,34,1)"

    fig.update_xaxes(range=[0, 120], visible=False)
    fig.update_yaxes(range=[0, 80], visible=False, scaleanchor="x", scaleratio=1)

    fig.update_layout(
        title=title,
        paper_bgcolor=pitch_color,
        plot_bgcolor=pitch_color,
        margin=dict(l=10, r=10, t=45 if title else 10, b=10),
        height=height,
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
        legend_title_text="",
    )
    return fig


def shotmap_figure(df_shots, color_col="corner_team", title="Shotmap"):
    fig = draw_pitch(go.Figure(), title=title)

    if df_shots.empty:
        return fig

    plot_df = df_shots.copy()
    plot_df["shot_xg_plot"] = plot_df["shot_xg"].fillna(0)
    plot_df["marker_size"] = np.clip(plot_df["shot_xg_plot"] * 95 + 10, 10, 36)

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
                    opacity=0.86,
                    line=dict(color="white", width=1),
                ),
                text=[
                    f"{row['Match']}<br>"
                    f"Team: {row['corner_team']}<br>"
                    f"Taker: {row['Taker']}<br>"
                    f"Shooter: {row['Shooter']}<br>"
                    f"Outcome: {row['SP_outcome']}<br>"
                    f"xG: {0 if pd.isna(row['shot_xg']) else row['shot_xg']:.3f}<br>"
                    f"Minute: {int(row['Minute']) if pd.notna(row['Minute']) else ''}"
                    for _, row in sub.iterrows()
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )

    return fig


def delivery_map_figure(df_events, color_col="delivery_zone", title="Delivery Map"):
    fig = draw_pitch(go.Figure(), title=title, height=620)

    if df_events.empty:
        return fig

    plot_df = df_events.dropna(subset=["pass_location_x", "pass_location_y", "pass_end_location_x", "pass_end_location_y"]).copy()
    if plot_df.empty:
        return fig

    categories = plot_df[color_col].fillna("Unknown").astype(str).unique().tolist()
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.Set2 + px.colors.qualitative.Safe
    color_map = {cat: palette[i % len(palette)] for i, cat in enumerate(categories)}

    for cat in categories:
        sub = plot_df[plot_df[color_col].fillna("Unknown").astype(str) == cat]
        for _, row in sub.iterrows():
            fig.add_trace(
                go.Scatter(
                    x=[row["pass_location_x"], row["pass_end_location_x"]],
                    y=[row["pass_location_y"], row["pass_end_location_y"]],
                    mode="lines+markers",
                    line=dict(color=color_map[cat], width=2),
                    marker=dict(size=[6, 8], color=[color_map[cat], color_map[cat]]),
                    name=str(cat),
                    legendgroup=str(cat),
                    showlegend=False,
                    text=(
                        f"{row['Match']}<br>"
                        f"Team: {row['corner_team']}<br>"
                        f"Taker: {row['Taker']}<br>"
                        f"Technique: {row['pass_technique']}<br>"
                        f"Delivery Zone: {row['delivery_zone']}<br>"
                        f"End Zone: {row['end_zone']}<br>"
                        f"Outcome: {row['SP_outcome']}<br>"
                        f"Minute: {int(row['Minute']) if pd.notna(row['Minute']) else ''}"
                    ),
                    hovertemplate="%{text}<extra></extra>",
                )
            )

    for cat in categories:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="lines",
                line=dict(color=color_map[cat], width=3),
                name=str(cat),
                legendgroup=str(cat),
                showlegend=True,
            )
        )

    return fig


def delivery_end_heatmap(df_events, title="Delivery End Locations"):
    plot_df = df_events.dropna(subset=["pass_end_location_x", "pass_end_location_y"]).copy()
    if plot_df.empty:
        return go.Figure()

    fig = px.density_heatmap(
        plot_df,
        x="pass_end_location_x",
        y="pass_end_location_y",
        nbinsx=18,
        nbinsy=14,
        title=title,
        hover_data=["corner_team", "Taker", "pass_technique", "delivery_zone", "end_zone"],
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, range=[0, 80])
    fig.update_xaxes(range=[0, 120])
    fig.update_layout(height=520)
    return fig


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
            short_corners=("is_short_corner", "sum"),
            target_box_deliveries=("is_goal_kick_zone_delivery", "sum"),
            six_yard_deliveries=("is_six_yard_delivery", "sum"),
            penalty_area_deliveries=("is_penalty_area_delivery", "sum"),
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
    ts["six_yard_delivery_rate"] = ts["six_yard_deliveries"] / ts["corners_taken"].replace(0, np.nan)
    ts["penalty_area_delivery_rate"] = ts["penalty_area_deliveries"] / ts["corners_taken"].replace(0, np.nan)
    ts["short_corner_rate"] = ts["short_corners"] / ts["corners_taken"].replace(0, np.nan)
    return ts


# =========================================================
# DATA LOAD
# =========================================================
@st.cache_data
def load_data():
    if not os.path.exists(FILE_NAME):
        raise FileNotFoundError(f"{FILE_NAME} not found. Put it in the same folder as this app.")
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
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

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
    df["is_short_corner"] = df["pass_technique"].astype(str).str.contains("short", case=False, na=False)

    df["delivery_zone"] = df["pass_end_location_y"].apply(left_right_from_y)
    df["corner_side"] = df["pass_location_y"].apply(corner_side_from_start_y)
    df["delivery_length"] = df.apply(
        lambda r: delivery_length(r["pass_location_x"], r["pass_location_y"], r["pass_end_location_x"], r["pass_end_location_y"]),
        axis=1,
    )
    df["end_zone"] = df.apply(lambda r: zone_from_end_location(r["pass_end_location_x"], r["pass_end_location_y"]), axis=1)

    df["is_goal_kick_zone_delivery"] = (
        (df["pass_end_location_x"].between(114, 120, inclusive="both")) &
        (df["pass_end_location_y"].between(30, 50, inclusive="both"))
    )
    df["is_six_yard_delivery"] = df["end_zone"].eq("6-yard box")
    df["is_penalty_area_delivery"] = df["end_zone"].eq("Penalty area")

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
            short_corners=("is_short_corner", "sum"),
            six_yard_deliveries=("is_six_yard_delivery", "sum"),
            penalty_area_deliveries=("is_penalty_area_delivery", "sum"),
        )
        .reset_index()
    )

    def count_team_corners(match_id, team_name):
        if pd.isna(team_name):
            return np.nan
        return int(((df["match_id"] == match_id) & (df["corner_team"] == team_name)).sum())

    match_summary["home_corners"] = match_summary.apply(lambda r: count_team_corners(r["match_id"], r["home_team"]), axis=1)
    match_summary["away_corners"] = match_summary.apply(lambda r: count_team_corners(r["match_id"], r["away_team"]), axis=1)
    match_summary["shot_rate"] = match_summary["shots_from_corners"] / match_summary["total_corners"].replace(0, np.nan)
    match_summary["xg_per_corner"] = match_summary["total_xg"] / match_summary["total_corners"].replace(0, np.nan)

    team_summary = build_team_summary(df)

    return df, match_summary, team_summary


# =========================================================
# LOAD
# =========================================================
st.title("⚽ Allsvenskan Set Piece Studio")
st.markdown(
    '<span class="badge">League Dashboard</span>'
    '<span class="badge">Team Intelligence</span>'
    '<span class="badge">Shotmap</span>'
    '<span class="badge">Delivery Map</span>'
    '<span class="badge">Set Piece Lab</span>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="subtle">Deep corner analysis with team filters, shotmaps, delivery maps, zone profiling, taker intelligence, and match-level breakdowns.</div>',
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
st.sidebar.title("Filters")
page = st.sidebar.radio(
    "Select Page",
    ["League Overview", "Team Analysis", "Match Explorer", "Set Piece Lab", "Data Center"],
)

all_teams = sorted([t for t in team_summary["team"].dropna().unique().tolist() if str(t).strip()])
selected_team = st.sidebar.selectbox("Team", ["All Teams"] + all_teams)

all_takers = sorted([str(t) for t in df["Taker"].dropna().astype(str).unique().tolist() if str(t).strip()])
selected_takers = st.sidebar.multiselect("Taker", all_takers)

all_matches = sorted(df["Match"].dropna().astype(str).unique().tolist())
selected_matches = st.sidebar.multiselect("Matches", all_matches)

minute_min = int(df["Minute"].min()) if not df["Minute"].dropna().empty else 0
minute_max = int(df["Minute"].max()) if not df["Minute"].dropna().empty else 120

if minute_max <= minute_min:
    minute_range = (minute_min, minute_max)
    st.sidebar.caption(f"Minute Range: {minute_min}")
else:
    minute_range = st.sidebar.slider(
        "Minute Range",
        min_value=minute_min,
        max_value=minute_max,
        value=(minute_min, minute_max),
    )

if len(match_summary) > 0:
    min_corners = int(match_summary["total_corners"].min())
    max_corners = int(match_summary["total_corners"].max())
else:
    min_corners, max_corners = 0, 0

if max_corners <= min_corners:
    corner_range = (min_corners, max_corners)
    st.sidebar.caption(f"Match Corner Range: {min_corners}")
else:
    corner_range = st.sidebar.slider(
        "Match Corner Range",
        min_value=min_corners,
        max_value=max_corners,
        value=(min_corners, max_corners),
    )

show_shot_only = st.sidebar.checkbox("Shot outcomes only", value=False)
show_inswing_only = st.sidebar.checkbox("Inswingers only", value=False)
show_outswing_only = st.sidebar.checkbox("Outswingers only", value=False)
show_short_only = st.sidebar.checkbox("Short corners only", value=False)

all_delivery_zones = [z for z in ["Near Post Zone", "Central Zone", "Far Post Zone", "Unknown"] if z in df["delivery_zone"].astype(str).unique().tolist()]
selected_delivery_zones = st.sidebar.multiselect("Delivery Zone", all_delivery_zones)

all_end_zones = [z for z in ["6-yard box", "Penalty area", "Deep box", "Outside danger zone", "Unknown"] if z in df["end_zone"].astype(str).unique().tolist()]
selected_end_zones = st.sidebar.multiselect("End Zone", all_end_zones)

all_setups = sorted([str(x) for x in df["Defensive_setup"].dropna().astype(str).unique().tolist() if str(x).strip()])
selected_setups = st.sidebar.multiselect("Defensive Setup", all_setups)

st.sidebar.markdown("---")
st.sidebar.caption("Style: analyst desk / broadcast dashboard")


# =========================================================
# GLOBAL FILTERS
# =========================================================
league_match_df = match_summary[
    (match_summary["total_corners"] >= corner_range[0]) &
    (match_summary["total_corners"] <= corner_range[1])
].copy()

league_event_df = df[df["match_id"].isin(league_match_df["match_id"].unique())].copy()
league_event_df = league_event_df[
    (league_event_df["Minute"].fillna(0) >= minute_range[0]) &
    (league_event_df["Minute"].fillna(0) <= minute_range[1])
]

if selected_team != "All Teams":
    league_event_df = league_event_df[league_event_df["corner_team"] == selected_team]
if selected_takers:
    league_event_df = league_event_df[league_event_df["Taker"].astype(str).isin(selected_takers)]
if selected_matches:
    league_event_df = league_event_df[league_event_df["Match"].astype(str).isin(selected_matches)]
if show_shot_only:
    league_event_df = league_event_df[league_event_df["led_to_shot"]]
if show_inswing_only and not show_outswing_only:
    league_event_df = league_event_df[league_event_df["is_inswinger"]]
if show_outswing_only and not show_inswing_only:
    league_event_df = league_event_df[league_event_df["is_outswinger"]]
if show_short_only:
    league_event_df = league_event_df[league_event_df["is_short_corner"]]
if selected_delivery_zones:
    league_event_df = league_event_df[league_event_df["delivery_zone"].isin(selected_delivery_zones)]
if selected_end_zones:
    league_event_df = league_event_df[league_event_df["end_zone"].isin(selected_end_zones)]
if selected_setups:
    league_event_df = league_event_df[league_event_df["Defensive_setup"].astype(str).isin(selected_setups)]

league_match_df = league_match_df[league_match_df["match_id"].isin(league_event_df["match_id"].unique())]
league_team_df = build_team_summary(league_event_df) if not league_event_df.empty else build_team_summary(df.iloc[0:0].copy())


def team_insight_table(source_df):
    if source_df.empty:
        return pd.DataFrame()
    out = (
        source_df.groupby(["corner_team", "delivery_zone", "end_zone"], dropna=False)
        .agg(
            corners=("match_id", "size"),
            shots=("led_to_shot", "sum"),
            total_xg=("shot_xg", "sum"),
            fast_shots=("is_fast_shot", "sum"),
        )
        .reset_index()
    )
    out["shot_rate"] = out["shots"] / out["corners"].replace(0, np.nan)
    return out.sort_values(["corner_team", "corners"], ascending=[True, False])


# =========================================================
# PAGE: LEAGUE OVERVIEW
# =========================================================
if page == "League Overview":
    tabs = st.tabs(["League Snapshot", "Team Rankings", "Shotmap", "Delivery Map"])

    with tabs[0]:
        render_kpis(league_event_df, league_match_df)

        c1, c2 = st.columns(2)
        with c1:
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

        with c2:
            st.markdown('<div class="section-title">Efficiency: Shot Rate vs xG per Match</div>', unsafe_allow_html=True)
            if not league_team_df.empty:
                fig = px.scatter(
                    league_team_df,
                    x="shot_rate",
                    y="xg_per_match",
                    size="corners_taken",
                    hover_name="team",
                    hover_data=["matches", "corners_per_match", "fast_shot_rate", "box_delivery_rate"],
                )
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('<div class="section-title">League Match Board</div>', unsafe_allow_html=True)
        board_cols = [
            c for c in [
                "Match", "home_team", "away_team", "home_corners", "away_corners",
                "total_corners", "shots_from_corners", "fast_shots", "total_xg", "shot_rate"
            ] if c in league_match_df.columns
        ]
        st.dataframe(
            league_match_df[board_cols].sort_values(["total_corners", "shots_from_corners"], ascending=False).reset_index(drop=True),
            use_container_width=True,
            height=430,
        )

    with tabs[1]:
        ranking_tabs = st.tabs(["Teams", "Takers", "Zones"])

        with ranking_tabs[0]:
            st.dataframe(
                league_team_df.sort_values(["total_xg", "shot_rate", "corners_taken"], ascending=False).reset_index(drop=True),
                use_container_width=True,
                height=540,
            )

        with ranking_tabs[1]:
            taker_table = (
                league_event_df.groupby("Taker", dropna=False)
                .agg(
                    corners=("match_id", "size"),
                    shots=("led_to_shot", "sum"),
                    fast_shots=("is_fast_shot", "sum"),
                    total_xg=("shot_xg", "sum"),
                    six_yard_deliveries=("is_six_yard_delivery", "sum"),
                )
                .reset_index()
            )
            taker_table["shot_rate"] = taker_table["shots"] / taker_table["corners"].replace(0, np.nan)
            taker_table["six_yard_rate"] = taker_table["six_yard_deliveries"] / taker_table["corners"].replace(0, np.nan)
            st.dataframe(taker_table.sort_values(["corners", "total_xg"], ascending=False).reset_index(drop=True), use_container_width=True, height=520)

        with ranking_tabs[2]:
            zone_table = team_insight_table(league_event_df)
            st.dataframe(zone_table.reset_index(drop=True), use_container_width=True, height=520)

    with tabs[2]:
        st.markdown('<div class="section-title">League Shotmap</div>', unsafe_allow_html=True)
        shot_df = league_event_df.dropna(subset=["shot_location_x", "shot_location_y"]).copy()
        shot_color = st.selectbox("Shotmap color by", ["corner_team", "Shooter", "Taker", "pass_technique"], index=0, key="league_shot_color")
        fig = shotmap_figure(shot_df, color_col=shot_color, title="League Shotmap — Corner Shots")
        st.plotly_chart(fig, use_container_width=True)

        if not shot_df.empty:
            st.dataframe(
                shot_df[["Match", "corner_team", "Taker", "Shooter", "SP_outcome", "shot_xg", "shot_location_x", "shot_location_y", "Minute"]]
                .sort_values(["shot_xg", "Minute"], ascending=[False, True])
                .reset_index(drop=True),
                use_container_width=True,
                height=300,
            )

    with tabs[3]:
        st.markdown('<div class="section-title">League Delivery Map</div>', unsafe_allow_html=True)
        delivery_color = st.selectbox("Delivery map color by", ["delivery_zone", "end_zone", "pass_technique", "corner_team", "Taker"], index=0, key="league_delivery_color")
        fig = delivery_map_figure(league_event_df, color_col=delivery_color, title="League Delivery Map — Corners")
        st.plotly_chart(fig, use_container_width=True)

        heatmap_fig = delivery_end_heatmap(league_event_df, title="League Delivery End-Location Heatmap")
        if len(heatmap_fig.data) > 0:
            st.plotly_chart(heatmap_fig, use_container_width=True)

        delivery_summary = (
            league_event_df.groupby(["corner_team", "pass_technique", "delivery_zone", "end_zone"], dropna=False)
            .agg(corners=("match_id", "size"), shots=("led_to_shot", "sum"), total_xg=("shot_xg", "sum"))
            .reset_index()
        )
        delivery_summary["shot_rate"] = delivery_summary["shots"] / delivery_summary["corners"].replace(0, np.nan)
        st.dataframe(delivery_summary.sort_values(["corners", "total_xg"], ascending=False).reset_index(drop=True), use_container_width=True, height=320)


# =========================================================
# PAGE: TEAM ANALYSIS
# =========================================================
elif page == "Team Analysis":
    if selected_team == "All Teams":
        st.info("Select a specific team in the sidebar for detailed team intelligence.")
    else:
        team_event_df = league_event_df[league_event_df["corner_team"] == selected_team].copy()
        team_match_df = league_match_df[league_match_df["match_id"].isin(team_event_df["match_id"].unique())].copy()
        team_row_df = league_team_df[league_team_df["team"] == selected_team].copy()

        team_tabs = st.tabs(["Snapshot", "Shotmap", "Delivery Map", "Takers & Shooters", "Match-by-Match"])

        with team_tabs[0]:
            st.subheader(f"Team Snapshot — {selected_team}")
            render_kpis(team_event_df, team_match_df)

            if not team_row_df.empty:
                row = team_row_df.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    metric_card("Volume Percentile", f"{percentile_rank(league_team_df['corners_per_match'], row['corners_per_match']):.0f}", "th")
                with c2:
                    metric_card("Shot Rate Percentile", f"{percentile_rank(league_team_df['shot_rate'], row['shot_rate']):.0f}", "th")
                with c3:
                    metric_card("xG per Match Percentile", f"{percentile_rank(league_team_df['xg_per_match'], row['xg_per_match']):.0f}", "th")
                with c4:
                    metric_card("6-yard Delivery Rate", f"{row['six_yard_delivery_rate'] * 100:.1f}", "%")

            c1, c2 = st.columns(2)
            with c1:
                outcome_df = team_event_df.groupby("outcome_bucket", dropna=False).size().reset_index(name="events").sort_values("events", ascending=False)
                fig = px.bar(outcome_df, x="outcome_bucket", y="events", title="Outcome Profile")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                zone_df = team_event_df.groupby("end_zone", dropna=False).size().reset_index(name="corners").sort_values("corners", ascending=False)
                fig = px.bar(zone_df, x="end_zone", y="corners", title="End-Zone Profile")
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

        with team_tabs[1]:
            shot_color = st.selectbox("Shotmap color by", ["Shooter", "Taker", "pass_technique", "Match"], index=0, key="team_shot_color")
            team_shots = team_event_df.dropna(subset=["shot_location_x", "shot_location_y"]).copy()
            fig = shotmap_figure(team_shots, color_col=shot_color, title=f"Shotmap — {selected_team}")
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                team_shots[["Match", "Taker", "Shooter", "pass_technique", "SP_outcome", "shot_xg", "shot_location_x", "shot_location_y", "Minute"]]
                .sort_values(["shot_xg", "Minute"], ascending=[False, True])
                .reset_index(drop=True),
                use_container_width=True,
                height=320,
            )

        with team_tabs[2]:
            delivery_color = st.selectbox("Delivery map color by", ["delivery_zone", "end_zone", "pass_technique", "Taker", "Match"], index=0, key="team_delivery_color")
            fig = delivery_map_figure(team_event_df, color_col=delivery_color, title=f"Delivery Map — {selected_team}")
            st.plotly_chart(fig, use_container_width=True)

            heatmap_fig = delivery_end_heatmap(team_event_df, title=f"Delivery End-Location Heatmap — {selected_team}")
            if len(heatmap_fig.data) > 0:
                st.plotly_chart(heatmap_fig, use_container_width=True)

            team_delivery_summary = (
                team_event_df.groupby(["Taker", "pass_technique", "delivery_zone", "end_zone"], dropna=False)
                .agg(corners=("match_id", "size"), shots=("led_to_shot", "sum"), total_xg=("shot_xg", "sum"), fast_shots=("is_fast_shot", "sum"))
                .reset_index()
            )
            team_delivery_summary["shot_rate"] = team_delivery_summary["shots"] / team_delivery_summary["corners"].replace(0, np.nan)
            st.dataframe(team_delivery_summary.sort_values(["corners", "total_xg"], ascending=False).reset_index(drop=True), use_container_width=True, height=320)

        with team_tabs[3]:
            taker_table = (
                team_event_df.groupby("Taker", dropna=False)
                .agg(
                    corners=("match_id", "size"),
                    shots=("led_to_shot", "sum"),
                    fast_shots=("is_fast_shot", "sum"),
                    total_xg=("shot_xg", "sum"),
                    inswingers=("is_inswinger", "sum"),
                    outswingers=("is_outswinger", "sum"),
                    short_corners=("is_short_corner", "sum"),
                    six_yard_deliveries=("is_six_yard_delivery", "sum"),
                )
                .reset_index()
            )
            taker_table["shot_rate"] = taker_table["shots"] / taker_table["corners"].replace(0, np.nan)
            taker_table["six_yard_rate"] = taker_table["six_yard_deliveries"] / taker_table["corners"].replace(0, np.nan)
            shooter_table = (
                team_event_df.groupby("Shooter", dropna=False)
                .agg(shots=("led_to_shot", "sum"), total_xg=("shot_xg", "sum"))
                .reset_index()
                .sort_values(["shots", "total_xg"], ascending=False)
            )
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="section-title">Taker Leaderboard</div>', unsafe_allow_html=True)
                st.dataframe(taker_table.sort_values(["corners", "shots"], ascending=False).reset_index(drop=True), use_container_width=True, height=420)
            with c2:
                st.markdown('<div class="section-title">Shooter Leaderboard</div>', unsafe_allow_html=True)
                st.dataframe(shooter_table.reset_index(drop=True), use_container_width=True, height=420)

        with team_tabs[4]:
            match_team_table = (
                team_event_df.groupby("Match", dropna=False)
                .agg(
                    corners=("match_id", "size"),
                    shots=("led_to_shot", "sum"),
                    fast_shots=("is_fast_shot", "sum"),
                    total_xg=("shot_xg", "sum"),
                    six_yard_deliveries=("is_six_yard_delivery", "sum"),
                )
                .reset_index()
            )
            match_team_table["shot_rate"] = match_team_table["shots"] / match_team_table["corners"].replace(0, np.nan)
            st.dataframe(match_team_table.sort_values(["corners", "total_xg"], ascending=False).reset_index(drop=True), use_container_width=True, height=500)


# =========================================================
# PAGE: MATCH EXPLORER
# =========================================================
elif page == "Match Explorer":
    available_matches = sorted(league_match_df["Match"].dropna().unique().tolist()) if not league_match_df.empty else []
    selected_match = st.selectbox("Select Match", ["All Matches"] + available_matches)

    match_event_df = league_event_df.copy()
    match_board_df = league_match_df.copy()
    if selected_match != "All Matches":
        match_board_df = match_board_df[match_board_df["Match"] == selected_match]
        match_event_df = match_event_df[match_event_df["match_id"].isin(match_board_df["match_id"].unique())]

    tabs = st.tabs(["Match Board", "Timeline", "Shotmap", "Delivery Map", "Event Feed"])

    with tabs[0]:
        render_kpis(match_event_df, match_board_df)
        board_cols = [c for c in ["Match", "home_team", "away_team", "home_corners", "away_corners", "total_corners", "shots_from_corners", "fast_shots", "total_xg", "shot_rate"] if c in match_board_df.columns]
        st.dataframe(match_board_df[board_cols].sort_values(["total_corners", "shots_from_corners"], ascending=False).reset_index(drop=True), use_container_width=True, height=420)

    with tabs[1]:
        if not match_event_df.empty:
            minute_df = match_event_df.groupby("Minute", dropna=False).size().reset_index(name="corner_events").sort_values("Minute")
            fig = px.bar(minute_df, x="Minute", y="corner_events", title="Corner Timeline")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        shot_color = st.selectbox("Shotmap color by", ["corner_team", "Shooter", "Taker", "pass_technique"], index=0, key="match_shot_color")
        fig = shotmap_figure(match_event_df.dropna(subset=["shot_location_x", "shot_location_y"]), color_col=shot_color, title=f"Shotmap — {selected_match}")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        delivery_color = st.selectbox("Delivery map color by", ["corner_team", "delivery_zone", "end_zone", "pass_technique", "Taker"], index=0, key="match_delivery_color")
        fig = delivery_map_figure(match_event_df, color_col=delivery_color, title=f"Delivery Map — {selected_match}")
        st.plotly_chart(fig, use_container_width=True)

    with tabs[4]:
        show_cols = [c for c in ["Match", "corner_team", "Taker", "Shooter", "Minute", "Second", "SP_outcome", "shot_xg", "Defensive_setup", "pass_technique", "delivery_zone", "end_zone"] if c in match_event_df.columns]
        st.dataframe(match_event_df[show_cols].sort_values(["Minute", "Second"]).reset_index(drop=True), use_container_width=True, height=620)


# =========================================================
# PAGE: SET PIECE LAB
# =========================================================
elif page == "Set Piece Lab":
    tabs = st.tabs(["Efficiency", "Timing", "Delivery Lab", "Defensive Looks", "Zone Analysis"])

    with tabs[0]:
        efficiency_table = league_team_df.copy()
        efficiency_table["shot_rate_pct"] = efficiency_table["shot_rate"] * 100
        efficiency_table["fast_shot_rate_pct"] = efficiency_table["fast_shot_rate"] * 100
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(efficiency_table, x="corners_per_match", y="shot_rate_pct", size="total_xg", hover_name="team", hover_data=["xg_per_match", "fast_shot_rate_pct", "box_delivery_rate"])
            fig.update_layout(height=430, xaxis_title="Corners per Match", yaxis_title="Shot Rate %")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.scatter(efficiency_table, x="six_yard_delivery_rate", y="xg_per_match", size="shots_from_corners", hover_name="team", hover_data=["shot_rate_pct", "fast_shot_rate_pct", "short_corner_rate"])
            fig.update_layout(height=430, xaxis_title="6-yard Delivery Rate", yaxis_title="xG per Match")
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(efficiency_table.sort_values(["xg_per_match", "shot_rate"], ascending=False).reset_index(drop=True), use_container_width=True, height=420)

    with tabs[1]:
        phase_order = ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"]
        phase_team = league_event_df.groupby(["corner_team", "phase"], dropna=False).size().reset_index(name="corners")
        phase_team["phase"] = pd.Categorical(phase_team["phase"], categories=phase_order, ordered=True)
        fig = px.bar(phase_team.sort_values(["phase", "corners"]), x="phase", y="corners", color="corner_team", title="Corner Timing by Team")
        fig.update_layout(height=460)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        delivery_df = league_event_df.groupby("pass_technique", dropna=False).agg(corners=("match_id", "size"), shots=("led_to_shot", "sum"), total_xg=("shot_xg", "sum")).reset_index()
        delivery_df["shot_rate"] = delivery_df["shots"] / delivery_df["corners"].replace(0, np.nan)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(delivery_df.sort_values("corners", ascending=False), x="pass_technique", y="corners", hover_data=["shots", "shot_rate", "total_xg"], title="Technique Volume")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(delivery_df.sort_values("shot_rate", ascending=False), x="pass_technique", y="shot_rate", hover_data=["corners", "shots", "total_xg"], title="Technique Efficiency")
            fig.update_layout(height=420)
            st.plotly_chart(fig, use_container_width=True)

        fig = delivery_end_heatmap(league_event_df, title="Delivery End-Location Heatmap")
        if len(fig.data) > 0:
            st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        defensive_df = league_event_df.groupby("Defensive_setup", dropna=False).agg(corners=("match_id", "size"), shots=("led_to_shot", "sum"), total_xg=("shot_xg", "sum")).reset_index()
        defensive_df["shot_rate"] = defensive_df["shots"] / defensive_df["corners"].replace(0, np.nan)
        c1, c2 = st.columns(2)
        with c1:
            fig = px.bar(defensive_df.sort_values("corners", ascending=False).head(15), x="Defensive_setup", y="corners", hover_data=["shots", "total_xg", "shot_rate"], title="Most Common Defensive Setups")
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            fig = px.bar(defensive_df.sort_values("shot_rate", ascending=False).head(15), x="Defensive_setup", y="shot_rate", hover_data=["corners", "shots", "total_xg"], title="Defensive Setup Shot Rate Allowed")
            fig.update_layout(height=430)
            st.plotly_chart(fig, use_container_width=True)
        st.dataframe(defensive_df.sort_values("corners", ascending=False).reset_index(drop=True), use_container_width=True, height=380)

    with tabs[4]:
        zone_table = team_insight_table(league_event_df)
        st.dataframe(zone_table.reset_index(drop=True), use_container_width=True, height=560)


# =========================================================
# PAGE: DATA CENTER
# =========================================================
elif page == "Data Center":
    tabs = st.tabs(["Raw Events", "Team Table", "Match Table", "Shot Events", "Delivery Events"])

    with tabs[0]:
        st.dataframe(league_event_df.reset_index(drop=True), use_container_width=True, height=620)
    with tabs[1]:
        st.dataframe(league_team_df.reset_index(drop=True), use_container_width=True, height=620)
    with tabs[2]:
        st.dataframe(league_match_df.reset_index(drop=True), use_container_width=True, height=620)
    with tabs[3]:
        st.dataframe(league_event_df[league_event_df["led_to_shot"]].reset_index(drop=True), use_container_width=True, height=620)
    with tabs[4]:
        delivery_cols = [c for c in ["Match", "corner_team", "Taker", "Minute", "pass_technique", "pass_location_x", "pass_location_y", "pass_end_location_x", "pass_end_location_y", "delivery_zone", "end_zone", "SP_outcome"] if c in league_event_df.columns]
        st.dataframe(league_event_df[delivery_cols].reset_index(drop=True), use_container_width=True, height=620)
