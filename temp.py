# -------------------- Imports & Config --------------------
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Set Piece Dashboard", layout="wide", page_icon="⚽")

# -------------------- Global UI Settings --------------------
APP_NAME = "Set Piece Intelligence"
BASE_PATH = os.path.dirname(__file__)
FILTER_COLS = [
    "competition.country_name",
    "competition.competition_name",
    "season.season_name",
]

# -------------------- Theme + Pro Styling --------------------
PRO_STYLE = """
<style>
  :root{
    --blue:#1a73e8;
    --red:#ed3b3b;
    --green:#4caf50;
    --dark:#202020;
    --muted:#6b7280;
    --bg:#ffffff;
    --surface:#ffffff;
    --surface2:#fbfbfd;
    --border:#e7e7ef;
    --shadow: 0 1px 2px rgba(0,0,0,.06);
    --radius: 14px;
  }

  /* App background */
  .main { background: var(--bg); }
  section[data-testid="stSidebar"]{
    border-right: 1px solid var(--border);
    background: linear-gradient(180deg, #fff 0%, #fcfcff 100%);
  }

  /* Typography */
  h1,h2,h3,h4{
    color: var(--dark);
    letter-spacing: -0.4px;
    font-weight: 850;
  }

  /* Buttons */
  .stButton>button{
    background: var(--blue);
    color: #fff;
    border: none;
    border-radius: 10px;
    font-weight: 800;
    padding: .55rem .9rem;
  }
  .stButton>button:hover{ background: #0d5bba; }

  /* Pills / chips */
  .chipwrap{ display:flex; flex-wrap:wrap; gap:8px; margin: 4px 0 0 0;}
  .chip{
    display:inline-flex;
    align-items:center;
    gap:6px;
    font-size: 12px;
    color: #111827;
    background: #f4f6fb;
    border: 1px solid var(--border);
    padding: 6px 10px;
    border-radius: 999px;
  }
  .chip b{ color: var(--blue); font-weight: 900; }

  /* Cards */
  .card{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 14px 14px;
  }
  .card h3{ margin: 0 0 8px 0; }

  /* Metric containers */
  [data-testid="metric-container"]{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 14px;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"]{
    gap: 0px;
    border-bottom: 2px solid var(--border);
  }
  .stTabs [data-baseweb="tab"]{
    padding: 10px 16px;
    font-weight: 850;
    color: var(--muted);
  }
  .stTabs [aria-selected="true"]{
    color: var(--blue);
    border-bottom: 3px solid var(--blue);
  }

  /* Leaderboard */
  .leaderboard-table{
    width: 100%;
    border-collapse: collapse;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    box-shadow: var(--shadow);
  }
  .leaderboard-table th{
    background: #111827;
    color: #fff;
    text-align: left;
    padding: 12px;
    font-weight: 900;
    font-size: 0.93rem;
  }
  .leaderboard-table td{
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    font-size: 0.92rem;
  }
  .leaderboard-table tr:nth-child(even){ background: var(--surface2); }

  /* Footer */
  .footer{
    margin-top: 22px;
    padding-top: 12px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 0.86rem;
  }

  /* Compact spacing */
  div.block-container{ padding-top: 1.3rem; padding-bottom: 1.3rem; }
</style>
"""
st.markdown(PRO_STYLE, unsafe_allow_html=True)

# -------------------- Utility Helpers --------------------
def safe_unique(series: pd.Series):
    if series is None:
        return []
    vals = series.dropna().astype(str).str.strip()
    vals = vals[(vals != "") & (vals != "nan") & (vals != "None")]
    return sorted(vals.unique().tolist())

def fmt_num(x, decimals=2):
    try:
        return f"{float(x):.{decimals}f}"
    except Exception:
        return "—"

def make_chips(chips: list[tuple[str, str]]):
    # chips: [(label, value), ...]
    html = '<div class="chipwrap">' + "".join(
        [f'<span class="chip">{k}: <b>{v}</b></span>' for k, v in chips if v and v != "All"]
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

def kpi_row(items: list[tuple[str, str, str | None]]):
    """
    items: [(title, value, help_text), ...]
    """
    cols = st.columns(len(items))
    for c, (title, value, help_text) in zip(cols, items):
        with c:
            st.metric(title, value, help=help_text)

# -------------------- Data Loading (cached + fast parsing) --------------------
@st.cache_data(ttl=3600)
def load_shots_data() -> pd.DataFrame:
    df = pd.read_excel(os.path.join(BASE_PATH, "db.xlsx"))

    for col in FILTER_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("").str.strip()

    # Vectorized location parsing: "[x, y, z]" -> x,y,z
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

    if "shot.statsbomb_xg" in df.columns:
        df["shot.statsbomb_xg"] = pd.to_numeric(df["shot.statsbomb_xg"], errors="coerce")

    subset_cols = [c for c in [
        "location_x", "location_y", "shot.statsbomb_xg",
        "team.name", "player.name", "Match", "shot.body_part.name"
    ] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)

    df = df[df["location_x"].notna() & df["shot.statsbomb_xg"].notna()].copy()
    return df

@st.cache_data(ttl=3600)
def load_corner_data() -> pd.DataFrame:
    df = pd.read_excel(os.path.join(BASE_PATH, "corner_passes_and_shots_with_metadata.xlsx"))
    df.columns = df.columns.astype(str).str.strip()

    for col in FILTER_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").fillna("").str.strip()

    if "location" in df.columns:
        loc = df["location"].astype("string").fillna("").str.strip()
        parts = loc.str.strip("[]()").str.split(",", expand=True)
        df["location_x"] = pd.to_numeric(parts[0], errors="coerce") if parts.shape[1] > 0 else np.nan
        df["location_y"] = pd.to_numeric(parts[1], errors="coerce") if parts.shape[1] > 1 else np.nan
    else:
        df["location_x"] = np.nan
        df["location_y"] = np.nan

    if "pass.end_location" in df.columns:
        pel = df["pass.end_location"].astype("string").fillna("").str.split(",", expand=True)
        df["pass_end_x"] = pd.to_numeric(pel[0], errors="coerce") if pel.shape[1] > 0 else np.nan
        df["pass_end_y"] = pd.to_numeric(pel[1], errors="coerce") if pel.shape[1] > 1 else np.nan
    else:
        df["pass_end_x"] = np.nan
        df["pass_end_y"] = np.nan

    if "shot.statsbomb_xg" in df.columns:
        df["shot.statsbomb_xg"] = pd.to_numeric(df["shot.statsbomb_xg"], errors="coerce").fillna(0)

    return df

@st.cache_data(ttl=3600)
def build_corner_summary(df_corner: pd.DataFrame) -> pd.DataFrame:
    df = df_corner.copy()

    sort_col = "index" if "index" in df.columns else ("event_id" if "event_id" in df.columns else None)
    if sort_col and "possession" in df.columns:
        df = df.sort_values(["possession", sort_col]).reset_index(drop=True)
    elif "possession" in df.columns:
        df = df.sort_values(["possession"]).reset_index(drop=True)

    if "event_type" not in df.columns or "possession" not in df.columns:
        return pd.DataFrame()

    df["is_shot"] = df["event_type"].eq("Shot")
    df["xg"] = df.get("shot.statsbomb_xg", 0)
    pos_xg = df.groupby("possession")["xg"].sum()

    df["next_event_type"] = df.groupby("possession")["event_type"].shift(-1)
    df["shot_within_3"] = (
        df.groupby("possession")["is_shot"]
          .transform(lambda s: s.shift(-1).rolling(3, min_periods=1).max())
          .fillna(False)
          .astype(bool)
    )

    corner_passes = df[df["event_type"].eq("CornerPass")].copy()
    if corner_passes.empty:
        return pd.DataFrame()

    y = corner_passes.get("location_y", pd.Series(dtype=float))
    corner_passes["side"] = np.select([y.eq(0.1), y.eq(80)], ["Left", "Right"], default="Unknown")

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
        default="First contact - no shot",
    )

    corner_passes["xG"] = corner_passes["possession"].map(pos_xg).fillna(0)

    out = pd.DataFrame({
        "corner_index": corner_passes.index,
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

# -------------------- Sidebar (Pro Nav + Controls) --------------------
with st.sidebar:
    st.markdown(f"## ⚽ {APP_NAME}")
    st.caption("Professional, fast set-piece analytics.")

    nav = st.radio("Navigation", ["Shots", "Corner Routines", "Penalties"], index=0, key="nav_section")

    st.markdown("---")
    with st.expander("Controls", expanded=False):
        st.caption("Use these to improve performance and workflow.")
        use_compact = st.toggle("Compact tables", value=True, help="Smaller tables render faster.")
        show_data_default = st.toggle("Show data expanders by default", value=False)
        if st.button("Reset all inputs"):
            for k in list(st.session_state.keys()):
                if k not in ("nav_section",):
                    del st.session_state[k]
            st.rerun()

# -------------------- Header Area --------------------
top = st.container()
with top:
    left, right = st.columns([3, 1])
    with left:
        st.title(APP_NAME)
        st.caption("High-signal set-piece dashboards with clean UI and optimized compute.")
    with right:
        st.markdown("<div class='card'><b>Status</b><br>✅ Ready</div>", unsafe_allow_html=True)

st.markdown("")

# -------------------- Shots Section --------------------
def render_shots():
    with st.spinner("Loading shots data…"):
        df = load_shots_data()

    df_goals = df[(df.get("shot.outcome.name") == "Goal") & (df["location_x"] >= 60)].copy()
    if df_goals.empty:
        st.warning("No goals found with location_x >= 60.")
        return

    # --- Sidebar Filters (in a form = smooth) ---
    with st.sidebar:
        with st.form("shots_filters_form", border=False):
            st.markdown("### Filters")

            set_piece_vals = ["All"] + safe_unique(df_goals.get("play_pattern.name"))
            team_vals = ["All"] + safe_unique(df_goals.get("team.name"))
            pos_vals = ["All"] + safe_unique(df_goals.get("position.name"))
            nation_vals = ["All"] + safe_unique(df_goals.get("competition.country_name"))
            league_vals = ["All"] + safe_unique(df_goals.get("competition.competition_name"))
            season_vals = ["All"] + safe_unique(df_goals.get("season.season_name"))
            match_vals = ["All"] + safe_unique(df_goals.get("Match"))
            body_vals = ["All"] + safe_unique(df_goals.get("shot.body_part.name"))

            f_set_piece = st.selectbox("Set Piece", set_piece_vals, key="shots_set_piece")
            f_team = st.selectbox("Team", team_vals, key="shots_team")
            f_pos = st.selectbox("Position", pos_vals, key="shots_pos")
            f_nation = st.selectbox("Nation", nation_vals, key="shots_nation")
            f_league = st.selectbox("League", league_vals, key="shots_league")
            f_season = st.selectbox("Season", season_vals, key="shots_season")
            f_match = st.selectbox("Match", match_vals, key="shots_match")
            f_body = st.selectbox("Body Part", body_vals, key="shots_body")

            f_first_time = st.selectbox("First-Time Shot", ["All", "Yes", "No"], key="shots_first_time")

            xg_min = float(np.nanmin(df_goals["shot.statsbomb_xg"].values))
            xg_max = float(np.nanmax(df_goals["shot.statsbomb_xg"].values))
            f_xg = st.slider(
                "xG Range", xg_min, xg_max,
                (max(xg_min, 0.0), min(xg_max, 1.0)),
                0.01,
                key="shots_xg"
            )

            # Optional: quick search box for player (for table/filtering)
            player_search = st.text_input("Quick player search (optional)", value="", key="shots_player_search")

            apply = st.form_submit_button("Apply")

    # --- Apply filters ---
    filtered = df_goals.copy()

    def apply_eq(col, val):
        nonlocal filtered
        if val != "All" and col in filtered.columns:
            filtered = filtered[filtered[col] == val]

    apply_eq("play_pattern.name", f_set_piece)
    apply_eq("team.name", f_team)
    apply_eq("position.name", f_pos)
    apply_eq("competition.country_name", f_nation)
    apply_eq("competition.competition_name", f_league)
    apply_eq("season.season_name", f_season)
    apply_eq("Match", f_match)
    apply_eq("shot.body_part.name", f_body)

    if f_first_time != "All" and "shot.first_time" in filtered.columns:
        filtered = filtered[filtered["shot.first_time"] == (f_first_time == "Yes")]

    filtered = filtered[filtered["shot.statsbomb_xg"].between(*f_xg)]

    if player_search.strip():
        if "player.name" in filtered.columns:
            ps = player_search.strip().lower()
            filtered = filtered[filtered["player.name"].astype(str).str.lower().str.contains(ps, na=False)]

    # --- Content ---
    header = st.container()
    with header:
        st.markdown("### Shots · Set Piece Goals")
        make_chips([
            ("Set Piece", f_set_piece),
            ("Team", f_team),
            ("League", f_league),
            ("Season", f_season),
            ("Nation", f_nation),
            ("xG", f"{f_xg[0]:.2f}–{f_xg[1]:.2f}")
        ])

    if filtered.empty:
        st.warning("No goals found for these filters.")
        return

    # KPIs (with useful help)
    top_team = filtered["team.name"].mode().iat[0] if "team.name" in filtered.columns else "—"
    common_type = filtered["play_pattern.name"].mode().iat[0] if "play_pattern.name" in filtered.columns else "—"
    kpi_row([
        ("Total Goals", str(int(len(filtered))), "Number of goals in filtered set-piece sample"),
        ("Avg xG", fmt_num(filtered["shot.statsbomb_xg"].mean(), 3), "Mean xG for these goals"),
        ("Top Team", str(top_team), "Most frequent team in filtered results"),
        ("Most Common Type", str(common_type), "Most frequent set-piece type")
    ])

    st.markdown("")

    # Tabs
    tab0, tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", "Goal Map", "Goal Placement", "Leaderboard", "Data"
    ])

    # --- Overview ---
    with tab0:
        colA, colB = st.columns(2)

        with colA:
            st.markdown("<div class='card'><h3>Goals by Team</h3></div>", unsafe_allow_html=True)
            team_counts = filtered["team.name"].value_counts().reset_index()
            team_counts.columns = ["Team", "Goals"]
            fig_team = px.bar(team_counts, x="Team", y="Goals", template="plotly_white")
            fig_team.update_layout(height=420, margin=dict(t=20, b=20))
            fig_team.update_traces(marker_line_width=0)
            st.plotly_chart(fig_team, use_container_width=True)
            st.markdown('<div class="annotation">Distribution of set-piece goals by team</div>', unsafe_allow_html=True)

        with colB:
            st.markdown("<div class='card'><h3>Goals by Set Piece Type</h3></div>", unsafe_allow_html=True)
            type_counts = filtered["play_pattern.name"].value_counts().reset_index()
            type_counts.columns = ["Set Piece Type", "Goals"]
            fig_type = px.bar(type_counts, x="Set Piece Type", y="Goals", template="plotly_white")
            fig_type.update_layout(height=420, margin=dict(t=20, b=20))
            fig_type.update_traces(marker_line_width=0)
            st.plotly_chart(fig_type, use_container_width=True)
            st.markdown('<div class="annotation">Which set pieces produce the most goals</div>', unsafe_allow_html=True)

        # Extra: xG distribution (quick signal)
        st.markdown("<div class='card'><h3>xG Distribution</h3></div>", unsafe_allow_html=True)
        fig_hist = px.histogram(filtered, x="shot.statsbomb_xg", nbins=20, template="plotly_white")
        fig_hist.update_layout(height=280, margin=dict(t=20, b=20))
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown('<div class="annotation">Shape of the xG profile for your filtered goals</div>', unsafe_allow_html=True)

    # --- Goal Map ---
    with tab1:
        st.markdown("<div class='card'><h3>Goal Locations (Attacking Half)</h3></div>", unsafe_allow_html=True)

        filtered_half = filtered[filtered["location_x"] >= 60].copy()
        if filtered_half.empty:
            st.info("No goals in the attacking half for these filters.")
        else:
            filtered_half["plot_x"] = filtered_half["location_y"]
            filtered_half["plot_y"] = 120 - filtered_half["location_x"]

            hover_text = (
                "Player: " + filtered_half.get("player.name", "").astype(str)
                + "<br>Team: " + filtered_half.get("team.name", "").astype(str)
                + "<br>xG: " + filtered_half["shot.statsbomb_xg"].round(2).astype(str)
                + "<br>Body Part: " + filtered_half.get("shot.body_part.name", "").astype(str)
                + "<br>Match: " + filtered_half.get("Match", "").astype(str)
            )

            fig = go.Figure()
            fig.update_layout(
                xaxis=dict(range=[0, 80], showgrid=False, zeroline=False, visible=False, scaleanchor="y"),
                yaxis=dict(range=[0, 60], showgrid=False, zeroline=False, visible=False),
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=720,
                margin=dict(t=20, b=20),
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
                    opacity=0.85,
                ),
                text=hover_text,
                hoverinfo="text",
                name="Goals",
            ))

            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<div class="annotation">WebGL scatter = smoother even with lots of goals</div>', unsafe_allow_html=True)

            # Quick drill-down
            players = safe_unique(filtered_half.get("player.name"))
            if players:
                selected_player = st.selectbox("Drill down: Player", players, key="shots_drill_player")
                show_cols = [c for c in [
                    "player.name", "team.name", "play_pattern.name", "shot.statsbomb_xg",
                    "shot.body_part.name", "Match", "competition.competition_name", "season.season_name"
                ] if c in filtered_half.columns]
                st.dataframe(
                    filtered_half[filtered_half["player.name"] == selected_player][show_cols],
                    use_container_width=True,
                    height=260 if use_compact else 420
                )

    # --- Goal Placement ---
    with tab2:
        st.markdown("<div class='card'><h3>Goal Placement (shot.end_location)</h3></div>", unsafe_allow_html=True)

        if "shot.end_location" not in filtered.columns:
            st.info("shot.end_location column not found.")
        else:
            endloc = filtered["shot.end_location"].astype("string").fillna("").str.split(",", expand=True)
            filtered["shot.end_location_x"] = pd.to_numeric(endloc[0], errors="coerce") if endloc.shape[1] > 0 else np.nan
            filtered["shot.end_location_y"] = pd.to_numeric(endloc[1], errors="coerce") if endloc.shape[1] > 1 else np.nan
            filtered["shot.end_location_z"] = pd.to_numeric(endloc[2], errors="coerce") if endloc.shape[1] > 2 else np.nan

            GOAL_WIDTH, GOAL_HEIGHT = 7.32, 2.44
            LEFT_POST_Y, RIGHT_POST_Y = 36.8, 43.2

            goals = filtered.dropna(subset=["shot.end_location_y"]).copy()
            goals = goals[(goals["shot.end_location_y"] >= LEFT_POST_Y) & (goals["shot.end_location_y"] <= RIGHT_POST_Y)]

            if goals.empty:
                st.info("No goals with shot.end_location_y inside goalposts found.")
            else:
                goals["goal_x_m"] = (goals["shot.end_location_y"] - LEFT_POST_Y) * (GOAL_WIDTH / (RIGHT_POST_Y - LEFT_POST_Y))
                goals["goal_z_m"] = goals["shot.end_location_z"].fillna(0)
                xg = goals["shot.statsbomb_xg"].fillna(0)

                marker_size = np.interp(xg, (xg.min(), xg.max()), (7, 22)) if len(goals) > 1 else np.full(len(goals), 12)

                fig = go.Figure()
                fig.add_shape(type="rect", x0=0, y0=0, x1=GOAL_WIDTH, y1=GOAL_HEIGHT, line=dict(color="black", width=3))
                fig.add_shape(type="line", x0=0, y0=GOAL_HEIGHT/2, x1=GOAL_WIDTH, y1=GOAL_HEIGHT/2, line=dict(color="#6b7280", dash="dash"))
                fig.add_shape(type="line", x0=GOAL_WIDTH/3, y0=0, x1=GOAL_WIDTH/3, y1=GOAL_HEIGHT, line=dict(color="#6b7280", dash="dash"))
                fig.add_shape(type="line", x0=2*GOAL_WIDTH/3, y0=0, x1=2*GOAL_WIDTH/3, y1=GOAL_HEIGHT, line=dict(color="#6b7280", dash="dash"))

                fig.add_trace(go.Scattergl(
                    x=goals["goal_x_m"],
                    y=goals["goal_z_m"],
                    mode="markers",
                    marker=dict(
                        size=marker_size,
                        color=xg,
                        colorscale="Bluered",
                        showscale=True,
                        colorbar=dict(title="xG"),
                        line=dict(width=0.6, color="black"),
                        opacity=0.86
                    ),
                    text=goals.get("player.name", "").astype(str),
                    hovertemplate="Player: %{text}<br>Width: %{x:.2f} m<br>Height: %{y:.2f} m<br>xG: %{marker.color:.3f}<extra></extra>",
                    name="Goals"
                ))

                fig.update_layout(
                    height=620,
                    margin=dict(t=20, b=20),
                    xaxis=dict(title="Goal Width (m)", range=[0, GOAL_WIDTH], showgrid=False, zeroline=False),
                    yaxis=dict(title="Goal Height (m)", range=[0, GOAL_HEIGHT], showgrid=False, zeroline=False),
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    yaxis_scaleanchor="x"
                )

                st.plotly_chart(fig, use_container_width=True)
                st.markdown('<div class="annotation">Placement of goals within the goalmouth (zones + xG)</div>', unsafe_allow_html=True)

    # --- Leaderboard ---
    with tab3:
        st.markdown("<div class='card'><h3>Performance Leaderboard</h3></div>", unsafe_allow_html=True)

        metric = st.selectbox(
            "Rank players by",
            ["Total Goals", "Total xG", "Average xG per Goal", "Goals per Match"],
            key="shots_lb_metric"
        )

        grp = filtered.groupby(["player.name", "team.name"], dropna=False).agg(
            Total_Goals=("player.name", "count"),
            Total_xG=("shot.statsbomb_xg", "sum"),
            Matches=("Match", "nunique")
        ).reset_index()

        grp["Avg_xG_per_Goal"] = grp["Total_xG"] / grp["Total_Goals"]
        grp["Goals_per_Match"] = grp["Total_Goals"] / grp["Matches"].replace(0, np.nan)

        if metric == "Total Goals":
            grp = grp.sort_values("Total_Goals", ascending=False); metric_col = "Total_Goals"
        elif metric == "Total xG":
            grp = grp.sort_values("Total_xG", ascending=False); metric_col = "Total_xG"
        elif metric == "Average xG per Goal":
            grp = grp.sort_values("Avg_xG_per_Goal", ascending=False); metric_col = "Avg_xG_per_Goal"
        else:
            grp = grp.sort_values("Goals_per_Match", ascending=False); metric_col = "Goals_per_Match"

        grp[metric_col] = grp[metric_col].round(3)

        topn = st.slider("Rows", 10, 50, 20, 5, key="shots_lb_rows")
        top = grp.head(topn).reset_index(drop=True)

        st.write(
            f"""
            <table class="leaderboard-table">
                <thead>
                    <tr>
                        <th>Rank</th><th>Player</th><th>Team</th>
                        <th>{metric}</th><th>Total Goals</th><th>Total xG</th><th>Matches</th>
                    </tr>
                </thead>
                <tbody>
            """,
            unsafe_allow_html=True
        )
        for i, row in top.iterrows():
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
        st.markdown('<div class="annotation">Rankings update instantly after applying filters</div>', unsafe_allow_html=True)

        st.download_button(
            "Download leaderboard (CSV)",
            data=grp.to_csv(index=False),
            file_name="shots_leaderboard.csv",
            key="dl_shots_leaderboard"
        )

    # --- Data Tab (kept lightweight) ---
    with tab4:
        st.markdown("<div class='card'><h3>Filtered Data</h3></div>", unsafe_allow_html=True)
        show_cols = [c for c in [
            "player.name", "team.name", "play_pattern.name", "position.name", "shot.statsbomb_xg",
            "shot.body_part.name", "Match", "competition.competition_name", "season.season_name",
            "location_x", "location_y"
        ] if c in filtered.columns]

        st.download_button(
            "Download filtered goals (CSV)",
            data=filtered[show_cols].to_csv(index=False),
            file_name="filtered_set_piece_goals.csv",
            key="dl_shots_filtered"
        )

        # Use expander to avoid heavy render by default
        expanded = bool(show_data_default)
        with st.expander("Show table", expanded=expanded):
            st.dataframe(
                filtered[show_cols],
                use_container_width=True,
                height=360 if use_compact else 520
            )

# -------------------- Corner Routines Section --------------------
def render_corners():
    with st.spinner("Loading corner data…"):
        df_corner = load_corner_data()

    if df_corner.empty:
        st.error("No data loaded for corner routines analysis.")
        return

    if "event_type" not in df_corner.columns or "possession" not in df_corner.columns:
        st.error("Corner file must include 'event_type' and 'possession' columns.")
        return

    with st.spinner("Building corner summary…"):
        summary = build_corner_summary(df_corner)

    if summary.empty:
        st.info("No corner passes found (event_type == CornerPass).")
        return

    st.markdown("### Corner Routines")
    st.caption("Outcome classification + delivery end locations, optimized for speed.")

    # Sidebar Filters
    with st.sidebar:
        with st.form("corner_filters_form", border=False):
            st.markdown("### Filters")

            team_vals = ["All"] + safe_unique(summary["team.name"])
            tech_vals = ["All"] + safe_unique(summary["pass_technique"])
            side_vals = ["All", "Left", "Right", "Unknown"]
            height_vals = ["All"] + safe_unique(summary["pass_height"])
            body_vals = ["All"] + safe_unique(summary["pass_body_part"])
            outcome_vals = ["All"] + safe_unique(summary["pass_outcome"])
            class_vals = ["All"] + safe_unique(summary["classification"])
            league_vals = ["All"] + safe_unique(summary["competition.competition_name"])
            season_vals = ["All"] + safe_unique(summary["season.season_name"])

            f_team = st.selectbox("Team", team_vals, key="corner_team")
            f_tech = st.selectbox("Technique", tech_vals, key="corner_tech")
            f_side = st.selectbox("Side", side_vals, key="corner_side")
            f_height = st.selectbox("Pass Height", height_vals, key="corner_height")
            f_body = st.selectbox("Body Part", body_vals, key="corner_body")
            f_outcome = st.selectbox("Pass Outcome", outcome_vals, key="corner_outcome")
            f_class = st.selectbox("Classification", class_vals, key="corner_class")
            f_league = st.selectbox("League", league_vals, key="corner_league")
            f_season = st.selectbox("Season", season_vals, key="corner_season")

            apply = st.form_submit_button("Apply")

    filtered = summary.copy()
    if f_team != "All": filtered = filtered[filtered["team.name"] == f_team]
    if f_tech != "All": filtered = filtered[filtered["pass_technique"] == f_tech]
    if f_side != "All": filtered = filtered[filtered["side"] == f_side]
    if f_height != "All": filtered = filtered[filtered["pass_height"] == f_height]
    if f_body != "All": filtered = filtered[filtered["pass_body_part"] == f_body]
    if f_outcome != "All": filtered = filtered[filtered["pass_outcome"] == f_outcome]
    if f_class != "All": filtered = filtered[filtered["classification"] == f_class]
    if f_league != "All": filtered = filtered[filtered["competition.competition_name"] == f_league]
    if f_season != "All": filtered = filtered[filtered["season.season_name"] == f_season]

    make_chips([
        ("Team", f_team), ("Technique", f_tech), ("Side", f_side),
        ("League", f_league), ("Season", f_season)
    ])

    if filtered.empty:
        st.info("No corners found for these filters.")
        return

    # Metrics (unique possession)
    unique_pos = filtered["possession"].dropna().unique().tolist()
    total_corners = int(len(filtered))
    total_xg = float(filtered.drop_duplicates("possession")["xG"].sum()) if unique_pos else 0.0
    df_shots = df_corner[df_corner["possession"].isin(unique_pos) & df_corner["event_type"].eq("Shot")]
    total_shots = int(len(df_shots))
    avg_xg_shot = (total_xg / total_shots) if total_shots > 0 else np.nan

    kpi_row([
        ("Total Corners", str(total_corners), "Number of corner deliveries in filtered sample"),
        ("Shots from Corners", str(total_shots), "Count of shots in same possessions"),
        ("Total xG", fmt_num(total_xg, 2), "Sum of possession xG from those corners"),
        ("Avg xG / Shot", fmt_num(avg_xg_shot, 3) if total_shots > 0 else "N/A", "Total xG divided by shots")
    ])

    tab0, tab1, tab2 = st.tabs(["Map", "Breakdown", "Data"])

    with tab0:
        st.markdown("<div class='card'><h3>Corner Pass End Locations (Vertical Attacking Half)</h3></div>", unsafe_allow_html=True)
        valid = filtered.dropna(subset=["pass_end_x", "pass_end_y"])
        if valid.empty:
            st.info("No valid pass end locations available.")
        else:
            color_map = {
                "First contact - direct shot": "red",
                "First contact - shot within 3 seconds": "blue",
                "No first contact - shot": "green",
                "First contact - no shot": "orange",
                "No first contact - no shot": "gray"
            }

            pitch_length, pitch_width = 120, 80
            fig = go.Figure()
            fig.update_layout(
                height=980,
                margin=dict(t=20, b=20),
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
                    dict(type="path", path=f"M {pitch_width/2 - 10},{102} A 10,10 0 0,1 {pitch_width/2 + 10},{102}", line=dict(color="black", width=2)),
                ]
            )

            for classification, grp in valid.groupby("classification"):
                fig.add_trace(go.Scattergl(
                    x=grp["pass_end_y"],
                    y=grp["pass_end_x"],
                    mode="markers",
                    name=classification,
                    marker=dict(
                        size=10,
                        color=color_map.get(classification, "gray"),
                        opacity=0.85,
                        line=dict(width=1, color="black")
                    ),
                    hovertemplate=(
                        "Team: %{customdata[0]}<br>"
                        "Player: %{customdata[1]}<br>"
                        "Classification: %{customdata[2]}<br>"
                        "xG (possession): %{customdata[3]:.2f}<extra></extra>"
                    ),
                    customdata=grp[["team.name", "player.name", "classification", "xG"]].values
                ))

            st.plotly_chart(fig, use_container_width=True)
            st.markdown('<div class="annotation">Grouped by outcome classification (WebGL for smooth pan/zoom)</div>', unsafe_allow_html=True)

    with tab1:
        st.markdown("<div class='card'><h3>Outcome Breakdown</h3></div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)

        with c1:
            counts = filtered["classification"].value_counts().reset_index()
            counts.columns = ["Classification", "Count"]
            fig = px.bar(counts, x="Classification", y="Count", template="plotly_white")
            fig.update_layout(height=360, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            team_xg = filtered.drop_duplicates("possession").groupby("team.name")["xG"].sum().reset_index()
            team_xg = team_xg.sort_values("xG", ascending=False)
            fig2 = px.bar(team_xg, x="team.name", y="xG", template="plotly_white")
            fig2.update_layout(height=360, margin=dict(t=20, b=20), xaxis_title="Team", yaxis_title="Total xG")
            st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.download_button(
            "Download filtered corners (CSV)",
            data=filtered.to_csv(index=False),
            file_name="filtered_corner_summary.csv",
            key="dl_corner_filtered"
        )
        expanded = bool(show_data_default)
        with st.expander("Show table", expanded=expanded):
            st.dataframe(filtered, use_container_width=True, height=360 if use_compact else 520)

# -------------------- Penalties Placeholder --------------------
def render_penalties():
    st.markdown("### Penalties")
    st.info("Penalty section not included in your provided dataset/code. Add it and I’ll match the same pro layout + speed patterns.")
    st.markdown("<div class='card'><h3>Suggested Additions</h3><ul><li>Keeper dive direction heatmap</li><li>Placement zones + xG</li><li>Player conversion leaderboard</li></ul></div>", unsafe_allow_html=True)

# -------------------- Router --------------------
if nav == "Shots":
    render_shots()
elif nav == "Corner Routines":
    render_corners()
else:
    render_penalties()

st.markdown("<div class='footer'>© Football Analytics Team</div>", unsafe_allow_html=True)
