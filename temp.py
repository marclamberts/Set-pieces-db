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
# THEME / VISUAL SYSTEM
# =========================================================
BG = "#0b1220"
CARD = "#121a2b"
CARD_2 = "#162033"
BORDER = "rgba(255,255,255,0.08)"
TEXT = "#f4f7fb"
MUTED = "#9fb0c7"
ACCENT = "#58a6ff"
GREEN_PITCH = "#123c22"

px.defaults.template = "plotly_dark"

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            radial-gradient(circle at top right, rgba(88,166,255,0.10), transparent 28%),
            linear-gradient(180deg, #09101d 0%, #0b1220 100%);
        color: {TEXT};
    }}
    .block-container {{
        max-width: 1600px;
        padding-top: 1rem;
        padding-bottom: 1rem;
    }}
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #0e1627 0%, #0b1220 100%);
        border-right: 1px solid {BORDER};
    }}
    .hero {{
        padding: 18px 22px;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(88,166,255,0.18), rgba(88,166,255,0.05));
        border: 1px solid rgba(88,166,255,0.18);
        margin-bottom: 14px;
    }}
    .hero-title {{
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.05;
        margin-bottom: 0.3rem;
    }}
    .hero-sub {{
        color: {MUTED};
        font-size: 0.97rem;
    }}
    .pill {{
        display: inline-block;
        padding: 0.25rem 0.65rem;
        border-radius: 999px;
        background: rgba(88,166,255,0.14);
        color: #dcecff;
        border: 1px solid rgba(88,166,255,0.18);
        font-size: 0.78rem;
        margin-right: 0.4rem;
    }}
    .kpi-card {{
        background: linear-gradient(180deg, {CARD} 0%, {CARD_2} 100%);
        border: 1px solid {BORDER};
        border-radius: 20px;
        padding: 16px 16px;
        min-height: 105px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.16);
    }}
    .kpi-label {{
        color: {MUTED};
        text-transform: uppercase;
        font-size: 0.74rem;
        letter-spacing: 0.08em;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        color: {TEXT};
        font-weight: 800;
        font-size: 1.8rem;
        line-height: 1.05;
    }}
    .section-title {{
        font-size: 1.08rem;
        font-weight: 800;
        margin: 0.2rem 0 0.8rem 0;
    }}
    .section-sub {{
        color: {MUTED};
        margin-top: -0.4rem;
        margin-bottom: 0.9rem;
        font-size: 0.9rem;
    }}
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 18px;
        overflow: hidden;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# LOGIN
# =========================================================
def login_screen():
    st.markdown('<div class="hero"><div class="hero-title">⚽ Set Piece Studio</div><div class="hero-sub">Premium corner analysis for Allsvenskan 2025</div></div>', unsafe_allow_html=True)
    with st.form("login_form", clear_on_submit=False):
        c1, c2, c3 = st.columns([1.1, 1.1, 1])
        with c1:
            username = st.text_input("Name")
        with c2:
            password = st.text_input("Password", type="password")
        with c3:
            st.write("")
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
# HELPERS
# =========================================================
def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


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


def figure_layout(fig, height=420, title=None):
    fig.update_layout(
        height=height,
        title=title,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=48 if title else 10, b=8),
        legend_title_text="",
        font=dict(color=TEXT),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


def draw_pitch(fig, title=None, height=560):
    line_color = "rgba(255,255,255,0.58)"
    pitch_color = GREEN_PITCH
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
        font=dict(color="white"),
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
                marker=dict(size=sub["marker_size"], color=color_map[cat], opacity=0.88, line=dict(color="white", width=1)),
                text=[
                    f"{row['Match']}<br>Team: {row['corner_team']}<br>Taker: {row['Taker']}<br>Shooter: {row['Shooter']}<br>Outcome: {row['SP_outcome']}<br>xG: {0 if pd.isna(row['shot_xg']) else row['shot_xg']:.3f}<br>Minute: {int(row['Minute']) if pd.notna(row['Minute']) else ''}"
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
                    line=dict(color=color_map[cat], width=2.3),
                    marker=dict(size=[6, 8], color=[color_map[cat], color_map[cat]]),
                    name=str(cat),
                    legendgroup=str(cat),
                    showlegend=False,
                    text=(
                        f"{row['Match']}<br>Team: {row['corner_team']}<br>Taker: {row['Taker']}<br>Technique: {row['pass_technique']}<br>Delivery Zone: {row['delivery_zone']}<br>End Zone: {row['end_zone']}<br>Outcome: {row['SP_outcome']}<br>Minute: {int(row['Minute']) if pd.notna(row['Minute']) else ''}"
                    ),
                    hovertemplate="%{text}<extra></extra>",
                )
            )
    for cat in categories:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=color_map[cat], width=3), name=str(cat), legendgroup=str(cat), showlegend=True))
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
    return figure_layout(fig, height=520, title=title)


def outcome_pie_figure(df_events, title="Outcome Split"):
    if df_events.empty:
        return go.Figure()
    summary = df_events.groupby("outcome_bucket", dropna=False).size().reset_index(name="corners")
    fig = px.pie(summary, names="outcome_bucket", values="corners", title=title, hole=0.55)
    return figure_layout(fig, height=380, title=title)


def technique_pie_figure(df_events, title="Delivery Technique Split"):
    if df_events.empty:
        return go.Figure()
    summary = df_events.groupby("pass_technique", dropna=False).size().reset_index(name="corners")
    fig = px.pie(summary, names="pass_technique", values="corners", title=title, hole=0.55)
    return figure_layout(fig, height=380, title=title)


def cumulative_timeline_figure(df_events, color_col="corner_team", title="Cumulative Corners Over Time"):
    if df_events.empty:
        return go.Figure()
    base = (
        df_events.groupby(["Minute", color_col], dropna=False)
        .size()
        .reset_index(name="corners")
        .sort_values([color_col, "Minute"])
    )
    base["cumulative_corners"] = base.groupby(color_col)["corners"].cumsum()
    fig = px.line(base, x="Minute", y="cumulative_corners", color=color_col, markers=True, title=title)
    return figure_layout(fig, height=400, title=title)


def team_scatter_figure(team_df, x_col, y_col, size_col, title):
    if team_df.empty:
        return go.Figure()
    fig = px.scatter(
        team_df,
        x=x_col,
        y=y_col,
        size=size_col,
        hover_name="team",
        hover_data=["corners_taken", "matches", "total_xg", "fast_shot_rate", "box_delivery_rate"],
        title=title,
    )
    return figure_layout(fig, height=420, title=title)


def phase_heatmap_figure(df_events, title="Corner Timing Heatmap"):
    if df_events.empty:
        return go.Figure()
    tmp = df_events.groupby(["corner_team", "phase"], dropna=False).size().reset_index(name="corners")
    pivot = tmp.pivot(index="corner_team", columns="phase", values="corners").fillna(0)
    phase_order = [p for p in ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"] if p in pivot.columns]
    pivot = pivot.reindex(columns=phase_order)
    fig = px.imshow(pivot, aspect="auto", title=title, labels=dict(x="Phase", y="Team", color="Corners"), text_auto=True)
    fig.update_layout(height=max(380, 40 * max(6, len(pivot.index))))
    return figure_layout(fig, height=max(380, 40 * max(6, len(pivot.index))), title=title)


def end_zone_bar_figure(df_events, group_col="corner_team", title="End Zone Volume"):
    if df_events.empty:
        return go.Figure()
    summary = df_events.groupby([group_col, "end_zone"], dropna=False).size().reset_index(name="corners")
    fig = px.bar(summary, x=group_col, y="corners", color="end_zone", title=title)
    return figure_layout(fig, height=420, title=title)


def minute_histogram_figure(df_events, title="Corner Minute Distribution"):
    if df_events.empty:
        return go.Figure()
    fig = px.histogram(df_events, x="Minute", nbins=20, title=title)
    return figure_layout(fig, height=380, title=title)


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


def add_advanced_features(source_df):
    df2 = source_df.copy()
    if df2.empty:
        for col, dtype in {
            "venue_split": "object",
            "delivery_length_band": "object",
            "xg_created": "float",
            "goal_from_corner": "bool",
            "delivery_success_proxy": "bool",
        }.items():
            if col not in df2.columns:
                df2[col] = pd.Series(dtype=dtype)
        return df2
    df2["venue_split"] = np.where(df2["is_home_corner"], "Home", np.where(df2["is_away_corner"], "Away", "Unknown"))
    df2["delivery_length_band"] = pd.cut(df2["delivery_length"], bins=[-0.1, 8, 16, 28, 200], labels=["Short", "Medium", "Long", "Very Long"], right=True).astype(str)
    df2["xg_created"] = df2["shot_xg"].fillna(0)
    df2["goal_from_corner"] = df2["shot_outcome"].astype(str).str.contains("goal", case=False, na=False)
    df2["delivery_success_proxy"] = df2["led_to_shot"].fillna(False) | df2["is_first_contact_shot"].fillna(False) | df2["is_goal_kick_zone_delivery"].fillna(False)
    return df2


def team_insight_table(source_df):
    if source_df.empty:
        return pd.DataFrame()
    out = (
        source_df.groupby(["corner_team", "delivery_zone", "end_zone"], dropna=False)
        .agg(corners=("match_id", "size"), shots=("led_to_shot", "sum"), total_xg=("shot_xg", "sum"), fast_shots=("is_fast_shot", "sum"))
        .reset_index()
    )
    out["shot_rate"] = out["shots"] / out["corners"].replace(0, np.nan)
    out["xg_per_corner"] = out["total_xg"] / out["corners"].replace(0, np.nan)
    return out.sort_values(["corner_team", "corners"], ascending=[True, False])


def taker_summary_table(source_df):
    if source_df.empty:
        return pd.DataFrame()
    out = (
        source_df.groupby(["corner_team", "Taker"], dropna=False)
        .agg(
            corners=("match_id", "size"),
            shots=("led_to_shot", "sum"),
            fast_shots=("is_fast_shot", "sum"),
            first_contact_shots=("is_first_contact_shot", "sum"),
            goals=("goal_from_corner", "sum"),
            total_xg=("xg_created", "sum"),
            inswingers=("is_inswinger", "sum"),
            outswingers=("is_outswinger", "sum"),
            short_corners=("is_short_corner", "sum"),
            six_yard_deliveries=("is_six_yard_delivery", "sum"),
            penalty_area_deliveries=("is_penalty_area_delivery", "sum"),
        )
        .reset_index()
    )
    out["shot_rate"] = out["shots"] / out["corners"].replace(0, np.nan)
    out["xg_per_corner"] = out["total_xg"] / out["corners"].replace(0, np.nan)
    out["goal_rate"] = out["goals"] / out["corners"].replace(0, np.nan)
    return out.sort_values(["corners", "total_xg"], ascending=False)


def match_pattern_table(source_df):
    if source_df.empty:
        return pd.DataFrame()
    out = (
        source_df.groupby(["Match", "corner_team", "venue_split"], dropna=False)
        .agg(corners=("match_id", "size"), shots=("led_to_shot", "sum"), fast_shots=("is_fast_shot", "sum"), goals=("goal_from_corner", "sum"), total_xg=("xg_created", "sum"), six_yard_deliveries=("is_six_yard_delivery", "sum"), short_corners=("is_short_corner", "sum"))
        .reset_index()
    )
    out["shot_rate"] = out["shots"] / out["corners"].replace(0, np.nan)
    out["xg_per_corner"] = out["total_xg"] / out["corners"].replace(0, np.nan)
    return out.sort_values(["total_xg", "corners"], ascending=False)


def team_report_card(source_df, league_team_df, selected_team_name):
    if source_df.empty or selected_team_name == "All Teams":
        return pd.DataFrame()
    team_row = league_team_df[league_team_df["team"] == selected_team_name]
    if team_row.empty:
        return pd.DataFrame()
    row = team_row.iloc[0]
    metrics = [
        ("Corners/Match", row.get("corners_per_match"), percentile_rank(league_team_df["corners_per_match"], row.get("corners_per_match"))),
        ("Shot Rate", row.get("shot_rate"), percentile_rank(league_team_df["shot_rate"], row.get("shot_rate"))),
        ("xG/Match", row.get("xg_per_match"), percentile_rank(league_team_df["xg_per_match"], row.get("xg_per_match"))),
        ("Fast Shot Rate", row.get("fast_shot_rate"), percentile_rank(league_team_df["fast_shot_rate"], row.get("fast_shot_rate"))),
        ("6Y Delivery Rate", row.get("six_yard_delivery_rate"), percentile_rank(league_team_df["six_yard_delivery_rate"], row.get("six_yard_delivery_rate"))),
        ("Short Corner Rate", row.get("short_corner_rate"), percentile_rank(league_team_df["short_corner_rate"], row.get("short_corner_rate"))),
    ]
    return pd.DataFrame(metrics, columns=["metric", "value", "percentile"])

# =========================================================
# DATA LOAD / PREP
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

    required = {"match_id": match_id_col, "match": match_col, "pass_team_name": team_col, "minute": minute_col, "second": second_col}
    missing = [name for name, col in required.items() if col is None]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    rename_map = {match_id_col: "match_id", match_col: "Match", team_col: "corner_team", minute_col: "Minute", second_col: "Second"}
    optional_map = {
        outcome_col: "SP_outcome", xg_col: "shot_xg", taker_col: "Taker", shooter_col: "Shooter", defensive_setup_col: "Defensive_setup",
        shot_team_col: "shot_team_name", pass_x_col: "pass_location_x", pass_y_col: "pass_location_y", pass_end_x_col: "pass_end_location_x",
        pass_end_y_col: "pass_end_location_y", shot_x_col: "shot_location_x", shot_y_col: "shot_location_y", shot_z_col: "shot_location_z",
        pass_technique_col: "pass_technique", pass_height_col: "pass_height", pass_body_col: "pass_body_part", shot_body_col: "shot_body_part",
        shot_outcome_col: "shot_outcome", pass_outcome_col: "pass_outcome", pass_position_col: "pass_position", shot_position_col: "shot_position",
    }
    for k, v in optional_map.items():
        if k is not None:
            rename_map[k] = v
    df = df.rename(columns=rename_map)

    defaults = {
        "SP_outcome": "", "shot_xg": np.nan, "Taker": np.nan, "Shooter": np.nan, "Defensive_setup": np.nan, "shot_team_name": np.nan,
        "pass_location_x": np.nan, "pass_location_y": np.nan, "pass_end_location_x": np.nan, "pass_end_location_y": np.nan,
        "shot_location_x": np.nan, "shot_location_y": np.nan, "shot_location_z": np.nan, "pass_technique": np.nan, "pass_height": np.nan,
        "pass_body_part": np.nan, "shot_body_part": np.nan, "shot_outcome": np.nan, "pass_outcome": np.nan, "pass_position": np.nan, "shot_position": np.nan,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    numeric_cols = ["Minute", "Second", "shot_xg", "pass_location_x", "pass_location_y", "pass_end_location_x", "pass_end_location_y", "shot_location_x", "shot_location_y", "shot_location_z"]
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
    df["delivery_length"] = df.apply(lambda r: delivery_length(r["pass_location_x"], r["pass_location_y"], r["pass_end_location_x"], r["pass_end_location_y"]), axis=1)
    df["end_zone"] = df.apply(lambda r: zone_from_end_location(r["pass_end_location_x"], r["pass_end_location_y"]), axis=1)
    df["is_goal_kick_zone_delivery"] = ((df["pass_end_location_x"].between(114, 120, inclusive="both")) & (df["pass_end_location_y"].between(30, 50, inclusive="both")))
    df["is_six_yard_delivery"] = df["end_zone"].eq("6-yard box")
    df["is_penalty_area_delivery"] = df["end_zone"].eq("Penalty area")
    df["phase"] = pd.cut(df["event_minute"], bins=[-0.1, 15, 30, 45, 60, 75, 120], labels=["0-15", "16-30", "31-45", "46-60", "61-75", "76+"], right=True).astype(str)

    match_summary = (
        df.groupby(["match_id", "Match", "home_team", "away_team"], dropna=False)
        .agg(total_corners=("match_id", "size"), shots_from_corners=("led_to_shot", "sum"), first_contact_shots=("is_first_contact_shot", "sum"), fast_shots=("is_fast_shot", "sum"), total_xg=("shot_xg", "sum"), avg_xg=("shot_xg", "mean"), unique_takers=("Taker", pd.Series.nunique), inswingers=("is_inswinger", "sum"), outswingers=("is_outswinger", "sum"), short_corners=("is_short_corner", "sum"), six_yard_deliveries=("is_six_yard_delivery", "sum"), penalty_area_deliveries=("is_penalty_area_delivery", "sum"))
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
try:
    raw_df = load_data()
    df, match_summary, team_summary = prepare_data(raw_df)
except Exception as e:
    st.error("Failed to load or prepare the Excel file.")
    st.exception(e)
    st.stop()

st.markdown(
    '<div class="hero">'
    '<div class="hero-title">⚽ Allsvenskan Set Piece Studio</div>'
    '<div class="hero-sub">A cleaner split between <b>Data</b> and <b>Data Visualisation</b>, with sharper dashboard visuals, clearer navigation, and more analyst-friendly tables.</div>'
    '<div style="margin-top:10px">'
    '<span class="pill">Executive Dashboard</span>'
    '<span class="pill">Visualisation Studio</span>'
    '<span class="pill">Data Hub</span>'
    '<span class="pill">Match Explorer</span>'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# =========================================================
# SIDEBAR FILTERS
# =========================================================
st.sidebar.title("Studio Controls")
page = st.sidebar.radio("Go to", ["Executive Dashboard", "Visualisation Studio", "Team Analysis", "Match Explorer", "Data Hub"])

all_teams = sorted([t for t in team_summary["team"].dropna().unique().tolist() if str(t).strip()])
selected_team = st.sidebar.selectbox("Team", ["All Teams"] + all_teams)
all_takers = sorted([str(t) for t in df["Taker"].dropna().astype(str).unique().tolist() if str(t).strip()])
selected_takers = st.sidebar.multiselect("Taker", all_takers)
all_matches = sorted(df["Match"].dropna().astype(str).unique().tolist())
selected_matches = st.sidebar.multiselect("Matches", all_matches)

minute_min = int(df["Minute"].min()) if not df["Minute"].dropna().empty else 0
minute_max = int(df["Minute"].max()) if not df["Minute"].dropna().empty else 120
minute_range = (minute_min, minute_max) if minute_max <= minute_min else st.sidebar.slider("Minute Range", min_value=minute_min, max_value=minute_max, value=(minute_min, minute_max))

min_corners = int(match_summary["total_corners"].min()) if len(match_summary) > 0 else 0
max_corners = int(match_summary["total_corners"].max()) if len(match_summary) > 0 else 0
corner_range = (min_corners, max_corners) if max_corners <= min_corners else st.sidebar.slider("Match Corner Range", min_value=min_corners, max_value=max_corners, value=(min_corners, max_corners))

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
venue_filter = st.sidebar.multiselect("Home / Away", ["Home", "Away", "Unknown"], default=["Home", "Away", "Unknown"])
phase_filter = st.sidebar.multiselect("Phase", ["0-15", "16-30", "31-45", "46-60", "61-75", "76+"], default=["0-15", "16-30", "31-45", "46-60", "61-75", "76+"])
outcome_filter = st.sidebar.multiselect("Outcome Bucket", ["Shot ≤3s", "First Contact Shot", "Shot", "No First Contact", "Other", "Unknown"], default=["Shot ≤3s", "First Contact Shot", "Shot", "No First Contact", "Other", "Unknown"])

# =========================================================
# GLOBAL FILTERS
# =========================================================
league_match_df = match_summary[(match_summary["total_corners"] >= corner_range[0]) & (match_summary["total_corners"] <= corner_range[1])].copy()
league_event_df = df[df["match_id"].isin(league_match_df["match_id"].unique())].copy()
league_event_df = add_advanced_features(league_event_df)
league_event_df = league_event_df[(league_event_df["Minute"].fillna(0) >= minute_range[0]) & (league_event_df["Minute"].fillna(0) <= minute_range[1])]

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
if venue_filter:
    league_event_df = league_event_df[league_event_df["venue_split"].isin(venue_filter)]
if phase_filter:
    league_event_df = league_event_df[league_event_df["phase"].isin(phase_filter)]
if outcome_filter:
    league_event_df = league_event_df[league_event_df["outcome_bucket"].isin(outcome_filter)]

league_match_df = league_match_df[league_match_df["match_id"].isin(league_event_df["match_id"].unique())]
league_team_df = build_team_summary(league_event_df) if not league_event_df.empty else build_team_summary(add_advanced_features(df.iloc[0:0].copy()))

# =========================================================
# PAGE: EXECUTIVE DASHBOARD
# =========================================================
if page == "Executive Dashboard":
    render_kpis(league_event_df, league_match_df)

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown('<div class="section-title">Corner Volume by Team</div><div class="section-sub">Top-level volume view for filtered sample</div>', unsafe_allow_html=True)
        if not league_team_df.empty:
            fig = px.bar(league_team_df.sort_values("corners_taken", ascending=False), x="team", y="corners_taken", hover_data=["matches", "corners_per_match", "shots_from_corners", "shot_rate", "xg_per_match"])
            st.plotly_chart(figure_layout(fig, 420), use_container_width=True)
    with c2:
        st.markdown('<div class="section-title">Efficiency Map</div><div class="section-sub">Shot rate versus xG per match</div>', unsafe_allow_html=True)
        if not league_team_df.empty:
            fig = px.scatter(league_team_df, x="shot_rate", y="xg_per_match", size="corners_taken", hover_name="team", hover_data=["matches", "corners_per_match", "fast_shot_rate", "box_delivery_rate"])
            st.plotly_chart(figure_layout(fig, 420), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(outcome_pie_figure(league_event_df, title="Outcome Split"), use_container_width=True)
    with c4:
        st.plotly_chart(technique_pie_figure(league_event_df, title="Delivery Technique Split"), use_container_width=True)

    st.markdown('<div class="section-title">League Match Board</div><div class="section-sub">This is the analysis table part of the overview</div>', unsafe_allow_html=True)
    board_cols = [c for c in ["Match", "home_team", "away_team", "home_corners", "away_corners", "total_corners", "shots_from_corners", "fast_shots", "total_xg", "shot_rate"] if c in league_match_df.columns]
    st.dataframe(league_match_df[board_cols].sort_values(["total_corners", "shots_from_corners"], ascending=False).reset_index(drop=True), use_container_width=True, height=420)

# =========================================================
# PAGE: VISUALISATION STUDIO
# =========================================================
elif page == "Visualisation Studio":
    st.markdown('<div class="section-title">Data Visualisation Studio</div><div class="section-sub">All charts and pitch visuals live here. The tables live in Data Hub.</div>', unsafe_allow_html=True)
    viz_tabs = st.tabs(["Shotmap", "Delivery Map", "Team Comparison", "Timing", "Zones"])

    with viz_tabs[0]:
        shot_df = league_event_df.dropna(subset=["shot_location_x", "shot_location_y"]).copy()
        shot_color = st.selectbox("Shotmap color by", ["corner_team", "Shooter", "Taker", "pass_technique"], index=0)
        st.plotly_chart(shotmap_figure(shot_df, color_col=shot_color, title="League Shotmap — Corner Shots"), use_container_width=True)

    with viz_tabs[1]:
        delivery_color = st.selectbox("Delivery map color by", ["delivery_zone", "end_zone", "pass_technique", "corner_team", "Taker"], index=0)
        st.plotly_chart(delivery_map_figure(league_event_df, color_col=delivery_color, title="League Delivery Map — Corners"), use_container_width=True)
        heatmap_fig = delivery_end_heatmap(league_event_df, title="Delivery End-Location Heatmap")
        if len(heatmap_fig.data) > 0:
            st.plotly_chart(heatmap_fig, use_container_width=True)

    with viz_tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(team_scatter_figure(league_team_df, "corners_per_match", "shot_rate", "corners_taken", "Corners per Match vs Shot Rate"), use_container_width=True)
        with c2:
            st.plotly_chart(team_scatter_figure(league_team_df, "six_yard_delivery_rate", "xg_per_match", "shots_from_corners", "6Y Delivery Rate vs xG per Match"), use_container_width=True)

    with viz_tabs[3]:
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(cumulative_timeline_figure(league_event_df, color_col="corner_team", title="League Cumulative Corners by Team"), use_container_width=True)
        with c2:
            st.plotly_chart(minute_histogram_figure(league_event_df, title="League Corner Minute Distribution"), use_container_width=True)
        fig = phase_heatmap_figure(league_event_df, title="League Timing Heatmap")
        if len(fig.data) > 0:
            st.plotly_chart(fig, use_container_width=True)

    with viz_tabs[4]:
        fig = end_zone_bar_figure(league_event_df, group_col="corner_team", title="Team End-Zone Volume")
        if len(fig.data) > 0:
            st.plotly_chart(fig, use_container_width=True)

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
        team_tabs = st.tabs(["Snapshot", "Visuals", "Takers", "Matches", "Report Card"])

        with team_tabs[0]:
            render_kpis(team_event_df, team_match_df)
            zone_df = team_event_df.groupby("end_zone", dropna=False).size().reset_index(name="corners").sort_values("corners", ascending=False)
            fig = px.bar(zone_df, x="end_zone", y="corners", title=f"{selected_team} End-Zone Profile")
            st.plotly_chart(figure_layout(fig, 380), use_container_width=True)

        with team_tabs[1]:
            c1, c2 = st.columns(2)
            with c1:
                team_shots = team_event_df.dropna(subset=["shot_location_x", "shot_location_y"]).copy()
                st.plotly_chart(shotmap_figure(team_shots, color_col="Shooter", title=f"Shotmap — {selected_team}"), use_container_width=True)
            with c2:
                st.plotly_chart(delivery_map_figure(team_event_df, color_col="delivery_zone", title=f"Delivery Map — {selected_team}"), use_container_width=True)

        with team_tabs[2]:
            st.dataframe(taker_summary_table(team_event_df).reset_index(drop=True), use_container_width=True, height=500)

        with team_tabs[3]:
            st.dataframe(match_pattern_table(team_event_df).reset_index(drop=True), use_container_width=True, height=500)

        with team_tabs[4]:
            report_df = team_report_card(team_event_df, league_team_df, selected_team)
            st.dataframe(report_df, use_container_width=True, height=320)
            if not team_row_df.empty:
                row = team_row_df.iloc[0]
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    metric_card("Volume Percentile", f"{percentile_rank(league_team_df['corners_per_match'], row['corners_per_match']):.0f}", "th")
                with c2:
                    metric_card("Shot Rate Percentile", f"{percentile_rank(league_team_df['shot_rate'], row['shot_rate']):.0f}", "th")
                with c3:
                    metric_card("xG / Match Percentile", f"{percentile_rank(league_team_df['xg_per_match'], row['xg_per_match']):.0f}", "th")
                with c4:
                    metric_card("6Y Delivery Rate", f"{row['six_yard_delivery_rate'] * 100:.1f}", "%")

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

    tabs = st.tabs(["Summary", "Timeline", "Shotmap", "Delivery Map", "Event Feed"])
    with tabs[0]:
        render_kpis(match_event_df, match_board_df)
        board_cols = [c for c in ["Match", "home_team", "away_team", "home_corners", "away_corners", "total_corners", "shots_from_corners", "fast_shots", "total_xg", "shot_rate"] if c in match_board_df.columns]
        st.dataframe(match_board_df[board_cols].reset_index(drop=True), use_container_width=True, height=320)
    with tabs[1]:
        if not match_event_df.empty:
            minute_df = match_event_df.groupby("Minute", dropna=False).size().reset_index(name="corner_events").sort_values("Minute")
            fig = px.bar(minute_df, x="Minute", y="corner_events", title="Corner Timeline")
            st.plotly_chart(figure_layout(fig, 400), use_container_width=True)
    with tabs[2]:
        st.plotly_chart(shotmap_figure(match_event_df.dropna(subset=["shot_location_x", "shot_location_y"]), color_col="corner_team", title=f"Shotmap — {selected_match}"), use_container_width=True)
    with tabs[3]:
        st.plotly_chart(delivery_map_figure(match_event_df, color_col="corner_team", title=f"Delivery Map — {selected_match}"), use_container_width=True)
    with tabs[4]:
        show_cols = [c for c in ["Match", "corner_team", "Taker", "Shooter", "Minute", "Second", "SP_outcome", "shot_xg", "Defensive_setup", "pass_technique", "delivery_zone", "end_zone"] if c in match_event_df.columns]
        st.dataframe(match_event_df[show_cols].sort_values(["Minute", "Second"]).reset_index(drop=True), use_container_width=True, height=560)

# =========================================================
# PAGE: DATA HUB
# =========================================================
elif page == "Data Hub":
    st.markdown('<div class="section-title">Data Hub</div><div class="section-sub">This is the data part. Raw tables, summaries, filtering output, and export-ready views.</div>', unsafe_allow_html=True)
    tabs = st.tabs(["Raw Events", "Team Table", "Match Table", "Shot Events", "Delivery Events", "Taker Summary", "Zone Table"])

    with tabs[0]:
        st.dataframe(league_event_df.reset_index(drop=True), use_container_width=True, height=620)
    with tabs[1]:
        st.dataframe(league_team_df.reset_index(drop=True), use_container_width=True, height=620)
    with tabs[2]:
        st.dataframe(league_match_df.reset_index(drop=True), use_container_width=True, height=620)
    with tabs[3]:
        st.dataframe(league_event_df[league_event_df["led_to_shot"]].reset_index(drop=True), use_container_width=True, height=620)
    with tabs[4]:
        delivery_cols = [c for c in ["Match", "corner_team", "Taker", "Minute", "pass_technique", "pass_location_x", "pass_location_y", "pass_end_location_x", "pass_end_location_y", "delivery_zone", "end_zone", "delivery_length_band", "SP_outcome"] if c in league_event_df.columns]
        st.dataframe(league_event_df[delivery_cols].reset_index(drop=True), use_container_width=True, height=620)
    with tabs[5]:
        st.dataframe(taker_summary_table(league_event_df).reset_index(drop=True), use_container_width=True, height=620)
    with tabs[6]:
        st.dataframe(team_insight_table(league_event_df).reset_index(drop=True), use_container_width=True, height=620)

    csv = league_event_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered events as CSV", data=csv, file_name="filtered_corner_events.csv", mime="text/csv")
