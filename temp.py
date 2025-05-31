import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import ast

st.set_page_config(page_title="Interactive Goal Map", layout="centered")
st.title("⚽ Goal Map by Set Piece Type")

# Load and merge Excel files
def load_data():
    path_1 = os.path.join(os.path.dirname(__file__), "events2.xlsx")
    path_2 = os.path.join(os.path.dirname(__file__), "merged_output.xlsx")  # optional second file
    try:
        df1 = pd.read_excel(path_1)
    except FileNotFoundError:
        st.error("File 'events2.xlsx' not found.")
        st.stop()
    try:
        df2 = pd.read_excel(path_2)
        df = pd.concat([df1, df2], ignore_index=True)
    except FileNotFoundError:
        df = df1  # fallback if only one file exists
    return df

df = load_data()

# Parse location column
def parse_location(loc):
    try:
        return ast.literal_eval(loc) if isinstance(loc, str) else loc
    except:
        return [None, None, None]

df[['location_x', 'location_y', 'location_z']] = df['location'].apply(parse_location).apply(pd.Series)

# Ensure all necessary columns exist
required_cols = {
    "shot.outcome.name", "location_x", "location_y", "play_pattern.name",
    "team.name", "position.name", "shot.statsbomb_xg", "Match",
    "shot.first_time", "shot.body_part.name",
    "competition.country_name", "competition.competition_name", "season.season_name"
}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    st.error(f"Missing required columns: {', '.join(missing_cols)}")
    st.stop()

# Base filter: goals in upper half
filtered_df = df[(df["shot.outcome.name"] == "Goal") & (df["location_x"] >= 60)].copy()

# Sidebar filters
st.sidebar.header("Filters")
set_piece_options = ["All", "From Corner", "From Throw In", "From Free Kick"]
selected_pattern = st.sidebar.selectbox("Set Piece Type", set_piece_options)

# First-time shot filter
first_time_options = ["All", "True", "False"]
selected_first_time = st.sidebar.selectbox("First-Time Shot", first_time_options)

# Body part filter
body_parts = ["All"] + sorted(filtered_df["shot.body_part.name"].dropna().unique())
selected_body_part = st.sidebar.selectbox("Body Part Used", body_parts)

# Team, Match, Position filters
teams = ["All"] + sorted(filtered_df["team.name"].dropna().unique())
matches = ["All"] + sorted(filtered_df["Match"].dropna().unique())
positions = ["All"] + sorted(filtered_df["position.name"].dropna().unique())

selected_team = st.sidebar.selectbox("Team", teams)
selected_match = st.sidebar.selectbox("Match", matches)
selected_position = st.sidebar.selectbox("Position", positions)

# Nation, League, Season filters
nations = ["All"] + sorted(filtered_df["competition.country_name"].dropna().unique())
leagues = ["All"] + sorted(filtered_df["competition.competition_name"].dropna().unique())
seasons = ["All"] + sorted(filtered_df["season.season_name"].dropna().unique())

selected_nation = st.sidebar.selectbox("Nation", nations)
selected_league = st.sidebar.selectbox("League", leagues)
selected_season = st.sidebar.selectbox("Season", seasons)

# xG Range slider
min_xg = float(filtered_df["shot.statsbomb_xg"].min())
max_xg = float(filtered_df["shot.statsbomb_xg"].max())
xg_range = st.sidebar.slider("xG Range", min_value=0.0, max_value=round(max_xg + 0.05, 2),
                             value=(round(min_xg, 2), round(max_xg, 2)), step=0.01)

# Apply filters
if selected_pattern != "All":
    filtered_df = filtered_df[filtered_df["play_pattern.name"] == selected_pattern]
if selected_team != "All":
    filtered_df = filtered_df[filtered_df["team.name"] == selected_team]
if selected_match != "All":
    filtered_df = filtered_df[filtered_df["Match"] == selected_match]
if selected_position != "All":
    filtered_df = filtered_df[filtered_df["position.name"] == selected_position]
if selected_first_time != "All":
    filtered_df = filtered_df[filtered_df["shot.first_time"] == (selected_first_time == "True")]
if selected_body_part != "All":
    filtered_df = filtered_df[filtered_df["shot.body_part.name"] == selected_body_part]
if selected_nation != "All":
    filtered_df = filtered_df[filtered_df["competition.country_name"] == selected_nation]
if selected_league != "All":
    filtered_df = filtered_df[filtered_df["competition.competition_name"] == selected_league]
if selected_season != "All":
    filtered_df = filtered_df[filtered_df["season.season_name"] == selected_season]

filtered_df = filtered_df[filtered_df["shot.statsbomb_xg"].between(xg_range[0], xg_range[1])]

if filtered_df.empty:
    st.warning("No goals found for this combination of filters.")
    st.stop()

# Optional data table
if st.checkbox("Show data table"):
    st.dataframe(filtered_df[[
        "team.name", "Match", "position.name", 
        "play_pattern.name", "shot.first_time", "shot.body_part.name",
        "shot.statsbomb_xg", "location_x", "location_y",
        "competition.country_name", "competition.competition_name", "season.season_name"
    ]])

# Prepare pitch coordinates
x = filtered_df["location_y"]
y = filtered_df["location_x"] - 60

# Hover text
hover_texts = [
    f"<b>Team:</b> {t}<br><b>Match:</b> {m}<br><b>Pos:</b> {p}<br><b>xG:</b> {xg:.2f}"
    f"<br><b>First-Time:</b> {ft}<br><b>Body:</b> {bp}"
    for t, m, p, xg, ft, bp in zip(
        filtered_df["team.name"],
        filtered_df["Match"],
        filtered_df["position.name"],
        filtered_df["shot.statsbomb_xg"],
        filtered_df["shot.first_time"],
        filtered_df["shot.body_part.name"]
    )
]

# Create Plotly pitch (Washington Post-style)
plot = go.Figure()

plot.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(size=10, color='#FFD700', line=dict(color='black', width=1.2)),
    text=hover_texts,
    hoverinfo='text',
    name='Goals'
))

pitch_shapes = [
    dict(type="rect", x0=0, y0=0, x1=80, y1=60, line=dict(color="#CCCCCC", width=2)),
    dict(type="rect", x0=18, y0=42, x1=62, y1=60, line=dict(color="#AAAAAA", width=1)),
    dict(type="rect", x0=30, y0=54, x1=50, y1=60, line=dict(color="#AAAAAA", width=1)),
    dict(type="line", x0=36, y0=60, x1=44, y1=60, line=dict(color="#AAAAAA", width=4)),
    dict(type="circle", x0=38.5, y0=48.5, x1=41.5, y1=51.5, line=dict(color="#AAAAAA", width=1)),
]

plot.update_layout(
    title=dict(text=f"Goals from {selected_pattern}", font=dict(size=22, family="Georgia")),
    xaxis=dict(range=[0, 80], showgrid=False, zeroline=False, visible=False),
    yaxis=dict(range=[0, 60], showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
    shapes=pitch_shapes,
    plot_bgcolor='#F9F9F9',
    paper_bgcolor='#F9F9F9',
    height=700,
    margin=dict(l=20, r=20, t=50, b=20),
    font=dict(family="Georgia", size=14, color="#333333")
)

st.plotly_chart(plot, use_container_width=True)
