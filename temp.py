import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import ast

st.set_page_config(page_title="Interactive Vertical Pitch", layout="centered")
st.title("⚽ Interactive Goal Map by Set Piece Type")

# Load local Excel file
excel_path = os.path.join(os.path.dirname(__file__), "events2.xlsx")
try:
    df = pd.read_excel(excel_path)
except FileNotFoundError:
    st.error("Excel file not found. Make sure 'events2.xlsx' is in the same folder as this script.")
    st.stop()

# Function to parse 'location' column
def parse_location(loc):
    try:
        if isinstance(loc, str):
            return ast.literal_eval(loc)
        else:
            return loc
    except:
        return [None, None, None]

# Parse location into columns
df[['location_x', 'location_y', 'location_z']] = df['location'].apply(parse_location).apply(pd.Series)

# Check required columns
required_cols = {
    "shot.outcome.name", "location_x", "location_y", "play_pattern.name",
    "player.name", "team.name", "position.name", "shot.statsbomb_xg", "Match"
}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    st.error(f"Excel must contain columns: {', '.join(missing_cols)}")
    st.stop()

# Sidebar filter: Set piece type
set_piece_options = ["From Corner", "From Throw In", "From Free Kick"]
selected_pattern = st.sidebar.selectbox("Set Piece Type", set_piece_options)

# Initial filtering: only goals from selected set piece and upper half
filtered_df = df[
    (df["shot.outcome.name"] == "Goal") &
    (df["play_pattern.name"] == selected_pattern) &
    (df["location_x"] >= 60)
].copy()

if filtered_df.empty:
    st.warning("No goals found for this set piece and pitch half.")
    st.stop()

# Dropdown filters with "All" option
players = ["All"] + sorted(filtered_df["player.name"].dropna().unique())
teams = ["All"] + sorted(filtered_df["team.name"].dropna().unique())
matches = ["All"] + sorted(filtered_df["Match"].dropna().unique())
positions = ["All"] + sorted(filtered_df["position.name"].dropna().unique())

selected_player = st.sidebar.selectbox("Player", players, index=0)
selected_team = st.sidebar.selectbox("Team", teams, index=0)
selected_match = st.sidebar.selectbox("Match", matches, index=0)
selected_position = st.sidebar.selectbox("Position", positions, index=0)

# xG Range slider
min_xg = float(filtered_df["shot.statsbomb_xg"].min())
max_xg = float(filtered_df["shot.statsbomb_xg"].max())
xg_range = st.sidebar.slider("xG Range", min_value=0.0, max_value=round(max_xg + 0.05, 2),
                             value=(round(min_xg, 2), round(max_xg, 2)), step=0.01)

# Apply filters conditionally
if selected_player != "All":
    filtered_df = filtered_df[filtered_df["player.name"] == selected_player]
if selected_team != "All":
    filtered_df = filtered_df[filtered_df["team.name"] == selected_team]
if selected_match != "All":
    filtered_df = filtered_df[filtered_df["Match"] == selected_match]
if selected_position != "All":
    filtered_df = filtered_df[filtered_df["position.name"] == selected_position]

# xG range filter
filtered_df = filtered_df[
    filtered_df["shot.statsbomb_xg"].between(xg_range[0], xg_range[1])
]

if filtered_df.empty:
    st.warning("No goals found for this combination of filters.")
    st.stop()

# Optional data table
if st.checkbox("Show data table"):
    st.dataframe(filtered_df[[
        "player.name", "team.name", "Match", "position.name", 
        "play_pattern.name", "shot.statsbomb_xg", "location_x", "location_y"
    ]])

# Prepare coordinates
x = filtered_df["location_y"]
y = filtered_df["location_x"] - 60

# Hover text
hover_texts = [
    f"Player: {p}<br>Team: {t}<br>Position: {pos}<br>xG: {xg:.2f}<br>Match: {match}"
    for p, t, pos, xg, match in zip(
        filtered_df["player.name"],
        filtered_df["team.name"],
        filtered_df["position.name"],
        filtered_df["shot.statsbomb_xg"],
        filtered_df["Match"]
    )
]

# Build pitch
plot = go.Figure()

plot.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(size=10, color='gold', line=dict(color='black', width=1.5)),
    text=hover_texts,
    hoverinfo='text',
    name='Goals'
))

pitch_shapes = [
    dict(type="rect", x0=0, y0=0, x1=80, y1=60, line=dict(color="white", width=3)),
    dict(type="rect", x0=18, y0=42, x1=62, y1=60, line=dict(color="white", width=2)),
    dict(type="rect", x0=30, y0=54, x1=50, y1=60, line=dict(color="white", width=2)),
    dict(type="line", x0=36, y0=60, x1=44, y1=60, line=dict(color="white", width=5)),
    dict(type="circle", x0=38.5, y0=48.5, x1=41.5, y1=51.5, line=dict(color="white", width=2)),
]

plot.update_layout(
    title=f"Goals from {selected_pattern}",
    xaxis=dict(range=[0, 80], showgrid=False, zeroline=False, visible=False),
    yaxis=dict(range=[0, 60], showgrid=False, zeroline=False, visible=False, scaleanchor="x"),
    shapes=pitch_shapes,
    plot_bgcolor='#144A29',
    paper_bgcolor='#144A29',
    height=700,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(plot, use_container_width=True)
