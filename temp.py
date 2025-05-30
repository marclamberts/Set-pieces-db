import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go
import ast

st.set_page_config(page_title="Interactive Vertical Pitch", layout="centered")
st.title("⚽ Interactive Goal Map by Set Piece Type")

# Load local Excel file
excel_path = os.path.join(os.path.dirname(__file__), "events.xlsx")
try:
    df = pd.read_excel(excel_path)
except FileNotFoundError:
    st.error("Excel file not found. Make sure 'events.xlsx' is in the same folder as this script.")
    st.stop()

# Function to parse 'location' column (e.g. '[x, y, z]' string) into list of floats
def parse_location(loc):
    try:
        if isinstance(loc, str):
            return ast.literal_eval(loc)
        else:
            return loc
    except:
        return [None, None, None]

# Split 'location' column into three columns
df[['location_x', 'location_y', 'location_z']] = df['location'].apply(parse_location).apply(pd.Series)

# Required columns to proceed
required_cols = {
    "shot.outcome.name", "location_x", "location_y", "play_pattern.name",
    "player.name", "team.name", "position.name", "shot.statsbomb_xg", "Match"
}

missing_cols = required_cols - set(df.columns)
if missing_cols:
    st.error(f"Excel must contain columns: {', '.join(missing_cols)}")
    st.stop()

# Sidebar filter for set piece type
set_piece_options = ["From Corner", "From Throw In", "From Free Kick"]
selected_pattern = st.sidebar.selectbox("Set Piece Type", set_piece_options)

# Filter for goals from selected set piece in upper pitch (location_x >= 60)
filtered_df = df[
    (df["shot.outcome.name"] == "Goal") &
    (df["play_pattern.name"] == selected_pattern) &
    (df["location_x"] >= 60)
].copy()

if filtered_df.empty:
    st.warning("No goals found for this set piece and pitch half.")
    st.stop()

# Prepare coordinates for plot
x = filtered_df["location_y"]        # pitch width 0-80
y = filtered_df["location_x"] - 60   # shift length 60-120 to 0-60

# Build hover text including Match info
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

# Create pitch plot
plot = go.Figure()

plot.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(
        size=10,
        color='gold',
        line=dict(color='black', width=1.5)
    ),
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
