import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Interactive Vertical Pitch", layout="centered")
st.title("⚽ Interactive Goal Map by Set Piece Type")

# Load local CSV
csv_path = os.path.join(os.path.dirname(__file__), "events.csv")
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    st.error("CSV file not found. Make sure 'events.csv' is in the same folder as this script.")
    st.stop()

# Ensure required columns exist
required_cols = {
    "outcome_name", "location_x", "location_y", "play_pattern_name",
    "player_name", "team_name", "player_position_name", "statsbomb_xg"
}
if not required_cols.issubset(df.columns):
    st.error(f"CSV must contain columns: {', '.join(required_cols)}")
    st.stop()

# Sidebar filter
set_piece_options = ["From Corner", "From Throw In", "From Free Kick"]
selected_pattern = st.sidebar.selectbox("Set Piece Type", set_piece_options)

# Filter data
filtered_df = df[
    (df["outcome_name"] == "Goal") &
    (df["play_pattern_name"] == selected_pattern)
].copy()

# Swap axes to make it vertical (StatsBomb pitch: x is vertical)
x = filtered_df["location_y"]
y = filtered_df["location_x"]

# Create Plotly figure
plot = go.Figure()

# Add goal markers with hover text
plot.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(size=10, color='gold', line=dict(color='black', width=1.5)),
    text=[
        f"Player: {p}<br>Team: {t}<br>Position: {pos}<br>xG: {xg:.2f}"
        for p, t, pos, xg in zip(
            filtered_df["player_name"],
            filtered_df["team_name"],
            filtered_df["player_position_name"],
            filtered_df["statsbomb_xg"]
        )
    ],
    hoverinfo='text',
    name='Goals'
))

# Manual pitch shapes (half pitch — attacking up)
pitch_shapes = [
    # Outer boundaries
    dict(type="rect", x0=0, y0=0, x1=80, y1=60, line=dict(color="white", width=2)),

    # Penalty box
    dict(type="rect", x0=18, y0=42, x1=62, y1=60, line=dict(color="white", width=1)),

    # 6-yard box
    dict(type="rect", x0=30, y0=54, x1=50, y1=60, line=dict(color="white", width=1)),

    # Goal line
    dict(type="line", x0=36, y0=60, x1=44, y1=60, line=dict(color="white", width=4)),

    # Penalty spot
    dict(type="circle", x0=38.5, y0=48.5, x1=41.5, y1=51.5, line=dict(color="white", width=1)),

    # Optional arc or center line could go here
]

# Layout for upper half pitch (flip y-axis)
plot.update_layout(
    title=f"Goals from {selected_pattern}",
    xaxis=dict(range=[0, 80], showgrid=False, zeroline=False, visible=False),
    yaxis=dict(range=[60, 0], showgrid=False, zeroline=False, visible=False),  # flipped Y-axis
    shapes=pitch_shapes,
    plot_bgcolor='#144A29',
    paper_bgcolor='#144A29',
    height=700,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(plot, use_container_width=True)
