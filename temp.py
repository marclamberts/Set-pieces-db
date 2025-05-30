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

# Filter for upper half goals from selected set piece
filtered_df = df[
    (df["outcome_name"] == "Goal") &
    (df["play_pattern_name"] == selected_pattern) &
    (df["location_x"] >= 60)  # Only show upper/attacking half
].copy()

# Convert StatsBomb (x=length, y=width) to vertical layout (y becomes x)
x = filtered_df["location_y"]    # width → horizontal axis
y = filtered_df["location_x"]    # length → vertical axis

# Create plotly figure
plot = go.Figure()

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

# Pitch shapes for upper attacking half (StatsBomb: 0–120 x, 0–80 y)
pitch_shapes = [
    # Outer box
    dict(type="rect", x0=0, y0=60, x1=80, y1=120, line=dict(color="white", width=2)),

    # Penalty box
    dict(type="rect", x0=18, y0=102, x1=62, y1=120, line=dict(color="white", width=1)),

    # Six-yard box
    dict(type="rect", x0=30, y0=114, x1=50, y1=120, line=dict(color="white", width=1)),

    # Goal line
    dict(type="line", x0=36, y0=120, x1=44, y1=120, line=dict(color="white", width=4)),

    # Penalty spot
    dict(type="circle", x0=38.5, y0=108.5, x1=41.5, y1=111.5, line=dict(color="white", width=1)),
]

# Final layout with vertical pitch attacking top
plot.update_layout(
    title=f"Goals from {selected_pattern}",
    xaxis=dict(range=[0, 80], showgrid=False, zeroline=False, visible=False),
    yaxis=dict(range=[120, 60], showgrid=False, zeroline=False, visible=False),  # Y reversed
    shapes=pitch_shapes,
    plot_bgcolor='#144A29',
    paper_bgcolor='#144A29',
    height=700,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(plot, use_container_width=True)
