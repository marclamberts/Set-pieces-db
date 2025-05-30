import streamlit as st
import pandas as pd
import os
from mplsoccer import VerticalPitch
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
    "player_name", "team_name", "player_position_name", "statsbombxg"
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

# Create pitch layout using mplsoccer (for consistency with StatsBomb coords)
pitch = VerticalPitch(pitch_type='statsbomb', half=True)
fig, ax = pitch.draw()
xlim, ylim = ax.get_xlim(), ax.get_ylim()  # Needed for consistent scaling
fig.clf()  # We only needed limits, clear the fig

# Create Plotly figure
plot = go.Figure()

# Add half pitch lines using shapes
# (Optional: You can use an image background instead)

# Add goal markers with tooltips
plot.add_trace(go.Scatter(
    x=filtered_df["location_x"],
    y=filtered_df["location_y"],
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

# Set pitch dimensions manually (StatsBomb half-pitch: 0–120 x, 0–80 y)
plot.update_layout(
    title=f"Goals from {selected_pattern}",
    xaxis=dict(range=[0, 120], showgrid=False, zeroline=False, visible=False),
    yaxis=dict(range=[0, 80], showgrid=False, zeroline=False, visible=False),
    plot_bgcolor='#144A29',
    paper_bgcolor='#144A29',
    height=700,
    margin=dict(l=20, r=20, t=40, b=20)
)

st.plotly_chart(plot, use_container_width=True)
