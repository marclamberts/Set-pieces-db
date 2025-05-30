import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Interactive Vertical Pitch", layout="centered")
st.title("⚽ Interactive Goal Map by Set Piece Type")

# Load local CSV
csv_path = os.path.join(os.path.dirname(__file__), "events.xlsx")
try:
    df = pd.read_excel(csv_path)
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

# Shift function to normalize pitch y-coordinates so goal line is at y=0 (top)
def shift_y(y_val):
    return y_val - 60

# Shift the data y values accordingly to match new pitch coordinates
y = filtered_df["location_x"] - 60  # shift location_x down by 60
x = filtered_df["location_y"]       # width stays same (0 to 80)

# Create plotly figure
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
    text=[
        f"Player: {p}<br>Team: {t}<br>Position: {pos}<br>xG: {xg:.2f}"
        for p, t, pos, xg in zip(
            filtered_df["player_name"],
            filtered_df["team_name"],
            filtered_df["player_position_name"],
            filtered_df["statsbomb_xg"],
            filtered_df["Match"]
        )
    ],
    hoverinfo='text',
    name='Goals'
))

# Sharp pitch shapes, all y-coordinates shifted by -60
pitch_shapes = [
    # Outer boundaries (80x60)
    dict(type="rect", x0=0, y0=0, x1=80, y1=60, line=dict(color="white", width=3)),

    # Penalty box
    dict(type="rect", x0=18, y0=42, x1=62, y1=60, line=dict(color="white", width=2)),

    # Six-yard box
    dict(type="rect", x0=30, y0=54, x1=50, y1=60, line=dict(color="white", width=2)),

    # Goal line
    dict(type="line", x0=36, y0=60, x1=44, y1=60, line=dict(color="white", width=5)),

    # Penalty spot (circle)
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
