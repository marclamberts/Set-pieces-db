
import streamlit as st
import pandas as pd
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt
import os

st.set_page_config(page_title="Vertical Half Pitch with mplsoccer", layout="centered")
st.title("⚽ Vertical Half Pitch (Using mplsoccer)")

# Load local CSV
csv_path = os.path.join(os.path.dirname(__file__), "events.csv")  # Adjust if needed
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    st.error("CSV file not found. Make sure 'events.csv' is in the same folder as this script.")
    st.stop()

# Ensure required columns are present
required_cols = {"outcome_name", "location_x", "location_y"}
if not required_cols.issubset(df.columns):
    st.error(f"CSV must contain columns: {', '.join(required_cols)}")
    st.stop()

# Filter for goals
goals_df = df[df["outcome_name"] == "Goal"]

# Create vertical half pitch
pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#144A29',
                      line_color='white', half=True)
fig, ax = pitch.draw(figsize=(6, 8))

# Plot goal locations
pitch.scatter(goals_df["location_x"], goals_df["location_y"],
              ax=ax, s=100, color='gold', edgecolors='black', linewidth=1.5, zorder=3)

st.pyplot(fig)
