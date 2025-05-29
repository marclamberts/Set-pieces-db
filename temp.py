import streamlit as st
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Styled Vertical Half Pitch", layout="centered")

# Title
st.title("Styled Vertical Half Pitch")

# Create custom styled pitch
pitch = VerticalPitch(
    half=True,
    pitch_type='statsbomb',
    pitch_color='#144A29',       # dark green
    line_color='#ffffff',        # white lines
    line_zorder=2,
    linewidth=2,
    stripe=True,                 # adds grass stripes
    stripe_color='#0E3B1E'       # darker green for stripes
)

# Draw the pitch
fig, ax = pitch.draw(figsize=(6, 8))

# Show in Streamlit
st.pyplot(fig)
