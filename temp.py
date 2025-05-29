import streamlit as st
from mplsoccer import Pitch
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="Half Vertical Pitch", layout="centered")

# Title
st.title("⚽ Vertical Half Pitch (mplsoccer)")

# Create figure
pitch = Pitch(half=True, pitch_type='statsbomb', orientation='vertical')
fig, ax = pitch.draw(figsize=(6, 8))

# Display with Streamlit
st.pyplot(fig)
