
import streamlit as st
from mplsoccer import VerticalPitch
import matplotlib.pyplot as plt

st.set_page_config(page_title="Vertical Half Pitch with mplsoccer", layout="centered")
st.title("⚽ Vertical Half Pitch (Using mplsoccer)")

# Create a vertical half pitch
pitch = VerticalPitch(pitch_type='statsbomb', pitch_color='#144A29',
              line_color='white', half=True)

fig, ax = pitch.draw(figsize=(6, 8))

st.pyplot(fig)
