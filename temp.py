
import streamlit as st
from mplsoccer import Pitch
import matplotlib.pyplot as plt

st.set_page_config(page_title="Vertical Half Pitch with mplsoccer", layout="centered")
st.title("⚽ Vertical Half Pitch (Using mplsoccer)")

# Create a vertical half pitch
pitch = Pitch(pitch_type='statsbomb', pitch_color='#144A29',
              line_color='white', half=True, orientation='vertical')

fig, ax = pitch.draw(figsize=(6, 8))

st.pyplot(fig)
