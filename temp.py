import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set page config
st.set_page_config(page_title="Manual Vertical Half Pitch", layout="centered")
st.title("⚽ Vertical Half Pitch (No mplsoccer)")

# Create figure
fig, ax = plt.subplots(figsize=(6, 8))
ax.set_facecolor('#144A29')  # Pitch color

# Pitch dimensions (StatsBomb: 120 x 80 yards — use half)
pitch_length = 120
pitch_width = 80

# Draw outline
ax.add_patch(patches.Rectangle((0, 0), pitch_width, pitch_length / 2,
                               linewidth=2, edgecolor='white', facecolor='none'))

# Center circle (only part of it shows on half)
center_circle = plt.Circle((pitch_width / 2, pitch_length / 2), 10,
                           color='white', fill=False, linewidth=2)
ax.add_patch(center_circle)

# Penalty box
ax.add_patch(patches.Rectangle(((pitch_width - 44) / 2, 0), 44, 18,
                               linewidth=2, edgecolor='white', facecolor='none'))

# Six-yard box
ax.add_patch(patches.Rectangle(((pitch_width - 20) / 2, 0), 20, 6,
                               linewidth=2, edgecolor='white', facecolor='none'))

# Penalty spot
ax.plot(pitch_width / 2, 12, 'wo', markersize=3)

# Arc at top of penalty area
arc = patches.Arc((pitch_width / 2, 12), 20, 20, angle=0, theta1=308, theta2=232,
                  color='white', linewidth=2)
ax.add_patch(arc)

# Formatting
ax.set_xlim(0, pitch_width)
ax.set_ylim(0, pitch_length / 2)
ax.axis('off')

# Show in Streamlit
st.pyplot(fig)
