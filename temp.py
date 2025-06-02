import streamlit as st
import pandas as pd
import os
import ast
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mplsoccer import VerticalPitch

st.set_page_config(page_title="Throw-In Pass Visualizer", layout="wide")

@st.cache_data
def load_ti_data():
    base_path = os.path.dirname(__file__)
    df_ti = pd.read_excel(os.path.join(base_path, "TI.xlsx"))
    return df_ti

def extract_xy(loc):
    try:
        if isinstance(loc, str):
            loc = ast.literal_eval(loc)
        if isinstance(loc, (list, tuple)):
            return [loc[0] if len(loc) > 0 else None,
                    loc[1] if len(loc) > 1 else None]
    except Exception:
        pass
    return [None, None]

st.title("🌀 Throw-ins Leading to Shots")

# Load and filter data
ti = load_ti_data()
passes = ti[(ti["type.name"] == "Pass") & (ti["play_pattern.name"] == "From Throw In")].copy()
shots = ti[ti["type.name"] == "Shot"].copy()
throwins = passes[passes["possession"].isin(shots["possession"])].copy()

# Map xG from shots to throw-ins
xg_map = shots.groupby("possession")["shot.statsbomb_xg"].max()
throwins["shot_xG"] = throwins["possession"].map(xg_map)

# Extract coordinates
throwins[["location_x", "location_y"]] = throwins["location"].apply(extract_xy).apply(pd.Series)
throwins[["pass.end_location_x", "pass.end_location_y"]] = throwins["pass.end_location"].apply(extract_xy).apply(pd.Series)

# Draw pitch and plot arrows
pitch = VerticalPitch(pitch_type='statsbomb', half=True, pitch_color='white', line_color='black')
fig, ax = pitch.draw(figsize=(6, 5))  # Smaller pitch size

norm = mcolors.Normalize(vmin=throwins["shot_xG"].min(), vmax=throwins["shot_xG"].max())
cmap = cm.get_cmap("coolwarm")

for _, row in throwins.dropna(subset=["location_x", "location_y", "pass.end_location_x", "pass.end_location_y"]).iterrows():
    color = cmap(norm(row["shot_xG"])) if pd.notnull(row["shot_xG"]) else "gray"
    pitch.arrows(
        row["location_x"], row["location_y"],
        row["pass.end_location_x"], row["pass.end_location_y"],
        width=2, headwidth=4, color=color, ax=ax
    )

# Add colorbar legend
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label("xG of Resulting Shot")

st.pyplot(fig)

# Show detailed data
st.subheader("📋 Data Table: Throw-ins Leading to Shots")
st.dataframe(throwins[[
    "team.name", "player.name", "possession", "location_x", "location_y",
    "pass.end_location_x", "pass.end_location_y", "shot_xG"
]].dropna())
