import streamlit as st
import pandas as pd
import os
import ast
import matplotlib.pyplot as plt
from mplsoccer import Pitch

# -------------------- Config --------------------
st.set_page_config(layout="wide")

PASSWORD = "PrincessWay2526"
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    password = st.text_input("Enter password to continue:", type="password")
    if password == PASSWORD:
        st.session_state.authenticated = True
        st.success("Access granted.")
    else:
        st.stop()

# -------------------- Helper Functions --------------------
def safe_parse_location(loc):
    try:
        parsed = ast.literal_eval(loc) if isinstance(loc, str) else loc
        if isinstance(parsed, (list, tuple)):
            return (parsed + [None] * 3)[:3]
        return [None, None, None]
    except:
        return [None, None, None]

@st.cache_data
def load_ti_data():
    base_path = os.path.dirname(__file__)
    return pd.read_excel(os.path.join(base_path, "TI.xlsx"))

# -------------------- Load Data --------------------
ti = load_ti_data()

# Parse locations
ti["location"] = ti["location"].apply(safe_parse_location)
ti[["location_x", "location_y", "location_z"]] = pd.DataFrame(ti["location"].tolist(), index=ti.index)

ti["pass.end_location"] = ti["pass.end_location"].apply(safe_parse_location)
ti[["pass.end_location_x", "pass.end_location_y", "pass.end_location_z"]] = pd.DataFrame(ti["pass.end_location"].tolist(), index=ti.index)

# -------------------- Filter for Throw-Ins Ending in Shots --------------------
ti_throwin = ti[
    (ti["type.name"] == "Pass") &
    (ti["play_pattern.name"] == "From Throw In")
].copy()

# Assign possession ID if available
if "possession" in ti.columns:
    grouped = ti.groupby("possession")

    shot_possessions = ti[
        ti["type.name"] == "Shot"
    ]["possession"].dropna().unique()

    ti_throwin = ti_throwin[
        ti_throwin["possession"].isin(shot_possessions)
    ]

# -------------------- Streamlit App --------------------
st.markdown("## Throw-ins Leading to Shots")

tab1, tab2 = st.tabs(["📊 Visualisation", "📋 Data"])

with tab1:
    st.subheader("Passes from Throw-ins (leading to shots)")

    pitch = Pitch(pitch_type='statsbomb', half=True, pitch_color='white', line_color='black')
    fig, ax = pitch.draw(figsize=(10, 7))

    for _, row in ti_throwin.iterrows():
        if pd.notna(row["location_x"]) and pd.notna(row["pass.end_location_x"]):
            pitch.arrow(
                row["location_x"], row["location_y"],
                row["pass.end_location_x"], row["pass.end_location_y"],
                width=1.5, headwidth=5, headlength=5,
                color="crimson", ax=ax, alpha=0.7
            )

    st.pyplot(fig)

with tab2:
    st.dataframe(ti_throwin)
