import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="Set Piece Studio",
    page_icon="⚽",
    layout="wide",
)

FILE_PATHS = [
    "SWE SP.xlsx",
    "/mnt/data/SWE SP.xlsx"
]

# =========================================================
# HELPERS
# =========================================================
def safe_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def find_col(df, names):
    cols = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols:
            return cols[n.lower()]
    return None

def parse_xy(cell, i):
    try:
        parts = str(cell).split(",")
        return float(parts[i])
    except:
        return np.nan

def sp_type_map(x):
    s = str(x).lower()
    if "corner" in s:
        return "Corner"
    if "free" in s:
        return "Free Kick"
    if "throw" in s:
        return "Throw-In"
    return "Other"

# =========================================================
# LOAD DATA (NO UPLOAD)
# =========================================================
@st.cache_data
def load_data():
    for path in FILE_PATHS:
        if os.path.exists(path):
            raw = pd.read_excel(path)
            return prepare(raw), path
    raise FileNotFoundError("Put SWE SP.xlsx in the app folder")

def prepare(df):
    df.columns = [str(c).strip() for c in df.columns]

    team = find_col(df, ["team", "team.name"])
    sp   = find_col(df, ["SP_Type", "set_piece_type"])
    xg   = find_col(df, ["shot_xg", "shot.statsbomb_xg"])
    minc = find_col(df, ["minute"])
    sec  = find_col(df, ["second"])
    match= find_col(df, ["match"])

    df_out = pd.DataFrame()

    df_out["team"] = df[team]
    df_out["Match"] = df[match] if match else "Match"
    df_out["Minute"] = safe_numeric(df[minc]) if minc else 0
    df_out["Second"] = safe_numeric(df[sec]) if sec else 0
    df_out["shot_xg"] = safe_numeric(df[xg]) if xg else 0
    df_out["type"] = df[sp].apply(sp_type_map)

    # shot detection
    df_out["shot"] = df_out["shot_xg"] > 0

    # coordinates
    if "location.shot" in df.columns:
        df_out["x"] = df["location.shot"].apply(lambda x: parse_xy(x,0))
        df_out["y"] = df["location.shot"].apply(lambda x: parse_xy(x,1))
    else:
        df_out["x"] = np.nan
        df_out["y"] = np.nan

    return df_out

df, path_used = load_data()

# =========================================================
# UI HELPERS
# =========================================================
def metric(label, value):
    st.metric(label, value)

def pitch(fig):
    fig.update_xaxes(range=[0,80], visible=False)
    fig.update_yaxes(range=[60,120], visible=False)
    return fig

def shotmap(data):
    fig = go.Figure()
    d = data.dropna(subset=["x","y"])
    if not d.empty:
        fig.add_trace(go.Scatter(
            x=d["y"],
            y=d["x"],
            mode="markers",
            marker=dict(size=12, opacity=0.7)
        ))
    return pitch(fig)

# =========================================================
# STATE
# =========================================================
if "page" not in st.session_state:
    st.session_state["page"] = "home"

def go(page):
    st.session_state["page"] = page

# =========================================================
# HOME
# =========================================================
if st.session_state["page"] == "home":

    st.title("⚽ Set Piece Studio")
    st.caption(f"Loaded: {path_used}")

    c1, c2, c3 = st.columns(3)

    with c1:
        if st.button("Free Kick"):
            go("Free Kick")

    with c2:
        if st.button("Corner"):
            go("Corner")

    with c3:
        if st.button("Throw-In"):
            go("Throw-In")

    summary = df.groupby("type").agg(
        events=("type","count"),
        xg=("shot_xg","sum")
    ).reset_index()

    st.bar_chart(summary.set_index("type")["events"])

# =========================================================
# SEGMENT VIEW
# =========================================================
else:

    page = st.session_state["page"]
    seg = df[df["type"] == page]

    if st.button("← Back"):
        go("home")

    st.header(page)

    if seg.empty:
        st.warning("No data")
        st.stop()

    # KPIs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric("Events", len(seg))
    with c2:
        metric("Matches", seg["Match"].nunique())
    with c3:
        metric("Shots", seg["shot"].sum())
    with c4:
        metric("xG", round(seg["shot_xg"].sum(),2))

    # visuals
    tabs = st.tabs(["Overview","Shotmap","Teams"])

    with tabs[0]:
        st.bar_chart(seg.groupby("team")["shot_xg"].sum())

    with tabs[1]:
        st.plotly_chart(shotmap(seg), use_container_width=True)

    with tabs[2]:
        team = seg.groupby("team").agg(
            events=("team","count"),
            shots=("shot","sum"),
            xg=("shot_xg","sum")
        )
        st.dataframe(team)
