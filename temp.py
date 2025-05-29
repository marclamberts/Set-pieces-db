# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st

# Set page config
st.set_page_config(page_title="My Streamlit App", layout="centered")

# Title and Description
st.title("🚀 Welcome to My Streamlit App")
st.write("This is a simple Streamlit app to demonstrate basic functionality.")

# Input widgets
name = st.text_input("Enter your name:")
age = st.slider("Select your age", 0, 100, 25)

# Conditional output
if name:
    st.success(f"Hello, {name}! You are {age} years old.")

# Data display
st.subheader("Sample Data Table")
st.dataframe({
    "Name": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35],
    "Location": ["New York", "San Francisco", "London"]
})
