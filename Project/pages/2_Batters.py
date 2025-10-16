import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns

st.set_page_config(layout="wide", page_title="Batter Hall of Fame Predictor")

#Load data
batting_url = "https://raw.githubusercontent.com/EthanLabombard/cmse830_fds/refs/heads/main/Project/Data/batter_hof.csv"
batter_hof = pd.read_csv(batting_url)


st.markdown(
    f"""
    <div style='text-align: center; padding-bottom: 10px;'>
        <h1 style='color: #041E42;'>🏟️Batter Hall of Fame Predictor🏟️</h1>
    </div>
    """,
    unsafe_allow_html=True
)



#st.header("🏟️Batters🏟️")

color = {"Y": "green", "N": "red"}
alphas = {"Y": 1, "N": 0.3}

# ------------------------------------------
# Page title
# ------------------------------------------
#st.title("Batter Hall of Fame Analysis Dashboard")

# ------------------------------------------
# Plot 1: Plotly Scatter (Career Hits vs Batting Average)
# ------------------------------------------
st.header("Career Hits and Batting Averages by Hall of Fame Status")
st.markdown("""
This scatter plot shows how career hits relate to batting average for players, 
colored by their Hall of Fame induction status. Green points represent inducted players, 
while red points represent non-inducted players.
""")

fig1 = px.scatter(
    batter_hof,
    x="H",
    y="AVG",
    color="inducted",
    color_discrete_map=color,
    opacity=0.8,
    labels={"H": "Career Hits", "AVG": "Career Batting Average"},
    title="Career Hits and Batting Averages by Hall of Fame Status"
)
st.plotly_chart(fig1, use_container_width=True)

# ------------------------------------------
# Plot 2: Violin Plot (Distribution of Home Runs)
# ------------------------------------------
st.header("Distribution of Home Runs by Hall of Fame Status")
st.markdown("""
This violin plot displays the distribution of home runs among inducted and non-inducted players.
Each violin shows the data spread and density, with embedded box plots and individual data points.
""")

fig2 = px.violin(
    batter_hof,
    x="inducted",
    y="HR",
    color="inducted",
    box=True,
    points="all",
    color_discrete_map=color,
    title="Distribution of Home Runs by Hall of Fame Status"
)
st.plotly_chart(fig2, use_container_width=True)

# ------------------------------------------
# Plot 3: Interactive Correlation Heatmap
# ------------------------------------------
st.header("Correlation Between Batter Statistics and Hall of Fame Induction")
st.markdown("""
This heatmap allows you to explore how different batting statistics correlate with 
Hall of Fame induction.
You can customize which statistics to include using the selection below.
""")

# Ensure numeric mapping exists
batter_hof["inducted_numeric"] = batter_hof["inducted"].map({"Y": 1, "N": 0})

# Select numeric variables for correlation
numeric_cols = ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP", "AVG", "OBP", "inducted_numeric"]
default_selection = ["H", "BB", "HR", "AVG", "RBI"]  # preset
selected_vars = st.multiselect(
    "Select variables to compare with Hall of Fame induction:",
    options=[c for c in numeric_cols if c != "inducted_numeric"],
    default=default_selection
)

# Always include inducted_numeric
corr_vars = selected_vars + ["inducted_numeric"]

corr = batter_hof[corr_vars].corr()

fig3, ax3 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", vmin=-1, vmax=1, ax=ax3)
ax3.set_title("Correlation Between Selected Statistics and Hall of Fame Induction")

st.pyplot(fig3, use_container_width=True)