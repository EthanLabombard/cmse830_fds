import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns

st.set_page_config(layout="wide", page_title="Pitcher Hall of Fame Predictor")

#Load data
pitching_url = "https://raw.githubusercontent.com/EthanLabombard/cmse830_fds/refs/heads/main/Project/Data/pitcher_hof.csv"
pitcher_hof = pd.read_csv(pitching_url)


st.markdown(
    f"""
    <div style='text-align: center; padding-bottom: 10px;'>
        <h1 style='color: #041E42;'>🧢Pitcher Hall of Fame Predictor🧢</h1>
    </div>
    """,
    unsafe_allow_html=True
)


#st.header("🧢Pitchers🧢")

color = {"Y": "green", "N": "red"}
alphas = {"Y": 1, "N": 0.3}

# ----------------------------------------------------
# Page section header
# ----------------------------------------------------
#st.header("Pitcher Hall of Fame Analysis")

# ----------------------------------------------------
# Plot 1: Plotly Scatter (ERA vs Strikeouts)
# ----------------------------------------------------
st.subheader("Career ERA and Strikeouts by Hall of Fame Status")
st.markdown("""
This scatter plot shows how Earned Run Average relates to career strikeouts 
for pitchers, colored by Hall of Fame induction status.  
Pitchers with lower ERAs and higher strikeout totals are generally more likely to be inducted.
""")

# Filter out extreme ERA values for cleaner visualization
fil_pitch = pitcher_hof[pitcher_hof["ERA"] <= 8]

fig4 = px.scatter(
    fil_pitch,
    x="ERA",
    y="SO",
    color="inducted",
    color_discrete_map=color,
    opacity=0.8,
    labels={"ERA": "Career ERA", "SO": "Career Strikeouts"},
    title="Career ERA and Strikeouts by Hall of Fame Status"
)
st.plotly_chart(fig4, use_container_width=True)

# ----------------------------------------------------
# Plot 2: Violin Plot (Wins distribution)
# ----------------------------------------------------
st.subheader("Distribution of Wins by Hall of Fame Status")
st.markdown("""
This violin plot displays the distribution of career wins for inducted and non-inducted pitchers.  
Each violin shows the overall spread and density of wins, while the box inside represents 
the interquartile range and median. Individual points show specific player values.
""")

fig5 = px.violin(
    pitcher_hof,
    x="inducted",
    y="W",
    color="inducted",
    box=True,
    points="all",
    color_discrete_map=color,
    title="Distribution of Wins by Hall of Fame Status"
)
st.plotly_chart(fig5, use_container_width=True)

# ----------------------------------------------------
# Plot 3: Interactive Correlation Heatmap
# ----------------------------------------------------
st.subheader("Correlation Between Pitcher Statistics and Hall of Fame Induction")
st.markdown("""
Use the selector below to explore how various pitching statistics correlate 
with Hall of Fame induction.
""")

# Ensure numeric mapping exists
pitcher_hof["inducted_numeric"] = pitcher_hof["inducted"].map({"Y": 1, "N": 0})

# Numeric columns to choose from
numeric_cols = ["W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts", "H", "ER", "HR", "BB", "SO", "BAOpp", "ERA", "IBB", "WP", "HBP", "BK", "BFP", "GF", "R", "SH", "SF", "GIDP", "inducted_numeric"]
default_selection = ["ERA", "SO", "W", "L", "SV"]  # preset

selected_vars = st.multiselect(
    "Select pitcher statistics to compare with Hall of Fame induction:",
    options=[c for c in numeric_cols if c != "inducted_numeric"],
    default=default_selection
)

# Always include inducted_numeric
corr_vars = selected_vars + ["inducted_numeric"]

# Compute correlation matrix
corr = pitcher_hof[corr_vars].corr()

# Create heatmap
fig6, ax6 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", vmin=-1, vmax=1, ax=ax6)
ax6.set_title("Correlation Between Selected Pitcher Statistics and Hall of Fame Induction")
st.pyplot(fig6, use_container_width=True)