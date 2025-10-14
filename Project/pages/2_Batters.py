import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns

st.set_page_config(layout="wide", page_title="Batter Hall of Fame Predictor")

#Load data
batting_url = "https://raw.githubusercontent.com/EthanLabombard/cmse830_fds/refs/heads/main/Data/batter_hof.csv"
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

col1, col2, col3 = st.columns(3)

color = {"Y": "green", "N": "red"}
alphas = {"Y": 1, "N": .3}
fig1, ax1 = plt.subplots()
for induct, players in batter_hof.groupby("inducted"):
    ax1.scatter(players["H"], players["AVG"], label = induct, c=color[induct], alpha=alphas[induct])
ax1.set_xlabel("Career Hits")
ax1.set_ylabel("Career Batting Average")
ax1.set_title("Career Hits and Batting Averages by Hall of Fame Status")
ax1.legend()

fig2 = px.violin(batter_hof, x = "inducted", y = "HR", color="inducted", box=True, points="all", color_discrete_map=color, title="Distribution of Home Runs by Hall of Fame Status")

batter_hof["inducted_numeric"] = batter_hof["inducted"].map({"Y": 1, "N": 0})
corr = batter_hof[["H", "BB", "HR", "AVG", "RBI", "inducted_numeric"]].corr()
fig3, ax3 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", vmin=-1, vmax=1, ax=ax3)
ax3.set_title("Correlation Between Batter Statistics and Hall of Fame Induction")


with col1: st.pyplot(fig1, use_container_width=True)
with col2: st.plotly_chart(fig2, use_container_width=True)
with col3: st.pyplot(fig3, use_container_width=True)