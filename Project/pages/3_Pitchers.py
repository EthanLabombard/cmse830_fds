import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns

st.set_page_config(layout="wide", page_title="Pitcher Hall of Fame Predictor")

#Load data
pitching_url = "https://raw.githubusercontent.com/EthanLabombard/cmse830_fds/refs/heads/main/Data/pitcher_hof.csv"
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

col4, col5, col6 = st.columns(3)

fil_pitch = pitcher_hof[(pitcher_hof["ERA"] <= 8)]
color = {"Y": "green", "N": "red"}
alphas = {"Y": 1, "N": .3}
fig4, ax4 = plt.subplots()
for induct, players in fil_pitch.groupby("inducted"):
    ax4.scatter(players["ERA"], players["SO"], label = induct, c=color[induct], alpha=alphas[induct])
ax4.set_xlabel("Career ERA")
ax4.set_ylabel("Career Strikeouts")
ax4.set_title("Career ERA and Strikeouts by Hall of Fame Status")
ax4.legend()

fig5 = px.violin(pitcher_hof, x = "inducted", y = "W", color="inducted", box=True, points="all", color_discrete_map=color, title="Distribution of Wins by Hall of Fame Status")

pitcher_hof["inducted_numeric"] = pitcher_hof["inducted"].map({"Y": 1, "N": 0})
corr = pitcher_hof[["ERA", "SO", "W", "L", "SV", "inducted_numeric"]].corr()
fig6, ax6 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", vmin=-1, vmax=1, ax=ax6)
ax6.set_title("Correlation Between Pitcher Statistics and Hall of Fame Induction")


with col4: st.pyplot(fig4, use_container_width=True)
with col5: st.plotly_chart(fig5, use_container_width=True)
with col6: st.pyplot(fig6, use_container_width=True)