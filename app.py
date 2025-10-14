import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns

st.set_page_config(layout="wide", page_title="Baseball Hall of Fame Predictor")

#Load data
batting_url = "https://raw.githubusercontent.com/EthanLabombard/cmse830_fds/refs/heads/main/Data/batter_hof.csv"
pitching_url = "https://raw.githubusercontent.com/EthanLabombard/cmse830_fds/refs/heads/main/Data/pitcher_hof.csv"
batting_hof = pd.read_csv(batting_url)
pitching_hof = pd.read_csv(pitching_url)


st.markdown(
    f"""
    <div style='text-align: center; padding-bottom: 10px;'>
        <h1 style='color: #041E42;'>Baseball Hall of Fame Predictor Dashboard</h1>
    </div>
    """,
    unsafe_allow_html=True
)


#Tabs for navigation
tab1, tab2 = st.tabs(["⚾ Batters", "🧢 Pitcher"])

#Tab 1
with tab1:
    st.header("⚾ Batters")

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
    sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", ax=ax3)
    ax3.set_title("Correlation Between Batter Statistics and Hall of Fame Induction")


    with col1: st.pyplot(fig1, use_container_width=True)
    with col2: st.plotly_chart(fig2, use_container_width=True)
    with col3: st.pyplot(fig3, use_container_width=True)

# --- TAB 2: School Search ---
with tab2:
    st.header("🧢 Pitcher")

    col4, col5, col6 = st.columns(3)

    color = {"Y": "green", "N": "red"}
    alphas = {"Y": 1, "N": .3}
    fig4, ax4 = plt.subplots()
    for induct, players in pitcher_hof.groupby("inducted"):
        ax1.scatter(players["ERA"], players["SO"], label = induct, c=color[induct], alpha=alphas[induct])
    ax4.set_xlabel("Career ERA")
    ax4.set_ylabel("Career Strikeouts")
    ax4.set_title("Career ERA and Strikeouts by Hall of Fame Status")
    ax4.legend()

    fig5 = px.violin(pitcher_hof, x = "inducted", y = "W", color="inducted", box=True, points="all", color_discrete_map=color, title="Distribution of Wins by Hall of Fame Status")
    
    pitcher_hof["inducted_numeric"] = pitcher_hof["inducted"].map({"Y": 1, "N": 0})
    corr = pitcher_hof[["ERA", "SO", "W", "L", "SV", "inducted_numeric"]].corr()
    fig6, ax6 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", ax=ax6)
    ax6.set_title("Correlation Between Pitcher Statistics and Hall of Fame Induction")


    with col4: st.pyplot(fig1, use_container_width=True)
    with col5: st.plotly_chart(fig2, use_container_width=True)
    with col6: st.pyplot(fig3, use_container_width=True)