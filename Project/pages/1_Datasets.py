import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
import io

st.set_page_config(layout="wide", page_title="Baseball Dataset Analysis")

batting_url = "https://raw.githubusercontent.com/EthanLabombard/cmse830_fds/refs/heads/main/Project/Data/Batting.csv"
pitching_url = "https://raw.githubusercontent.com/EthanLabombard/cmse830_fds/refs/heads/main/Project/Data/Pitching.csv"
batting = pd.read_csv(batting_url)
pitching = pd.read_csv(pitching_url)


st.markdown(
    f"""
    <div style='text-align: center; padding-bottom: 10px;'>
        <h1 style='color: #041E42;'>Baseball Dataset Analysis</h1>
    </div>
    """,
    unsafe_allow_html=True
)

dataset_choice = st.radio("Select a dataset:", ("Batting", "Pitching"),horizontal=True)


if dataset_choice == "Batting":
    df = batting
else:
    df = pitching

view_choice = st.radio("Choose a view:", ("Data Info", "Missing Values Visualization"), horizontal=True)

if dataset_choice == "Batting":
    df = batting
    st.markdown(
        """
        Below is the evaluations for the raw batting data, to clean the data K-Nearest Neighbors evaluation was completed to fill missing values for the stolen base and strikeout columns. The rest of the missing data was set to 0 as these stats were either set at NaN because it did not occur or because it wasn't a possible statistic at that time.
        """
    )
else:
    df = pitching
    st.markdown(
        """
        Below is the evaluations for the raw pitching data, to clean the pitching dataset K-Nearest Neighbors evaluation was completed to fill missing values for the intentional walks and hit by pitch columns. The rest of the missing data was set to 0 as these stats were either set at NaN because it did not occur or because it wasn't a possible statistic at that time.
        """
    )

if view_choice == "Data Info":
    st.subheader(f"{dataset_choice} Dataset Information")
    excluded_cols = ["yearID", "stint"]
    st.dataframe(df.drop(columns=excluded_cols, errors="ignore").describe())

else:
    st.subheader(f"Missing Values Visualization — {dataset_choice}")
    nan_mask = df.isna()
    nan_array = nan_mask.astype(int).to_numpy()
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(nan_array.T, interpolation='nearest', aspect='auto', cmap='viridis')
    ax.set_xlabel("Player ID")
    ax.set_ylabel("Features")
    ax.set_title(f"Visualizing Missing Values in the {dataset_choice} Dataset")
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)
    num_players = nan_array.shape[0]
    ax.set_xticks(np.linspace(0, num_players - 1, min(10, num_players)).astype(int))
    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.colorbar(im, ax=ax, label="Missing (1) / Present (0)")
    st.pyplot(fig)