import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
import io

st.set_page_config(layout="wide", page_title="Baseball Dataset Analysis")

@st.cache_data
def load_baseball_data():
    batting_url = "https://raw.githubusercontent.com/EthanLabombard/cmse830_fds/refs/heads/main/Project/Data/Batting.csv"
    pitching_url = "https://raw.githubusercontent.com/EthanLabombard/cmse830_fds/refs/heads/main/Project/Data/Pitching.csv"
    batting = pd.read_csv(batting_url)
    pitching = pd.read_csv(pitching_url)
    return batting, pitching

batting, pitching = load_baseball_data()

st.session_state["batting_raw"] = batting
st.session_state["pitching_raw"] = pitching


st.markdown(
    f"""
    <div style='text-align: center; padding-bottom: 10px;'>
        <h1 style='color: #041E42;'>Baseball Dataset Analysis</h1>
    </div>
    """,
    unsafe_allow_html=True
)


dataset_choice = st.radio("Select a dataset:", ("Batting", "Pitching"), horizontal=True)

df = batting if dataset_choice == "Batting" else pitching


if dataset_choice == "Batting":
    st.markdown(
        """
        Below is the evaluation of the raw batting data.
        Missing values were imputed with K-Nearest Neighbors for:
        - SB (Stolen Bases)
        - SO (Strikeouts)  
        
        All other missing values were set to 0.
        """
    )
else:
    st.markdown(
        """
        Below is the evaluation of the raw pitching data.
        Missing values were imputed with K-Nearest Neighbors for:
        - IBB (Intentional Walks)
        - HBP (Hit by Pitch)  

        All other missing values were set to 0.
        """
    )


view_choice = st.radio("Choose a view:", ("Data Info", "Missing Values Visualization"), horizontal=True)


if view_choice == "Data Info":
    st.subheader(f"{dataset_choice} Dataset Information")

    excluded_cols = ["yearID", "stint"]
    st.dataframe(
        df.drop(columns=excluded_cols, errors="ignore").describe()
    )

else:
    st.subheader(f"Missing Values Visualization — {dataset_choice}")

    nan_mask = df.isna()
    nan_array = nan_mask.astype(int).to_numpy()

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(nan_array.T, interpolation='nearest', aspect='auto', cmap='viridis')

    ax.set_xlabel("Player Index")
    ax.set_ylabel("Features")
    ax.set_title(f"Visualizing Missing Values in the {dataset_choice} Dataset")
    ax.set_yticks(range(len(df.columns)))
    ax.set_yticklabels(df.columns)

    # Avoid overcrowding X-axis
    num_rows = nan_array.shape[0]
    ax.set_xticks(
        np.linspace(0, num_rows - 1, min(10, num_rows)).astype(int)
    )

    ax.grid(True, axis="y", linestyle="--", alpha=0.7)
    plt.colorbar(im, ax=ax, label="Missing (1) / Present (0)")

    st.pyplot(fig)
