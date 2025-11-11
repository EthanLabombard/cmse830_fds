import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

st.set_page_config(layout="wide", page_title="Data Cleaning, Encoding, and Imputation")

batting_url = "https://raw.githubusercontent.com/EthanLabombard/cmse830_fds/refs/heads/main/Project/Data/Batting.csv"
pitching_url = "https://raw.githubusercontent.com/EthanLabombard/cmse830_fds/refs/heads/main/Project/Data/Pitching.csv"
hof_url = "https://raw.githubusercontent.com/EthanLabombard/cmse830_fds/refs/heads/main/Project/Data/HallOfFame.csv"
batting = pd.read_csv(batting_url)
pitching = pd.read_csv(pitching_url)
hof = pd.read_csv(hof_url)

st.header("1️⃣ Data Overview")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Batting Data")
    st.dataframe(batting.head())
with col2:
    st.subheader("Pitching Data")
    st.dataframe(pitching.head())
with col3:
    st.subheader("Hall of Fame Data")
    st.dataframe(hof.head())

# --- Imputation Helper Function ---
def perform_knn_imputation(df, columns, label):
    st.subheader(f"2️⃣ KNN Imputation for {label}")
    st.write(f"Selected columns for imputation: `{columns}`")

    numeric_df = df[columns]
    missing_before = numeric_df.isnull().sum()

    st.write("Missing values before imputation:")
    st.write(missing_before)

    scaler = StandardScaler()
    imputer = KNNImputer(n_neighbors=5)

    # Fit on non-missing data
    df_no_missing = numeric_df.dropna()
    scaled = pd.DataFrame(scaler.fit_transform(df_no_missing), columns=df_no_missing.columns)
    imputer.fit(scaled)

    # Impute all data
    scaled_full = pd.DataFrame(scaler.transform(numeric_df.fillna(0)), columns=columns, index=numeric_df.index)
    imputed_scaled = imputer.transform(scaled_full)
    imputed = pd.DataFrame(scaler.inverse_transform(imputed_scaled), columns=columns, index=numeric_df.index)

    # Compare distributions
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(numeric_df[columns[1]].dropna(), kde=True, color='blue', alpha=0.5, label='Original (non-missing)')
    sns.histplot(imputed.loc[numeric_df[columns[1]].isnull(), columns[1]], kde=True, color='red', alpha=0.5, label='Imputed')
    plt.title(f"Distribution of Original vs Imputed '{columns[1]}'")
    plt.legend()
    st.pyplot(fig)

    st.write("Statistics comparison after imputation:")
    c1, c2 = st.columns(2)
    with c1:
        st.write("Original data:")
        st.write(numeric_df.describe())
    with c2:
        st.write("Imputed data:")
        st.write(imputed.describe())

    return imputed

# --- Apply Imputation ---
batting_imputed = perform_knn_imputation(batting, ["SB", "SO"], "Batting")
pitching_imputed = perform_knn_imputation(pitching, ["IBB", "HBP"], "Pitching")

# --- Replace with Imputed Data and Fill Remaining Missing Values ---
batting["SB"] = batting_imputed["SB"]
batting["SO"] = batting_imputed["SO"]
pitching["IBB"] = pitching_imputed["IBB"]
pitching["HBP"] = pitching_imputed["HBP"]

batting.fillna(0, inplace=True)
pitching.fillna(0, inplace=True)

# --- Feature Engineering ---
st.header("3️⃣ Feature Engineering")
batting["AVG"] = batting["H"] / batting["AB"].replace(0, np.nan)
batting["OBP"] = (
    (batting["H"] + batting["BB"] + batting["IBB"] + batting["HBP"]) /
    (batting["AB"] + batting["BB"] + batting["IBB"] + batting["HBP"] + batting["SH"] + batting["SF"])
)

mask = hof["category"] == "Player"
hof = hof[mask]

st.write("Filtered Hall of Fame data (only players):")
st.dataframe(hof.head())

# --- Aggregate Player Data ---
st.header("4️⃣ Aggregation by Player")
st.markdown("Each player's career stats are summed and merged with Hall of Fame data.")

batting_sum = batting.groupby("playerID", as_index=False).agg({
    "yearID": "count",
    "G": "sum", "AB": "sum", "R": "sum", "H": "sum",
    "2B": "sum", "3B": "sum", "HR": "sum", "RBI": "sum", 
    "SB": "sum", "CS": "sum", "BB": "sum", "SO": "sum",
    "IBB": "sum", "HBP": "sum", "SH": "sum", "SF": "sum",
    "GIDP": "sum", "OBP": "mean", "AVG": "mean"
})
batter_hof = pd.merge(batting_sum, hof, on="playerID", how="outer")

pitcher_sum = pitching.groupby("playerID", as_index=False).agg({
    "yearID": "count", "W": "sum", "L": "sum", "G": "sum", "GS": "sum",
    "CG": "sum", "SHO": "sum", "SV": "sum", "IPouts": "sum", "H": "sum",
    "ER": "sum", "HR": "sum", "BB": "sum", "SO": "sum", "BAOpp": "mean",
    "ERA": "mean", "IBB": "sum", "WP": "sum", "HBP": "sum", "BK": "sum",
    "BFP": "sum", "GF": "sum", "R": "sum", "SH": "sum", "SF": "sum", "GIDP": "sum"
})
pitcher_hof = pd.merge(pitcher_sum, hof, on="playerID", how="outer")

# --- Derived Stats ---
batter_hof["AVG"] = batter_hof["H"] / batter_hof["AB"].replace(0, np.nan)
batter_hof["OBP"] = (
    (batter_hof["H"] + batter_hof["BB"] + batter_hof["IBB"] + batter_hof["HBP"]) /
    (batter_hof["AB"] + batter_hof["BB"] + batter_hof["IBB"] + batter_hof["HBP"] + batter_hof["SH"] + batter_hof["SF"])
)
pitcher_hof["BAOpp"] = pitcher_hof["H"] / (
    pitcher_hof["BFP"] - pitcher_hof["BB"] - pitcher_hof["IBB"] - pitcher_hof["HBP"] - pitcher_hof["SH"] - pitcher_hof["SF"]
)
pitcher_hof["ERA"] = pitcher_hof["ER"] / (pitcher_hof["IPouts"] / 27)

# --- Fill Missing Values ---
for df in [batter_hof, pitcher_hof]:
    df["category"] = df["category"].fillna("Player")
    df["inducted"] = df["inducted"].fillna("N")
    df["ballots"] = df["ballots"].fillna(0)
    df["votes"] = df["votes"].fillna(0)

# --- Display Results ---
st.header("5️⃣ Final Cleaned Datasets")
st.subheader("Batting + HOF Merged")
st.dataframe(batter_hof.head())

st.subheader("Pitching + HOF Merged")
st.dataframe(pitcher_hof.head())

# --- Export Option ---
st.header("6️⃣ Save Results")
if st.button("💾 Export Cleaned Data to CSV"):
    batter_hof.to_csv("batter_hof.csv", index=False)
    pitcher_hof.to_csv("pitcher_hof.csv", index=False)
    st.success("Cleaned datasets saved as 'batter_hof.csv' and 'pitcher_hof.csv'.")