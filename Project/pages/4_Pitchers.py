import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression

st.set_page_config(layout="wide", page_title="Pitcher Hall of Fame Predictor")

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


st.subheader("Career ERA and Strikeouts by Hall of Fame Status")
st.markdown("""
This scatter plot shows how Earned Run Average relates to career strikeouts 
for pitchers, colored by Hall of Fame induction status.  
Pitchers with lower ERAs and higher strikeout totals are generally more likely to be inducted.
""")

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
    labels={"W": "Wins", "inducted": "Inducted"},
    title="Distribution of Wins by Hall of Fame Status"
)
st.plotly_chart(fig5, use_container_width=True)

st.subheader("Correlation Between Pitcher Statistics and Hall of Fame Induction")
st.markdown("""
Use the selector below to explore how various pitching statistics correlate 
with Hall of Fame induction.
""")

pitcher_hof["inducted_numeric"] = pitcher_hof["inducted"].map({"Y": 1, "N": 0})

numeric_cols = ["W", "L", "G", "GS", "CG", "SHO", "SV", "IPouts", "H", "ER", "HR", "BB", "SO", "BAOpp", "ERA", "IBB", "WP", "HBP", "BK", "BFP", "GF", "R", "SH", "SF", "GIDP", "inducted_numeric"]
default_selection = ["ERA", "SO", "W", "L", "SV"]

selected_vars = st.multiselect(
    "Select pitcher statistics to compare with Hall of Fame induction:",
    options=[c for c in numeric_cols if c != "inducted_numeric"],
    default=default_selection
)

corr_vars = selected_vars + ["inducted_numeric"]

corr = pitcher_hof[corr_vars].corr()

fig6, ax6 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", vmin=-1, vmax=1, ax=ax6)
ax6.set_title("Correlation Between Selected Pitcher Statistics and Hall of Fame Induction")
st.pyplot(fig6, use_container_width=True)


st.header("Player Archetype Radar Charts")
st.markdown("""
This radar chart clusters pitchers by some key statistics with a custumizable number of clusters possible.
For the preset 4 clusters the player archetypes are:
- 0  - Starting Pitchers; these pitchers throw many innings and tend to have high earned runs and high wins.
- 1  - Aces; the best starting pitcers on the team, specialize in low earned runs, high strikeouts, and high wins.
- 2  - Bullpen; the relief pitchers who tend to pitch few innings, tend to have high earned runs and high walks.
- 3  - Closers; pitchers who finish games, these pitchers specialize in low earned runs and high saves.
""")
key_cols = ["W", "SV", "ER", "HR", "BB", "SO"]
pitcher_hof_clean = pitcher_hof.dropna(subset=key_cols)

k = st.sidebar.slider("Number of Archetypes (Clusters)", 2, 8, 4)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(pitcher_hof_clean[key_cols])

kmeans = KMeans(n_clusters=k, random_state=42)
pitcher_hof_clean["cluster"] = kmeans.fit_predict(X_scaled)

cluster_profiles = pitcher_hof_clean.groupby("cluster")[key_cols].mean()

cluster_choice = st.selectbox("Select a player archetype:", cluster_profiles.index)

profile = cluster_profiles.loc[cluster_choice]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r = profile.values,
    theta = key_cols,
    fill = "toself",
    name = f"Archetype {cluster_choice}"
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(visible=True),
    ),
    showlegend=False,
    title=f"Radar Chart: Archetype {cluster_choice}"
)

st.plotly_chart(fig)



st.header("Hall of Fame Probability Tree")
st.markdown("""
This probability tree splits along various statistics that pitchers may have. The leaf nodes can be used to predict which class the player should be categorized within.
""")
key_cols = ["W", "SV", "IPouts", "ER", "HR", "BB", "SO"]

pitcher_hof_clean = pitcher_hof.dropna(subset=key_cols + ["inducted_numeric"])

X = pitcher_hof_clean[key_cols]
y = pitcher_hof_clean["inducted_numeric"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


max_depth = st.sidebar.slider("Tree Depth", 2, 8, 4)

clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
clf.fit(X_train, y_train)

st.subheader("Decision Tree Visualization")

fig, ax = plt.subplots(figsize=(22, 14))  # Lots of room for labels
tree.plot_tree(
    clf,
    feature_names=key_cols,
    class_names=["Non-HOF", "HOF"],
    filled=True,
    rounded=True,
    fontsize=8
)
st.pyplot(fig)



st.header("Hall of Fame Logistic Model")
st.markdown("""
This model is a logistic regression model that predicts whether a pitcher will make the hall of fame based upon the inputed statistics. The default statistics are the career statistics for Justin Verlander, a player widely considered to be a future hall of fame inductee.
""")

key_cols = ["W", "SV", "IPouts", "ER", "HR", "BB", "SO"]
target = "inducted_numeric"

pitcher_hof_clean = pitcher_hof.dropna(subset=key_cols + [target])

X = pitcher_hof_clean[key_cols]
y = pitcher_hof_clean[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=200, class_weight={0: 1, 1: 10})
model.fit(X_scaled, y)

st.sidebar.header("Enter Player Statistics")
integer_features = ["W", "SV", "ER", "HR", "BB", "SO"]
float_features = ["IPouts"]
default_values = {
    "W": 266,
    "SV": 0,
    "ER": 1317,
    "HR": 371,
    "BB": 1004,
    "SO": 3553,
    "IPouts": 3567.67
}

user_input = {}

for feature in key_cols:
    if feature in integer_features:
        user_input[feature] = st.sidebar.number_input(
            feature,
            min_value=0,
            max_value=5000,
            value=int(default_values[feature]),
            step=1,
            format="%d"
        )
    else:
        user_input[feature] = st.sidebar.number_input(
            feature,
            value=float(default_values[feature]),
            step=0.01,
            format="%.3f"
        )

input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

prob = model.predict_proba(input_scaled)[0, 1]
pred = model.predict(input_scaled)[0]
st.subheader("Prediction")
st.write(f"**Probability of Hall of Fame Induction:** `{prob:.3f}`")

st.write(
    "**Prediction:** Likely Hall of Famer" if pred == 1 else "**Prediction:** Not Likely Hall of Famer"
)
