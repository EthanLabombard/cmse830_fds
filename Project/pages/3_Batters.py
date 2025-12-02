import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import graphviz
from sklearn.linear_model import LogisticRegression

st.set_page_config(layout="wide", page_title="Batter Hall of Fame Predictor")

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
    labels={"HR": "Home Runs", "inducted": "Inducted"},
    title="Distribution of Home Runs by Hall of Fame Status"
)
st.plotly_chart(fig2, use_container_width=True)


st.header("Correlation Between Batter Statistics and Hall of Fame Induction")
st.markdown("""
This heatmap allows you to explore how different batting statistics correlate with 
Hall of Fame induction.
You can customize which statistics to include using the selection below.
""")

batter_hof["inducted_numeric"] = batter_hof["inducted"].map({"Y": 1, "N": 0})

numeric_cols = ["G", "AB", "R", "H", "2B", "3B", "HR", "RBI", "SB", "CS", "BB", "SO", "IBB", "HBP", "SH", "SF", "GIDP", "AVG", "OBP", "inducted_numeric"]
default_selection = ["H", "BB", "HR", "AVG", "RBI"]
selected_vars = st.multiselect(
    "Select variables to compare with Hall of Fame induction:",
    options=[c for c in numeric_cols if c != "inducted_numeric"],
    default=default_selection
)

corr_vars = selected_vars + ["inducted_numeric"]

corr = batter_hof[corr_vars].corr()

fig3, ax3 = plt.subplots()
sns.heatmap(corr, annot=True, cmap="viridis", fmt=".2f", vmin=-1, vmax=1, ax=ax3)
ax3.set_title("Correlation Between Selected Statistics and Hall of Fame Induction")

st.pyplot(fig3, use_container_width=True)


st.header("Player Archetype Radar Charts")
st.markdown("""
This radar chart clusters batters by some key statistics with a custumizable number of clusters possible.
For the preset 4 clusters the player archetypes are:
- 0  - Short Stint Players; players who didn't play long or well in the major leagues.
- 1  - Power Hitters; players who specialized in hitting home runs.
- 2  - Speedy On-Base Players; players who specialized in getting on base and stealing bases.
- 3  - Slow On-Base Players; players who specialized in getting on base but didn't steal bases.
""")
key_cols = ["H", "HR", "RBI", "SB", "BB", "SO"]
batter_hof_clean = batter_hof.dropna(subset=key_cols)

k = st.sidebar.slider("Number of Archetypes (Clusters)", 2, 8, 4)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(batter_hof_clean[key_cols])

kmeans = KMeans(n_clusters=k, random_state=42)
batter_hof_clean["cluster"] = kmeans.fit_predict(X_scaled)

cluster_profiles = batter_hof_clean.groupby("cluster")[key_cols].mean()

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
This probability tree splits along various statistics that batters may have. The leaf nodes can be used to predict which class the player should be categorized within.
""")
key_cols = ["H", "HR", "RBI", "SB", "BB", "SO"]

batter_hof_clean = batter_hof.dropna(subset=key_cols + ["inducted_numeric"])

X = batter_hof_clean[key_cols]
y = batter_hof_clean["inducted_numeric"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


max_depth = st.sidebar.slider("Tree Depth", 2, 8, 4)

clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
clf.fit(X_train, y_train)

st.subheader("Decision Tree Visualization")

dot_data = export_graphviz(
    clf,
    out_file=None,
    feature_names=key_cols,
    class_names=["Non-HOF", "HOF"],
    filled=True,
    rounded=True,
    special_characters=True
)

graph = graphviz.Source(dot_data)
st.graphviz_chart(dot_data)


st.header("Hall of Fame Logistic Model")
st.markdown("""
This model is a logistic regression model that predicts whether a batter will make the hall of fame based upon the inputed statistics. The default statistics are the career statistics for Miguel Cabrera, a player widely considered to be a future hall of fame inductee.
""")

key_cols = ["H", "HR", "RBI", "SB", "BB", "SO", "AVG"]
target = "inducted_numeric"

batter_hof_clean = batter_hof.dropna(subset=key_cols + [target])

X = batter_hof_clean[key_cols]
y = batter_hof_clean[target]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = LogisticRegression(max_iter=200, class_weight={0: 1, 1: 5})
model.fit(X_scaled, y)

st.sidebar.header("Enter Player Statistics")
integer_features = ["H", "HR", "RBI", "SB", "BB", "SO"]
float_features = ["AVG"]
default_values = {
    "H": 3174,
    "HR": 511,
    "RBI": 1881,
    "SB": 40,
    "BB": 1258,
    "SO": 2105,
    "AVG": .306
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
