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

clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42, class_weight={0: 1, 1: 10})
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
st.header("Model Evaluation & Comparison")
st.markdown("""
Below are thorough evaluation metrics for both the Decision Tree and Logistic Regression models.
These measures help demonstrate model selection, validation, and performance comparison.
""")

# ----------------------------
# 1. Predictions for both models
# ----------------------------
tree_preds = clf.predict(X_test)
tree_probs = clf.predict_proba(X_test)[:, 1]

log_preds = model.predict(X_scaled)
log_probs = model.predict_proba(X_scaled)[:, 1]

# ----------------------------
# 2. Confusion Matrix + Metrics
# ----------------------------
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_curve, auc
)

st.subheader("Confusion Matrices")

col1, col2 = st.columns(2)

with col1:
    st.write("### Decision Tree")
    cm_tree = confusion_matrix(y_test, tree_preds)
    st.write(cm_tree)

with col2:
    st.write("### Logistic Regression")
    cm_log = confusion_matrix(y, log_preds)
    st.write(cm_log)

# ----------------------------
# 3. Classification Reports
# ----------------------------
st.subheader("Classification Reports")

cr_tree = classification_report(y_test, tree_preds, output_dict=True)
cr_log = classification_report(y, log_preds, output_dict=True)

col1, col2 = st.columns(2)
with col1:
    st.write("### Decision Tree")
    st.json(cr_tree)

with col2:
    st.write("### Logistic Regression")
    st.json(cr_log)

# ----------------------------
# 4. ROC Curves & AUC
# ----------------------------
st.subheader("ROC Curve Comparison")

tree_fpr, tree_tpr, _ = roc_curve(y_test, tree_probs)
tree_auc = auc(tree_fpr, tree_tpr)

log_fpr, log_tpr, _ = roc_curve(y, log_probs)
log_auc = auc(log_fpr, log_tpr)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(tree_fpr, tree_tpr, label=f"Decision Tree (AUC = {tree_auc:.3f})")
ax.plot(log_fpr, log_tpr, label=f"Logistic Regression (AUC = {log_auc:.3f})")
ax.plot([0,1],[0,1],"k--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve Comparison")
ax.legend()
st.pyplot(fig)

# ----------------------------
# 5. Cross-Validation
# ----------------------------
from sklearn.model_selection import cross_val_score

st.subheader("Cross-Validation Scores")

tree_cv = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
log_cv = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

col1, col2 = st.columns(2)
with col1:
    st.write("### Decision Tree CV Accuracy")
    st.write(tree_cv)
    st.write(f"Mean: {tree_cv.mean():.3f}")

with col2:
    st.write("### Logistic Regression CV Accuracy")
    st.write(log_cv)
    st.write(f"Mean: {log_cv.mean():.3f}")

# ----------------------------
# 6. Basic Hyperparameter Tuning (Grid Search)
# ----------------------------
st.subheader("Hyperparameter Tuning (Grid Search)")

from sklearn.model_selection import GridSearchCV

param_grid = {
    "max_depth": [2, 4, 6, 8, 10],
    "criterion": ["gini", "entropy"]
}

grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy'
)
grid.fit(X, y)

st.write("Best Parameters:", grid.best_params_)
st.write("Best Cross-Validation Score:", grid.best_score_)
st.markdown("""
These results show that the decision tree tends to predict more heavly that a player will be in the hall of fame than the logistic regression model currently does. Since the two models have a very similar overall accuracy score, the prefered model would be the decision tree because this model better matches the overall purpose of this project.
""")