import streamlit as st

st.set_page_config(layout="wide", page_title="Baseball Hall of Fame Dashboard")

st.markdown(
    """
    <div style='text-align: center; padding-bottom: 10px;'>
        <h1 style='color: #041E42;'>Baseball Hall of Fame Dashboard</h1>
        <p>Select a page from the sidebar to explore batter and pitcher statistics.</p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("""
This project will explore the relationship between baseball player performance and induction into the Hall of Fame. This will involve exploring the statistics for both batters and pitchers to gain a true measure of what it takes to be a Hall of Fame inductee. 
""")

st.markdown("""
### For Navigation
There are four pages as a part of this project:
- This page is the Home page and includes information about the project and how best to use this app. 
- The Datasets page includes some information and key statistics about the datasets used in this project. 
- The Batters page focuses on the relationship between batting statistics and Hall of Fame induction. 
- The Pitchers page includes information about the relationship between pitching statistics and Hall of Fame Induction.
""")

st.markdown(
"""
### Batting Dataset Acronyms and Explanations:  
- G: The number of games played that season by the player.
- AB: The number of at bats the player had that season.
- R: The number of runs scored by the player that season.
- H: The number of hits the player got that season.
- 2B: The number of doubles the player got that season.
- 3B: The number of triples the player got that season.
- HR: The number of home runs the player got that season.
- RBI: The number of runs batted in by the player that season.
- SB: The number of stolen bases by the player that season.
- CS: The number of times the player got caught stealing that season.
- BB: The number of times that the player got a base on balls that season
- SO: The number of times that the player struck out that season.
- IBB: The number of times that the player got intentionally walked that season.
- HBP: The number of times that the player got hit by a pitch that season.
- SH: The number of sacrifice hits that the player hit that season.
- SF: The number of sacrifice flys that the player hit that season.
- GIDP: The number of times that the player grounded into a double play that season.
"""
)

st.markdown(
"""
### Pitching Dataset Acronyms and Explanations: 
- W: The number of wins that the pitcher earned that season.
- L: The number of losses that the pitcher earned that season.
- G: The number of games that the pitcher played that season.
- GS: The number of games that the pitcher started that season.
- CG: The number of complete games thrown by the pitcher that season.
- SHO: The number of shutouts thrown by the pitcher that season.
- SV: The number of saves made by the pitcher that season.
- IPouts: The number of innings pitched by the player that season.
- H: The number of hits allowed by the pitcher that season.
- ER: The number of earned runs allowed by the pitcher that season.
- HR: The number of home runs allowed by the pitcher that season.
- BB: The number of base on balls allowed by the pitcher that season.
- SO: The number of batters struck out by the pitcher that season.
- BAOpp: The batting average of batters facing the pitcher that season.
- ERA: The earned run average of the pitcher that season.
- IBB: The number of intentional walks given by the pitcher that season.
- WP: The number of wild pitches thrown by the pitcher that season.
- HBP: The number of batters hit by pitches thrown by the pitcher that season.
- BK: The number of balks created by the pitcher that season.
- BFP: The number of batters faced by the pitcher that season.
- GF: The number of games finished by the pitcher that season.
- R: The number of runs allowed by the pitcher that season.
- SH: The number of sacrifice hits allowed by the pitcher that season.
- SF: The number of sacrifice flys allowed by the pitcher that season.
- GIDP: The number of times batters grounded into a double play against the pitcher that season.
"""
)