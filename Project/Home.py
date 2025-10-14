import streamlit as st

st.set_page_config(layout="wide", page_title="Baseball Hall of Fame Dashboard")

st.markdown(
    """
    <div style='text-align: center; padding-bottom: 10px;'>
        <h1 style='color: #041E42;'>⚾Baseball Hall of Fame Dashboard⚾</h1>
        <p>Select a page from the sidebar to explore batter and pitcher statistics.</p>
    </div>
    """,
    unsafe_allow_html=True
)