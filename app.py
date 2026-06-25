import streamlit as st

st.set_page_config(
    page_title="Keep Track",
    page_icon="📘",
    layout="wide"
)

home_page = st.Page(
    "pages/home.py",
    title="Home",
)

vector_page = st.Page(
    "pages/vector_database_and_embedding.py",
    title="Vector Database and Embedding",
    
)

distance_page = st.Page(
    "pages/distance_metrics_and_loss_functions.py",
    title="Distance Metrics and Loss Functions",
)

pg = st.navigation(
    {
        "Keep Track": [home_page, vector_page, distance_page]
    }
)

pg.run()



