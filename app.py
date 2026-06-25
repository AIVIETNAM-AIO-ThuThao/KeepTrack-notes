import streamlit as st
from datetime import datetime

st.set_page_config(
    page_title="Keep Track",
    page_icon="📘",
    layout="wide"
)

today = datetime.now()

# ===== TITLE =====
st.title("Keep Track")

# ===== DAY COUNTER / DATE =====
col1, col2 = st.columns([1, 3])

with col1:
    st.info("**Day 3**")

with col2:
    st.caption(today.strftime("%B %d, %Y"))

# ===== INTRODUCTION =====
st.markdown("""
This page is used to **keep track of study materials and lecture notes**.

It serves as a personal learning note page where important concepts, definitions, explanations, and examples are collected in a structured format for later review.
""")

# ===== CONTENTS =====
st.markdown("## Topics Covered")

st.markdown("""
### 1. Vector Database and Embedding Indexing
- Overview of vector databases
- Embedding representation
- Flat Index
- Inverted File Index (IVF)
- Product Quantization (PQ)

### 2. Distance Metrics and Loss Functions
- Euclidean distance
- Manhattan distance
- Cosine similarity
- Common loss functions in embedding learning
""")

# ===== NAVIGATION NOTE =====
st.info("Use the sidebar on the left to open each note page.")

# ===== FOOTER =====
st.markdown("---")
st.write("Keep Track • Lecture Notes")
