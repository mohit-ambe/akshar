import runpy
import streamlit as st

st.set_page_config(page_title="akshar", layout="centered")

st.title("akshar")

st.markdown("""
    <style>
    div[data-baseweb="tab-list"] {
        justify-content: center;
    }
    </style>
    """, unsafe_allow_html=True, )

demo_tab, about_tab = st.tabs(["Demo", "About"])

with about_tab:
    runpy.run_path("streamlit/about.py", run_name="__main__")

with demo_tab:
    runpy.run_path("streamlit/demo.py", run_name="__main__")