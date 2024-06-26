import streamlit as st

from GUI.session_renderer import SessionRenderer
from GUI.shared import load_global_singleton
from GUI.navbar import navbar


st.set_page_config(
    page_title=f"LLM Agent | Complete New Benchmark",
)
if not "global_singleton" in st.session_state:
    global_singleton = load_global_singleton()
    st.session_state["global_singleton"] = global_singleton
else: 
    global_singleton = st.session_state["global_singleton"]
navbar(global_singleton)

session = SessionRenderer(global_singleton, isBenchmark=True)
session.render()
