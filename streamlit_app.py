import streamlit as st
from GUI.session_renderer import SessionRenderer
from GUI.benchmark_renderer import BenchmarkRenderer
from GUI.shared import load_global_singleton
from GUI.navbar import navbar

if not "global_singleton" in st.session_state:
    global_singleton = load_global_singleton()
    st.session_state["global_singleton"] = global_singleton
else: 
    global_singleton = st.session_state["global_singleton"]

navbar(global_singleton)
# Main Content

session = BenchmarkRenderer(global_singleton)
session.render()


