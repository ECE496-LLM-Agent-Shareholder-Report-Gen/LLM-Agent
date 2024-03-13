import streamlit as st
from GUI.session_renderer import BenchmarkRenderer
from GUI.shared import load_global_singleton, navbar

if not "global_singleton" in st.session_state:
    global_singleton = load_global_singleton()
    st.session_state["global_singleton"] = global_singleton
else: 
    global_singleton = st.session_state["global_singleton"]

print("global:", global_singleton.__dict__)
navbar(global_singleton)

session = BenchmarkRenderer(global_singleton)
session.render()
