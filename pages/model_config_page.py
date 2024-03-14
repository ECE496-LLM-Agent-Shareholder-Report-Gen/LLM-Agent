import streamlit as st
from GUI.llm_select_renderer import LLMRenderer
from GUI.shared import load_global_singleton, navbar

if not "global_singleton" in st.session_state:
    global_singleton = load_global_singleton()
    st.session_state["global_singleton"] = global_singleton
else: 
    global_singleton = st.session_state["global_singleton"]

navbar(global_singleton)

llm = LLMRenderer(global_singleton)
llm.render()
