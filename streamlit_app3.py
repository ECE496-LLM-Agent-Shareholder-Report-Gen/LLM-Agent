import streamlit as st
from GUI.session_renderer import SessionRenderer
from GUI.shared import load_global_singleton, navbar
#from GUI.llm_select_renderer import

if not "global_singleton" in st.session_state:
    global_singleton = load_global_singleton()
    st.session_state["global_singleton"] = global_singleton
else: 
    global_singleton = st.session_state["global_singleton"]

#print("global:", global_singleton.__dict__)

#llm_selection = llm

navbar(global_singleton)

session = SessionRenderer(global_singleton)
session.render()
