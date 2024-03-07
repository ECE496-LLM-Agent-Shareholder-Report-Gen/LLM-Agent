import streamlit as st
from GUI.session_renderer import SessionRenderer
from GUI.shared import load_global_singleton, navbar
from global_singleton import GlobalSingleton
from model_loader import LLMModelLoader, EmbeddingsLoader



if not "global_singleton" in st.session_state:
    global_singleton = load_global_singleton()
    st.session_state["global_singleton"] = global_singleton
else: 
    global_singleton = st.session_state["global_singleton"]


navbar(global_singleton)
# Main Content

session = SessionRenderer(global_singleton)
session.render()


