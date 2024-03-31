import streamlit as st
from GUI.embedding_renderer import EmbeddingRenderer
from GUI.shared import load_global_singleton
from GUI.navbar import navbar


if not "global_singleton" in st.session_state:
    global_singleton = load_global_singleton()
    st.session_state["global_singleton"] = global_singleton
else: 
    global_singleton = st.session_state["global_singleton"]

navbar(global_singleton)
llm = EmbeddingRenderer(global_singleton)
llm.render()