import streamlit as st
from GUI.llm_select_renderer import LLMRenderer
from GUI.shared import load_global_singleton
from GUI.navbar import navbar


if not "global_singleton" in st.session_state:
    global_singleton = load_global_singleton()
    st.session_state["global_singleton"] = global_singleton
else: 
    global_singleton = st.session_state["global_singleton"]

#navbar(global_singleton)
#old_llm = global_singleton.llm
st.set_page_config(
    page_title=f"LLM Agent | Model Selection",
)
llm = LLMRenderer(global_singleton)
llm.render()
navbar(global_singleton)


#st.rerun()
