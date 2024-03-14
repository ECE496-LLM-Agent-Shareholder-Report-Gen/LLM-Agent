import streamlit as st
from GUI.llm_select_renderer import LLMRenderer
from GUI.shared import load_global_singleton, navbar

if not "global_singleton" in st.session_state:
    print("yarrrrrrrrrrrrrrrak")

    global_singleton = load_global_singleton()
    print("test",global_singleton.llm)
    st.session_state["global_singleton"] = global_singleton
else: 
    global_singleton = st.session_state["global_singleton"]
print("baaaaaaaaaaaaas",global_singleton.llm)
#print("global ama model_config_page de:", global_singleton.__dict__)

navbar(global_singleton)

llm = LLMRenderer(global_singleton)
llm.render()