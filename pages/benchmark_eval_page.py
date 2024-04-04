import streamlit as st
from GUI.benchmark_eval_renderer import BenchmarkEvalRenderer
from GUI.shared import load_global_singleton
from GUI.navbar import navbar

if not "global_singleton" in st.session_state:
    global_singleton = load_global_singleton()
    st.session_state["global_singleton"] = global_singleton
else: 
    global_singleton = st.session_state["global_singleton"]
if global_singleton.benchmark_session_manager.active_session:
    st.set_page_config(
        page_title=f"LLM Agent | {global_singleton.benchmark_session_manager.active_session.name}",
    )
else:
    st.set_page_config(
        page_title="LLM Agent | No Benchmark Selected",
    )
navbar(global_singleton)

session = BenchmarkEvalRenderer(global_singleton)
session.render()
