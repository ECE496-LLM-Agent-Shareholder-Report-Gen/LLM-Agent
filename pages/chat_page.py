import streamlit as st

from GUI.chat_renderer import ChatRenderer
from GUI.shared import load_global_singleton
from GUI.navbar import navbar


if not "global_singleton" in st.session_state:
    global_singleton = load_global_singleton()
    st.session_state["global_singleton"] = global_singleton
else: 
    global_singleton = st.session_state["global_singleton"]
if global_singleton.chat_session_manager.active_session:
    st.set_page_config(
        page_title=f"LLM Agent | {global_singleton.chat_session_manager.active_session.name}",
    )
else:
    st.set_page_config(
        page_title="LLM Agent | No Chat Session Selected",
    )
navbar(global_singleton)

chat = ChatRenderer(global_singleton)
chat.render()
