import streamlit as st

from global_singleton import GlobalSingleton
from model_loader import LLMModelLoader, EmbeddingsLoader

@st.cache_resource
def load_llm():
    print("loading llm")
    with st.spinner(text="Loading and Llama 2 model – hang tight! This should take 1-2 minutes."):
        llm_loader = LLMModelLoader("llama-2-13b-chat", streaming=False, temperature=0)
        return llm_loader.load_ollama(model="llama2-13b-chat")

@st.cache_resource
def load_word_embedder():
    print("loading embeddings")
    with st.spinner(text="Loading and Embeddings model – hang tight! This should take 1-2 minutes."):
        embeddings_loader = EmbeddingsLoader()
        return embeddings_loader.load_bge()


@st.cache_data
def load_global_singleton():
    global_singleton = GlobalSingleton(content_path="./content/companies",session_path="./saved_session.json", )
    global_singleton.llm = load_llm()
    global_singleton.embeddings = load_word_embedder()
    global_singleton.load_file_manager()
    global_singleton.load_session_manager()
    print("done loading global singleton!!!")
    return global_singleton


def link_clicked(session, global_singleton):
    global_singleton.session_manager.active_session = session

def navbar(global_singleton):
    # Sidebar
    with st.sidebar:
        if global_singleton.session_manager.sessions:
            switch_page = False
            for session_name, session in global_singleton.session_manager.sessions.items():
                switch_page = st.button(session_name, key=session_name, on_click=link_clicked, args=[session, global_singleton], use_container_width=True)
            if switch_page:
                st.switch_page("pages/chat_page.py")
            st.divider()
            new_session = st.button("＋ Create new Session", use_container_width=True)
            if new_session:
                st.switch_page("pages/session_page.py")
            save_sessions = st.button("Save sessions", use_container_width=True)
            if save_sessions:
                global_singleton.session_manager.save()

        else:
            st.markdown("No sessions")