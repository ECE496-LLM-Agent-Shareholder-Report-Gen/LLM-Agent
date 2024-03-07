import streamlit as st
from GUI.session_renderer import SessionPage
from global_singleton import GlobalSingleton
from model_loader import LLMModelLoader, EmbeddingsLoader

@st.cache_data
def load_global_singleton():
    global_singleton = GlobalSingleton()
    # with st.spinner(text="Loading and Llama 2 model – hang tight! This should take 1-2 minutes."):
    #     llm_loader = LLMModelLoader("llama-2-13b-chat", streaming=True, temperature=0)
    #     global_singleton.llm = llm_loader.load()
    # with st.spinner(text="Loading and Embeddings model – hang tight! This should take 1-2 minutes."):
    #     embeddings_loader = EmbeddingsLoader()
    #     global_singleton.embeddings = embeddings_loader.load_bge()
    # print("done loading global singleton!!!")
    return global_singleton

global_singleton = load_global_singleton()


# Sidebar
with st.sidebar:
    st.title('Sidebar')
    'Session 1'
    'Session 2'
    'Session 3'

# Main Content
session = SessionPage(global_singleton)
session.render()
