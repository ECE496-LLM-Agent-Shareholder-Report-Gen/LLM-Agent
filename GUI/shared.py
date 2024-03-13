import streamlit as st
import os
from global_singleton import GlobalSingleton
from model_loader import LLMModelLoader, EmbeddingsLoader

#@st.cache_resource
def load_llm(_global_singleton):
    print("loading llm:yaninda yazicak", _global_singleton)
    print("yine icerde bu sefer", _global_singleton.llm)
    #print(_global_singleton)
    if _global_singleton.llm_type=="LLAMA":
        load_llm_llama(_global_singleton.llm_path)
        print("asdasdasdasdasdasdasdas")
        print(_global_singleton.llm)
        print(_global_singleton)
    elif _global_singleton.llm_type=="Huggingface":
        load_llm_huggingface(_global_singleton.hug_llm_name, _global_singleton.hug_api_key)
        #return load_huggingface(_self.global_singleton.hug_llm_name) bu methoda st.cache_resource eklemek gerekebilir ekle
    elif _global_singleton.llm_type=="Openai":
        load_llm_openai(_global_singleton.opai_api_key)
    else:
        load_llm_default()
        print("def asdasdasdasdasdasdasd")
        print(_global_singleton.llm)
        print(_global_singleton)
"""
@st.cache_resource
def load_llm():
    print("loading default llm")
    with st.spinner(text="Loading and Llama 2 model – hang tight! This should take 1-2 minutes.") as spinner:
        llm_loader = LLMModelLoader(streaming=False, temperature=0)#"llama2-13b-chat", streaming=False, temperature=0)
        return llm_loader.load_ollama(model="llama2")
"""

@st.cache_resource
def load_llm_default():
    print("loading default llm")
    with st.spinner(text="Loading and Llama 2 model – hang tight! This should take 1-2 minutes.") as spinner:
        llm_loader = LLMModelLoader(streaming=False, temperature=0)#"llama2-13b-chat", streaming=False, temperature=0)
        return llm_loader.load_ollama(model="llama2")

@st.cache_resource
def load_llm_llama(llm_path):
    print("loading llama llm")
    with st.spinner(text="Loading and Llama 2 model – hang tight! This should take 1-2 minutes.") as spinner:
        llm_loader = LLMModelLoader(llm_path, streaming=False, temperature=0)
        return llm_loader.load()

@st.cache_resource
def load_llm_huggingface(huggingface_model_name, huggingface_api_key):
    print("huggingface load llm in icine girdi")
    pass

@st.cache_resource
def load_llm_openai(openai_api_key):
    print("open ai llm loadun icine girdi")
    pass

            
@st.cache_resource
def load_word_embedder():
    print("loading embeddings")
    with st.spinner(text="Loading and Embeddings model – hang tight! This should take 1-2 minutes."):
        embeddings_loader = EmbeddingsLoader()
        return embeddings_loader.load_bge()


@st.cache_data
def load_global_singleton():
    global_singleton = GlobalSingleton(content_path="./content/companies",session_path="./saved_session.json", )
    global_singleton.llm = load_llm(global_singleton)
    #global_singleton.llm = load_llm(global_singleton)
    #print("loading global singleton", global_singleton.llm)
    #print("load global singleton icinde", global_singleton)
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
            st.subheader("Sessions", divider="grey")
            for session_name, session in global_singleton.session_manager.sessions.items():
                switch_page = st.button(session_name, key=session_name, on_click=link_clicked, args=[session, global_singleton], use_container_width=True)
            if switch_page:
                st.switch_page("pages/chat_page.py")
            new_session = st.button("＋ Create new Session", use_container_width=True)
            if new_session:
                st.switch_page("pages/session_page.py")
            save_sessions = st.button("Save sessions", use_container_width=True)
            if save_sessions:
                global_singleton.session_manager.save()

            # benchmarks
            st.subheader("Benchmarks", divider="grey")
            for session_name, session in global_singleton.session_manager.sessions.items():
                switch_page = st.button(session_name, key=f"b_{session_name}", on_click=link_clicked, args=[session, global_singleton], use_container_width=True)
            if switch_page:
                st.switch_page("pages/benchmark_page.py")
            new_benchmark = st.button("＋ New Benchmark", use_container_width=True, key="b_create")
            if new_benchmark:
                st.switch_page("pages/benchmark_page.py")
            save_benchmarks = st.button("Save benchmarks", use_container_width=True, key="b_save")
            if save_benchmarks:
                pass

            # choose llm
            st.divider()
            select_llm = False
            select_llm = st.button("Choose/Change LLM", use_container_width=True)
            if select_llm:
                st.switch_page("pages/model_config_page.py")
            
        else:
            st.markdown("No sessions")
            st.divider()
            select_llm = False
            select_llm = st.button("Choose/Change LLM", use_container_width=True)
            if select_llm:
                st.switch_page("pages/model_config_page.py")
