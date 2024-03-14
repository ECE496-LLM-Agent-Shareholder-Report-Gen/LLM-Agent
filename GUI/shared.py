import streamlit as st
import os
from global_singleton import GlobalSingleton
from model_loader import LLMModelLoader, EmbeddingsLoader
from transformers import AutoModel, AutoModelForCausalLM
from langchain_openai import OpenAI

#@st.cache_resource
def load_llm(_global_singleton):
    if _global_singleton.llm_type=="LLAMA":
        return load_llm_llama(_global_singleton.llm_path)
    elif _global_singleton.llm_type=="Huggingface":
        return load_llm_huggingface(_global_singleton.hug_llm_name, _global_singleton.hug_api_key)
    elif _global_singleton.llm_type=="Openai":
        return load_llm_openai(_global_singleton.opai_api_key)
    else:
        return load_llm_default()

@st.cache_resource
def load_llm_default():
    print("loading default llm")
    with st.spinner(text="Loading and Llama 2 model – hang tight! This should take 1-2 minutes.") as spinner:
        llm_loader = LLMModelLoader(streaming=False, temperature=0)#"llama2-13b-chat", streaming=False, temperature=0)
        return llm_loader.load_ollama(model="llama2-13b-chat")

@st.cache_resource
def load_llm_llama(llm_path):
    print("loading llama llm")
    with st.spinner(text="Loading and Llama 2 model – hang tight! This should take 1-2 minutes.") as spinner:
        llm_loader = LLMModelLoader(llm_path, streaming=False, temperature=0)
        return llm_loader.load()

@st.cache_resource
#not checking wheter huggingface_model_name and huggingface_api_key are valid or not
def load_llm_huggingface(huggingface_model_name, huggingface_api_key):
    print("huggingface load llm in icine girdi")
    with st.spinner(text="Loading model from HuggingFace – hang tight! This should take 1-2 minutes.") as spinner:
        model = AutoModelForCausalLM.from_pretrained(huggingface_model_name, token = "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")
        #model = AutoModel.from_pretrained(huggingface_model_name, token = "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")#, api_key=huggingface_api_key)
        print(model)
        return model

@st.cache_resource
def load_llm_openai(openai_api_key):
    print("open ai llm loadun icine girdi")
    model = OpenAI(openai_api_key = "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")
    print(model)
    return model
            
@st.cache_resource
def load_word_embedder():
    print("loading embeddings")
    with st.spinner(text="Loading and Embeddings model – hang tight! This should take 1-2 minutes."):
        embeddings_loader = EmbeddingsLoader()
        return embeddings_loader.load_bge()


#@st.cache_data
def load_global_singleton():
    global_singleton = GlobalSingleton(content_path="./content/companies",chat_session_path="./saved_session.json", benchmark_session_path="./benchmark_session.json")
    global_singleton.llm = load_llm(global_singleton)
    global_singleton.embeddings = load_word_embedder()
    global_singleton.load_file_manager()
    global_singleton.load_chat_session_manager()
    global_singleton.load_benchmark_session_manager()
    global_singleton.load_index_generator()
    return global_singleton


def link_clicked(session, session_manager):
    session_manager.active_session = session

def navbar(global_singleton):
    # Sidebar
    with st.sidebar:
        st.subheader("Sessions", divider="grey")
        if global_singleton.chat_session_manager and global_singleton.chat_session_manager.sessions:
            switch_page = False
            for session_name, session in global_singleton.chat_session_manager.sessions.items():
                switch_page = st.button(session_name, key=session_name, on_click=link_clicked, args=[session, global_singleton.chat_session_manager], use_container_width=True)
            if switch_page:
                st.switch_page("pages/chat_page.py")
            save_sessions = st.button("Save sessions", use_container_width=True)
            if save_sessions:
                global_singleton.chat_session_manager.save()
        else:
            st.markdown("No sessions")
        new_session = st.button("＋ Create new Session", use_container_width=True)
        if new_session:
            st.switch_page("pages/session_page.py")
        st.subheader("Benchmarks", divider="grey")
        if global_singleton.benchmark_session_manager and global_singleton.benchmark_session_manager.sessions:
            # benchmarks
            for session_name, session in global_singleton.benchmark_session_manager.sessions.items():
                switch_page = st.button(session_name, key=f"b_{session_name}", on_click=link_clicked, args=[session, global_singleton.benchmark_session_manager], use_container_width=True)
            if switch_page:
                st.switch_page("pages/benchmark_page.py")
            save_benchmarks = st.button("Save benchmarks", use_container_width=True, key="b_save")
            if save_benchmarks:
                pass
        else:
            st.markdown("No benchmarks")
        new_benchmark = st.button("＋ New Benchmark", use_container_width=True, key="b_create")
        if new_benchmark:
            st.switch_page("pages/benchmark_page.py")


            

        # choose llm
        st.divider()
        select_llm = False
        select_llm = st.button("Choose/Change LLM", use_container_width=True)
        if select_llm:
            st.switch_page("pages/model_config_page.py")
