import streamlit as st
import os
from global_singleton import GlobalSingleton
from model_loader import LLMModelLoader, EmbeddingsLoader
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_openai import OpenAI, OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings, HuggingFaceEmbeddings
from sentence_transformers.cross_encoder import CrossEncoder
import torch
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline


def load_llm(_global_singleton):
    if _global_singleton.llm_path is not None:
        return load_llm_llama(_global_singleton.llm_path)
    elif _global_singleton.hug_llm_name is not None:
        return load_llm_huggingface(_global_singleton.hug_llm_name, _global_singleton.hug_api_key)
    elif _global_singleton.opai_llm_name is not None:
        return load_llm_openai(_global_singleton.opai_llm_name, _global_singleton.opai_api_key)
    else:
        return load_llm_default()

"""
def load_llm(_global_singleton):
    if _global_singleton.llm_type=="LLAMA":
        return load_llm_llama(_global_singleton.llm_path)
    elif _global_singleton.llm_type=="Huggingface":
        return load_llm_huggingface(_global_singleton.hug_llm_name, _global_singleton.hug_api_key)
    elif _global_singleton.llm_type=="Openai":
        return load_llm_openai(_global_singleton.opai_api_key)
    else:
        return load_llm_default()
"""
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
    print("loading huggingface llm: ", huggingface_model_name)
    with st.spinner(text="Loading model from HuggingFace – hang tight! This should take 1-2 minutes.") as spinner:
        token = "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"
        tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name,#"deepset/roberta-base-squad2",
                                                  #device_map='auto',
                                                  token = token,
                                                  cache_dir = "/groups/acmogrp/Large-Language-Model-Agent/app/cache_dir")
        model = AutoModelForCausalLM.from_pretrained(huggingface_model_name,#"deepset/roberta-base-squad2",
                                                    #device_map='auto',
                                                    torch_dtype=torch.float16,
                                                    load_in_8bit=True,
                                                    token = token,
                                                    cache_dir = "/groups/acmogrp/Large-Language-Model-Agent/app/cache_dir")
        #return model
        pipe = pipeline("text-generation",
                            model = model,
                            tokenizer = tokenizer,
                            #device_map = "auto",
                            min_new_tokens = -1,
                            top_k = 30,
                            #todo: change max_new_tokens since 385 not enough, but if tokens exceed some amount it crash, limit exceed
                            max_new_tokens = 385
                            )
        return HuggingFacePipeline(pipeline=pipe, model_kwargs = {"temperature": 0.1})
                                   #token = token
                                   #)
                                   #cache_dir = "/groups/acmogrp/Large-Language-Model-Agent/app/cache_dir")
        """return HuggingFacePipeline.from_model_id(model_id = model,
                                                     task = "text-generation",
                                                     tokenizer = tokenizer,
                                                     #device_map = "auto",
                                                     min_new_tokens = -1,
                                                     top_k = 30,
                                                     token = token)"""
        #model = AutoModelForCausalLM.from_pretrained(huggingface_model_name, token = "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", cache_dir = "/groups/acmogrp/Large-Language-Model-Agent/app/cache_dir")

@st.cache_resource
def load_llm_openai(opai_llm_name, openai_api_key):
    print("open ai llm loadun icine girdi")
    with st.spinner(text="Loading model from HuggingFace – hang tight! This should take 1-2 minutes.") as spinner:
        model = ChatOpenAI(opai_llm_name, openai_api_key = "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")
        #model = ChatOpenAI(model_name = "gpt-3.5-turbo", openai_api_key = "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")
        return model

############ word embedders:
"""
@st.cache_resource
def load_word_embedder():
    print("loading embeddings")
    with st.spinner(text="Loading and Embeddings model – hang tight! This should take 1-2 minutes."):
        embeddings_loader = EmbeddingsLoader()
        return embeddings_loader.load_bge()
"""

#todo: 1- initiate global_singleton.openai embedding name
def load_word_embedding(_global_singleton):
    if _global_singleton.embedding_type=="OpenAI":
        return load_word_embedder_openai(_global_singleton.opai_api_key)
    elif _global_singleton.embedding_type=="Huggingface":
        return load_word_embedder_huggingface(_global_singleton.hug_embedding_name)
    else:
        return load_word_embedder_default()

#todo: openai embeddings (get input from st.selectbox and this select box has options as "text-embedding-ada-002","text-embedding-3-large","text-embedding-3-small") or just ada embedding model
@st.cache_resource
def load_word_embedder_openai(openai_api_key):
    print("loading openai embeddings")
    with st.spinner(text="Loading and OpenAI Embeddings model – hang tight! This should take 1-2 minutes."):
        #todo: change api key with the users entry
        embeddings = OpenAIEmbeddings(openai_api_key="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")
        return embeddings

#todo: huggingface embeddings (locally downloaded, user chooses folder/file) or just show select options as what is in that path yada heryerden istedigin bi path secbilcegin file uploader
@st.cache_resource
def load_word_embedder_huggingface(model_name):
    print("loading huggingface embeddings")
    with st.spinner(text="Loading and HuggingFace Embeddings model – hang tight! This should take 1-2 minutes."):
        #model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            #model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return hf

@st.cache_resource
def load_word_embedder_default():
    print("loading default bge embeddings")
    with st.spinner(text="Loading and Embeddings model – hang tight! This should take 1-2 minutes."):
        embeddings_loader = EmbeddingsLoader()
        return embeddings_loader.load_bge()

##### cross encoder
#not being called anywhere rn
@st.cache_resource
def load_cross_encoder():
    print("loading cross encoder")
    with st.spinner(text="Loading and Cross Encoder model – hang tight! This should take 1-2 minutes."):
        ce_file_path = "/groups/acmogrp/Large-Language-Model-Agent/language_models/cross_encoder/BAAI_bge-reranker-large"
        return CrossEncoder(model_name=ce_file_path)

def load_global_singleton():
    global_singleton = GlobalSingleton(content_path="./content/companies",chat_session_path="./saved_session.json", benchmark_session_path="./benchmark_session.json")
    global_singleton.llm = load_llm(global_singleton)
    #global_singleton.embeddings = load_word_embedder()
    global_singleton.embeddings = load_word_embedding(global_singleton)
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
        st.write("Current LLM: ", global_singleton.llm)
        st.write("Current LLM Type: ", global_singleton.llm_type)
        #choose embedder
        st.divider()
        select_embedder = False
        select_embedder = st.button("Choose/Change Embedder", use_container_width=True)
        if select_embedder:
            st.switch_page("pages/embedding_config_page.py")
        st.write("Current Embedder: ", global_singleton.embeddings)
        st.write("Current Embedder Type: ", global_singleton.embedding_type)
        #st.write("Current Huggingface Embedder Name: ", global_singleton.hug_embedding_name)

