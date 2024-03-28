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
from streamlit_extras.stylable_container import stylable_container
import openai


def load_llm(_global_singleton):
    if _global_singleton.llm_path is not None:
        return load_llm_llama(_global_singleton.llm_path)
    elif _global_singleton.hug_llm_name is not None:
        return load_llm_huggingface(_global_singleton.hug_llm_name, _global_singleton.hug_api_key)
    elif _global_singleton.opai_llm_name is not None:
        if _global_singleton.opai_api_key is None:
            return load_llm_openai(_global_singleton.opai_llm_name, "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")
        else:
            return load_llm_openai(_global_singleton.opai_llm_name, _global_singleton.opai_api_key)
    else:
        return load_llm_default(_global_singleton)

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
def load_llm_default(_global_singleton):
    # print("loading default llm")
    with st.spinner(text="Loading and Llama 2 model – hang tight! This should take 1-2 minutes.") as spinner:
        llm_loader = LLMModelLoader(streaming=False, temperature=0)#"llama2-13b-chat", streaming=False, temperature=0)
        llm_model = "llama 2 13b chat"
        return llm_loader.load_ollama(model="llama2-13b-chat"), llm_model

@st.cache_resource
def load_llm_llama(llm_path):
    # print("loading llama llm")
    with st.spinner(text="Loading and Llama 2 model – hang tight! This should take 1-2 minutes.") as spinner:
        llm_loader = LLMModelLoader(llm_path, streaming=False, temperature=0)
        #print(3)
        return llm_loader.load(), llm_path

@st.cache_resource
#not checking wheter huggingface_model_name and huggingface_api_key are valid or not
def load_llm_huggingface(huggingface_model_name, huggingface_api_key):
    # print("loading huggingface llm: ", huggingface_model_name)
    with st.spinner(text="Loading model from HuggingFace – hang tight! This should take 1-2 minutes.") as spinner:
        #token = "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"
        token = huggingface_api_key
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
        return HuggingFacePipeline(pipeline=pipe, model_kwargs = {"temperature": 0.1}), huggingface_model_name
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
    # print("open ai llm loadun icine girdi")
    with st.spinner(text="Loading model from HuggingFace – hang tight! This should take 1-2 minutes.") as spinner:
        model = ChatOpenAI(model_name = opai_llm_name, openai_api_key = openai_api_key)
        #model = ChatOpenAI(model_name = "gpt-3.5-turbo", openai_api_key = "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")
        return model, opai_llm_name

############ word embedders:
"""
#sil bunu sonra for dev purposes
def load_llm(_global_singleton):
    if _global_singleton.llm_path is not None:
        return load_llm_llama(_global_singleton.llm_path)
    elif _global_singleton.hug_llm_name is not None:
        return load_llm_huggingface(_global_singleton.hug_llm_name, _global_singleton.hug_api_key)
    elif _global_singleton.opai_llm_name is not None:
        if _global_singleton.opai_api_key is None:
            return load_llm_openai(_global_singleton.opai_llm_name, "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")
        else:
            return load_llm_openai(_global_singleton.opai_llm_name, _global_singleton.opai_api_key)
    else:
        return load_llm_default(_global_singleton)
"""

#todo: 1- initiate global_singleton.openai embedding name
def load_word_embedding(_global_singleton):
    if _global_singleton.embedding_type=="OpenAI":
        return load_word_embedder_openai(_global_singleton.opai_embedding_name,_global_singleton.opai_api_key)
    elif _global_singleton.embedding_type=="Huggingface":
        return load_word_embedder_huggingface(_global_singleton.hug_embedding_name)
    else:
        return load_word_embedder_default()

""" 
def load_word_embedding(_global_singleton):
    if _global_singleton.opai_embedding_name is not None:
        return load_word_embedder_openai(_global_singleton.opai_embedding_name,_global_singleton.opai_api_key)
    elif _global_singleton.hug_embedding_name is not None:
        return load_word_embedder_huggingface(_global_singleton.hug_embedding_name)
    else:
        return load_word_embedder_default()
"""

#todo: openai embeddings (get input from st.selectbox and this select box has options as "text-embedding-ada-002","text-embedding-3-large","text-embedding-3-small") or just ada embedding model
@st.cache_resource
def load_word_embedder_openai(opai_embedding_name, openai_api_key):
    # print("loading openai embeddings")
    with st.spinner(text="Loading and OpenAI Embeddings model – hang tight! This should take 1-2 minutes."):
        #embeddings = OpenAIEmbeddings(openai_api_key="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")
        embeddings = OpenAIEmbeddings(model_name = opai_embedding_name, openai_api_key=openai_api_key)
        return embeddings, opai_embedding_name

#todo: huggingface embeddings (locally downloaded, user chooses folder/file) or just show select options as what is in that path yada heryerden istedigin bi path secbilcegin file uploader
@st.cache_resource
def load_word_embedder_huggingface(model_name):
    # print("loading huggingface embeddings")
    with st.spinner(text="Loading and HuggingFace Embeddings model – hang tight! This should take 1-2 minutes."):
        #model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}
        hf = HuggingFaceEmbeddings(
            model_name=model_name,
            #model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        return hf, model_name

@st.cache_resource
def load_word_embedder_default():
    # print("loading default bge embeddings")
    with st.spinner(text="Loading and Embeddings model – hang tight! This should take 1-2 minutes."):
        embeddings_loader = EmbeddingsLoader()
        return embeddings_loader.load_bge(), embeddings_loader.model_name

##### cross encoder
@st.cache_resource
def load_cross_encoder():
    # print("loading cross encoder")
    with st.spinner(text="Loading and Cross Encoder model – hang tight! This should take 1-2 minutes."):
        ce_file_path = "/groups/acmogrp/Large-Language-Model-Agent/language_models/cross_encoder/BAAI_bge-reranker-large"
        cross_encoder_model = "BAAI/bge-reranker-large"
        return CrossEncoder(model_name=ce_file_path), cross_encoder_model

def load_global_singleton():
    global_singleton = GlobalSingleton(content_path="./content/companies",chat_session_path="./saved_session.json", benchmark_session_path="./benchmark_session.json")
    global_singleton.llm, global_singleton.llm_model = load_llm(global_singleton)
    #global_singleton.embeddings = load_word_embedder()
    global_singleton.embeddings, global_singleton.embeddings_model = load_word_embedding(global_singleton)
    global_singleton.cross_encoder, global_singleton.cross_encoder_model = load_cross_encoder()
    global_singleton.load_file_manager(index_name=global_singleton.embeddings_model)
    global_singleton.load_chat_session_manager()
    global_singleton.load_benchmark_session_manager()
    global_singleton.load_index_generator(index_name=global_singleton.embeddings_model)
    return global_singleton


def link_clicked(session, session_manager):
    if session_manager.active_session != None:
        session_manager.active_session.deinitialize()
    session_manager.active_session = session

def navbar(global_singleton):
    # Sidebar
    with st.sidebar:
        st.subheader("Sessions", divider="grey")
        if global_singleton.chat_session_manager and global_singleton.chat_session_manager.sessions:
            switch_page = False
            for session_name, session in global_singleton.chat_session_manager.sessions.items():
                switch_page = False
                if session == global_singleton.chat_session_manager.active_session:
                     with stylable_container(
                        key="active_session",
                            css_styles="""
                                button {
                                    background-color: white;
                                    color: black;
                                }
                                """,
                        ):
                         switch_page = st.button(session_name, 
                                            key=session_name, 
                                            on_click=link_clicked, 
                                            args=[session, global_singleton.chat_session_manager], 
                                            use_container_width=True)
                         
                else:
                    switch_page = st.button(session_name, 
                                            key=session_name, 
                                            on_click=link_clicked, 
                                            args=[session, global_singleton.chat_session_manager], 
                                            use_container_width=True)
            if switch_page:
                if global_singleton.benchmark_session_manager.active_session:
                    global_singleton.benchmark_session_manager.active_session.deinitialize()
                st.switch_page("pages/chat_page.py")
            save_sessions = st.button("Save sessions", use_container_width=True)
            if save_sessions:
                global_singleton.chat_session_manager.save()
        else:
            st.markdown("No sessions")
        new_session = st.button("＋ Create new Session", use_container_width=True)
        if new_session:
            if global_singleton.benchmark_session_manager.active_session:
                    global_singleton.benchmark_session_manager.active_session.deinitialize()
                    global_singleton.benchmark_session_manager.active_session = None
            st.switch_page("pages/session_page.py")
        st.subheader("Benchmarks", divider="grey")
        if global_singleton.benchmark_session_manager and global_singleton.benchmark_session_manager.sessions:
            # benchmarks
            for session_name, session in global_singleton.benchmark_session_manager.sessions.items():
                switch_page = False
                if session == global_singleton.benchmark_session_manager.active_session:
                     with stylable_container(
                        key="active_session",
                            css_styles="""
                                button {
                                    background-color: white;
                                    color: black;
                                }
                                """,
                        ):
                        switch_page_b = st.button(session_name, 
                                            key=f"b_{session_name}", 
                                            on_click=link_clicked, 
                                            args=[session, global_singleton.benchmark_session_manager], 
                                            use_container_width=True)
                         
                else:
                    switch_page_b = st.button(session_name, 
                                            key=f"b_{session_name}", 
                                            on_click=link_clicked, 
                                            args=[session, global_singleton.benchmark_session_manager], 
                                            use_container_width=True)
            if switch_page_b:
                if global_singleton.chat_session_manager.active_session:
                    global_singleton.chat_session_manager.active_session.deinitialize()
                    global_singleton.chat_session_manager.active_session = None


                st.switch_page("pages/benchmark_eval_page.py")
            save_benchmarks = st.button("Save benchmarks", use_container_width=True, key="b_save")
            if save_benchmarks:
                global_singleton.benchmark_session_manager.save()

        else:
            st.markdown("No benchmarks")
        new_benchmark = st.button("＋ New Benchmark", use_container_width=True, key="b_create")
        if new_benchmark:
            if global_singleton.chat_session_manager.active_session:
                    global_singleton.chat_session_manager.active_session.deinitialize()
                    global_singleton.chat_session_manager.active_session = None
            st.switch_page("pages/benchmark_page.py")

        # choose llm
        st.header("LLMs", divider="grey")
        st.markdown(f"LLM: {global_singleton.llm_model}")
        st.markdown(f"Embeddings: {global_singleton.embeddings_model}")
        st.markdown(f"Cross Encoder: {global_singleton.cross_encoder_model}")
        select_llm = st.button("Load LLMs", use_container_width=True)
        if select_llm:
            st.switch_page("pages/model_config_page.py")

def check_openai_api_key(api_key):
    client = openai.OpenAI(api_key=api_key)
    try:
        client.models.list()
    except openai.AuthenticationError:
        return False
    else:
        return True