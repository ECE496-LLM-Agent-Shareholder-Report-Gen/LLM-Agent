import streamlit as st

from utility.global_singleton import GlobalSingleton
from agent.model_loader import EmbeddingsLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from sentence_transformers.cross_encoder import CrossEncoder

from langchain_openai import  OpenAIEmbeddings, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.chat_models import ChatOllama

import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now you can import your desired module
import config


import torch

"""
from numba import cuda
device = cuda.get_current_device()
device.reset()
"""

def load_llm(_global_singleton):
    if _global_singleton.llm_path is not None:
        return load_llm_llama(_global_singleton.llm_path, _global_singleton.llm_temp)
    elif _global_singleton.hug_llm_name is not None:
        return load_llm_huggingface(_global_singleton.hug_llm_name, _global_singleton.hug_api_key, _global_singleton)
    elif _global_singleton.opai_llm_name is not None:
        #if _global_singleton.opai_api_key is None:
            #return load_llm_openai(_global_singleton.opai_llm_name, "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")
        return load_llm_openai(_global_singleton.opai_llm_name, _global_singleton.opai_api_key, _global_singleton.llm_temp)
    else:
        return load_llm_default(_global_singleton)

#@st.cache_resource
def load_llm_default(_global_singleton):
    # print("loading default llm")
    with st.spinner(text="Loading default llama2-13b-chat model – hang tight! This should take 1-2 minutes.") as spinner:
        return ChatOllama(base_url=config.OLLAMA_URL, model="llama2-13b-chat", verbose =True), "llama 2 13b chat"

#@st.cache_resource
def load_llm_llama(llm_path, llm_temp):
    # print("loading llama llm")
    with st.spinner(text="Loading "+llm_path+" – hang tight! This should take 1-2 minutes.") as spinner:
        return ChatOllama(base_url=config.OLLAMA_URL, model=llm_path, verbose=True, temperature = llm_temp), llm_path

def load_llm_huggingface(huggingface_model_name, huggingface_api_key, _global_singleton):
    with st.spinner(text="Loading "+huggingface_model_name+" from HuggingFace – hang tight! This should take 1-2 minutes.") as spinner:
        #token = "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"
        token = huggingface_api_key
        tokenizer = AutoTokenizer.from_pretrained(huggingface_model_name,#"deepset/roberta-base-squad2",
                                                  #device_map='auto',
                                                  token = token,
                                                  cache_dir = "/groups/acmogrp/Large-Language-Model-Agent/app/cache_dir",
                                                  truncation = True)
       # _global_singleton.hug_tokenizer = tokenizer
        
        model = AutoModelForCausalLM.from_pretrained(huggingface_model_name,#"deepset/roberta-base-squad2",
                                                    #device_map='auto',
                                                    torch_dtype=torch.float16,
                                                    load_in_8bit=True,
                                                    token = token,
                                                    cache_dir = "/groups/acmogrp/Large-Language-Model-Agent/app/cache_dir")
                                                    #temperature = _global_singleton.llm_temp)
       # _global_singleton.hug_model = model
        #return model
        pipe = pipeline("text-generation",
                            model = model,
                            tokenizer = tokenizer,
                            #device_map = "auto",
                            min_new_tokens = -1,
                            #top_k = 30,
                            #todo: change max_new_tokens since 385 not enough, but if tokens exceed some amount it crash, limit exceed
                            #max_new_tokens = 385
                            max_length = 400
                            )
       # _global_singleton.hug_pipe = pipe
        return HuggingFacePipeline(pipeline=pipe, model_kwargs = {"temperature": _global_singleton.llm_temp}), huggingface_model_name

#@st.cache_resource
def load_llm_openai(opai_llm_name, openai_api_key,llm_temp):
    # print("open ai llm loadun icine girdi")
    with st.spinner(text="Loading "+opai_llm_name+" from OpenAI – hang tight! This should take 1-2 minutes.") as spinner:
        model = ChatOpenAI(model_name = opai_llm_name, openai_api_key = openai_api_key, temperature = llm_temp)
        return model, opai_llm_name


############ word embedders:
#todo: 1- initiate global_singleton.openai embedding name


def load_word_embedding(_global_singleton):
    if _global_singleton.opai_embedding_name is not None:
        return load_word_embedder_openai(_global_singleton.opai_embedding_name,_global_singleton.opai_api_key)
    elif _global_singleton.hug_embedding_name is not None:
        return load_word_embedder_huggingface(_global_singleton.hug_embedding_name)
    else:
        return load_word_embedder_default()
    


#todo: openai embeddings (get input from st.selectbox and this select box has options as "text-embedding-ada-002","text-embedding-3-large","text-embedding-3-small") or just ada embedding model
#@st.cache_resource
def load_word_embedder_openai(opai_embedding_name, openai_api_key):
    with st.spinner(text="Loading "+opai_embedding_name+" from OpenAI Embedding models – hang tight! This should take 1-2 minutes."):
        return OpenAIEmbeddings(model = opai_embedding_name, openai_api_key=openai_api_key,dimensions = 1024), opai_embedding_name

#todo: huggingface embeddings (locally downloaded, user chooses folder/file) or just show select options as what is in that path yada heryerden istedigin bi path secbilcegin file uploader
#@st.cache_resource
def load_word_embedder_huggingface(model_namee):
    with st.spinner(text="Loading "+ model_namee+" from HuggingFace Embedding models – hang tight! This should take 1-2 minutes."):
        encode_kwargs = {'normalize_embeddings': True}
        return HuggingFaceEmbeddings(
            model_name=model_namee,
            encode_kwargs=encode_kwargs,
            cache_folder =config.MODEL_DIR
        ), model_namee

#@st.cache_resource
def load_word_embedder_default():
    with st.spinner(text="Loading default embedding model – hang tight! This should take 1-2 minutes."):
        try:
            embeddings_loader = EmbeddingsLoader()
            return embeddings_loader.load_bge(), embeddings_loader.model_name
        except:
            return None, None

##### cross encoder
@st.cache_resource
def load_cross_encoder():
    with st.spinner(text="Loading the Cross Encoder model – hang tight! This should take 1-2 minutes."):
        ce_file_path = "/groups/acmogrp/Large-Language-Model-Agent/language_models/cross_encoder/BAAI_bge-reranker-large"
        cross_encoder_model = "BAAI/bge-reranker-large"
        return CrossEncoder(model_name=ce_file_path), cross_encoder_model

def load_global_singleton():
    global_singleton = GlobalSingleton(content_path="./content/companies",chat_session_path="./saved_session.json", benchmark_session_path="./benchmark_session.json")
    global_singleton.llm, global_singleton.llm_model = load_llm(global_singleton)
    global_singleton.embeddings, global_singleton.embeddings_model = load_word_embedding(global_singleton)
    global_singleton.cross_encoder, global_singleton.cross_encoder_model = load_cross_encoder()
    global_singleton.load_file_manager(index_name=global_singleton.embeddings_model)
    global_singleton.load_chat_session_manager()
    global_singleton.load_benchmark_session_manager()
    global_singleton.load_index_generator(index_name=global_singleton.embeddings_model)
    return global_singleton



