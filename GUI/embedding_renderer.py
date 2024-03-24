import streamlit as st
import os
from global_singleton import GlobalSingleton
from GUI.shared import load_word_embedding
from GUI.test_open_key import check_openai_api_key, check_hug_key


class EmbeddingRenderer:
    def __init__(self, global_singleton, dev=False):
        self.global_singleton = global_singleton
        self.dev = dev
        self.embedding_type = ""
        self.opai_embedding_name = ""
        self.hug_api_key = ""
        self.hug_embedding_name = ""
        self.opai_api_key = ""

    def render(self):
        if self.dev:
            self.render_info_for_dev()
            st.divider()
            self.render_header()
        self.render_embedding_selector()
        self.render_load()

    def render_header(self):
        st.title("Choose the Embedding you would like to use")

    def render_info_for_dev(self):
        st.title("self.global_singleton.embeddings")
        st.markdown(self.global_singleton.embeddings)
        st.divider()
        st.title("session_state global singleton.embeddings")
        st.markdown(st.session_state["global_singleton"].embeddings)
        st.divider()
        st.markdown(self.global_singleton)
        st.divider()

    def render_embedding_selector(self):
        st.subheader("Choose your Embeddings Model", divider = "grey")
        left_col, right_col = st.columns(2)
        with left_col:
            self.embedding_type = st.selectbox("Select your Embedding type", options=["None", "Huggingface","Openai"])
        with right_col:
            #if passing api key and model name only, load huggingfaceapi call, otherwise let the user upload the model and use it 
            if self.embedding_type == "Huggingface":
                self.hug_embedding_name = st.text_input("Please enter the name of the model on Huggingface", placeholder="Model name")
                self.hug_api_key = st.text_input("Please enter your Huggingface API access key", placeholder="Key")
                self.global_singleton.embedding_type = self.embedding_type
                self.global_singleton.hug_api_key = self.hug_api_key
                self.global_singleton.hug_embedding_name = self.hug_embedding_name

                self.global_singleton.opai_embedding_name = None

            if self.embedding_type == "Openai":
                #self.opai_embedding_name = st.text_input("Please enter the name of the model on Openai", placeholder="Model name")
                self.opai_embedding_name = st.selectbox("Embedding model", options=["text-embedding-3-small","text-embedding-3-large", "text-embedding-ada-002"])#options=list(LLMModelLoader.AVAILABLE_MODELS.keys()))#, key="llm_path")
                self.opai_api_key = st.text_input("Please enter your openai api access key", placeholder="Key")

                self.global_singleton.embedding_type = self.embedding_type
                self.global_singleton.opai_api_key = self.opai_api_key
                self.global_singleton.opai_embedding_name = self.opai_embedding_name

                self.global_singleton.hug_embedding_name = None

    def render_load(self):
        load_embedding_button = False
        load_embedding_button = st.button("Load Embedding", use_container_width = True)
        if load_embedding_button:
            #for openai
            if self.global_singleton.opai_embedding_name is not None:
                if self.global_singleton.opai_api_key is None:
                    st.error("Please enter an OpenAI API key (not set by the user)")
                else:
                    if check_openai_api_key(self.global_singleton.opai_api_key):
                        self.global_singleton.embeddings = load_word_embedding(self.global_singleton)
                    else:
                        st.error("Invalid OpenAI API key")
            #for huggingface
            elif self.global_singleton.hug_embedding_name is not None:
                if self.global_singleton.hug_api_key is None:
                    st.error("Please enter a Huggingface API key (not set by the user)")
                else:
                    if check_hug_key(self.global_singleton.hug_api_key):
                        self.global_singleton.embeddings = load_word_embedding(self.global_singleton)
                    else:
                        st.error("Invalid Huggingface API key")
            else:
                print("we def dont see this normally")
                st.error("managed to break the app!")
                self.global_singleton.embeddings = load_word_embedding(self.global_singleton)
                    

"""
    def render_load(self):
        load_embedding_button = False
        load_embedding_button = st.button("Load Embedding", use_container_width = True)
        if load_embedding_button:
            self.global_singleton.embeddings = load_word_embedding(self.global_singleton)
            load_embedding_button = False"""