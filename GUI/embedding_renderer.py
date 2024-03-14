import streamlit as st
import os
from global_singleton import GlobalSingleton
from GUI.shared import load_word_embedding

class EmbeddingRenderer:
    def __init__(self, global_singleton):
        self.global_singleton = global_singleton
        self.embedding_type = ""
        self.opai_embedding_name = ""
        self.hug_api_key = ""
        self.hug_embedding_name = ""
        self.opai_api_key = ""

    def render(self):
        self.render_info_for_dev()
        st.divider()
        self.render_embedding_selector()
        st.divider()
        self.render_load()
        st.divider()

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
        st.title("Choose the Embedding you would like to use")
        st.divider()
        st.subheader("Choose your Embedding", divider = "grey")
        left_col, right_col = st.columns(2)
        with left_col:
            self.embedding_type = st.selectbox("Select your Embedding type", options=["Huggingface","Openai"])
        with right_col:
            #if passing api key and model name only, load huggingfaceapi call, otherwise let the user upload the model and use it 
            if self.embedding_type == "Huggingface":
                self.hug_embedding_name = st.text_input("Please enter the name of the model on Huggingface", placeholder="Model name")
                self.hug_api_key = st.text_input("Please enter your Huggingface API access key", placeholder="Key")
                st.session_state["global_singleton"].hug_api_key = self.hug_api_key
                st.session_state["global_singleton"].embedding_type = self.embedding_type
                st.session_state["global_singleton"].hug_embedding_name = self.hug_embedding_name
                self.global_singleton.hug_api_key = self.hug_api_key
                self.global_singleton.hug_embedding_name = self.hug_embedding_name
                self.global_singleton.embedding_type = self.embedding_type
            if self.embedding_type == "Openai":
                self.opai_embedding_name = st.text_input("Please enter the name of the model on Openai", placeholder="Model name")
                self.opai_api_key = st.text_input("Please enter your openai api access key", placeholder="Key")
                st.session_state["global_singleton"].opai_api_key = self.opai_api_key
                st.session_state["global_singleton"].embedding_type = self.embedding_type
                st.session_state["global_singleton"].opai_embedding_name = self.opai_embedding_name
                self.global_singleton.opai_api_key = self.opai_api_key
                self.global_singleton.embedding_type = self.embedding_type
                self.global_singleton.opai_embedding_name = self.opai_embedding_name

    def render_load(self):
        load_embedding_button = False
        load_embedding_button = st.button("Load Embedding", use_container_width = True)
        if load_embedding_button:
            self.global_singleton.embeddings = load_word_embedding(self.global_singleton)
            load_embedding_button = False
