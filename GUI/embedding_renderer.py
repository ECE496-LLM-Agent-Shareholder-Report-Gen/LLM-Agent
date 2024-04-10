import streamlit as st

from GUI.shared import load_word_embedding
from utility.test_open_key import check_openai_api_key


class EmbeddingRenderer:
    def __init__(self, global_singleton, dev=False):
        self.global_singleton = global_singleton
        self.dev = dev
        self.embedding_type = ""
        self.opai_embedding_name = ""
        self.hug_api_key = ""
        self.hug_embedding_name = ""
        self.opai_api_key = ""
        self.embedder_load_button_block = False

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
            if self.embedding_type == "None":
                self.embedder_load_button_block = True
            #if passing api key and model name only, load huggingfaceapi call, otherwise let the user upload the model and use it 
            if self.embedding_type == "Huggingface":
                self.embedder_load_button_block = False
                self.hug_embedding_name = st.text_input("Embedding Model", placeholder="Model name",key="huggingface_embed_name")
                self.global_singleton.embedding_type = self.embedding_type

                self.global_singleton.hug_embedding_name = self.hug_embedding_name

                self.global_singleton.opai_embedding_name = None

            if self.embedding_type == "Openai":
                self.embedder_load_button_block = False
                #self.opai_embedding_name = st.text_input("Please enter the name of the model on Openai", placeholder="Model name")
                #self.opai_embedding_name = st.selectbox("Embedding Model", options=["text-embedding-3-small","text-embedding-3-large", "text-embedding-ada-002"],key="opai_embed_name")#options=list(LLMModelLoader.AVAILABLE_MODELS.keys()))#, key="llm_path")
                self.opai_embedding_name = st.selectbox("Embedding Model", options=["text-embedding-3-small","text-embedding-3-large"],key="opai_embed_name")#options=list(LLMModelLoader.AVAILABLE_MODELS.keys()))#, key="llm_path")

                self.opai_api_key = st.text_input("Openai API access key", placeholder="Key",type='password',key="opai_api_key_embed")
                self.global_singleton.embedding_type = self.embedding_type
                self.global_singleton.opai_api_key = self.opai_api_key
                self.global_singleton.opai_embedding_name = self.opai_embedding_name

                self.global_singleton.hug_embedding_name = None

    def render_load(self):
        load_embedding_button = False
        load_embedding_button = st.button("Load Embedding", use_container_width = True, disabled = self.embedder_load_button_block,key="load_embedding_button")
        if load_embedding_button:
            #for openai
            if self.global_singleton.opai_embedding_name is not None:
                if self.global_singleton.opai_api_key is None:
                    st.error("Please enter an OpenAI API key (not set by the user)")
                else:
                    if check_openai_api_key(self.global_singleton.opai_api_key):
                        self.global_singleton.embeddings, self.global_singleton.embeddings_model = load_word_embedding(self.global_singleton)
                    else:
                        st.error("Invalid OpenAI API key")
            #for huggingface
            elif self.global_singleton.hug_embedding_name is not None:
                    self.global_singleton.embeddings, self.global_singleton.embeddings_model = load_word_embedding(self.global_singleton)
            else:
                st.error("managed to break the app!")
                self.global_singleton.embeddings, self.global_singleton.embeddings_model = load_word_embedding(self.global_singleton)
                    
