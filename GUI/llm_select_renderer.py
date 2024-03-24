import streamlit as st
import os
from GUI.embedding_renderer import EmbeddingRenderer
from global_singleton import GlobalSingleton
from model_loader import LLMModelLoader
from GUI.shared import load_global_singleton, load_llm
from GUI.test_open_key import check_openai_api_key, check_hug_key

#from GUI.shared import 
#def not working might have messed up other parts, if you see this message and need me to fix it back message me on discord - mert
class LLMRenderer:
    #def __init__(self, global_singleton, llm_path = None, llm_type = None, hug_api_key = None, hug_llm_name = None, opai_api_key = None):
    def __init__(self, global_singleton,llm_path = "", llm_type = "", hug_api_key = "", hug_llm_name = "", opai_api_key = "", opai_llm_name = "",  dev=False):
        self.global_singleton = global_singleton
        self.embedding_renderer = self.load_embedding_renderer(global_singleton, _dev=dev)
        self.dev = dev
        self.llm_path = llm_path
        self.llm_type = llm_type
        self.hug_api_key = hug_api_key
        self.hug_llm_name = hug_llm_name
        self.opai_api_key = opai_api_key
        self.opai_llm_name = opai_llm_name
        if "llm_path" not in st.session_state:
            st.session_state["llm_path"] = ""
        if "llm_type" not in st.session_state:
            st.session_state["llm_type"] = ""
        if "hug_api_key" not in st.session_state:
            st.session_state["hug_api_key"] = ""
        if "hug_llm_name" not in st.session_state:
            st.session_state["hug_llm_name"] = ""
        if "opai_api_key" not in st.session_state:
            st.session_state["opai_api_key"] = ""
        if "opai_llm_name" not in st.session_state:
            st.session_state["opai_llm_name"] = ""

    @st.cache_resource
    def load_embedding_renderer(_self, _global_singleton, _dev=False):
        return EmbeddingRenderer(_global_singleton, dev=_dev)

    def render(self):
        if self.dev:
            self.render_info_for_dev()
            st.divider()
        self.render_header()
        self.render_llm_selector()
        self.render_load()
        self.embedding_renderer.render()

    def render_info_for_dev(self):
        if "llm_path" not in st.session_state:
            st.session_state["llm_path"] = ""
        if "llm_type" not in st.session_state:
            st.session_state["llm_type"] = ""
        if "hug_api_key" not in st.session_state:
            st.session_state["hug_api_key"] = ""
        if "hug_llm_name" not in st.session_state:
            st.session_state["hug_llm_name"] = ""
        if "opai_api_key" not in st.session_state:
            st.session_state["opai_api_key"] = ""
        if "opai_llm_name" not in st.session_state:
            st.session_state["opai_llm_name"] = ""
        st.title("self.global_singleton.llm")
        st.markdown(self.global_singleton.llm)
        st.divider()
        st.title("session_state global singleton.llm")
        st.markdown(st.session_state["global_singleton"].llm)
        st.divider()
        st.markdown(st.session_state["llm_type"])
        st.divider()
        st.markdown(self.global_singleton)
        st.divider()

    def render_header(self):
        st.title("Choose the LLM you would like to chat with")

    def render_llm_selector(self):
        if "llm_path" not in st.session_state:
            st.session_state["llm_path"] = ""
        if "llm_type" not in st.session_state:
            st.session_state["llm_type"] = ""
        if "hug_api_key" not in st.session_state:
            st.session_state["hug_api_key"] = ""
        if "hug_llm_name" not in st.session_state:
            st.session_state["hug_llm_name"] = ""
        if "opai_api_key" not in st.session_state:
            st.session_state["opai_api_key"] = ""
        if "opai_llm_name" not in st.session_state:
            st.session_state["opai_llm_name"] = ""
        
        if "llm_type_options" not in st.session_state:
            st.session_state["llm_type_options"] = ["None", "LLAMA","Huggingface", "Openai"]   
        st.subheader("Choose your Large Language Model", divider = "grey")

        left_col, right_col = st.columns(2)
        with left_col:
            self.llm_type = st.selectbox("Select your LLM type", options=["None", "LLAMA", "Huggingface", "Openai"])#, key="llm_type")
        with right_col:
            if self.llm_type == "LLAMA":
                self.llm_path = st.selectbox("LLM", options=list(LLMModelLoader.AVAILABLE_MODELS.keys()))#, key="llm_path")
                st.session_state["global_singleton"].llm_path = self.llm_path
                st.session_state["global_singleton"].llm_type = self.llm_type
                self.global_singleton.llm_type = self.llm_type
                self.global_singleton.llm_path = self.llm_path

                self.global_singleton.hug_llm_name = None
                st.session_state["global_singleton"].hug_llm_name = None
                self.global_singleton.opai_llm_name = None
                st.session_state["global_singleton"].opai_llm_name = None

            if self.llm_type == 'Huggingface':
                self.hug_api_key = st.text_input("Please enter your Huggingface API access key", placeholder="Key")#, key="huggingface_api_key")
                self.hug_llm_name = st.text_input("Please enter the name of the model on Huggingface", placeholder="Model name")#, key = "huggingface_model_name")
                st.session_state["global_singleton"].hug_llm_name = self.hug_llm_name
                st.session_state["global_singleton"].hug_api_key = self.hug_api_key
                st.session_state["global_singleton"].llm_type = self.llm_type
                self.global_singleton.hug_llm_name = self.hug_llm_name
                self.global_singleton.hug_api_key = self.hug_api_key
                self.global_singleton.llm_type = self.llm_type

                self.global_singleton.llm_path = None
                st.session_state["global_singleton"].llm_path = None
                self.global_singleton.opai_llm_name = None
                st.session_state["global_singleton"].opai_llm_name = None

            if self.llm_type == "Openai":
                self.opai_api_key = st.text_input("Please enter your openai api access key", placeholder="Key")#, key="opai_api_key")
                self.opai_llm_name = st.selectbox("LLM", options=["gpt-4-turbo-preview","gpt-3.5-turbo"])#options=list(LLMModelLoader.AVAILABLE_MODELS.keys()))#, key="llm_path")
                st.session_state["global_singleton"].opai_api_key = self.opai_api_key
                st.session_state["global_singleton"].llm_type = self.llm_type
                self.global_singleton.opai_api_key = self.opai_api_key
                self.global_singleton.llm_type = self.llm_type
                st.session_state["global_singleton"].opai_llm_name = self.opai_llm_name
                self.global_singleton.opai_llm_name = self.opai_llm_name

                self.global_singleton.llm_path = None
                st.session_state["global_singleton"].llm_path = None
                self.global_singleton.hug_llm_name = None
                st.session_state["global_singleton"].hug_llm_name = None
            
    def render_load(self):
        load_llm_button = False
        load_llm_button = st.button("Load LLM", use_container_width = True)
        if load_llm_button:
            #for openai
            if self.global_singleton.opai_llm_name is not None:
                if self.global_singleton.opai_api_key is None:
                    st.error("Please enter an OpenAI API key (not set by the user)")
                else:
                    if check_openai_api_key(self.global_singleton.opai_api_key):
                        self.global_singleton.llm = load_llm(self.global_singleton)
                        st.session_state["global_singleton"].llm = self.global_singleton.llm
                        #load_llm_button = False
                    else:
                        st.error("Invalid OpenAI API key")
            #for huggingface
            elif self.global_singleton.hug_llm_name is not None:
                if self.global_singleton.hug_api_key is None:
                    st.error("Please enter a Huggingface API key (not set by the user)")
                else:
                    if check_hug_key(self.global_singleton.hug_api_key):
                        self.global_singleton.llm = load_llm(self.global_singleton)
                        st.session_state["global_singleton"].llm = self.global_singleton.llm
                        #load_llm_button = False
                    else:
                        st.error("Invalid Huggingface API key")
            else:
                self.global_singleton.llm = load_llm(self.global_singleton)
                st.session_state["global_singleton"].llm = self.global_singleton.llm
                #load_llm_button = False

"""
    #@st.cache_resource
    def load_llm_llama(self):
        if self.llm_path in LLMModelLoader.AVAILABLE_MODELS:
            print("load llm loadingde llm loading")
            with st.spinner(text="Loading and Llama 2 model – hang tight! This should take 1-2 minutes."):
                llm_loader = LLMModelLoader(model_name=self.llm_path, streaming=False, temperature=0)
                llm = llm_loader.load()
                print("llm load() success", llm)
                #st.session_state["global_singleton"].llm = llm
                self.global_singleton.llm = llm
                print("donezerrooo")
                #print("loading: ", self.llm_path)
        else:
            st.error("Selected LLM is not available")
            print("yarragi sapindan tuttuk")
"""
"""
    @st.cache_resource
    def update_llm_global_singleton(_self):
        _self.global_singleton.llm = _self.load_llm_llama(st.session_state["llm_path"])
        st.session_state["global_singleton"] = _self.global_singleton
"""
"""
    def render_load(self):
        load_llm = False
        load_llm = st.button("Load LLM", use_container_width = True)
        if load_llm:
            if st.session_state["llm_type"] == "LLAMA":
                self.update_llm_global_singleton()
                print("amciiik")
                #st.session_state["global_singleton"] = self.global_singleton
                #self.global_singleton.llm 
                
                #st.session_state.global_singleton.llm = self.global_singleton
            if st.session_state["llm_type"] == 'Huggingface':
                pass
            if st.session_state["llm_type"] == "Openai":
                pass

    @st.cache_resource
    def load_llm_llama(_self, _model_name):
        print("load llm loadingde llm loading")
        with st.spinner(text="Loading and Llama 2 model – hang tight! This should take 1-2 minutes."):
            llm_loader = LLMModelLoader(_model_name, streaming=False, temperature=0)
            print("loading: ", _model_name)
            return llm_loader.load()#(model=_model_name)

    @st.cache_resource
    def update_llm_global_singleton(_self):
        _self.global_singleton.llm = _self.load_llm_llama(st.session_state["llm_path"])
        st.session_state["global_singleton"] = _self.global_singleton
"""
"""
    def render_load(self):
        load_llm_button = False
        load_llm_button = st.button("Load LLM", use_container_width = True)
        if load_llm_button:
            self.global_singleton.llm = load_llm(self.global_singleton)
            st.session_state["global_singleton"].llm = self.global_singleton.llm
            load_llm_button = False
            print("butona basildi global singleton.llm ici :////////", self.global_singleton.llm, self.global_singleton.hug_llm_name)
            print("iceri girmis render loadla loadladiktan sonra", self.global_singleton)
"""