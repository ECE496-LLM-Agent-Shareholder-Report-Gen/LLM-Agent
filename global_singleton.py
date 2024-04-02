from session import BenchmarkSession, ChatSession
from util import FileManager, SessionManager
from pre_processing import IndexGenerator
import streamlit as st
"""
class GlobalSingleton:

    def __init__(self, content_path='./content/companies', session_path='./saved_sessions.json'):
        self.llm = None
        self.embeddings = None
        self.cross_encoder = None
        self.file_manager = FileManager(content_path)
        self.session_manager = SessionManager(session_path)
        self.index_generator = IndexGenerator()
    
        #vars for llm load:
        self.llm_path = None
        self.llm_type = None
        self.hug_llm_name = None
        self.hug_api_key = None
        self.opai_api_key = None
        
    def load_file_manager(self):
        self.file_manager.load()
    
    def load_session_manager(self):
        self.session_manager.load()
"""

class GlobalSingleton:
    _instance = None
    llm = None
    embeddings = None
    cross_encoder = None
    file_manager = None
    index_generator = None
    chat_session_manager = None
    benchmark_session_manager = None
    content_path = None
    chat_session_path = None
    benchmark_session_path = None
    #vars for llm_load
    llm_path = None
    llm_type = None
    llm_model = None
    hug_llm_name = None
    hug_api_key = None
    opai_api_key = None
    opai_llm_name = None
    #vars for embedding_load
    embedding_type = None
    embeddings_model = None
    cross_encoder_model = None
    opai_embedding_name = None
    hug_embedding_name = None 
    #for memory management
    hug_tokenizer = None
    hug_model = None
    hug_pipe = None
    #test, for updating navbar
    #navbar = None

    def __new__(cls, content_path='./content/companies', chat_session_path='./saved_sessions.json', benchmark_session_path='./benchmark_session.json'):
        if cls._instance is None:
            print('Creating the object')
            cls._instance = super(GlobalSingleton, cls).__new__(cls)
            cls._instance.llm = None
            cls._instance.embeddings = None
            cls._instance.cross_encoder = None
            cls._instance.file_manager = None
            cls._instance.index_generator = None
            cls._instance.chat_session_manager = None
            cls._instance.benchmark_session_manager = None
            cls._instance.content_path = content_path
            cls._instance.chat_session_path = chat_session_path
            cls._instance.benchmark_session_path = benchmark_session_path
            cls._instance.llm_path = None
            cls._instance.llm_model = None
            cls._instance.llm_type = None
            cls._instance.hug_llm_name = None
            cls._instance.hug_api_key = None
            cls._instance.opai_api_key = None
            cls._instance.opai_llm_name = None
            cls._instance.embeddings_model = None
            cls._instance.cross_encoder_model = None
            cls._instance.embedding_type = None
            cls._instance.opai_embedding_name = None
            cls._instance.hug_embedding_name = None
            cls._instance.hug_tokenizer = None
            cls._instance.hug_model = None
            cls._instance.hug_pipe = None
            #cls._instance.navbar = None
        # print(cls._instance, cls._instance.llm)
        return cls._instance
        
    def load_index_generator(self,index_name=None):
        self.index_generator = IndexGenerator(index_name=index_name)
    
    def load_file_manager(self, index_name=None):
        self.file_manager = FileManager(self.content_path, index_name=index_name)
        self.file_manager.load()
    
    def load_chat_session_manager(self):
        self.chat_session_manager = SessionManager(self.chat_session_path, _session_cls=ChatSession)
        self.chat_session_manager.load()
    
    def load_benchmark_session_manager(self):
        self.benchmark_session_manager = SessionManager(self.benchmark_session_path, _session_cls=BenchmarkSession)
        self.benchmark_session_manager.load()
"""
    @staticmethod
    def get_instance():
        if not hasattr(GlobalSingleton, "_instance"):
            GlobalSingleton._instance = GlobalSingleton()
        return GlobalSingleton._instance
"""