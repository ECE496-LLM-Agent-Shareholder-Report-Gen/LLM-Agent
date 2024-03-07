from util import FileManager, SessionManager
from pre_processing import IndexGenerator

class GlobalSingleton:

    def __init__(self, content_path='./content/companies', session_path='./saved_sessions.json'):
        self.llm = None
        self.embeddings = None
        self.cross_encoder = None
        self.file_manager = FileManager(content_path)
        self.session_manager = SessionManager(session_path)
        self.index_generator = IndexGenerator()
    
    def load_file_manager(self):
        self.file_manager.load()
    
    def load_session_manager(self):
        self.session_manager.load()
