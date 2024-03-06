from .util import FileManager, SessionManager

class GlobalSingleton:

    def __init__(self, content_path='./content', session_path='./saved_sessions.json'):
        self.llm = None
        self.word_embedder = None
        self.cross_encoder = None
        self.session_manager = SessionManager(session_path)
        self.file_manager = FileManager(content_path)
        pass
