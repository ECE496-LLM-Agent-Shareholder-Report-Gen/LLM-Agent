from random import randint
import streamlit as st
from session import Report, Session

import os

"""Everything pertaining to the sessions for the GUI. 
This class creates the session itself, and is responsible for
populating Sessions."""
class ChatRenderer:

    def __init__(self, global_singleton):
        self.global_singleton = global_singleton
        if global_singleton.chat_session_manager and global_singleton.chat_session_manager.active_session:
            self.session = global_singleton.chat_session_manager.active_session 
            print("Chatting with active session: ", self.session)
            if not self.session.initialized:
                print("initialized session!")
                with st.spinner("initializing session..."):
                    self.session.initialize(global_singleton.index_generator, global_singleton.file_manager, global_singleton.llm, global_singleton.embeddings)
        else:
            print("no active session!")
            st.switch_page("streamlit_app.py")
    
    def render(self):
        self.render_header()
        self.render_conversation()

    def render_header(self):
        st.title("Chat with the LLM")

    def render_conversation(self):
        # Initialize chat history

        # Display chat messages from history on app rerun
        for qa in self.session.conversation_history:
            with st.chat_message("user"):
                st.markdown(qa.question)
            with st.chat_message("ai"):
                st.markdown(qa.answer)

        # Accept user input
        question = st.chat_input("Say something")
        if question:
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(question)
            # Add user message to chat history
            stream = self.session.chatbot.stream(question)
            response = st.write_stream(stream)
            self.session.add_to_conversation(question, response)