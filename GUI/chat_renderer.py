from random import randint
import streamlit as st
from GUI.misc import render_session_info
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

        else:
            print("no active session!")
            st.switch_page("streamlit_app.py")

    def render(self):
        self.render_header()
        self.render_conversation()

    def render_header(self):
        st.title(self.session.name)
        with st.empty():
            render_session_info(self.session)
        col1, col2, col3 = st.columns([0.1, 0.2, 0.7])
        if not self.session.initialized:
            with col1:
                    init_soft = st.button("Load", help="Load from existing vector stores (if they exist), and create embeddings for the files that don't have any vector store")
                    if init_soft:
                        with st.spinner("Loading session..."):
                            self.session.initialize(self.global_singleton.index_generator, self.global_singleton.file_manager, self.global_singleton.llm, self.global_singleton.embeddings)
                            st.rerun()
            with col2:
                init_hard = st.button("Re-Initialize", help="Reload all vector stores, including those that already exist", key="hard_init_1")
                if init_hard:
                        with st.spinner("Re-Initializing session..."):
                            self.session.initialize(self.global_singleton.index_generator, self.global_singleton.file_manager, self.global_singleton.llm, self.global_singleton.embeddings, load=False)
                            st.rerun()
        else:
            init_hard = st.button("Re-Initialize", help="Reload all vector stores, including those that already exist", key="hard_init_2")
            if init_hard:
                with st.spinner("Re-Initializing session..."):
                    self.session.initialize(self.global_singleton.index_generator, self.global_singleton.file_manager, self.global_singleton.llm, self.global_singleton.embeddings, load=False)
                    st.rerun()


    def render_conversation(self):
        # Initialize chat history
        with st.chat_message("ai"):
            st.markdown("What would you like to know?")

        replay = None
        replay_q = None
        # Display chat messages from history on app rerun
        for idx, qa in enumerate(self.session.conversation_history):
            with st.chat_message("user"):
                st.markdown(qa.question)
                replay = st.button("↺", key=f"replay_{idx}")
                if replay:
                    replay_q = qa
            with st.chat_message("ai"):
                st.markdown(qa.answer)

        # Accept user input
        question = st.chat_input("Say something", disabled=not self.session.initialized)
        if question:
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("ai"):
                #full_response = self.session.chatbot.invoke(question)
                full_response = self.session.chatbot.st_render(question)
                self.session.add_to_conversation(question, full_response)
                st.rerun()
        if replay and replay_q:
             # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(replay_q.question)
            with st.chat_message("ai"):
                replays = replay_q.replays
                if replays == None:
                    replays = 0
                full_response = self.session.chatbot.st_render(replay_q.question, replays+1)
                full_response = full_response
                self.session.add_to_conversation(replay_q.question, full_response, replays+1)
                st.rerun()


