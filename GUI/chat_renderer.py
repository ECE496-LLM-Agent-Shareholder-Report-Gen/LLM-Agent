import json
from random import randint
import streamlit as st
from streamlit_extras.stylable_container import stylable_container

from streamlit_pdf_viewer import pdf_viewer
from GUI.misc import check_session_valid, render_session_info
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
            self.session_valid, self.missing_reports = check_session_valid(self.session.reports, file_manager=self.global_singleton.file_manager)
        else:
            st.switch_page("streamlit_app.py")

    def render(self):
        self.render_header()
        tab1, tab2 = st.tabs(["Conversation", "PDFs"])
        with tab1:
            self.render_conversation()
        with tab2:
            self.render_pdf()
        self.render_question()

    def download_conversation_history(self):
        conv_history = [qa.encode() for qa in self.session.conversation_history]
        json_data =json.dumps(conv_history)
        json_download = st.download_button("Export Conversation History", data=json_data, file_name=f"{self.session.name}.json", key="download-json", mime='text/json', help="Export conversationg history to JSON file format")

    def render_header(self):
        st.title(self.session.name)
        with st.empty():
            render_session_info(self.session, self.global_singleton)

        self.download_conversation_history()

    """ render conversation """
    def render_conversation(self):
        # Initialize chat history
        with stylable_container(key="no_border", css_styles="""
                                {
                                    border: none;
                                }"""):
            h_cont = st.container(height=700, border=None)
            with h_cont:
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

                if not self.session_valid:
                    st.warning(f"The current session is no longer valid because of the following missing reports: {self.missing_reports}. You can view this conversation's history but you can no longer ask questions in the current session.")

                
                if replay and replay_q:
                    # Display user message in chat message container
                    with st.chat_message("user"):
                        st.markdown(replay_q.question)
                    if not self.session.initialized:
                        with st.spinner("Loading session..."):
                            isllama = "llama" in self.global_singleton.llm_model.lower()
                            self.session.initialize(self.global_singleton.index_generator, self.global_singleton.file_manager, self.global_singleton.llm, self.global_singleton.embeddings, cross_encoder=self.global_singleton.cross_encoder, isllama=isllama)
                    with st.chat_message("ai"):
                        replays = replay_q.replays
                        if replays == None:
                            replays = 0
                        full_response, context = self.session.chatbot.st_render(replay_q.question, replays+1)
                        full_response = full_response
                        self.session.add_to_conversation(replay_q.question, full_response, replays+1, context=context)
                        st.rerun()

    """ render question input """
    def render_question(self):
        # Accept user input
        placeholder_text = "Ask a question"
        question = st.chat_input(placeholder_text, disabled=not self.session_valid)
        if question:
            if not self.session.initialized:
                with st.spinner("Loading session..."):
                    isllama = "llama" in self.global_singleton.llm_model.lower()
                    self.session.initialize(self.global_singleton.index_generator, self.global_singleton.file_manager, self.global_singleton.llm, self.global_singleton.embeddings, cross_encoder=self.global_singleton.cross_encoder, isllama=isllama)
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("ai"):
                #full_response = self.session.chatbot.invoke(question)
                full_response, context = self.session.chatbot.st_render(question)
                self.session.add_to_conversation(question, full_response, replays=0, context=context)
                st.rerun()

    """ render pdf and context """
    def render_pdf(self):
        report = None
        report_name_dict = {}
        for existing_file in self.session.reports:
            if existing_file.quarter:
                name = f"{existing_file.company} {existing_file.year} {existing_file.quarter} {existing_file.report_type}"
            else:
                name = f"{existing_file.company} {existing_file.year} {existing_file.report_type}"
            report_name_dict[name] = existing_file.file_path
        col1, col2 = st.columns(2)
        with col1:

            report = st.selectbox("Select Report", report_name_dict.keys())
        with col2:
            page = st.number_input("Page number", min_value=1, step=1)
        if report != None:
            h_cont = st.container(border=True)
            with h_cont:
                if page:
                    pdf_viewer(report_name_dict[report], width=670, pages_to_render=[page])

        question_context_dict = {}
        for idx, qa in enumerate(self.session.conversation_history):
            question_context_dict[qa.question] = qa.context
        st.divider()
        st.markdown("<b>View context for a specified question</b>", unsafe_allow_html=True)
        question = st.selectbox("Question", question_context_dict.keys())
        if question != None:
            if question_context_dict[question] != None:
                h_cont2 = st.container(height=700)
                with h_cont2:
                    st.markdown(question_context_dict[question].replace("$", "\$"))
            else:
                st.markdown("No context found for the specified question.")



