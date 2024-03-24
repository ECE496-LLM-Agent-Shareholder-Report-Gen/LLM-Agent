import streamlit as st

from GUI.misc import render_session_info

class BenchmarkEvalRenderer:

    def __init__(self, global_singleton):
        self.global_singleton = global_singleton
        self.similarity_scores = []
        if global_singleton.benchmark_session_manager and global_singleton.benchmark_session_manager.active_session:
            self.session = global_singleton.benchmark_session_manager.active_session

            if not f"benchmark_{self.session.name}" in st.session_state:
                st.session_state[f"benchmark_{self.session.name}"] = { "curr_idx": 0, "init": False}
        else:
            st.switch_page("streamlit_app.py")

    def render(self):
        self.render_header()
        self.render_question_expected()

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

    """ Render a single QAE box, without the title """
    def render_qae_box(self, question=None, expected=None, answer=None, similarity_score=None, response_time=None, cutoff=70, col1_width=0.2, col2_witdh=0.8):
        # do the question
        if question:
            question = question.replace("$", "\$")
            with st.container():
                col1, col2 = st.columns([col1_width, col2_witdh])
                with col1:
                    st.markdown(f"<b>Question:</b>", unsafe_allow_html=True)
                with col2:
                    if len(question) < cutoff:
                        st.markdown(f"{question}", unsafe_allow_html=True)
                    else:
                        with st.expander(label=f"{question[:cutoff]}..."):
                            st.markdown(f"{question}", unsafe_allow_html=True)

        # do the expected answer
        if expected:
            expected = expected.replace("$", "\$")
            with st.container():
                col1, col2 = st.columns([col1_width, col2_witdh])
                with col1:
                    st.markdown(f"<b>Expected:</b>", unsafe_allow_html=True)
                with col2:
                    if len(expected) < cutoff:
                        st.markdown(f"{expected}", unsafe_allow_html=True)
                    else:
                        with st.expander(label=f"{expected[:cutoff]}..."):
                            st.markdown(f"{expected}", unsafe_allow_html=True)

        # do the similarity score
        if similarity_score:
            with st.container():
                col1, col2 = st.columns([col1_width, col2_witdh])
                with col1:
                    st.markdown(f"<b>Similarity Score:</b>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"{similarity_score:.3f}", unsafe_allow_html=True)

        # do the response time
        if response_time:
            with st.container():
                col1, col2 = st.columns([col1_width, col2_witdh])
                with col1:
                    st.markdown(f"<b>Response Time:</b>", unsafe_allow_html=True)
                with col2:
                    st.markdown(f"{response_time}", unsafe_allow_html=True)

        # do the actual answer
        if answer:
            answer = answer.replace("$", "\$")
            with st.container():
                col1, col2 = st.columns([col1_width, col2_witdh])
                with col1:
                    st.markdown(f"<b>Answer:</b>", unsafe_allow_html=True)
                with col2:
                    if len(answer) < cutoff:
                        st.markdown(f"{answer}", unsafe_allow_html=True)
                    else:
                        with st.expander(label=f"{answer[:cutoff]}..."):
                            st.markdown(f"{answer}", unsafe_allow_html=True)



    """ renders the questions and answers """
    def render_question_expected(self):
        st.subheader('Questions and Answers', divider='grey')
        if self.session.initialized and not st.session_state[f"benchmark_{self.session.name}"]["init"]:
            run_test = st.button("Run Tests")
            if run_test:
                st.session_state[f"benchmark_{self.session.name}"]["init"] = True
                st.session_state[f"benchmark_{self.session.name}"]["curr_idx"] = 1
        init = st.session_state[f"benchmark_{self.session.name}"]["init"]
        containers = [st.container(border=True) for x in range(len(self.session.question_answer_expected.items()))]
        empty_containers = [st.empty() for x in range(len(self.session.question_answer_expected.items()))]

        container_to_stream_response = None
        container_to_stream_response_qae = None
        container_to_stream_response_cidx = 0

        container_to_stream_response_qid = 0

        for idx, (qid, qae) in enumerate(self.session.question_answer_expected.items()):
            curr_processing_qae =  st.session_state[f"benchmark_{self.session.name}"]["curr_idx"]
            with empty_containers[idx]: # running
                with containers[idx]:
                    if qid == curr_processing_qae and init:
                        st.markdown(f'<b><u>Q&A #{qid}</u> :running:</b>', unsafe_allow_html=True)
                        container_to_stream_response_qae = qae
                        container_to_stream_response = empty_containers[idx]
                        container_to_stream_response_qid = qid
                        container_to_stream_response_cidx = idx
                    elif qid < curr_processing_qae and init: #already ran
                        st.markdown(f'<b><u>Q&A #{qid}</u> :white_check_mark:</b>', unsafe_allow_html=True)

                    elif qid > curr_processing_qae or not init: # not yet run
                        st.markdown(f'<b><u>Q&A #{qid}</u> :clock2:</b>', unsafe_allow_html=True)

                    self.render_qae_box(question=qae.question,
                                        expected=qae.expected,
                                        answer=qae.answer,
                                        similarity_score=qae.similarity_score,
                                        response_time=qae.response_time)
                    self.similarity_scores.append(qae.similarity_score)

        st.subheader("Similarity Scores")
        st.line_chart(self.similarity_scores)

        if container_to_stream_response:
            final_response, score, response_time = self.session.chatbot.invoke_with_score(container_to_stream_response_qae.question,expected=container_to_stream_response_qae.expected,cross_encoder= self.global_singleton.cross_encoder)
            if final_response:
                self.session.update_qae(id=container_to_stream_response_qid,
                                        question=container_to_stream_response_qae.question,
                                        answer=final_response,
                                        expected=container_to_stream_response_qae.expected,
                                        similarity_score=score)
                st.session_state[f"benchmark_{self.session.name}"]["curr_idx"] += 1
                st.rerun()
            # with container_to_stream_response:
            #     temp_con =  st.container(border=True)
            #     with temp_con:
            #         st.markdown(f'<b><u>Q&A #{container_to_stream_response_qid}</u> :running:</b>', unsafe_allow_html=True)
            #         self.render_qae_box(question=container_to_stream_response_qae.question,
            #                                 expected=container_to_stream_response_qae.expected)
            #         col1, col2 = st.columns([0.2, 0.8])
            #         with col1:
            #             st.markdown(f"<b>Answer:</b>", unsafe_allow_html=True)
            #         with col2:
            #             final_response, score, response_time = self.session.chatbot.render_st_with_score(container_to_stream_response_qae.question, self.global_singleton.cross_encoder)
            #             if final_response:
            #                 self.session.update_qae(id=container_to_stream_response_qid,
            #                                         question=container_to_stream_response_qae.question,
            #                                         answer=final_response,
            #                                         expected=container_to_stream_response_qae.expected,
            #                                         similarity_score=score,
            #                                         response_time=response_time)
            #                 st.session_state[f"benchmark_{self.session.name}"]["curr_idx"] += 1
            #                 st.rerun()
