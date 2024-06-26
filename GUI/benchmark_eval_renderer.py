import json
import streamlit as st
import pandas as pd
import altair as alt

from GUI.misc import check_session_valid, render_session_info
from agent.cross_encode import compute_score

class BenchmarkEvalRenderer:

    def __init__(self, global_singleton):
        self.global_singleton = global_singleton
        if global_singleton.benchmark_session_manager and global_singleton.benchmark_session_manager.active_session:
            self.session = global_singleton.benchmark_session_manager.active_session

            if not f"benchmark_{self.session.name}" in st.session_state:
                st.session_state[f"benchmark_{self.session.name}"] = { "curr_idx": 0, "status": "not_started", "allowed_tests": "none", "eval_test_type": "none", "completed_idxs": [], "idxs_to_compute": []}

            
        else:
            st.switch_page("streamlit_app.py")

    def render(self):
        self.render_header()
        st.divider()
        self.render_chart()
        st.divider()
        self.render_question_expected()

    def get_qae_list(_self):
        return _self.session.qae_to_dict_list()
    
    def convert_dict_to_csv(_self, _data_list):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        df = pd.DataFrame(_data_list)
        return df.to_csv().encode('utf-8')

    def render_json_export(self, qae_list):
        json_data =json.dumps(qae_list)
        json_download = st.download_button("Export results to JSON", data=json_data, file_name=f"{self.session.name}.json", key="download-json", mime='text/json', help="Export evaluation results to JSON file format", disabled=st.session_state[f"benchmark_{self.session.name}"]["status"] == "running")
    
    def render_csv_export(self, qae_list):
        csv_data =self.convert_dict_to_csv(qae_list)
        csv_download = st.download_button("Export results to CSV", data=csv_data, file_name=f"{self.session.name}.csv", key="download-vsc", mime='text/csv', help="Export evaluation results to CSV file format", disabled=st.session_state[f"benchmark_{self.session.name}"]["status"] == "running")

    """ load vector stores and session """
    def load_session(self):
        if not self.session.initialized:
            with st.spinner("Loading session..."):
                isllama = "llama" in self.global_singleton.llm_model.lower()
                self.session.initialize(self.global_singleton.index_generator, self.global_singleton.file_manager, self.global_singleton.llm, self.global_singleton.embeddings, cross_encoder=self.global_singleton.cross_encoder, isllama=isllama)

    """ render header of benchmark  """
    def render_header(self):
        st.title(self.session.name)
        qae_list = self.get_qae_list()

        with st.empty():
            render_session_info(self.session, self.global_singleton)
        
        status, missing_reports = check_session_valid(self.session.reports, file_manager=self.global_singleton.file_manager)
        ce_exists  = True
        if self.global_singleton.cross_encoder == None:
            ce_exists = False

        if not status and not ce_exists:
            st.warning(f"The current benchmark is missing the following reports: {'; '.join(missing_reports)}. You will not be able to answer the questions using an LLM. There is also no cross-encoder loaded. You can load a cross-encoder by clicking on 'Load LLMs' located at the bottom of the left-hand navigation bar.") 
            st.session_state[f"benchmark_{self.session.name}"]["allowed_tests"] = "none"
        if not status and ce_exists:
            st.warning(f"The current benchmark is missing the following reports: {'; '.join(missing_reports)}. You will not be able to answer the questions using an LLM. However, you can still score the expected answers against the LLM answers (if they exist).") 
            st.session_state[f"benchmark_{self.session.name}"]["allowed_tests"] = "partial"
        st.divider()
        st.markdown("<b>Export your results either to a CSV or JSON file</b>", unsafe_allow_html=True)

        coljson, colcsv, _ = st.columns([0.3, 0.3, 0.4])
        with coljson:
            self.render_json_export(qae_list)
        with colcsv:
            self.render_csv_export(qae_list)

        
        if status and ce_exists:
            st.session_state[f"benchmark_{self.session.name}"]["allowed_tests"] = "all"
            

        if st.session_state[f"benchmark_{self.session.name}"]["status"] != "running":
            st.markdown("<b>Initialize the session and benchmark the LLM</b>", unsafe_allow_html=True)
            allowed_tests = st.session_state[f"benchmark_{self.session.name}"]["allowed_tests"]
            partial_col, full_col, _ = st.columns([0.3, 0.3, 0.4])
            with partial_col:
                if allowed_tests == "partial" or allowed_tests == "all":
                    partial_eval_test = st.button("Basic Evaluation", help="Evaluate any Q&As that do not already have a similarity score. Note that this ONLY computes the similarity score if possible. If the LLM Answer exists, then that will be used instead of answering the question again.")
                    if partial_eval_test:
                        self.load_session()
                        st.session_state[f"benchmark_{self.session.name}"]["status"] = "running"
                        st.session_state[f"benchmark_{self.session.name}"]["eval_test_type"] = "partial"
                        p_list, comp_idxs = self.compute_partial_eval_qae(self.session.question_answer_expected)
                        if len(p_list) == 0:
                            st.session_state[f"benchmark_{self.session.name}"]["status"] = "complete"
                        else:
                            st.session_state[f"benchmark_{self.session.name}"]["curr_idx"] = p_list[0]

                        st.session_state[f"benchmark_{self.session.name}"]["idxs_to_compute"] = p_list
                        st.session_state[f"benchmark_{self.session.name}"]["completed_idxs"] = comp_idxs
                        st.rerun()
            with full_col:
                if allowed_tests == "all":
                    full_eval_test = st.button("Complete Evaluation", help="Evaluate all Q&As. Running a complete evaluation will re-answer any questions with a similarity score.")
                    if full_eval_test:
                        self.load_session()
                        st.session_state[f"benchmark_{self.session.name}"]["status"] = "running"
                        st.session_state[f"benchmark_{self.session.name}"]["eval_test_type"] = "full"
                        full_list, comp_idxs = self.compute_full_eval_qae(self.session.question_answer_expected)
                        if len(full_list) == 0:
                            st.session_state[f"benchmark_{self.session.name}"]["status"] = "complete"
                        else:
                            st.session_state[f"benchmark_{self.session.name}"]["curr_idx"] = full_list[0]

                        st.session_state[f"benchmark_{self.session.name}"]["idxs_to_compute"] = full_list
                        st.session_state[f"benchmark_{self.session.name}"]["completed_idxs"] = comp_idxs
                        st.rerun()
        else:
            max_len = len(self.session.question_answer_expected)
            curr_len = len(st.session_state[f"benchmark_{self.session.name}"]["completed_idxs"])
            status = round(curr_len/max_len * 100)
            my_bar = st.progress(status, text="Please wait while the benchmark is complete")

    """ Render chart """
    def render_chart(self):
        chart_data = []
        average_str = "N/A"
        average = 0.0
        num_calc = 0
        for idx, qae in self.session.question_answer_expected.items():
            chart_data.append({"Q&A": idx, "Similarity Score": qae.similarity_score})
            if qae.similarity_score:
                num_calc += 1
                average = (qae.similarity_score + (average * (num_calc - 1))) / num_calc

        df = pd.DataFrame(chart_data)
        if average > 0.0:
            average_str = f"{average:.3f}"
        st.subheader("Similarity Scores", divider="grey")
        st.markdown(f"<b>Average: </b> {average_str}", unsafe_allow_html=True)
        if num_calc != 0:
            chart = alt.Chart(df).mark_bar().encode(
                x=alt.X('Q&A:O', axis=alt.Axis(title='Q&A',labelAngle=0)),
                y=alt.Y('Similarity Score:Q', axis=alt.Axis(title='Similarity Score')),
            )
            st.altair_chart(chart, use_container_width=True)
            # st.line_chart(df, x="Q&A", y="Similarity Score")

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

    """ Compute partial eval list """
    def compute_partial_eval_qae(self, qae_dict):
        partial_list = []
        completed_idxs = []
        for idx, qae in qae_dict.items():
            if qae.similarity_score == None:
                if st.session_state[f"benchmark_{self.session.name}"]["allowed_tests"] == "partial":
                    if qae.expected and qae.answer:
                        partial_list.append(idx)
                    else:
                        completed_idxs.append(idx)
                else:
                    partial_list.append(idx)
            else:
                completed_idxs.append(idx)
        return partial_list, completed_idxs


    
    def compute_full_eval_qae(self, qae_dict):
        full_list = []
        completed_idxs = []
        for idx, qae in qae_dict.items():
            full_list.append(idx)
        return full_list, completed_idxs

    """ renders the questions and answers """
    def render_question_expected(self):
        st.subheader('Questions and Answers', divider='grey')
        

        status = st.session_state[f"benchmark_{self.session.name}"]["status"]
        test_type = st.session_state[f"benchmark_{self.session.name}"]["eval_test_type"]
        comp_idxs = st.session_state[f"benchmark_{self.session.name}"]["completed_idxs"]
        containers = [st.container(border=True) for x in range(len(self.session.question_answer_expected.items()))]
        empty_containers = [st.empty() for x in range(len(self.session.question_answer_expected.items()))]

        container_to_stream_response = None
        curr_qae = None

        curr_qid = 0

        curr_processing_qae =  st.session_state[f"benchmark_{self.session.name}"]["curr_idx"]
        for idx, (qid, qae) in enumerate(self.session.question_answer_expected.items()):
            with empty_containers[idx]: # running
                with containers[idx]:
                    if qid == curr_processing_qae and status == "running":
                        st.markdown(f'<b><u>Q&A #{qid}</u> :running:</b>', unsafe_allow_html=True)
                        curr_qae = qae
                        container_to_stream_response = empty_containers[idx]
                        curr_qid = qid
                    elif (qid in comp_idxs) or (status == "complete"): #already ran
                        st.markdown(f'<b><u>Q&A #{qid}</u> :white_check_mark:</b>', unsafe_allow_html=True)

                    elif (not qid in comp_idxs) and status == "running": # not yet run
                        st.markdown(f'<b><u>Q&A #{qid}</u> :clock2:</b>', unsafe_allow_html=True)
                    elif status=="not_started":
                        st.markdown(f'<b><u>Q&A #{qid}</u></b>', unsafe_allow_html=True)

                    self.render_qae_box(question=qae.question,
                                        expected=qae.expected,
                                        answer=qae.answer,
                                        similarity_score=qae.similarity_score,
                                        response_time=qae.response_time)

       

        if container_to_stream_response:
            if test_type == "partial":
                if curr_qae.answer != None and curr_qae.expected != None: # just compute the score, don't answer the question
                    final_response = curr_qae.answer
                    score, response_time = compute_score(curr_qae.expected,answer=curr_qae.answer,cross_encoder= self.global_singleton.cross_encoder)
                else:
                    final_response, score, response_time = self.session.chatbot.invoke_with_score(curr_qae.question,expected=curr_qae.expected,cross_encoder= self.global_singleton.cross_encoder)
            else:
                final_response, score, response_time = self.session.chatbot.invoke_with_score(curr_qae.question,expected=curr_qae.expected,cross_encoder= self.global_singleton.cross_encoder)
            if final_response:
                if score:
                    score = score.astype(float)
                self.session.update_qae(id=curr_qid,
                                        question=curr_qae.question,
                                        answer=final_response,
                                        expected=curr_qae.expected,
                                        similarity_score=score)
                next_idx = st.session_state[f"benchmark_{self.session.name}"]["curr_idx"]

                # what to do if this is a partial test
                st.session_state[f"benchmark_{self.session.name}"]["idxs_to_compute"].remove(next_idx)
                st.session_state[f"benchmark_{self.session.name}"]["completed_idxs"].append(next_idx)
                p_list = st.session_state[f"benchmark_{self.session.name}"]["idxs_to_compute"]
                if len(p_list) == 0:
                    st.session_state[f"benchmark_{self.session.name}"]["status"] = "complete"
                else:
                    next_idx = p_list[0]
                    st.session_state[f"benchmark_{self.session.name}"]["curr_idx"] = next_idx
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
