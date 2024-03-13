import json
from random import randint
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from session import Report, Session

import os
import pandas as pd

"""Everything pertaining to the sessions for the GUI. 
This class creates the session itself, and is responsible for
populating Sessions."""
class BenchmarkRenderer:

    def __init__(self, global_singleton):
        self.session = Session()
        self.global_singleton = global_singleton
        if 'question_answers' not in st.session_state:
            st.session_state['question_answers'] = [["", ""]]
        
        
    def render(self):
        self.render_header()
        self.render_import()
        self.render_question_answers()
        st.divider()
        self.render_submit()

    def render_header(self):
        st.title('Benchmark your LLM')
        st.markdown('<b>Create a new benchmark session to evaluate your LLM</b>', unsafe_allow_html=True)
        st.session_state.name = st.text_input("Choose the name of the benchmark session", placeholder="Session name...", max_chars=50)


    def validate_json(self, json_data):
        if not isinstance(json_data, list):
            return False
        for obj in json_data:
            if not isinstance(obj, dict):
                return False
            if not 'question' in obj or not 'answer' in obj:
                return False
        return True

    def render_import(self):
        st.subheader('Import from a CSV of JSON', divider='grey')

        form = st.form(key="import_form", clear_on_submit=True, border=False)
        with form:
            uploaded_file = st.file_uploader("Import Q&As", type=['csv', 'json'])
            submitted = st.form_submit_button("Import")

            if submitted and uploaded_file:
                filename = uploaded_file.name
                if filename.endswith("csv"):
                    data = pd.read_csv(uploaded_file)
                    first_column_values = data.iloc[:, 0].values
                    second_column_values = data.iloc[:, 1].values

                    # Merge the values into an array of size-2 arrays
                    merged_array = [[col1, col2] for col1, col2 in zip(first_column_values, second_column_values)]
                    
                    # add imported array to the list of QAs
                    last_entry = None
                    if len(st.session_state.question_answers) > 0:
                        last_entry = st.session_state.question_answers[-1]
                        if last_entry[0].strip() == "" and last_entry[1].strip() == "":
                            st.session_state.question_answers.pop()
                    st.session_state.question_answers = st.session_state.question_answers + merged_array
                elif filename.endswith("json"):
                    json_data = json.load(uploaded_file)
                    # validate json data
                    valid = self.validate_json(json_data)
                    if not valid:
                        st.error("JSON file must be a list of objects containing 'question' and 'answer'")
                        return
                    flat_array = []
                    for obj in json_data:
                        flat_array.append([obj["question"], obj["answer"]])

                    # add imported array to the list of QAs
                    last_entry = None
                    if len(st.session_state.question_answers) > 0:
                        last_entry = st.session_state.question_answers[-1]
                        if last_entry[0].strip() == "" and last_entry[1].strip() == "":
                            st.session_state.question_answers.pop()
                    st.session_state.question_answers = st.session_state.question_answers + flat_array

            elif submitted:
                st.warning("No file uploaded.")

    """ renders the questions and answers """
    def render_question_answers(self):
        st.subheader('Questions and Answers', divider='grey')

        for idx, qa in enumerate(st.session_state.question_answers):

            with st.container(border=True):
                col1, col2 = st.columns([0.92,0.08])
                with col1:
                    st.markdown(f'<b>Q&A #{idx+1}</b>', unsafe_allow_html=True)
                with col2:
                    remove_qa = st.button(":x:", key=f"close_{idx+1}")
                    if remove_qa:
                         st.session_state.question_answers.remove(qa)
                         st.rerun()
                st.session_state.question_answers[idx][0] = st.text_area(f"Question", value= qa[0], key=f"question_{idx+1}")
                st.session_state.question_answers[idx][1] = st.text_area(f"Answer", value= qa[1], key=f"answer_{idx+1}")
        new_qa = st.button("Add Question")
        if new_qa:
            st.session_state.question_answers.append(["", ""])
            st.rerun()

    def render_submit(self):
        col1, col2 = st.columns([0.8, 0.2])
        
        with col1:
            with stylable_container(
            key="submit_button",
            css_styles="""
                button {
                    height: 3rem;
                }
                """,
            ):
                form_submitted = st.button("Submit", use_container_width=True)
                if form_submitted:
                    pass
        with col2:
            with stylable_container(
                key="clear_button",
                css_styles="""
                    button {
                        background-color: red;
                        color: white;
                        height: 3rem;
                    }
                    """,
            ):
                clear = st.button("Clear", use_container_width=True)
                if clear:
                    st.session_state.question_answers = [["", ""]]
                    st.rerun()
    