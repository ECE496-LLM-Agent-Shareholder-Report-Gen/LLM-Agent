import json
from random import randint
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from session import QAE, Report, Session

import os
import io
import csv
import pandas as pd

"""Everything pertaining to the sessions for the GUI. 
This class creates the session itself, and is responsible for
populating Sessions."""
class BenchmarkRenderer:

    def __init__(self, global_singleton):
        self.global_singleton = global_singleton
        if 'question_expected' not in st.session_state:
            st.session_state['question_expected'] = [["", "",  ""]]
        if 'b_form_continue' not in st.session_state:
            st.session_state['b_form_continue'] = False
        if 'b_name' not in st.session_state:
            st.session_state['b_name'] = ""
        
    def render(self):
        self.render_header()
        self.render_import()
        self.render_question_expected()
        st.divider()
        self.render_submit()

    def render_header(self):
        st.title('Benchmark your LLM')
        st.markdown('<b>Create a new benchmark session to evaluate your LLM</b>', unsafe_allow_html=True)
        st.session_state["b_name"] = st.text_input("Choose the name of the benchmark session", placeholder="Benchmark name...", value=st.session_state["b_name"], max_chars=50)


    def validate_json(self, json_data):
        if not isinstance(json_data, list):
            return False
        for obj in json_data:
            if not isinstance(obj, dict):
                return False
            if not 'question' in obj or not 'answer' in obj:
                return False
        return True

    """ Adds data to session state.question_expected """
    def add_to_question_expected(self, data_list):
        flat_array = []
        for obj in data_list:
            question = ""
            expected = ""
            llm_answer = ""
            # do question
            question_possibilities = ["question", "Question", "questions", "Questions", "QUESTIONS", "QUESTION"]
            expected_possibilities = ["expected", "Expected", "EXPECTED", "expecteds", "Expecteds", "EXPECTEDS", "expected answer", "Expected Answer", "EXPECTED ANSWER", "expected answers", "Expected Answers", "EXPECTED ANSWERS", "expected_answer","Expected_Answer", "EXPECTED_ANSWER","expected_answers", "Expected_Answers", "EXPECTED_ANSWERS"]
            llm_answer_possibilities = ["llm", "LLM", "answer", "Answer", "ANSWER", "answers", "Answers", "ANSWERS", "llm_answer", "LLM_answer", "LLM_ANSWER", "llm_answers", "LLM_answers", "LLM_ANSWERS"]
            for q in question_possibilities:
                if q in obj:
                    question = obj[q]
                    break
            for q in expected_possibilities:
                if q in obj:
                    expected = obj[q]
                    break
            for q in llm_answer_possibilities:
                if q in obj:
                    llm_answer = obj[q]
                    break
            
            flat_array.append([question, expected, llm_answer])
        
        # add imported array to the list of QAs
        last_entry = None
        if len(st.session_state.question_expected) > 0:
            last_entry = st.session_state.question_expected[-1]
            if last_entry[0].strip() == "" and last_entry[1].strip() == "" and last_entry[2].strip() == "":
                st.session_state.question_expected.pop()
        if len(flat_array) < len(data_list):
            st.warning("Some of the items in your uploaded file could not be imported. Please check the 'Help' and ensure that your uploaded file follows the correct format.")
        st.session_state.question_expected = st.session_state.question_expected + flat_array

    """ render help section """
    def render_help(self):
        dropdown = st.popover("Help :grey_question:")
        with dropdown:
            st.markdown("The following file formats are accepted: CSV and JSON")
            st.markdown("<b><u>NOTE:</u></b> you <b>MUST</b> include both the 'Question' and the 'Expected' answer. Including the LLM Answer is optional. You may include the LLM Answer if you just want to compare the LLM Answer against the Expected answer.", unsafe_allow_html=True)
            # CSV format
            st.markdown("<b><u>CSV format</u></b>", unsafe_allow_html=True)
            columns = ["Question", "Expected", "LLM answer"]
            data = [["Question #1", "Expected #1", "LLM Answer #1"],
                    ["Question #2", "Expected #2", "LLM Answer #2"],
                    ["Question #3", "Expected #3", "LLM Answer #3"],
                    ["...", "...", "..."]]
            df = pd.DataFrame(data, columns=columns)
            st.table(data=df)
            # json format
            st.markdown("<b><u>JSON format</u></b>", unsafe_allow_html=True)
            example = [{
                "question": "Question #1",
                "expected": "Expected #1",
                "llm_answer": "LLM Answer #1",
            },{
                "question": "Question #2",
                "expected": "Expected #2",
                "llm_answer": "LLM Answer #2",
            },{
                "question": "Question #3",
                "expected": "Expected #3",
                "llm_answer": "LLM Answer #3",
            },{
                "question": "...",
                "expected": "...",
                "llm_answer": "...",
            }]
            st.write(example)

    """ Render import section  """
    def render_import(self):
        st.subheader('Import from a CSV of JSON', divider='grey')
        self.render_help()
        form = st.form(key="import_form", clear_on_submit=True, border=False)
        with form:
            uploaded_file = st.file_uploader("Import Q&As", type=['csv', 'json'])
            submitted = st.form_submit_button("Import")

            if submitted and uploaded_file:
                filename = uploaded_file.name
                if filename.endswith("csv"):
                    try:
                        csvFile = csv.DictReader(io.StringIO(uploaded_file.getvalue().decode('utf-8')))
                        file_data = []
                        for line in csvFile:
                            file_data.append(line)
                        self.add_to_question_expected(file_data)
                        
                    except:
                        st.error("There was an error while reading the uploaded CSV file. Make sure it follows the proper format as described in the 'Help' section.")
                   
                elif filename.endswith("json"):
                    try:
                        json_data = json.load(uploaded_file)
                        self.add_to_question_expected(json_data)
                    except:
                        st.error("There was an error while reading the uploaded JSON file. Make sure it follows the proper format as described in the 'Help' section.")

            elif submitted:
                st.warning("No file uploaded.")

    """ renders the questions and answers """
    def render_question_expected(self):
        st.subheader('Questions and Answers', divider='grey')
        st.markdown("<b><u>NOTE:</u></b> You can choose to include either the 'Question' along with the 'Expected' or the 'Similarity Score' but you <b>MUST</b> include one. Including the 'LLM Answer' is optional.", unsafe_allow_html=True)
        for idx, qa in enumerate(st.session_state.question_expected):

            with st.container(border=True):
                col1, col2 = st.columns([0.92,0.08])
                with col1:
                    st.markdown(f'<b>Q&A #{idx+1}</b>', unsafe_allow_html=True)
                with col2:
                    remove_qa = st.button(":x:", key=f"close_{idx+1}")
                    if remove_qa:
                         st.session_state.question_expected.remove(qa)
                         st.rerun()
                st.session_state.question_expected[idx][0] = st.text_area(f"Question", value= qa[0], key=f"question_{idx+1}")
                st.session_state.question_expected[idx][1] = st.text_area(f"Expected Answer", value= qa[1], key=f"expected_{idx+1}")
                st.session_state.question_expected[idx][2] = st.text_area(f"LLM Answer", value= qa[2], key=f"answer_{idx+1}")
        new_qa = st.button("Add Question")
        if new_qa:
            st.session_state.question_expected.append(["", "",  ""])
            st.rerun()

    def create_question_expected_obj_list(self, qae_list):
        qae_dict = {}
        for idx, qe in enumerate(qae_list):
            question = qe[0].strip()
            if not question:
                question = None
            expected = qe[1].strip()
            if not expected:
                expected = None
            answer = qe[2].strip()
            if not answer:
                answer = None
            qae_dict[str(idx+1)] = QAE(question=question, expected=expected, answer=answer)
        return qae_dict


    """ Render the submit section """
    def render_submit(self):
        col1, col2 = st.columns([0.8, 0.2])
        form_submitted = False
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
                    st.session_state.question_expected = [["", "", ""]]
                    st.rerun()

        if form_submitted:
            # double clicked
            if st.session_state["b_form_continue"]:
                for idx in st.session_state["ids_to_remove"]:
                    st.session_state.question_expected.pop(int(idx)-1)

                    st.session_state.question_expected_final = self.create_question_expected_obj_list(st.session_state.question_expected)
                    # go to the next page
                st.switch_page("pages/benchmark_session_page.py")

            # check if the user has selected any questions
            if st.session_state["b_name"] and st.session_state["b_name"].strip():
                submittable = True
            else:
                st.error("No benchmark name was given. Please provide a benchmark name")
                submittable = False


            # check if session name is a duplicate
            if submittable:
                name = st.session_state["b_name"].strip()
                for ses in self.global_singleton.benchmark_session_manager.sessions:
                    if ses == name:
                        st.error(f"The benchmark name '{name}' is already being used. Please try another name.")
                        submittable = False

            if not st.session_state.question_expected or len(st.session_state.question_expected) == 0:
                st.warning("No Q&As were given.")
            else:
                missing_question_expecteds = []

                # get number of missing q&a's, scores
                for idx, qae in enumerate(st.session_state.question_expected):
                    for x in range(len(qae)):
                        qae[x] = qae[x].strip()
                    if (not qae[0] or not qae[1]): 
                        missing_question_expecteds.append(str(idx+1))

                # compile all the ids to remove
                ids_to_remove = []
                for idx in missing_question_expecteds:
                    ids_to_remove.append(idx)

                # provide a warning to the user that they will be removed
                if len(missing_question_expecteds) > 0:
                    if len(ids_to_remove) == len(st.session_state.question_expected) or not submittable:
                        st.warning(f"You have not provided a question/expected answer for the following Q&As: {', '.join(missing_question_expecteds)}.")
                    elif submittable:
                        st.warning(f"You have not provided a question/expected answer for the following Q&As: {', '.join(missing_question_expecteds)}. You can click 'Submit' again to remove these Q&As and continue on to the evaluation.")
                # provide a continue button
                if ((len(missing_question_expecteds) > 0) and len(ids_to_remove) < len(st.session_state.question_expected)) and submittable:
                    st.session_state["b_form_continue"] = True
                    st.session_state["ids_to_remove"] = ids_to_remove
                    

               

                if len(ids_to_remove) == 0 and submittable:
                    st.session_state.question_expected_final = self.create_question_expected_obj_list(st.session_state.question_expected)
                    st.switch_page("pages/benchmark_session_page.py")

        else:
            st.session_state["b_form_continue"] = False


               
        


               
    