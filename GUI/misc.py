import streamlit as st

import os

""" write stream from chatbot """
def write_stream(stream):
    result = ""
    container = st.empty()
    for chunk in stream:
        result += chunk.replace("$", "\$")
        container.write(result, unsafe_allow_html=True)
    return result


""" render info of the session """
def render_session_info(session, global_singleton, left_col_size = 0.35, right_col_size = 0.65):
    with st.popover("View Session Info"):
        with st.container():
            col1, col2 = st.columns([left_col_size, right_col_size])
            with col1:
                st.markdown("<b>LLM:</b> ", unsafe_allow_html=True)
            with col2: 
                st.markdown(global_singleton.llm_model)
                #st.markdown(global_singleton.llm)
        with st.container():
            col1, col2 = st.columns([left_col_size, right_col_size])
            with col1:
                st.markdown("<b>Word Embeddings:</b> ", unsafe_allow_html=True)
            with col2: 
                st.markdown(global_singleton.embeddings_model)
        with st.container():
            col1, col2 = st.columns([left_col_size, right_col_size])
            with col1:
                st.markdown("<b>Retriever Strategy:</b> ", unsafe_allow_html=True)
            with col2: 
                st.markdown(session.retrieval_strategy)
        with st.container():
            col1, col2 = st.columns([left_col_size, right_col_size])
            with col1:
                st.markdown("<b>k:</b> ", unsafe_allow_html=True)
            with col2: 
                st.markdown(session.k)
        if session.k_i:
            with st.container():
                col1, col2 = st.columns([left_col_size, right_col_size])
                with col1:
                    st.markdown("<b>k_i:</b> ", unsafe_allow_html=True)
                with col2: 
                    st.markdown(session.k_i)
        with st.container():
            col1, col2 = st.columns([left_col_size, right_col_size])
            with col1:
                st.markdown("<b>LLM Chain:</b> ", unsafe_allow_html=True)
            with col2: 
                st.markdown(session.llm_chain)
       
        with st.container():
            col1, col2 = st.columns([left_col_size, right_col_size])
            with col1:
                st.markdown("<b>Reports:</b> ", unsafe_allow_html=True)
            with col2: 
                for existing_file in session.reports:
                    if existing_file.quarter:
                        st.markdown(f"<b>{existing_file.company} {existing_file.year} {existing_file.report_type} {existing_file.quarter}</b>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<b>{existing_file.company} {existing_file.year} {existing_file.report_type}</b>", unsafe_allow_html=True)



""" check if session is still valid """
def check_session_valid(reports, file_manager):
    status = True
    file_manager.load()
    missing_reports = []
    for report in reports:
        file_path = file_manager.get_file_path(report.company, report.year, report.report_type, report.quarter)
        if file_path == None:
            status =  False
            missing_report = f"{report.company}, {report.year}, {report.report_type}"
            if report.quarter:
                missing_report += f", {report.quarter}"
            break
    return status, missing_reports