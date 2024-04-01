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
def render_session_info(session, left_col_size = 0.3, right_col_size = 0.7):
    with st.popover("View Session Info"):
        with st.container():
            col1, col2 = st.columns([left_col_size, right_col_size])
            with col1:
                st.markdown("<b>Retriever Strategy</b>: ", unsafe_allow_html=True)
            with col2: 
                st.markdown(session.retrieval_strategy)
        with st.container():
            col1, col2 = st.columns([left_col_size, right_col_size])
            with col1:
                st.markdown("<b>k</b>: ", unsafe_allow_html=True)
            with col2: 
                st.markdown(session.k)
        if session.k_i:
            with st.container():
                col1, col2 = st.columns([left_col_size, right_col_size])
                with col1:
                    st.markdown("<b>k_i</b>: ", unsafe_allow_html=True)
                with col2: 
                    st.markdown(session.k_i)
        with st.container():
            col1, col2 = st.columns([left_col_size, right_col_size])
            with col1:
                st.markdown("<b>LLM Chain</b>: ", unsafe_allow_html=True)
            with col2: 
                st.markdown(session.llm_chain)
        with st.container():
            col1, col2 = st.columns([left_col_size, right_col_size])
            with col1:
                st.markdown("<b>Memory Enabled</b>: ", unsafe_allow_html=True)
            with col2: 
                st.markdown(session.memory_enabled)
        with st.container():
            col1, col2 = st.columns([left_col_size, right_col_size])
            with col1:
                st.markdown("<b>Reports</b>: ", unsafe_allow_html=True)
            with col2: 
                for existing_file in session.reports:
                    container = st.container(border=True)
                    with container:
                        if existing_file.quarter:
                            ticker_col, report_type_col, year_col, quarter_col = st.columns(4)
                        else:
                            ticker_col, report_type_col, year_col = st.columns(3)

                        with ticker_col:
                            st.markdown(f"{existing_file.company}")
                        with report_type_col:
                            st.markdown(f"{existing_file.report_type}", unsafe_allow_html=True)
                        with year_col:
                            st.markdown(f"{existing_file.year}")
                        if  existing_file.quarter:
                            with quarter_col:
                                st.markdown(f"{existing_file.quarter}")


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