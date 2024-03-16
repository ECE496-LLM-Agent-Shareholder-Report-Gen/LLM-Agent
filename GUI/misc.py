import streamlit as st

import os

def write_stream(stream):
    result = ""
    container = st.empty()
    for chunk in stream:
        result += chunk.replace("$", "\$")
        container.write(result, unsafe_allow_html=True)
    return result

def render_session_info(session, left_col_size = 0.2, right_col_size = 0.8):
    print(session)
    with st.popover("View Session Info"):
        with st.container():
            col1, col2 = st.columns([left_col_size, right_col_size])
            with col1:
                st.markdown("<b>Embeddings Model</b>: ", unsafe_allow_html=True)
            with col2: 
                st.markdown(session.embeddings_model_name)
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
                        loc_col, ticker_col, report_type_col, year_col, quarter_col = st.columns(5)
                        with loc_col:
                            file_name = os.path.basename(existing_file.file_path)
                            st.markdown(f"<b>{file_name}</b>", unsafe_allow_html=True)
                        with ticker_col:
                            st.markdown(f"Company: {existing_file.company}")
                        with report_type_col:
                            st.markdown(f"Report Type: {existing_file.report_type}", unsafe_allow_html=True)
                        with year_col:
                            st.markdown(f"Year: {existing_file.year}")
                        if  existing_file.quarter:
                            with quarter_col:
                                st.markdown(f"Quarter: {existing_file.quarter}")
