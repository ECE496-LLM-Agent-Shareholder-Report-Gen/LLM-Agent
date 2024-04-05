import datetime
from random import randint
import streamlit as st
from GUI.navbar import disable_active_session
from session import QAE, BenchmarkSession, ChatSession, Report, Session
from sec_api import QueryApi
import json, requests

import os

"""Everything pertaining to the sessions for the GUI.
This class creates the session itself, and is responsible for
populating Sessions."""
class SessionRenderer:

    def __init__(self, global_singleton, isBenchmark=False):
        self.global_singleton = global_singleton
        self.isBenchmark = isBenchmark
        if 'reports' not in st.session_state:
            st.session_state['reports'] = []
        if 'name' not in st.session_state:
            st.session_state['name'] = ""
        if 'llm_chain' not in st.session_state:
            st.session_state['llm_chain'] = ""
        if 'retrieval_strategy' not in st.session_state:
            st.session_state['retrieval_strategy'] = ""
        if 'k' not in st.session_state:
            st.session_state['k'] = 0
        if 'k_i' not in st.session_state:
            st.session_state['k_i'] = 0
        if 'memory_enabled' not in st.session_state:
            st.session_state['memory_enabled'] = False
        if "widget_key" not in st.session_state:
            st.session_state["widget_key"] = str(randint(1000, 100000000))
        if isBenchmark:
            if "question_expected_final" not in st.session_state:
                st.switch_page("pages/benchmark_page.py")

    def render(self):
        if self.isBenchmark:
            self.render_header_benchmark()
        else:
            self.render_header()
        st.divider()
        self.render_report_loader()
        st.divider()
        self.render_retrieval_strategy_selector()
        st.divider()
        self.render_llm_chain_selector()
        st.divider()
        if self.isBenchmark:
            self.render_create_benchmark()
        else:
            self.render_create()


    def render_header_benchmark(self):
        st.title('Complete Benchmark Creation')
        st.markdown('<b>Select the reports, retrieval strategy, and LLM chain</b>', unsafe_allow_html=True)
        back_to_qae = st.button("Back", key="b_back")
        if back_to_qae:
            st.switch_page("pages/benchmark_page.py")

    def render_header(self):
        st.title('Chat with Shareholder Reports')
        st.markdown('<b>Create a new session to chat with your shareholder reports</b>', unsafe_allow_html=True)
        st.session_state.name = st.text_input("Choose the name of the session", placeholder="Session name...", max_chars=50)

    # callback handler for removing reports
    def remove_report(self, file_path):
        for report in st.session_state.reports:
            if report.file_path == file_path:
                st.session_state.reports.remove(report)

    # check if report already in reports
    def check_in_reports(self, report, reports):
        if not any(r.company == report.company and r.year == report.year and r.report_type == report.report_type and r.quarter == report.quarter for r in reports):
            return True
        else:
            if report.quarter:
                st.warning(f"Report: {report.company}, {report.year}, {report.report_type}, {report.quarter} already added!")
            else:
                st.warning(f"Report: {report.company}, {report.year}, {report.report_type} already added!")
            return False

    def clear_uploaded_files(self):
        # st.session_state["company_ticker"] = ""
        # st.session_state["year"] = ""
        # st.session_state["report_type"] = "10K"
        # st.session_state["report_quarter"] = "Q1"
        # st.session_state["report_type_other"] = ""
        st.session_state["widget_key"] = str(randint(1000, 100000000))
        st.rerun()


    # renders the component that lets users choose what reports to query on
    def render_report_loader(self):
        st.subheader('Choose your Shareholder Reports', divider='grey')

        upload_report, sec_report, existing_report = st.tabs(["Upload Report", "Fetch From SEC EDGAR", "Use Existing Report"])
        
        with upload_report:
            upload_form = st.form(key="u_form", clear_on_submit=True, border=False)
            with upload_form:
                uploaded_file = st.file_uploader('Upload your own shareholder reports', key=st.session_state["widget_key"])
                st.markdown('Enter some info about the report')

                if "company_ticker" not in st.session_state:
                    st.session_state["company_ticker"] = ""
                if "year" not in st.session_state:
                    st.session_state["year"] = ""
                if "report_type" not in st.session_state:
                    st.session_state["report_type"] = "10K"
                if "report_type_other" not in st.session_state:
                    st.session_state["report_type_other"] = ""
                if "report_quarter" not in st.session_state:
                    st.session_state["report_quarter"] = "Q1"

                company_ticker = None
                year = None
                report_type = None
                report_quarter = None
                report_type_other = None

                left_col, right_col = st.columns(2)
                with left_col:
                    company_ticker = st.text_input("Company Ticker", placeholder="Company Ticker", max_chars=6, key="company_ticker")
                    report_type_options = ["10K","10Q", "Other"]
                    report_type = st.selectbox("Report Type", options=report_type_options, key="report_type")


                with right_col:
                    year = st.text_input("Year the report was filed", placeholder="Year", max_chars=4, key="year")
                    if report_type == '10Q':
                        report_quarter_options = ["Q1", "Q2", "Q3"]
                        report_quarter = st.selectbox("Quarter", options=report_quarter_options, key="report_quarter")
                    if report_type == 'Other':
                        report_type_other = st.text_input("Please specify report type", placeholder="Other Report Type...", key="report_type_other")

                save_report = st.checkbox("Would you like to save the report for future use?", value=True)

                up_submitted = st.form_submit_button("Add Report", use_container_width=True)

        
        with sec_report:
            sec_api_key = st.text_input("SEC API Key", placeholder="Enter API Key", type="password", key="sec_api_key")

            with st.form(clear_on_submit=True, key="sec_form"):
                left_col, right_col = st.columns(2)
                with left_col:
                    sec_filing_ticker = st.text_input("Company Ticker", placeholder="Enter Company Ticker...", max_chars=5, key="sec_filing_ticker_input")
                with right_col:
                    sec_filing_year = st.text_input("10K Year", placeholder="Enter Report Year...", max_chars=4, key="sec_filing_year_input")


                sec_save_report = st.checkbox("Would you like to save the report for future use?", value=True, key="sec_save")
                sec_submitted = st.form_submit_button("Add Report", use_container_width=True)

        with existing_report:
            saved_ticker_col, saved_year_col, saved_report_type_col, saved_quarter_col = st.columns(4)
            selected_companies = []
            selected_years = []
            selected_report_types =[]
            selected_quarters = []
            with saved_ticker_col:
                comps = self.global_singleton.file_manager.get_companies()
                comps.sort()
                selected_companies = st.multiselect("Company Ticker", comps)
                pass
            with saved_year_col:
                years = self.global_singleton.file_manager.get_years(selected_companies)
                years.sort()
                selected_years = st.multiselect("Year", years)
                pass
            with saved_report_type_col:
                report_types = self.global_singleton.file_manager.get_report_types(selected_companies, selected_years)
                report_types.sort()
                selected_report_types = st.multiselect("Report Type", report_types)
                pass
            with saved_quarter_col:
                quarters = self.global_singleton.file_manager.get_quarters(selected_companies, selected_years, selected_report_types)
                selected_quarters = st.multiselect("Quarter (if 10Q)", quarters)
                pass

            submitted = st.button("Add Report", use_container_width=True, key="existing_submit")

        if sec_submitted:
            #handle fetching from sec edgar       
            sec_filing_ticker = sec_filing_ticker.upper()
            report_type = '10K'
            report = Report(sec_filing_ticker, sec_filing_year, "10K")
            file_name = f'{sec_filing_ticker}_{sec_filing_year}.pdf'
            report.company = sec_filing_ticker
            report.year = sec_filing_year

            url_query = {
                "query": {
                    "query_string": {
                        "query": f'formType:"10-K" AND ticker:{sec_filing_ticker} AND filedAt:[{sec_filing_year}-01-01 TO {sec_filing_year}-12-31]',
                    }
                },
                "from": "0",
                "size": "1",
            }

            #setting up sec-api key
            queryApi = QueryApi(api_key=sec_api_key)


            response = queryApi.get_filings(url_query)
            url = json.dumps(response["filings"][0]["linkToFilingDetails"], indent=2).replace(
                '"', ""
            )

            endpoint = "https://api.sec-api.io/filing-reader"
            params = {
                "url": url,
                "token": sec_api_key,
                "type": "pdf",
            }

            response = requests.get(endpoint, params=params)

            tmp_location = os.path.join('/tmp', file_name)
            with open(tmp_location, 'wb') as f:
                f.write(response.content)

            if sec_save_report:
                file_path = self.global_singleton.file_manager.move_file(tmp_location, report.company, report.year, report.report_type)
                report.file_path = file_path
            else:
                report.file_path = tmp_location

            report.save = sec_save_report

            # Check if the report is already in st.session_state.reports
            if self.check_in_reports(report, st.session_state.reports) and len(st.session_state.reports) < 10:
                st.session_state.reports.append(report)
            elif len(st.session_state.reports) >= 10:
                st.warning(f"Failed to add report '{uploaded_file.name}', maximum number of reports (10) reached")
            
            self.clear_uploaded_files()

        if up_submitted:
            if uploaded_file:
                report = None
                if company_ticker and year and report_type:
                    if report_type == '10Q':
                        if report_quarter:
                            report = Report(company_ticker.upper(), year, report_type, report_quarter)
                        else:
                            st.warning("Failed to add from uploaded reports: Missing report info")

                    elif report_type.lower() == 'other':
                        report = Report(company_ticker.upper(), year, report_type_other)
                    else:
                        report = Report(company_ticker.upper(), year, report_type)

                    tmp_location = os.path.join('/tmp', uploaded_file.name)
                    with open(tmp_location, 'wb') as out:
                        out.write(uploaded_file.getvalue())
                    if save_report:
                        file_path = self.global_singleton.file_manager.move_file(tmp_location, report.company, report.year, report.report_type, report.quarter)
                        report.file_path = file_path
                    else:
                        report.file_path = tmp_location

                    report.save = save_report

                    # Check if the report is already in st.session_state.reports
                    if self.check_in_reports(report, st.session_state.reports) and len(st.session_state.reports) < 10:
                        st.session_state.reports.append(report)
                    elif len(st.session_state.reports) >= 10:
                        st.warning(f"Failed to add report '{uploaded_file.name}', maximum number of reports (10) reached")
                    company_ticker = None
                    year = None
                    report_type = None
                    report_quarter = None
                    report_type_other = None
                    self.clear_uploaded_files()
                else:
                    st.warning("Failed to add from uploaded reports: Missing report info")
            else:
                st.warning("No file uploaded.")

        # file(s) submitted
        if submitted:
            # handle reports that were added from saved reports
            if selected_companies:
                for selected_company in selected_companies:
                    if selected_years and selected_report_types:
                        for selected_year in selected_years:
                            for selected_report_type in selected_report_types:

                                if '10Q' in selected_report_types:
                                    if selected_quarters:
                                        for quarter in selected_quarters:
                                            file_path = self.global_singleton.file_manager.get_file_path(selected_company, selected_year, selected_report_type, quarter)
                                            report = Report(selected_company, selected_year, selected_report_type, quarter=quarter, file_path=file_path)
                                            # Check if the report is already in st.session_state.reports
                                            if self.check_in_reports(report, st.session_state.reports) and len(st.session_state.reports) < 10:
                                                st.session_state.reports.append(report)
                                            elif len(st.session_state.reports) >= 10:
                                                st.warning(f"Failed to add report '{os.path.basename(file_path)}', maximum number of reports (10) reached")
                                    else:
                                        st.warning("Failed to add from saved reports: Missing quarter for 10Qs")
                                else:
                                    file_path = self.global_singleton.file_manager.get_file_path(selected_company, selected_year, selected_report_type)
                                    report = Report(selected_company, selected_year, selected_report_type, file_path=file_path)
                                    # Check if the report is already in st.session_state.reports
                                    if self.check_in_reports(report, st.session_state.reports) and len(st.session_state.reports) < 10:
                                        st.session_state.reports.append(report)
                                    elif len(st.session_state.reports) >= 10:
                                        st.warning(f"Failed to add report '{os.path.basename(file_path)}', maximum number of reports (10) reached")
                    else:
                        st.warning("Failed to add from saved reports: Missing years/report types")


        # clear the inputs

        existing_files = st.session_state.reports
        if len(existing_files) == 10:
            st.info("Maximum Number of files reached (10)!")

        for existing_file in existing_files:
            container = st.container(border=True)
            with container:
                loc_col, ticker_col, report_type_col, year_col, quarter_col, close_col = st.columns(6)
                with loc_col:
                    file_name = os.path.basename(existing_file.file_path)
                    st.markdown(f"<b>{file_name}</b>", unsafe_allow_html=True)
                with ticker_col:
                    st.write(f"Company: {existing_file.company}")
                with report_type_col:
                    st.markdown(f"Report Type: {existing_file.report_type}", unsafe_allow_html=True)
                with year_col:
                    st.markdown(f"Year: {existing_file.year}")
                if  existing_file.quarter:
                    with quarter_col:
                        st.markdown(f"Quarter: {existing_file.quarter}")
                with close_col:
                    st.button(":x:", key=existing_file.file_path, on_click=self.remove_report, args=[existing_file.file_path])



    def render_retrieval_strategy_selector(self):
        st.subheader('Choose your Retrieval Strategy', divider='grey')
        strategies = [{
            'key': 'SRS',
            'title': 'Simple Retrieval Strategy',
            'description': 'This strategy passes an input to the vector store, it returns k documents to use in the as context.',
            'k_i_exists': False
        },
        {
            'key': 'ReRS',
            'title': 'Reranker Retrieval Strategy',
            'description': 'This strategy passes an input to the vector store, it finds k_i documents using the simple retrieval strategy, then re-ranks them using a cross-encoder, and returns k documents to use as context. <u>NOTE: a cross-encoder must be loaded in to use this strategy.</u>',
            'k_i_exists': True
        },

        {
            'key': 'RRS',
            'title': 'Random Retrieval Strategy',
            'description': 'This strategy passes an input to the vector store, returns k_i documents, and out of those k documents, it randomly chooses k to pass as context.',
            'k_i_exists': True
        }]
        container = st.container(border=False)
        with container:
            for strategy in strategies:
                strategy['c'] = st.markdown(f"<b>{strategy['title']}</b><br>{strategy['description']}", unsafe_allow_html=True)

            strategy_options = [strategy['title'] for strategy in strategies]
            st.session_state.retrieval_strategy = st.selectbox("Choose a Retrieval Strategy", options=strategy_options, key="retriever_select")
            if st.session_state.retrieval_strategy:
                # find chosen strategy in the list
                strat = None
                for strategy in strategies:
                    if strategy['title'] == st.session_state.retrieval_strategy:
                        strat = strategy

                col1, col2 = st.columns(2)
                with col1:
                    st.session_state.k = st.number_input("Number of documents to use as context (k)", min_value=0, step=1, value=8)
                if strat['k_i_exists']:
                    with col2:
                        st.session_state.k_i = st.number_input("Initial number of documents to fetch (k_i)", min_value=0, step=1, value=100)




    def render_llm_chain_selector(self):
        st.subheader('Choose your Language Model Chain', divider='grey' )
        container = st.container(border=False)
        strategies = [{
            'key': 'SC',
            'title': 'One-to-One Chain',
            'description': 'The question is passed to the retriever which queries a single vector store containing all the shareholder reports for the session. The retriever returns a context. The LLM answers the original question using this context.',
        },
        {
            'key': 'SSbC',
            'title': 'One-to-Many Chain',
            'description': 'The question is passed to the retriever which queries every vector store. Each vector store contains at most one shareholder report. The retriever returns a context, and the LLM answers the original question using this context.',
        },
        {
            'key': 'FC',
            'title': 'One-to-Many Multi-Query Chain',
            'description': 'The question is passed to the LLM and generates sub queries. Each sub query pertains to a single vector store. Each vector store contains at most one shareholder report. The retriever returns context for each sub query. The LLM answers the original question using this context.',
            'k_i_exists': True
        },
        {
            'key': 'SbC',
            'title': 'One-to-Many Multi-Query Stepback Chain',
            'description': 'The question is passed to the LLM and generates sub queries. Each sub query pertains to a single vector store. Each vector store contains at most one shareholder report. The retriever returns context for each sub query. The LLM answers each sub query using the context retrieved from that sub query. The LLM answers the original question using the answers from the sub queries.',
        },
        {
            'key': 'A',
            'title': 'ReAct Chain',
            'description': 'The LLM answers the question through reasoning and acting. If the LLM needs a context to answer the question, it must reason it out itself, and must invoke the retriever through its actions.',
        }]
        with container:
            for strategy in strategies:
                st.markdown(f"<b>{strategy['title']}</b><br>{strategy['description']}", unsafe_allow_html=True)

                strategy_options = [strategy['title'] for strategy in strategies]
            st.session_state.llm_chain = st.selectbox("Choose an LLM Chain", options=strategy_options, key='chain_select')

    def render_create(self):
        create_session = st.button("Create Session", use_container_width=True)
        if create_session:
            name = st.session_state["name"].strip()
            submittable = True
            if not name:
                submittable = False
                st.error("No session name was given. Please provide a session name")

            if submittable:
                for ses in self.global_singleton.chat_session_manager.sessions:
                    if ses == name:
                        st.error(f"The session name '{name}' is already being used. Please try another name.")
                        submittable = False
                try:
                    session = ChatSession(name=st.session_state.name,
                                        #embeddings_model_name=self.global_singleton.embeddings_model_name,
                                        llm_chain=st.session_state.llm_chain,
                                        retrieval_strategy=st.session_state.retrieval_strategy,
                                        reports=st.session_state.reports,
                                        memory_enabled=st.session_state.memory_enabled,
                                        k=st.session_state.k,
                                        k_i=st.session_state.k_i)
                    del st.session_state["reports"]
                    del st.session_state["name"]
                    del st.session_state["memory_enabled"]
                    del st.session_state["llm_chain"]
                    del st.session_state["retrieval_strategy"]
                    del st.session_state["k"]
                    del st.session_state["k_i"]
                    disable_active_session(self.global_singleton)
                    self.global_singleton.chat_session_manager.add_session(session)
                    self.global_singleton.chat_session_manager.set_active_session(session)
                    st.session_state["global_singleton"] = self.global_singleton
                    st.switch_page("pages/chat_page.py")
                except Exception as e:
                    print(e)
                    st.error("Something went wrong while creating the new session. This may have been due to a page reload in which some data was lost. Please try again.")

    def render_create_benchmark(self):
        
        create_session = st.button("Create Session", use_container_width=True)
        if create_session:
            try:
                # turn st.session_state.question_expected to a dict
                session = BenchmarkSession(name=st.session_state.b_name,
                                        #    embeddings_model_name=self.global_singleton.embeddings_model_name,
                                        llm_chain=st.session_state.llm_chain,
                                        retrieval_strategy=st.session_state.retrieval_strategy,
                                        question_answer_expected= st.session_state.question_expected_final,
                                        reports=st.session_state.reports,
                                        memory_enabled=st.session_state.memory_enabled,k=st.session_state.k,
                                        k_i=st.session_state.k_i)
                del st.session_state["reports"]
                del st.session_state["b_name"]
                del st.session_state["question_expected_final"]
                del st.session_state["question_expected"]
                del st.session_state["memory_enabled"]
                del st.session_state["llm_chain"]
                del st.session_state["retrieval_strategy"]
                del st.session_state["k"]
                del st.session_state["k_i"]
                self.global_singleton.benchmark_session_manager.add_session(session)
                disable_active_session(self.global_singleton)
                self.global_singleton.benchmark_session_manager.set_active_session(session)
                st.session_state["global_singleton"] = self.global_singleton
                st.switch_page("pages/benchmark_eval_page.py")
            except Exception as e:
                print(e)
                st.error("Something went wrong while creating the new session. This may have been due to a page reload in which some data was lost. Please try again.")
