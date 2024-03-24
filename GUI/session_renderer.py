from random import randint
import streamlit as st
from session import QAE, BenchmarkSession, ChatSession, Report, Session

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

    def render_header(self):
        st.title('Chat with Shareholder Reports')
        st.markdown('<b>Create a new session to chat with your shareholder reports</b>', unsafe_allow_html=True)
        st.session_state.name = st.text_input("Choose the name of the session", placeholder="Session name...", max_chars=50)

    # callback handler for removing reports
    def remove_report(self, file_path):
        print("Removing report...")
        for report in st.session_state.reports:
            if report.file_path == file_path:
                st.session_state.reports.remove(report)
                print("removed report: ", file_path)

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

        sec_filing = st.text_input("Or find an SEC filing by Company Ticker or CIK", placeholder="Company Ticker or CIK...", max_chars=20, key="sec_filing")
        # do sec filing related fetching...

        st.markdown("Or choose from previously saved reports")
        reports = ['AMD 2022 10K', 'AMD 2022 10Q Q1', 'AMD 2022 10Q Q2', 'AMD 2022 10Q Q3', 'INTC 2022 10K', 'INTC 2022 10Q Q1', 'INTC 2022 10Q Q2', 'INTC 2022 10Q Q3']
        saved_ticker_col, saved_year_col, saved_report_type_col, saved_quarter_col = st.columns(4)
        selected_companies = []
        selected_years = []
        selected_report_types =[]
        selected_quarters = []
        with saved_ticker_col:
            comps = self.global_singleton.file_manager.get_companies()
            selected_companies = st.multiselect("Company Ticker", comps)
            pass
        with saved_year_col:
            years = self.global_singleton.file_manager.get_years(selected_companies)
            selected_years = st.multiselect("Year", years)
            pass
        with saved_report_type_col:
            report_types = self.global_singleton.file_manager.get_report_types(selected_companies, selected_years)
            selected_report_types = st.multiselect("Report Type", report_types)
            pass
        with saved_quarter_col:
            quarters = self.global_singleton.file_manager.get_quarters(selected_companies, selected_years, selected_report_types)
            selected_quarters = st.multiselect("Quarter (if 10Q)", quarters)
            pass



        submitted = st.button("Add Report", use_container_width=True)

        # file(s) submitted
        if submitted:
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
            'key': 'PDR',
            'title': 'Parent Document Retrieval Strategy',
            'description': 'This strategy uses larger chunks (called parents) and smaller chunks (called children). The similarity search is done on the child chunks, and the parents of the child chunks are retrieved, up to k parent chunks.',
            'k_i_exists': False
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
            'title': 'Simple Chain',
            'description': 'This input is passed to the language model, and the language model answers it directly.',
        },
        {
            'key': 'A',
            'title': 'Agent Chain',
            'description': 'This input is passed to the language model, the language model then breaks the question down into smaller questions to ask about each shareholder report, and answers those questions. The answers to these smaller questions is fed as context to answer the original question',
        },
        {
            'key': 'SbC',
            'title': 'Stepback Chain',
            'description': 'This input is passed to the language model, the language model then breaks the question down into smaller questions to ask about each shareholder report, and answers those questions. The answers to these smaller questions is fed as context to answer the original question',
        },
        {
            'key': 'SSbC',
            'title': 'Simple Stepback Chain',
            'description': 'This input is passed to the language model, the language model passes the question to each shareholder report, combines the context, and uses the context  to answer the  question',
        },
        {
            'key': 'FC',
            'title': 'Fusion Chain',
            'description': 'This input is passed to the language model, the language model passes the breaks the question down, and passes the smaller questions to the respective shareholder report. Context is retrieved and used to answer the original question',
            'k_i_exists': True
        }]
        with container:
            for strategy in strategies:
                st.markdown(f"<b>{strategy['title']}</b><br>{strategy['description']}", unsafe_allow_html=True)

                strategy_options = [strategy['title'] for strategy in strategies]
            st.session_state.llm_chain = st.selectbox("Choose an LLM Chain", options=strategy_options, key='chain_select')

    def render_create(self):
        st.session_state.memory_enabled = st.checkbox("Would you like the language model to utilize previous Q&A's in the session to influence future answers (i.e., enable memory)?", key="mem_enabled" )
        create_session = st.button("Create Session", use_container_width=True)
        if create_session:
            print("creating session")
            session = ChatSession(name=st.session_state.name,
                                    #embeddings_model_name=self.global_singleton.embeddings_model_name,
                                    llm_chain=st.session_state.llm_chain,
                                    retrieval_strategy=st.session_state.retrieval_strategy,
                                    reports=st.session_state.reports,
                                    memory_enabled=st.session_state.memory_enabled,
                                    k=st.session_state.k,
                                    k_i=st.session_state.k_i)
            st.session_state.reports = []
            self.global_singleton.chat_session_manager.add_session(session)
            self.global_singleton.chat_session_manager.set_active_session(session)
            st.session_state["global_singleton"] = self.global_singleton
            st.switch_page("pages/chat_page.py")

    def render_create_benchmark(self):
        st.session_state.memory_enabled = st.checkbox("Would you like the language model to utilize previous Q&A's in the session to influence future answers (i.e., enable memory)?", key="mem_enabled" )
        create_session = st.button("Create Session", use_container_width=True)
        if create_session:
            # turn st.session_state.question_expected to a dict
            qae_dict = {}
            for idx, qe in enumerate(st.session_state.question_expected):
                qae_dict[idx+1] = QAE(question=qe[0], expected=qe[1])
            session = BenchmarkSession(name=st.session_state.name,
                                    #    embeddings_model_name=self.global_singleton.embeddings_model_name,
                                       llm_chain=st.session_state.llm_chain,
                                       retrieval_strategy=st.session_state.retrieval_strategy,
                                       question_answer_expected=qae_dict,
                                       reports=st.session_state.reports,
                                       memory_enabled=st.session_state.memory_enabled,k=st.session_state.k,
                                       k_i=st.session_state.k_i)
            st.session_state.reports = []
            self.global_singleton.benchmark_session_manager.add_session(session)
            self.global_singleton.benchmark_session_manager.set_active_session(session)
            st.session_state["global_singleton"] = self.global_singleton
            st.switch_page("pages/benchmark_eval_page.py")