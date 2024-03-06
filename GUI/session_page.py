import streamlit as st

import os

"""Everything pertaining to the sessions for the GUI. 
This class creates the page itself, and is responsible for
populating Sessions."""
class SessionPage:

    def __init__(self):
        self.conversation_history = []
        self.reports = []
        self.llm_chain = 'simple'
        self.retrieval_strategy = 'simple'
        self.name = 'Session'
        self.folder_path = '.'
        
    def render(self):
        self.render_header()
        st.divider()
        self.render_report_loader()
        st.divider()
        self.render_retrieval_strategy_selector()
        st.divider()
        self.render_llm_chain_selector()
        st.divider()
        self.render_create()

    def render_header(self):
        st.title('Chat with Shareholder Reports')
        st.markdown('<b>Create a new session to chat with your shareholder reports</b>', unsafe_allow_html=True)
        self.name = st.text_input("Choose the name of the session", placeholder="Session name...", max_chars=20)
        pass

    def render_report_loader(self):
        st.subheader('Choose your Shareholder Reports', divider='grey')
        selected_filename = st.file_uploader('Upload your own shareholder reports', key="files")
        
        sec_filing = st.text_input("Or find an SEC filing by Company Ticker or CIK", placeholder="Company Ticker or CIK...", max_chars=20, key="sec_ticker_cik")
        # do sec filing related fetching...

        reports = ['AMD 2022 10K', 'AMD 2022 10Q Q1', 'AMD 2022 10Q Q2', 'AMD 2022 10Q Q3', 'INTC 2022 10K', 'INTC 2022 10Q Q1', 'INTC 2022 10Q Q2', 'INTC 2022 10Q Q3']
        selected_reports = st.multiselect('Or choose from a list of previously saved reports', reports)

        st.markdown('Enter some info about the report')

        report_type = None
        company_ticker = None
        year = None
        report_quarter = None
        report_type_other = None
        
        left_col, right_col = st.columns(2)
        with left_col:
            company_ticker = st.text_input("Company Ticker", placeholder="Company Ticker", max_chars=6, key="company_ticker")
            report_type_options = ["10K","10Q", "Other"]
            report_type = st.selectbox("Report Type", options=report_type_options, key="report_type")
            if report_type == 'Other':
                report_type_other = st.text_input("Please specifiy report type", placeholder="Other Report Type...")

        with right_col:
            year = st.text_input("Year the report was filed", placeholder="Year", max_chars=4, key="year")
            if report_type == '10Q':
                report_quarter_options = ["Q1", "Q2", "Q3"]
                report_quarter = st.selectbox("Quarter", options=report_quarter_options, key="report_quarter")

        save_report = st.checkbox("Would you like to save the report for future use?")

        submitted = st.button("Add Report", use_container_width=True)
        
        
        
        existing_files = [{
            'report_loc': 'AMD_10K_2022.pdf',
            'ticker': 'AMD',
            'report_type': '10K',
            'year': '2022'},
            { 'report_loc': 'INTC_10Q_2022.pdf',
            'ticker': 'INTC',
            'report_type': '10Q',
            'year': '2022',
            'quarter': 'Q1'
        }]

        for existing_file in existing_files:
            container = st.container(border=True)
            with container:
                loc, ticker, report_type, year, quarter, close = st.columns(6)
                with loc:
                    st.markdown(f"<b>{existing_file['report_loc']}</b>", unsafe_allow_html=True)
                with ticker:
                    st.write(f"Company: {existing_file['ticker']}")
                with report_type:
                    st.markdown(f"Report Type: {existing_file['report_type']}", unsafe_allow_html=True)
                with year:
                    st.markdown(f"Year: {existing_file['year']}")
                if 'quarter' in existing_file:
                    with quarter:
                        st.markdown(f"Quarter: {existing_file['quarter']}")
                with close:
                    st.button(":x:", key=existing_file['report_loc'])



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
            chosen_strategy = st.selectbox("Choose a Retrieval Strategy", options=strategy_options, key="retriever_select")
            if chosen_strategy:
                # find chosen strategy in the list
                strat = None
                for strategy in strategies:
                    if strategy['title'] == chosen_strategy:
                        print("Strategy: ",strategy['title'] )
                        strat = strategy
                    
                col1, col2 = st.columns(2)
                with col1:
                    k = st.number_input("Number of documents to use as context (k)", min_value=0, step=1)
                    print(k)
                if strat['k_i_exists']:
                    with col2:
                        k_i = st.number_input("Initial number of documents to fetch (k_i)", min_value=0, step=1)




    def render_llm_chain_selector(self):
        st.subheader('Choose your Language Model Chain', divider='grey' )
        container = st.container(border=False)
        strategies = [{
            'key': 'SC',
            'title': 'Simple Chain',
            'description': 'This input is passed to the language model, and the language model answers it directly.',
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
                strategy['c'] = st.markdown(f"<b>{strategy['title']}</b><br>{strategy['description']}", unsafe_allow_html=True)
            
                strategy_options = [strategy['title'] for strategy in strategies]
            chosen_strategy = st.selectbox("Choose an LLM Chain", options=strategy_options, key='chain_select')

    def render_create(self):
        mem_enabled = st.checkbox("Would you like the language model to utilize previous Q&A's in the session to influence future answers (i.e., enable memory)?", key="mem_enabled" )
        create_session = st.button("Create Session", use_container_width=True)
