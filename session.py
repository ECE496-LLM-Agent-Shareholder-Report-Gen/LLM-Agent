import os
from langchain.document_loaders.pdf import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

import datetime
import json
from chatbots import SimpleChatbot

from retriever_strategies import CompositeRetrieverStrategy, SimpleRetrieverStrategy

""" Session
holds the state of the session, which represents
a user-LLM interaction """
class Session:

    def __init__(self, name=None, llm_chain=None, retrieval_strategy=None, conversation_history=[], reports=[], memory_enabled=False, k=None, k_i=None):
        self.conversation_history = conversation_history
        self.reports = reports
        self.llm_chain = llm_chain
        self.retrieval_strategy = retrieval_strategy
        if name == None or len(name.strip()) == 0:
            now = datetime.datetime.now()
            formatted_time = now.strftime("%Y-%m-%dT%H:%M:%S")
            self.name = formatted_time
        else:
            self.name = name
        self.memory_enabled = False
        self.k = k
        self.k_i = k_i
        self.memory_enabled = memory_enabled
        self.vectorstores = {}
        self.initialized = False
    
    def initialize(self, index_generator, file_manager, llm, embeddings):
        valid_retrieval_strategies = ["Simple Retrieval Strategy"]
        valid_llm_chains = ["Simple Chain"]
        if not self.retrieval_strategy or not self.retrieval_strategy in valid_retrieval_strategies:
            raise Exception("Invalid retrieval strategy: ", self.retrieval_strategy, ", must be one of ", valid_retrieval_strategies) 
        
        if not self.llm_chain or not self.llm_chain in valid_llm_chains:
            raise Exception("Invalid LLM Chain: ", self.llm_chain, ", must be one of ", valid_llm_chains) 

        if len(self.reports) == 0:
            raise Exception("No reports added") 

        # get vector stores from files
        self.populate_vectorstore(file_manager, index_generator, embeddings, self.reports)
        print("vectorstores: ", self.vectorstores)
        if self.retrieval_strategy == "Simple Retrieval Strategy":
            print("initializing simple retrieveer strategy")
            flat_vs = self.gen_vectorstore_flat()
            vectorstore = None
            iter_1 = True
            print("flat vs: ", flat_vs)
            for vs in flat_vs:
                if iter_1:
                    iter_1 = False
                    vectorstore = vs
                else:
                    index_generator.merge_vector_stores(vectorstore, vs)
            simple_retriever = SimpleRetrieverStrategy(vectorstore) 
            self.retriever_strategy_obj = CompositeRetrieverStrategy([simple_retriever], ["company", "year", "report type", "quarter"])
        else:
            print("No retriever strategy initialized")

        if self.llm_chain == "Simple Chain":
            print("initializing simple chatbot")
            self.chatbot = SimpleChatbot(self.retriever_strategy_obj, llm)
        else:
            print("No Chatbot initialized")
        self.initialized = True


    def add_to_conversation(self, question, answer):
        qa = QA(question, answer)
        if not isinstance(self.conversation_history, list): 
            self.conversation_history = []
        self.conversation_history.append(qa)

    def add_report(self, report):
        self.reports.append(report)

    def set_reports(self, reports):
        self.reports = reports

    def add_reports_dict(self, report_dict_list):
        for report_dict in report_dict_list:
            report = Report(**report_dict)
            self.add_report(report)

    def add_conversation_history(self, conversation_history_dict_list):
        for conversation_history_dict in conversation_history_dict_list:
            self.add_report(**conversation_history_dict)

    def generate_vectorstore(self, index_generator, embeddings, report):
        vectorstore = index_generator.generate_vector_store_pdf_file(embeddings, report.file_path, report.company, report.year, report.report_type, report.quarter)
        if report.save:
            index_path = os.path.join(os.path.dirname(report.file_path), "index")
            index_generator.save_vector_store(vectorstore, index_path)
        return vectorstore
            
    
    def load_vectorstore(self, file_manager, index_generator, embeddings, report):
        index_path = file_manager.get_index(report.company, report.year, report.report_type, report.quarter)
        if index_path:
            print("index path: ", index_path)
            return index_generator.load_vector_store(index_path, embeddings)
        else:
            return None
    
    def populate_vectorstore(self, file_manager, index_generator, embeddings, reports):
        for report in reports:
            vectorstore = self.load_vectorstore(file_manager, index_generator, embeddings, report)
            if vectorstore == None:
                vectorstore = self.generate_vectorstore(index_generator, embeddings, report)
            if report.quarter:
                dict_vectorstore = { report.year: { report.report_type: { report.quarter: vectorstore } } }
            else: 
                dict_vectorstore = { report.year: { report.report_type:  vectorstore } }
            self.vectorstores[report.company] = dict_vectorstore


    def dict_to_list(self, d, parent_keys=[]):
        lst = []
        for k, v in d.items():
            new_keys = parent_keys + [k]
            if isinstance(v, dict):
                lst.extend(self.dict_to_list(v, new_keys))
            else:
                lst.append(v)
        return lst

  
    def gen_vectorstore_flat(self):
        
        return self.dict_to_list(self.vectorstores)
    
   

    def encode(self):
        return vars(self)

    def to_dict(self, indent=None):
        json_str = json.dumps(self, default=lambda o: o.encode(), indent=indent)
        ses_dict = json.load(json_str)
        del ses_dict["vectorstores"]
        del ses_dict["retriever_strategy_obj"]
        del ses_dict["chatbot"]
        return ses_dict

    @classmethod
    def from_dict(cls, cls_dict):
        ses = cls(**cls_dict)
        ses.reports = []
        ses.conversation_history = []
        ses.add_reports_dict(cls_dict['reports'])
        ses.add_conversation_history(cls_dict['conversation_history'])
        return ses


""" QA
class that represents a question and an answer """
class QA:
    def __init__(self, question, answer):
        self.question = question
        self.answer = answer

    def encode(self):
        return vars(self)


""" Report
class that represents a company's shareholder report, metadata ONLY """
class Report:
    def __init__(self, company, year, report_type, quarter=None, file_path=None, save=True):
        self.company = company
        self.year = year
        self.report_type = report_type
        if report_type.lower() == '10q':
            assert quarter != None, "Quarter must exist for report type 10Q"
        self.quarter = quarter
        self.file_path = file_path
        self.save = save
    
    def encode(self):
        return vars(self)
