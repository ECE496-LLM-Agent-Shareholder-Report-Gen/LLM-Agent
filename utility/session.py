from agent.chatbots import FusionChatbot, SimpleChatbot, SimpleStepbackChatbot, StepbackChatbot, AgentChatbot
from agent.retriever_strategies import CompositeRetrieverStrategy, ReRankerRetrieverStrategy, SimpleRetrieverStrategy, StochasticRetrieverStrategy
from agent.template_formatter import LlamaTemplateFormatter

import os
import datetime


""" Session
holds the state of the session, which represents
a user-LLM interaction """
class Session:

    valid_retrieval_strategies = ["Simple Retrieval Strategy", "Reranker Retrieval Strategy", "Random Retrieval Strategy"]
    valid_llm_chains = ["Merged Chain", "Split Chain", "Multi-Query Split Chain", "Multi-Query Split Stepback Chain", "ReAct Chain"]

    def __init__(self,
                 name=None,
                 embeddings_model_name=None,
                 llm_chain=None,
                 retrieval_strategy=None,
                 reports=[],
                 memory_enabled=False,
                 k=None,
                 k_i=None,
                 *args, **kwargs):

        self.reports = reports
        self.llm_chain = llm_chain
        self.retrieval_strategy = retrieval_strategy
        if name == None or len(name.strip()) == 0:
            now = datetime.datetime.now()
            formatted_time = now.strftime("%Y-%m-%dT%H:%M:%S")
            self.name = formatted_time
        else:
            self.name = name
        self.embeddings_model_name = embeddings_model_name
        self.k = k
        self.k_i = k_i
        self.memory_enabled = memory_enabled
        self.vectorstores = {}
        self.initialized = False

    """ Initialize session for Q&A with LLM """
    def initialize(self, index_generator, file_manager, llm, embeddings, cross_encoder=None, load=True, isllama=False):

        if not self.retrieval_strategy or not self.retrieval_strategy in self.valid_retrieval_strategies:
            raise Exception("Invalid retrieval strategy: ", self.retrieval_strategy, ", must be one of ", ", ".self.valid_retrieval_strategies)

        if not self.llm_chain or not self.llm_chain in self.valid_llm_chains:
            raise Exception("Invalid LLM Chain: ", self.llm_chain, ", must be one of ", self.valid_llm_chains)

        # if len(self.reports) == 0:
        #     raise Exception("No reports added")

        # get vector stores from files
        self.populate_vectorstore(file_manager, index_generator, embeddings, self.reports, load=load)

        if self.retrieval_strategy == "Simple Retrieval Strategy":
            self.init_simple_retriever()
        elif self.retrieval_strategy == "Reranker Retrieval Strategy":
            self.init_reranker_retriever(cross_encoder)
        elif self.retrieval_strategy == "Random Retrieval Strategy":
            self.init_random_retriever()
        else:
            raise Exception("No retriever strategy initialized")

        print("############### \n")
        print("loading chain \n")

        if self.llm_chain == "Merged Chain":
            self.init_simple_chain(index_generator, llm, isllama=isllama)
        elif self.llm_chain == "ReAct Chain":
            self.init_agent_chain(index_generator, llm, isllama=isllama)
        elif self.llm_chain == "Multi-Query Split Chain":
            self.init_fusion_chain(llm, isllama=isllama)
        elif self.llm_chain == "Multi-Query Split Stepback Chain":
            self.init_stepback_chain(llm, isllama=isllama)
        elif self.llm_chain == "Split Chain":
            self.init_simple_stepback_chain(llm, isllama=isllama)
        else:
            raise Exception("No Chatbot initialized")

        self.initialized = True

    def deinitialize(self):
        self.initialized = False
        self.retriever_strategy_obj = None
        self.chatbot = None
        self.vectorstores = None

    """ Init simple retriever strategy """
    def init_simple_retriever(self):
        print("initializing simple retriever strategy")
        simple_retriever = SimpleRetrieverStrategy(k=self.k)
        self.retriever_strategy_obj = CompositeRetrieverStrategy([simple_retriever], ["company", "year", "report type", "quarter", "page"])

    """ Init reranker retriever strategy """
    def init_reranker_retriever(self, cross_encoder=None):
        print("initializing Reranker retriever strategy")
        if cross_encoder == None:
            raise Exception("Tried to initailize Reranker strategy without a cross encoder")
        reranker_retriever = ReRankerRetrieverStrategy(cross_encoder=cross_encoder, k=self.k, init_k=self.k_i)
        self.retriever_strategy_obj = CompositeRetrieverStrategy([reranker_retriever], ["company", "year", "report type", "quarter", "page"])

    """ Init random retriever strategy """
    def init_random_retriever(self):
        print("initializing Reranker retriever strategy")
        stochastic_retriever = StochasticRetrieverStrategy(k=self.k, fetch_k=self.k_i)
        self.retriever_strategy_obj = CompositeRetrieverStrategy([stochastic_retriever], ["company", "year", "report type", "quarter", "page"])

    """ Init Simple chain """
    def init_simple_chain(self, index_generator, llm, isllama=False):
        print("initializing simple chatbot")
        flat_vs = self.gen_vectorstore_flat()
        vectorstore = None
        iter_1 = True
        # create one vector store
        for vs in flat_vs:
            if iter_1:
                iter_1 = False
                vectorstore = vs
            else:
                index_generator.merge_vector_stores(vectorstore, vs)
        if isllama:
            self.chatbot = SimpleChatbot(self.retriever_strategy_obj, llm, vectorstore=vectorstore, template_formatter=LlamaTemplateFormatter())

        else:
            self.chatbot = SimpleChatbot(self.retriever_strategy_obj, llm, vectorstore=vectorstore)

        """ Init Simple chain """
    def init_agent_chain(self, index_generator, llm, isllama=False):
        print("initializing simple chatbot")
        flat_vs = self.gen_vectorstore_flat()
        vectorstore = None
        iter_1 = True
        # create one vector store
        for vs in flat_vs:
            if iter_1:
                iter_1 = False
                vectorstore = vs
            else:
                index_generator.merge_vector_stores(vectorstore, vs)
        if isllama:
            self.chatbot = AgentChatbot(self.retriever_strategy_obj, llm, vectorstore=vectorstore, template_formatter=LlamaTemplateFormatter())
        else:
            self.chatbot = AgentChatbot(self.retriever_strategy_obj, llm, vectorstore=vectorstore)


    """ Init Fusion chain """
    def init_fusion_chain(self, llm,  isllama=False):
        print("initializing fusion chatbot")
        if isllama:
            self.chatbot = FusionChatbot(self.retriever_strategy_obj, llm, vectorstores=self.vectorstores, max_k=self.k, template_formatter=LlamaTemplateFormatter())
        else:
            self.chatbot = FusionChatbot(self.retriever_strategy_obj, llm, vectorstores=self.vectorstores, max_k=self.k)

    """ Init Stepback chain """
    def init_stepback_chain(self, llm,  isllama=False):
        print("initializing stepback chatbot")
        if isllama:
            self.chatbot = StepbackChatbot(self.retriever_strategy_obj, llm, vectorstores=self.vectorstores, max_k=self.k, template_formatter=LlamaTemplateFormatter())
        else:
            self.chatbot = StepbackChatbot(self.retriever_strategy_obj, llm, vectorstores=self.vectorstores, max_k=self.k)

    """ Init Simple Stepback chain """
    def init_simple_stepback_chain(self, llm,  isllama=False):
        print("initializing simple stepback chatbot")
        if isllama:
            self.chatbot = SimpleStepbackChatbot(self.retriever_strategy_obj, llm, vectorstores=self.vectorstores, max_k=self.k, template_formatter=LlamaTemplateFormatter())
        else:
            self.chatbot = SimpleStepbackChatbot(self.retriever_strategy_obj, llm, vectorstores=self.vectorstores, max_k=self.k)

    def add_report(self, report):
        self.reports.append(report)

    def set_reports(self, reports):
        self.reports = reports

    def add_reports_dict(self, report_dict_list):
        for report_dict in report_dict_list:
            report = Report(**report_dict)
            self.add_report(report)

    def generate_vectorstore(self, index_generator, embeddings, report):
        vectorstore = index_generator.generate_vector_store_pdf_file(embeddings, report.file_path, report.company, report.year, report.report_type, report.quarter)
        if report.save:
            index_path = os.path.join(os.path.dirname(report.file_path), "index")
            index_generator.save_vector_store(vectorstore, index_path)
        return vectorstore


    def load_vectorstore(self, file_manager, index_generator, embeddings, report):
        if file_manager.index_exists(report.company, report.year, report.report_type, report.quarter) == False:
            return None
        index_path = file_manager.create_index(report.company, report.year, report.report_type, report.quarter)
        if index_path:
            print("index path: ", index_path)
            return index_generator.load_vector_store(index_path, embeddings)
        else:
            return None

    def populate_vectorstore(self, file_manager, index_generator, embeddings, reports, load=True):
        self.vectorstores = {}
        for report in reports:
            if load:
                vectorstore = self.load_vectorstore(file_manager, index_generator, embeddings, report)
            else:
                vectorstore = None
            if vectorstore == None:
                vectorstore = self.generate_vectorstore(index_generator, embeddings, report)
            if not report.company in self.vectorstores:
                self.vectorstores[report.company] = {}
            if not report.year in self.vectorstores[report.company]:
                self.vectorstores[report.company][report.year] = {}
            if report.quarter:
                if not report.report_type in self.vectorstores[report.company][report.year]:
                    self.vectorstores[report.company][report.year][report.report_type] = {}
                self.vectorstores[report.company][report.year][report.report_type][report.quarter] = vectorstore
            else:
                self.vectorstores[report.company][report.year][report.report_type] = vectorstore


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

    @classmethod
    def from_dict(cls, cls_dict):
        ses = cls(**cls_dict)
        ses.reports = []
        ses.add_reports_dict(cls_dict['reports'])
        return ses

    def to_dict(self):
        ses_dict = {}
        ses_dict["llm_chain"] = self.llm_chain
        ses_dict["retrieval_strategy"] = self.retrieval_strategy
        ses_dict["memory_enabled"] = self.memory_enabled
        ses_dict["embeddings_model_name"] = self.embeddings_model_name
        ses_dict["name"] = self.name
        ses_dict["k"] = self.k
        ses_dict["k_i"] = self.k_i
        ses_dict["reports"] = [report.encode() for report in self.reports]
        return ses_dict


""" Chat Session
holds the state of a chat session, which represents
a user-LLM interaction. It inherits from session, and uses
conversation history """
class ChatSession(Session):
    def __init__(self,
                 name=None,
                 embeddings_model_name=None,
                 llm_chain=None,
                 retrieval_strategy=None,
                 conversation_history=None,
                 reports=[],
                 memory_enabled=False,
                 k=None,
                 k_i=None,
                 *args,
                 **kwargs):
        super().__init__(name=name,
                         llm_chain=llm_chain,
                         embeddings_model_name=embeddings_model_name,
                         retrieval_strategy=retrieval_strategy,
                         reports=reports,
                         memory_enabled=memory_enabled,
                         k=k,
                         k_i=k_i,
                         *args,
                         **kwargs)
        if conversation_history == None:
            self.conversation_history = []
        else:
            self.conversation_history = conversation_history
            

    def add_to_conversation(self, question=None, answer=None, replays=0, **kwargs):
        qa = QA(question, answer, replays=replays, **kwargs)
        if not isinstance(self.conversation_history, list):
            self.conversation_history = []
        self.conversation_history.append(qa)

    def add_conversation_history(self, conversation_history_dict_list):
        for conversation_history_dict in conversation_history_dict_list:
            self.add_to_conversation(**conversation_history_dict)

    @classmethod
    def from_dict(cls, cls_dict):
        ses = super().from_dict(cls_dict)
        ses.conversation_history = []
        ses.add_conversation_history(cls_dict['conversation_history'])
        return ses

    def to_dict(self):
        ses_dict = super().to_dict()
        ses_dict["conversation_history"] = [qa.encode() for qa in self.conversation_history]
        return ses_dict



""" Benchmark Session
Holds the state of a benchmark session, which represents
an LLM evaluation interaction. it inherits from Session, but instead
of using conversation history it uses Question-Answer-Expected list """
class BenchmarkSession(Session):
    def __init__(self,
                 name=None,
                 embeddings_model_name=None,
                 llm_chain=None,
                 retrieval_strategy=None,
                 question_answer_expected={},
                 reports=[],
                 memory_enabled=False,
                 k=None,
                 k_i=None,
                 *args,
                 **kwargs):
        super().__init__(name=name,
                         embeddings_model_name=embeddings_model_name,
                         llm_chain=llm_chain,
                         retrieval_strategy=retrieval_strategy,
                         reports=reports,
                         memory_enabled=memory_enabled,
                         k=k,
                         k_i=k_i,
                         *args,
                         **kwargs)
        self.question_answer_expected = question_answer_expected

    def update_qae(self, id, question, answer, expected=None, similarity_score=None, response_time=None):
        qae = QAE(question, answer, expected=expected, similarity_score=similarity_score, response_time=response_time)
        self.question_answer_expected[str(id)] = qae

    def set_qae(self, question_answer_expected={}):
        print(question_answer_expected)
        self.question_answer_expected = question_answer_expected

    def set_qae_from_dict(self, qae_dict):
        for id, qae in qae_dict.items():
            self.update_qae(id, **qae)

    @classmethod
    def from_dict(cls, cls_dict):
        ses = super().from_dict(cls_dict)
        ses.question_answer_expected = {}
        ses.set_qae_from_dict(cls_dict['question_answer_expected'])
        return ses

    def to_dict(self):
        ses_dict = super().to_dict()
        temp_qae_dict = {}
        for id, qae in self.question_answer_expected.items():
            temp_qae_dict[id] = qae.encode()
        ses_dict["question_answer_expected"] = temp_qae_dict
        return ses_dict
    
    def qae_to_dict_list(self):
        qae_list = []
        for id, qae in self.question_answer_expected.items():
            qae_list.append({
                "question": qae.question,
                "expected": qae.expected,
                "llm_answer": qae.answer,
                "similarity_score": qae.similarity_score,
            })
        return qae_list







""" QA
class that represents a question and an answer """
class QA:
    def __init__(self, question, answer, context=None, replays=0):
        self.question = question
        self.answer = answer
        self.context = context
        self.replays = replays

    def encode(self):
        return vars(self)

""" QAE
class that represents a question, an answer, and an expeected answer """
class QAE:
    def __init__(self, question, answer=None, expected=None, similarity_score=None, response_time=None):
        self.question = question
        self.answer = answer
        self.expected = expected
        self.similarity_score=similarity_score
        self.response_time=response_time

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