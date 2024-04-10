from agent.chatbots import FusionChatbot, SimpleChatbot, SimpleStepbackChatbot, StepbackChatbot, AgentChatbot
from agent.retriever_strategies import CompositeRetrieverStrategy, ReRankerRetrieverStrategy, SimpleRetrieverStrategy, StochasticRetrieverStrategy
from agent.template_formatter import LlamaTemplateFormatter

import os
import datetime


""" Session
holds the state of the session, which represents
a user-LLM interaction """
class Session:
    """
    Represents a session for Q&A with a Language Model (LLM).

    Args:
        name (str): Optional name for the session. If not provided, a timestamp-based name is generated.
        embeddings_model_name (str): Name of the embeddings model.
        llm_chain (str): Type of LLM chain (e.g., "Merged Chain", "ReAct Chain").
        retrieval_strategy (str): Retrieval strategy (e.g., "Simple Retrieval Strategy", "Reranker Retrieval Strategy").
        reports (list): List of report objects.
        memory_enabled (bool): Whether memory is enabled.
        k (int): Parameter for retriever strategies.
        k_i (int): Parameter for retriever strategies.
        *args, **kwargs: Additional arguments.

    Attributes:
        reports (list): List of report objects.
        llm_chain (str): Type of LLM chain.
        retrieval_strategy (str): Retrieval strategy.
        name (str): Session name.
        embeddings_model_name (str): Name of the embeddings model.
        k (int): Parameter for retriever strategies.
        k_i (int): Parameter for retriever strategies.
        memory_enabled (bool): Whether memory is enabled.
        vectorstores (dict): Dictionary of vector stores.
        initialized (bool): Indicates if the session is initialized.
    """

    valid_retrieval_strategies = ["Simple Retrieval Strategy", "Reranker Retrieval Strategy", "Random Retrieval Strategy"]
    valid_llm_chains = ["Merged Chain", "Split Chain", "Multi-Query Split Chain", "Multi-Query Split Step-Back Chain", "ReAct Chain"]

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

    def initialize(self, index_generator, file_manager, llm, embeddings, cross_encoder=None, load=True, isllama=False):
        """
        Initializes the session for Q&A with the LLM.

        Args:
            index_generator: Index generator object.
            file_manager: File manager object.
            llm: Language model object.
            embeddings: Embeddings object.
            cross_encoder: Cross-encoder object (optional).
            load (bool): Whether to load existing vector stores.
            isllama (bool): Whether the session is using Llama template formatting.
        """

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
        elif self.llm_chain == "Multi-Query Split Step-Back Chain":
            self.init_stepback_chain(llm, isllama=isllama)
        elif self.llm_chain == "Split Chain":
            self.init_simple_stepback_chain(llm, isllama=isllama)
        else:
            raise Exception("No Chatbot initialized")

        self.initialized = True

    def deinitialize(self):
        """ Deinitializes the session. """
        self.initialized = False
        self.retriever_strategy_obj = None
        self.chatbot = None
        self.vectorstores = None

    def init_simple_retriever(self):
        """
        Initializes the simple retriever strategy.

        Returns: 
            None
        """
        print("initializing simple retriever strategy")
        simple_retriever = SimpleRetrieverStrategy(k=self.k)
        self.retriever_strategy_obj = CompositeRetrieverStrategy([simple_retriever], ["company", "year", "report type", "quarter", "page"])

    """ Init reranker retriever strategy """
    def init_reranker_retriever(self, cross_encoder=None):
        """
        Initializes the reranker retriever strategy.
    
        Args:
            cross_encoder (optional): A cross encoder model for reranking. Defaults to None.
        
        Raises:
            Exception: If cross_encoder is not provided.
        
        Returns: 
            None
        """
        print("initializing Reranker retriever strategy")
        if cross_encoder == None:
            raise Exception("Tried to initailize Reranker strategy without a cross encoder")
        reranker_retriever = ReRankerRetrieverStrategy(cross_encoder=cross_encoder, k=self.k, init_k=self.k_i)
        self.retriever_strategy_obj = CompositeRetrieverStrategy([reranker_retriever], ["company", "year", "report type", "quarter", "page"])

    def init_random_retriever(self):
        """
        Initializes the random retriever strategy.

        Returns:
            None
        """
        print("initializing Reranker retriever strategy")
        stochastic_retriever = StochasticRetrieverStrategy(k=self.k, fetch_k=self.k_i)
        self.retriever_strategy_obj = CompositeRetrieverStrategy([stochastic_retriever], ["company", "year", "report type", "quarter", "page"])

    def init_simple_chain(self, index_generator, llm, isllama=False):
        """
        Initializes the simple chatbot chain.

        Args:
            index_generator: An index generator.
            llm: A language model.
            isllama (optional): Whether the chatbot is using Llama template formatter. Defaults to False.

        Returns:
            None
        """
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

    def init_agent_chain(self, index_generator, llm, isllama=False):
        """
        Initializes the react chatbot chain.

        Args:
            index_generator: An index generator.
            llm: A language model.
            isllama (optional): Whether the chatbot is using Llama template formatter. Defaults to False.

        Returns:
            None
        """
        print("initializing react chatbot")
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

    def init_fusion_chain(self, llm,  isllama=False):
        """
        Initializes the fusion chatbot chain.

        Args:
            llm: A language model.
            isllama (optional): Whether the chatbot is using Llama template formatter. Defaults to False.

        Returns:
            None
        """
        print("initializing fusion chatbot")
        if isllama:
            self.chatbot = FusionChatbot(self.retriever_strategy_obj, llm, vectorstores=self.vectorstores, max_k=self.k, template_formatter=LlamaTemplateFormatter())
        else:
            self.chatbot = FusionChatbot(self.retriever_strategy_obj, llm, vectorstores=self.vectorstores, max_k=self.k)

    def init_stepback_chain(self, llm,  isllama=False):
        """
        Initializes the stepback chatbot chain.

        Args:
            llm: A language model.
            isllama (optional): Whether the chatbot is using Llama template formatter. Defaults to False.

        Returns:
            None
        """
        print("initializing stepback chatbot")
        if isllama:
            self.chatbot = StepbackChatbot(self.retriever_strategy_obj, llm, vectorstores=self.vectorstores, max_k=self.k, template_formatter=LlamaTemplateFormatter())
        else:
            self.chatbot = StepbackChatbot(self.retriever_strategy_obj, llm, vectorstores=self.vectorstores, max_k=self.k)

    def init_simple_stepback_chain(self, llm,  isllama=False):
        """
        Initializes the simple stepback chatbot chain.

        Args:
            llm: A language model.
            isllama (optional): Whether the chatbot is using Llama template formatter. Defaults to False.

        Returns:
            None
        """
        print("initializing simple stepback chatbot")
        if isllama:
            self.chatbot = SimpleStepbackChatbot(self.retriever_strategy_obj, llm, vectorstores=self.vectorstores, max_k=self.k, template_formatter=LlamaTemplateFormatter())
        else:
            self.chatbot = SimpleStepbackChatbot(self.retriever_strategy_obj, llm, vectorstores=self.vectorstores, max_k=self.k)

    def add_report(self, report):
        """
        Add report to the session

        Args: 
            report (Report): the report

        Returns: 
            None
        """
        self.reports.append(report)

    def set_reports(self, reports):
        """
        Add reports to the session

        Args: 
            report (List[Report]): the reports

        Returns: 
            None
        """
        self.reports = reports

    def add_reports_dict(self, report_dict_list):
        """
        Add reports to the session from a dictionary

        Args: 
            report (List[dict[Report]]): the reports

        Returns: 
            None
        """
        for report_dict in report_dict_list:
            report = Report(**report_dict)
            self.add_report(report)

    def generate_vectorstore(self, index_generator, embeddings, report):
        """
        Generates a vector store for the given report using the provided index generator and embeddings.

        Args:
            index_generator (IndexGenerator): An instance of the IndexGenerator class.
            embeddings (list): List of embeddings.
            report (Report): The report for which the vector store is generated.

        Returns:
            dict: The generated vector store.
        """
        vectorstore = index_generator.generate_vector_store_pdf_file(embeddings, report.file_path, report.company, report.year, report.report_type, report.quarter)
        if report.save:
            index_path = os.path.join(os.path.dirname(report.file_path), "index")
            index_generator.save_vector_store(vectorstore, index_path)
        return vectorstore


    def load_vectorstore(self, file_manager, index_generator, embeddings, report):
        """
        Loads a vector store for the given report using the provided file manager, index generator, and embeddings.

        Args:
            file_manager (FileManager): An instance of the FileManager class.
            index_generator (IndexGenerator): An instance of the IndexGenerator class.
            embeddings (list): List of embeddings.
            report (Report): The report for which the vector store is loaded.

        Returns:
            dict or None: The loaded vector store or None if it does not exist.
        """
        if file_manager.index_exists(report.company, report.year, report.report_type, report.quarter) == False:
            return None
        index_path = file_manager.create_index(report.company, report.year, report.report_type, report.quarter)
        if index_path:
            print("index path: ", index_path)
            return index_generator.load_vector_store(index_path, embeddings)
        else:
            return None

    def populate_vectorstore(self, file_manager, index_generator, embeddings, reports, load=True):
        """
        Populates the vector stores for the given reports using the provided file manager, index generator, and embeddings.

        Args:
            file_manager (FileManager): An instance of the FileManager class.
            index_generator (IndexGenerator): An instance of the IndexGenerator class.
            embeddings (list): List of embeddings.
            reports (list): List of Report instances.
            load (bool, optional): Whether to load existing vector stores. Defaults to True.
        """
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
        """
        Converts a nested dictionary to a flat list.

        Args:
            d (dict): The nested dictionary.
            parent_keys (list, optional): List of parent keys. Defaults to [].

        Returns:
            list: The flattened list.
        """
        lst = []
        for k, v in d.items():
            new_keys = parent_keys + [k]
            if isinstance(v, dict):
                lst.extend(self.dict_to_list(v, new_keys))
            else:
                lst.append(v)
        return lst


    def gen_vectorstore_flat(self):
        """
        Generates a flat list representation of the vector stores.

        Returns:
            list: The flat list representation.
        """
        return self.dict_to_list(self.vectorstores)



    def encode(self):
        """
        Encodes the instance as a dictionary.

        Returns:
            dict: The encoded dictionary.
        """
        return vars(self)

    @classmethod
    def from_dict(cls, cls_dict):
        """
        Creates an instance of the class from a dictionary.

        Args:
            cls_dict (dict): The dictionary containing class attributes.

        Returns:
            cls: An instance of the class.
        """
        ses = cls(**cls_dict)
        ses.reports = []
        ses.add_reports_dict(cls_dict['reports'])
        return ses

    def to_dict(self):
        """
        Converts the ChatbotInitializer object to a dictionary.

        Returns:
            dict: A dictionary representation of the ChatbotInitializer object.
        """
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
    """
    Represents a chat session with conversation history.

    Args:
        name (str, optional): Name of the chat session. Defaults to None.
        embeddings_model_name (str, optional): Name of the embeddings model. Defaults to None.
        llm_chain (LanguageModelChain, optional): Language model chain. Defaults to None.
        retrieval_strategy (str, optional): Retrieval strategy. Defaults to None.
        conversation_history (list, optional): List of QA objects representing conversation history. Defaults to None.
        reports (list, optional): List of report objects. Defaults to [].
        memory_enabled (bool, optional): Whether memory is enabled. Defaults to False.
        k (int, optional): Value of k. Defaults to None.
        k_i (int, optional): Value of k_i. Defaults to None.

    Attributes:
        conversation_history (list): List of QA objects representing conversation history.

    Methods:
        add_to_conversation(question=None, answer=None, replays=0, **kwargs):
            Adds a QA object to the conversation history.
        add_conversation_history(conversation_history_dict_list):
            Adds conversation history from a list of dictionaries.
        from_dict(cls, cls_dict):
            Creates a ChatSession object from a dictionary.
        to_dict():
            Converts the ChatSession object to a dictionary.
    """
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
        """
        Initializes a ChatSession object.

        Args:
            name (str, optional): Name of the chat session. Defaults to None.
            embeddings_model_name (str, optional): Name of the embeddings model. Defaults to None.
            llm_chain (LanguageModelChain, optional): Language model chain. Defaults to None.
            retrieval_strategy (str, optional): Retrieval strategy. Defaults to None.
            conversation_history (list, optional): List of QA objects representing conversation history. Defaults to None.
            reports (list, optional): List of report objects. Defaults to [].
            memory_enabled (bool, optional): Whether memory is enabled. Defaults to False.
            k (int, optional): Value of k. Defaults to None.
            k_i (int, optional): Value of k_i. Defaults to None.
        """
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
        """
        Adds a QA object to the conversation history.

        Args:
            question (str, optional): Question text. Defaults to None.
            answer (str, optional): Answer text. Defaults to None.
            replays (int, optional): Number of replays. Defaults to 0.
            **kwargs: Additional keyword arguments.

        Returns:
            None
        """
        qa = QA(question, answer, replays=replays, **kwargs)
        if not isinstance(self.conversation_history, list):
            self.conversation_history = []
        self.conversation_history.append(qa)

    def add_conversation_history(self, conversation_history_dict_list):
        """
        Adds conversation history from a list of dictionaries.

        Args:
            conversation_history_dict_list (list): List of dictionaries representing QA objects.

        Returns:
            None
        """
        for conversation_history_dict in conversation_history_dict_list:
            self.add_to_conversation(**conversation_history_dict)

    @classmethod
    def from_dict(cls, cls_dict):
        """
        Creates a ChatSession object from a dictionary.

        Args:
            cls_dict (dict): Dictionary representing a ChatSession object.

        Returns:
            ChatSession: A ChatSession object.
        """
        ses = super().from_dict(cls_dict)
        ses.conversation_history = []
        ses.add_conversation_history(cls_dict['conversation_history'])
        return ses

    def to_dict(self):
        """
        Converts the ChatSession object to a dictionary.

        Returns:
            dict: Dictionary representing the ChatSession object.
        """
        ses_dict = super().to_dict()
        ses_dict["conversation_history"] = [qa.encode() for qa in self.conversation_history]
        return ses_dict



""" Benchmark Session
Holds the state of a benchmark session, which represents
an LLM evaluation interaction. it inherits from Session, but instead
of using conversation history it uses Question-Answer-Expected list """
class BenchmarkSession(Session):
    """
    Represents a benchmark session with expected question-answer pairs.

    Args:
        name (str, optional): Name of the benchmark session. Defaults to None.
        embeddings_model_name (str, optional): Name of the embeddings model. Defaults to None.
        llm_chain (LanguageModelChain, optional): Language model chain. Defaults to None.
        retrieval_strategy (str, optional): Retrieval strategy. Defaults to None.
        question_answer_expected (dict, optional): Dictionary mapping question IDs to expected answers. Defaults to {}.
        reports (list, optional): List of report objects. Defaults to [].
        memory_enabled (bool, optional): Whether memory is enabled. Defaults to False.
        k (int, optional): Value of k. Defaults to None.
        k_i (int, optional): Value of k_i. Defaults to None.

    Attributes:
        question_answer_expected (dict): Dictionary mapping question IDs to expected answers.

    Methods:
        update_qae(id, question, answer, expected=None, similarity_score=None, response_time=None):
            Updates the question-answer-expected dictionary.
        set_qae(question_answer_expected={}):
            Sets the question-answer-expected dictionary.
        set_qae_from_dict(qae_dict):
            Sets question-answer-expected from a dictionary.
        from_dict(cls, cls_dict):
            Creates a BenchmarkSession object from a dictionary.
        to_dict():
            Converts the BenchmarkSession object to a dictionary.
        qae_to_dict_list():
            Converts the question-answer-expected dictionary to a list of dictionaries.
    """
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
        """
        Initializes a BenchmarkSession object.

        Args:
            name (str, optional): Name of the benchmark session. Defaults to None.
            embeddings_model_name (str, optional): Name of the embeddings model. Defaults to None.
            llm_chain (LanguageModelChain, optional): Language model chain. Defaults to None.
            retrieval_strategy (str, optional): Retrieval strategy. Defaults to None.
            question_answer_expected (dict, optional): Dictionary mapping question IDs to expected answers. Defaults to {}.
            reports (list, optional): List of report objects. Defaults to [].
            memory_enabled (bool, optional): Whether memory is enabled. Defaults to False.
            k (int, optional): Value of k. Defaults to None.
            k_i (int, optional): Value of k_i. Defaults to None.
        """
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
        """
        Updates the question-answer-expected dictionary.

        Args:
            id: Identifier for the question-answer pair.
            question (str): The question text.
            answer (str): The actual answer.
            expected (str, optional): The expected answer. Defaults to None.
            similarity_score (float, optional): The similarity score. Defaults to None.
            response_time (float, optional): The response time. Defaults to None.

        Returns:
            None
        """
        qae = QAE(question, answer, expected=expected, similarity_score=similarity_score, response_time=response_time)
        self.question_answer_expected[str(id)] = qae

    def set_qae(self, question_answer_expected={}):
        """
        Sets the question-answer-expected dictionary.

        Args:
            question_answer_expected (dict, optional): Dictionary mapping question IDs to expected answers. Defaults to {}.

        Returns:
            None
        """
        print(question_answer_expected)
        self.question_answer_expected = question_answer_expected

    def set_qae_from_dict(self, qae_dict):
        """
        Sets question-answer-expected from a dictionary.

        Args:
            qae_dict (dict): Dictionary representing question-answer-expected pairs.

        Returns:
            None
        """
        for id, qae in qae_dict.items():
            self.update_qae(id, **qae)

    @classmethod
    def from_dict(cls, cls_dict):
        """
        Creates a BenchmarkSession object from a dictionary.

        Args:
            cls_dict (dict): Dictionary representing a BenchmarkSession object.

        Returns:
            BenchmarkSession: A BenchmarkSession object.
        """
        ses = super().from_dict(cls_dict)
        ses.question_answer_expected = {}
        ses.set_qae_from_dict(cls_dict['question_answer_expected'])
        return ses

    def to_dict(self):
        """
        Converts the object to a dictionary representation.

        Returns:
            dict: A dictionary containing the serialized attributes of the object.
        """
        ses_dict = super().to_dict()
        temp_qae_dict = {}
        for id, qae in self.question_answer_expected.items():
            temp_qae_dict[id] = qae.encode()
        ses_dict["question_answer_expected"] = temp_qae_dict
        return ses_dict
    
    def qae_to_dict_list(self):
        """
        Converts the question-answer-expected (QAE) dictionary to a list of dictionaries.

        Returns:
            list: A list of dictionaries, each containing the QAE information.
                Each dictionary has the following keys:
                - "question": The question text.
                - "expected": The expected answer.
                - "llm_answer": The answer generated by a language model.
                - "similarity_score": The similarity score between the expected answer and the LLM answer.
        """
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
    """
    Represents a question and an answer.

    Attributes:
        question (str): The question text.
        answer (str): The answer text.
        context (str, optional): Additional context for the question-answer pair.
        replays (int, optional): Number of times the question has been replayed.
    """
    def __init__(self, question, answer, context=None, replays=0):
        self.question = question
        self.answer = answer
        self.context = context
        self.replays = replays

    def encode(self):
        """
        Encodes the QA object as a dictionary.

        Returns:
            dict: A dictionary containing the serialized attributes of the QA object.
        """
        return vars(self)

""" QAE
class that represents a question, an answer, and an expeected answer """
class QAE:
    """
    Represents a question, an answer, and an expected answer.

    Attributes:
        question (str): The question text.
        answer (str, optional): The answer generated by a language model.
        expected (str, optional): The expected answer.
        similarity_score (float, optional): The similarity score between the expected answer and the LLM answer.
        response_time (float, optional): The response time for generating the answer.
    """
    def __init__(self, question, answer=None, expected=None, similarity_score=None, response_time=None):
        self.question = question
        self.answer = answer
        self.expected = expected
        self.similarity_score=similarity_score
        self.response_time=response_time

    def encode(self):
        """
        Encodes the QAE object as a dictionary.

        Returns:
            dict: A dictionary containing the serialized attributes of the QAE object.
        """
        return vars(self)


""" Report
class that represents a company's shareholder report, metadata ONLY """
class Report:
    """
    Represents a company's shareholder report (metadata only).

    Attributes:
        company (str): The company name.
        year (int): The year of the report.
        report_type (str): The type of report (e.g., "10Q", "10K").
        quarter (int, optional): The quarter associated with the report (required for "10Q" reports).
        file_path (str, optional): The file path to the report.
        save (bool, optional): Whether to save the report.
    """
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
        """
        Encodes the Report object as a dictionary.

        Returns:
            dict: A dictionary containing the serialized attributes of the Report object.
        """
        return vars(self)
