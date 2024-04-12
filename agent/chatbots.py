import streamlit as st

# Langchain imports
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter


from agent.subquery_generator import SubQueryGenerator
from agent.template_formatter import NoTemplateFormatter

from llm_agents.agent import Agent
from llm_agents.llm_wrap import LLM_Wrapper
from llm_agents.tools.python_repl import PythonREPLTool
from llm_agents.tools.retriever_simple import RetrieverTool

import GUI.misc as Gmisc

import math
import time
from abc import ABC, abstractmethod



class Chatbot(ABC):
    """
    Abstract base class representing a chatbot.

    Attributes:
        retriever_strategy: The strategy for retrieving relevant information.
        llm: The language model instance.
        template_formatter: Formatter for templates (default is NoTemplateFormatter).
        memory_limit (int): Maximum memory size in characters.
        progressive_memory (bool): Whether to remove oldest memory when exceeding the limit.
        memory_active (bool): Whether memory functionality is active.

    Methods:
        invoke(self, question):
            Abstract method to invoke the chatbot with a question.

        st_render(self, question, skip=0):
            Abstract method to render a short text response.

        render_st_with_score(self, question, expected, cross_encoder=None):
            Renders a short text response and computes similarity score (if cross_encoder provided).

        invoke_with_score(self, question, expected, cross_encoder=None):
            Invokes the chatbot with a question and computes similarity score (if cross_encoder provided).

        update_memory(self, question, answer):
            Updates the chatbot's memory with a new question-answer pair.
    """

    def __init__(self, retriever_strategy, llm, template_formatter=NoTemplateFormatter(), memory_limit=3000, progressive_memory=True, memory_active=False):
        """
        Initializes the Chatbot.

        Args:
            retriever_strategy: The strategy for retrieving relevant information.
            llm: The language model instance.
            template_formatter: Formatter for templates (default is NoTemplateFormatter).
            memory_limit (int): Maximum memory size in characters.
            progressive_memory (bool): Whether to remove oldest memory when exceeding the limit.
            memory_active (bool): Whether memory functionality is active.
        """
        self.retriever_strategy = retriever_strategy
        self.llm = llm
        self.template = None
        self.memory = []
        self.mem_length = 0
        self.memory_active = memory_active
        self.progressive_memory = progressive_memory
        self.memory_limit = memory_limit
        self.template_formatter = template_formatter


    @abstractmethod
    def invoke(self, question):
        """
        Abstract method to invoke the chatbot with a question.

        Args:
            question (str): The user's question.

        Returns:
            str: The chatbot's response.
        """
        pass

    @abstractmethod
    def st_render(self, question, skip=0):
        """
        Abstract method to render a short text response.

        Args:
            question (str): The user's question.
            skip (int, optional): Number of responses to skip (default is 0).

        Returns:
            str: The rendered short text response.
        """
        pass

    def render_st_with_score(self, question, expected, cross_encoder=None):
        """
        Renders a short text response and computes similarity score (if cross_encoder provided).

        Args:
            question (str): The user's question.
            expected (str): The expected answer.
            cross_encoder: Cross-encoder model instance (optional).

        Returns:
            tuple: A tuple containing the response, similarity score, and duration (in seconds).
        """
        response = self.st_render(question)
        score = None
        start_time = time.time_ns()
        if cross_encoder:
            score = cross_encoder.predict([expected, response])
        stop_time = time.time_ns()
        duration = stop_time - start_time
        duration_s = duration/1e9
        return response, score, duration_s

    def invoke_with_score(self, question, expected, cross_encoder=None):
        """
        Invokes the chatbot with a question and computes similarity score (if cross_encoder provided).

        Args:
            question (str): The user's question.
            expected (str): The expected answer.
            cross_encoder: Cross-encoder model instance (optional).

        Returns:
            tuple: A tuple containing the response, similarity score, and duration (in seconds).
        """
        response = self.invoke(question)
        score = None
        start_time = time.time_ns()
        if cross_encoder:
            score = cross_encoder.predict([expected, response])
        stop_time = time.time_ns()
        duration = stop_time - start_time
        duration_s = duration/1e9
        return response, score, duration_s

    def update_memory(self, question, answer):
        """
        Updates the chatbot's memory with a new question-answer pair.

        Args:
            question (str): The user's question.
            answer (str): The chatbot's answer.

        Returns:
            None
        """
        # remove any curly braces from question and answer
        question = question.replace("{", "").replace("}","")
        answer = answer.replace("{", "").replace("}","")

        curr_mem_length = len(question) + len(answer)

        # check if it is worthwhile to delete memory
        if  curr_mem_length > self.memory_limit:
            return

        # check length of memory, make sure it does not exceed memory_limit
        if self.mem_length + curr_mem_length > self.memory_limit:
            if self.progressive_memory:
                # remove oldest memory until enough is available
                mem_left_to_clear = curr_mem_length
                while mem_left_to_clear > 0 and len(self.memory) > 0:
                    mem_left_to_clear -= self.memory[0][2]
                    self.mem_length -= self.memory[0][2]
                    self.memory.pop()
            else:
                # full reset
                self.memory = []
                self.mem_length = 0

        self.memory.append((question, answer, curr_mem_length))
        self.mem_length += curr_mem_length



""" simple """
class SimpleChatbot(Chatbot):
    """
    A simple chatbot for answering financial investment advisor questions related to shareholder reports.

    Attributes:
        system_message (str): A system message describing the role of the chatbot.
        instruction (str): Instructions for answering questions using context.
        parser (StrOutputParser): A parser for extracting relevant information from the chatbot's responses.
        template (PromptTemplate): A template for formatting chatbot responses.
        retriever_strategy (RetrieverStrategy): A strategy for retrieving relevant context.
        llm (LanguageModel): A language model for generating responses.
        vectorstore (VectorStore): A vector store for context retrieval.
        input_variables (list): List of input variables (e.g., ["question", "context"]).
        memory (list): A list of previous questions and answers stored in memory.
        mem_length (int): Maximum length of the memory.

    Methods:
        __init__(self, retriever_strategy, llm, vectorstore, *args, **kwargs):
            Initializes the SimpleChatbot instance.
        update_chain(self):
            Updates the processing chain for chatbot responses.
        update_memory(self, question, answer):
            Updates the chatbot's memory with a new question and answer.
        invoke(self, question):
            Invokes the chatbot to answer a given question.
        stream(self, question):
            Streams the chatbot's response to a given question.
        st_render(self, question, skip=0):
            Renders the chatbot's response using the language model and retrieves context.

    Note:
        - The chatbot is designed to answer questions related to shareholder reports.
        - The chatbot uses context provided in the "context" input variable to generate responses.
        - The chatbot updates its memory with previous questions and answers.
        - The chatbot retrieves context using the retriever strategy and language model.
    """

    system_message = """
You are a financial investment advisor who answers questions
about shareholder reports. You will be given a context and will answer the question using that context.
                """.strip()
    instruction = """
Context: "{context}"
\n\n
Question: "{question}"

\n\n
Make sure to source where you got the information from. This source should \
include the company, year, the report type, the quarter if possible, and page \
number as reported at the start of the Excerpt. Do NOT provide any URLs (i.e., https://...). 
                """.strip()

    def __init__(self, retriever_strategy, llm, vectorstore, *args, **kwargs):
        """
        Initializes a SimpleChatbot instance.

        Args:
            retriever_strategy (RetrieverStrategy): A strategy for retrieving relevant context.
            llm (LanguageModel): A language model for generating responses.
            vectorstore (VectorStore): A vector store for context retrieval.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(retriever_strategy, llm, *args, **kwargs)
        self.parser = StrOutputParser()
        self.template = self.template_formatter.init_template(
            system_message= self.system_message,
            instruction=self.instruction
        )
        self.retriever_strategy.set_vectorstore (vectorstore)
        self.input_variables = ["question", "context"]

        self.update_chain()


        self.memory = []
        self.mem_length = 0


    def update_chain(self):
        """
        Updates the processing chain for chatbot responses.
        """
        self.prompt = PromptTemplate(template=self.template, input_variables=self.input_variables)

        self.chain = (
             { "context": (RunnablePassthrough()
              | RunnableLambda(self.retriever_strategy.retrieve_context)),
              "question": RunnablePassthrough()}
             | self.prompt
             | self.llm
             | self.parser
        )


    def update_memory(self, question, answer):
        """
        Updates the chatbot's memory with a new question and answer.

        Args:
            question (str): The new question.
            answer (str): The corresponding answer.
        """
        super().update_memory(question, answer)

        # update template
        self.template = self.formatter.init_template_from_memory(system_message=self.system_message,
            instruction=self.instruction,
            memory=self.memory)

        self.update_chain()


    def invoke(self, question):
        """
        Invokes the chatbot to answer a given question.

        Args:
            question (str): The input question.

        Returns:
            str: The chatbot's response.
        """
        response = self.chain.invoke(question)
        if self.memory_active:
            self.update_memory(question, response)
        return response

    def stream(self, question):
        """
        Streams the chatbot's response to a given question.

        Args:
            question (str): The input question.

        Returns:
            Stream: The chatbot's response stream.
        """
        return self.chain.stream(question)

    """ render llm st response """
    def st_render(self, question, skip=0):
        """
        Renders the chatbot's response using the language model and retrieves context.

        Args:
            question (str): The input question.
            skip (int): Number of context items to skip (default is 0).

        Returns:
            tuple: A tuple containing the chatbot's response and the retrieved context.
        """
        self.retriever_strategy.set_skip(skip)
        stream = self.stream(question)
        response = Gmisc.write_stream(stream)
        context = self.retriever_strategy.get_recent_context()
        self.retriever_strategy.clear_recent_context()
        return response, context


class AgentChatbot(Chatbot):
    """
    A chatbot designed for answering financial investment advisor questions related to shareholder reports.

    Attributes:
        system_message (str): A system message describing the role of the chatbot.
        instruction (str): Instructions for answering questions using context.
        parser (StrOutputParser): A parser for extracting relevant information from the chatbot's responses.
        template (PromptTemplate): A template for formatting chatbot responses.
        retriever_strategy (RetrieverStrategy): A strategy for retrieving relevant context.
        llm (LanguageModel): A language model for generating responses.
        vectorstore (VectorStore): A vector store for context retrieval.
        input_variables (list): List of input variables (e.g., ["question", "context"]).
        memory (list): A list of previous questions and answers stored in memory.
        mem_length (int): Maximum length of the memory.
        llm_wrapper (LLM_Wrapper): A wrapper for the language model.
        agent (Agent): An agent combining language model and tools for chatbot functionality.

    Methods:
        __init__(self, retriever_strategy, llm, vectorstore, *args, **kwargs):
            Initializes the AgentChatbot instance.
        update_chain(self):
            Updates the processing chain for chatbot responses.
        update_memory(self, question, answer):
            Updates the chatbot's memory with a new question and answer.
        invoke(self, question):
            Invokes the chatbot to answer a given question.
        stream(self, question):
            Streams the chatbot's response to a given question.
        st_render(self, question, skip=0):
            Renders the chatbot's response using the language model and retrieves context.

    Note:
        - The chatbot is specifically tailored for answering questions related to shareholder reports.
        - It uses context provided in the "context" input variable to generate responses.
        - The chatbot updates its memory with previous questions and answers.
        - Retrieval of context is performed using the retriever strategy and language model.
    """

    system_message = """
                You are a financial investment advisor who answers questions
                about shareholder reports. You will be given a context and will answer the question using that context.
                """
    instruction = """
                Context: "{context}"
               \n\n
                Question: "{question}"

                \n\n
                Make sure to source where you got the information from.
                This source should include the company, year, the report type, and page number.
                """


    def __init__(self, retriever_strategy, llm, vectorstore, *args, **kwargs):
        """
        Initializes an AgentChatbot instance.

        Args:
            retriever_strategy (RetrieverStrategy): A strategy for retrieving relevant context.
            llm (LanguageModel): A language model for generating responses.
            vectorstore (VectorStore): A vector store for context retrieval.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(retriever_strategy, llm, *args, **kwargs)
        self.parser = StrOutputParser()
        self.template = self.template_formatter.init_template(
            system_message= self.system_message,
            instruction=self.instruction
        )
        self.retriever_strategy.set_vectorstore (vectorstore)
        self.input_variables = ["question", "context"]

        self.update_chain()


        self.memory = []
        self.mem_length = 0



    def update_chain(self):
        """
        Updates the processing chain for chatbot responses.
        """
        self.prompt = PromptTemplate(template=self.template, input_variables=self.input_variables)

        self.chain = (
              { "context": (RunnablePassthrough()
               | RunnableLambda(self.retriever_strategy.retrieve_context)),
               "question": RunnablePassthrough()}
              | self.prompt
              | self.llm
              | self.parser
         )

        self.llm_wrapper = LLM_Wrapper(llm=self.llm)

        self.agent = Agent(llm=self.llm_wrapper, tools=[PythonREPLTool(), RetrieverTool(rag_chain=self.chain)])


    def update_memory(self, question, answer):
        """
        Updates the chatbot's memory with a new question and answer.

        Args:
            question (str): The new question.
            answer (str): The corresponding answer.
        """
        super().update_memory(question, answer)

         # update template
        self.template = self.formatter.init_template_from_memory(system_message=self.system_message,
             instruction=self.instruction,
             memory=self.memory)

        self.update_chain()


    def invoke(self, question):
        """
        Invokes the chatbot to answer a given question.

        Args:
            question (str): The input question.

        Returns:
            str: The chatbot's response.
        """
        response = self.agent.run(question)
        return response

    def stream(self, question):
        """
        Streams the chatbot's response to a given question.

        Args:
            question (str): The input question.

        Yields:
            str: The chatbot's response.
        """
        yield self.invoke(question)



    def st_render(self, question, skip=0):
        """
        Renders the chatbot's response using the language model and retrieves context.

        Args:
            question (str): The input question.
            skip (int): Number of context items to skip (default is 0).

        Returns:
            tuple: A tuple containing the chatbot's response and the retrieved context.
        """
        self.retriever_strategy.set_skip(skip)
        stream = self.stream(question)
        response = Gmisc.write_stream(stream)
        context = self.retriever_strategy.get_recent_context()
        self.retriever_strategy.clear_recent_context()
        return response, context




""" FusionChatbot """
class FusionChatbot(Chatbot):
    """
    A chatbot designed for answering financial investment advisor questions related to shareholder reports.

    Attributes:
        result_system_message (str): A system message describing the role of the chatbot.
        result_instruction (str): Instructions for answering questions using context.
        parser (StrOutputParser): A parser for extracting relevant information from the chatbot's responses.
        result_template (PromptTemplate): A template for formatting chatbot responses.
        retriever_strategy (RetrieverStrategy): A strategy for retrieving relevant context.
        llm (LanguageModel): A language model for generating responses.
        vectorstores (dict): A dictionary of vector stores for context retrieval.
        max_k (int): Maximum number of retrieved context items.
        input_variables (list): List of input variables (e.g., ["question", "context"]).
        memory (list): A list of previous questions and answers stored in memory.
        mem_length (int): Maximum length of the memory.
        sub_query_generator (SubQueryGenerator): A generator for sub queries.

    Methods:
        __init__(self, retriever_strategy, llm, vectorstores={}, max_k=8, *args, **kwargs):
            Initializes the FusionChatbot instance.
        update_chain(self):
            Updates the processing chain for chatbot responses.
        update_memory(self, question, answer):
            Updates the chatbot's memory with a new question and answer.
        invoke(self, question):
            Invokes the chatbot to answer a given question.
        stream_sub_query_response(self, question):
            Streams the chatbot's sub query response to a given question.
        get_context(self, sub_query_response):
            Retrieves context from the sub query response.
        stream_result_response(self, question, context):
            Streams the chatbot's result response using the language model and retrieved context.
        st_render(self, question, skip=0):
            Renders the chatbot's response using the language model and retrieves context.

    Note:
        - The chatbot is specifically tailored for answering questions related to shareholder reports.
        - It uses context provided in the "context" input variable to generate responses.
        - The chatbot updates its memory with previous questions and answers.
        - Retrieval of context is performed using the retriever strategy and language model.
    """

    result_system_message = """
You are a helpful assistant. Answer questions given the context. \
Make sure to source where you got information from (given in the context). \
This source should include the company, year, the report type, (quarter if \
possible) and page number. Do NOT provide any URLs (i.e., https://...). 
        """

    result_instruction = """
Given the context: '{context}'\n\n
{question}\n\n
        """

    def __init__(self, retriever_strategy, llm, vectorstores={}, max_k=8, *args, **kwargs):
        """
        Initializes a FusionChatbot instance.

        Args:
            retriever_strategy (RetrieverStrategy): A strategy for retrieving relevant context.
            llm (LanguageModel): A language model for generating responses.
            vectorstores (dict): A dictionary of vector stores for context retrieval.
            max_k (int): Maximum number of retrieved context items.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(retriever_strategy, llm, *args, **kwargs)
        self.vectorstores = vectorstores
        self.parser = StrOutputParser()
        self.max_k = max_k
        self.sub_query_generator = SubQueryGenerator(retriever_strategy=retriever_strategy, llm=llm, vectorstores=vectorstores, max_k=max_k, template_formatter=self.template_formatter)


        self.result_template = self.template_formatter.init_template(
            system_message=self.result_system_message,
            instruction=self.result_instruction
        )

        self.input_variables = ["question", "context"]

        self.update_chain()

        self.memory = []
        self.mem_length = 0

    def update_chain(self):
        """
        Updates the processing chain for chatbot responses.
        """
        self.result_prompt = PromptTemplate(template=self.result_template, input_variables=self.input_variables)

        self.result_chain = (
            { "context": itemgetter("context"),
             "question": itemgetter("question")
             }
            | self.result_prompt
            | self.llm
            | self.parser
        )



    def invoke(self, question):
        """
        Invokes the chatbot to answer a given question.

        Args:
            question (str): The input question.

        Returns:
            str: The chatbot's response.
        """
        sub_query_response = self.sub_query_generator.invoke(question)
        context = self.sub_query_generator.retriever_multicontext(sub_query_response)
        if context == None:
            print("Could not retrieve context. The sub query generations might have failed, please try asking your question again.")
            return "N/A"
        result_response = self.result_chain.invoke({"context": context, "question": question})
        if self.memory_active:
            self.update_memory(question, result_response)
        return result_response

    def stream_sub_query_response(self, question):
        """
        Streams the sub-query response for the given question.
        Args:
            question (str): The input question.
        Returns:
            str: The sub-query response.
        """
        return self.sub_query_generator.stream(question)

    def get_context(self, sub_query_response):
        """
        Retrieves the context based on the sub-query response.
        Args:
            sub_query_response (str): The sub-query response.
        Returns:
            str: The retrieved context.
        """
        context = self.sub_query_generator.retriever_multicontext(sub_query_response)
        return context

    def stream_result_response(self, question, context):
        """
        Streams the result response for the given question and context.
        Args:
            question (str): The input question.
            context (str): The retrieved context.
        Returns:
            str: The result response.
        """
        return self.result_chain.stream({"context": context, "question": question})

    """ render llm st response """
    def st_render(self, question, skip=0):
        """
        Renders the LLM's response for the given question.
        Args:
            question (str): The input question.
            skip (int, optional): The number of sub-queries to skip. Defaults to 0.
        Returns:
            Tuple[str, str]: A tuple containing the full LLM response and the recent context.
        """
        all_responses = []

        # generate sub queries
        sub_query_stream = self.stream_sub_query_response(question)

        sub_query_response = Gmisc.write_stream(sub_query_stream)
        all_responses.append(sub_query_response)

        # check context
        self.retriever_strategy.set_skip(skip)
        context = self.get_context(sub_query_response)

        if context == None:
            all_responses.append("Failed to retrieve context, we might not have been able to parse the LLM's sub queries, please try again.")
        else:
            final_stream = self.stream_result_response(question, context)
            final_stream_response = Gmisc.write_stream(final_stream)
            all_responses.append(final_stream_response)

        full_response = "\n\n".join(all_responses)
        context = self.retriever_strategy.get_recent_context()
        self.retriever_strategy.clear_recent_context()
        return full_response, context


""" StepbackChatbot """
class StepbackChatbot(Chatbot):
    """
    A chatbot class for answering questions based on context and sub-queries.

    Attributes:
        retriever_strategy: An instance of the retriever strategy.
        llm: An instance of the Language Learning Model (LLM).
        vectorstores: A dictionary of vector stores.
        max_k: Maximum number of retrievals to consider.

    Methods:
        invoke(question: str) -> str:
            Invokes the chatbot to answer the given question.
            Args:
                question (str): The input question.
            Returns:
                str: The chatbot's response.

        st_render(question: str, skip: int = 0) -> Tuple[str, str]:
            Renders the chatbot's response for the given question.
            Args:
                question (str): The input question.
                skip (int, optional): The number of sub-queries to skip. Defaults to 0.
            Returns:
                Tuple[str, str]: A tuple containing the full chatbot response and the recent context.
    """

    result_system_message = """
You are a helpful assistant. You will be given a list of questions and their \
answers. References to the source where those answers were taken are embedded \
within the answers themselves. Answer the original question given the list of \
questions and answers. \n\n \
Make sure to source where you got the information from. This source should \
include the company, year, the report type, the quarter if possible, and page \
number as reported in the answer. Do NOT provide any URLs (i.e., https://...). 
        """.strip()

    result_instruction = """
Previous Questions and Answers: [{context}] \
\n\n\
Question: "{question}"\
        """.strip()

    simple_system_message = """
You are a helpful assistant. You will be given a context and will answer the \
question using that context. Make sure to source where you got the \
information from. This source should include the company, year, the report \
type, the quarter if possible, and page number as reported at the start of the \
Excerpt. Do NOT provide any URLs (i.e., https://...). 
        """.strip()

    simple_instruction = """
        Context: "{context}" \
        \n\n\
        Question: "{question}"\
        """.strip()


    """
    clist = {
        "company": {
            "year": {
                "report type": {
                    "10K": <report>,
                    "10Q": {
                        "1": <report>,
                        "2": <report>,
                        "3": <report>,
                    }
                }
            }
        }
        ...
    }
    """
    def __init__(self, retriever_strategy, llm, vectorstores={}, max_k=8, *args, **kwargs):
        super().__init__(retriever_strategy, llm, *args, **kwargs)
        self.vectorstores = vectorstores
        self.parser = StrOutputParser()
        self.max_k = max_k
        self.sub_query_generator = SubQueryGenerator(retriever_strategy=retriever_strategy, llm=llm, vectorstores=vectorstores, max_k=max_k, template_formatter=self.template_formatter)


        self.result_template = self.template_formatter.init_template(
            system_message=self.result_system_message,
            instruction=self.result_instruction
        )

        self.simple_template = self.template_formatter.init_template(
            system_message=self.simple_system_message,
            instruction=self.simple_instruction
        )

        self.input_variables = ["question", "context"]

        self.update_chain()

        self.memory = []
        self.mem_length = 0

    def update_chain(self):
        self.simple_prompt = PromptTemplate(template=self.simple_template, input_variables=["question", "context"])
        self.result_prompt = PromptTemplate(template=self.result_template, input_variables=["question", "context"])

        self.simple_chain = (
             { "context": itemgetter("context"),
              "question": itemgetter("question")}
             | self.simple_prompt
             | self.llm
             | self.parser
            )

        self.answer_chain = (
             RunnableLambda(self.sub_query_generator.retrieve_context_question_dict)
             | self.simple_prompt
             | self.llm
             | self.parser
        )

        self.result_chain = (
            self.result_prompt
            | self.llm
            | self.parser
        )


    def invoke(self, question):
        """
        Invokes the chatbot to answer the given question.

        Args:
            question (str): The user's question.

        Returns:
            str: The chatbot's response.
        """
        # get the broken down question
        sub_query_response = self.sub_query_generator.invoke(question)

        # get the matches
        all_matches =  self.sub_query_generator.parse_questions(sub_query_response)

        # answer the broken down questions
        all_responses = self.answer_chain.batch(all_matches)

        temp_mem = [f"Q: {all_matches[x][-1]}\n\nA: {all_responses[x]}" for x in range(len(all_matches))]
        context = "\n\n".join(temp_mem)


        if temp_mem == None or len(temp_mem) == 0:
            print("Oops! something went wrong")
            return "N/A"

        response = self.result_chain.invoke({"question": question, "context": context})
        if self.memory_active:
            self.update_memory(question, response)
        return response

    def stream_sub_query_gen(self, question):
        """
        Streams sub-query generation.

        Args:
            question (str): The user's question.

        Yields:
            Generator: Sub-query responses.
        """
        return self.sub_query_generator.stream(question)

    def stream_sub_query_responses(self, sub_query_response):
        """
        Streams sub-query responses.

        Args:
            sub_query_response (str): The sub-query response.

        Returns:
            List[Tuple[Generator, str, str]]: List of sub-query streams, questions, and sources.
        """
        # get the matches
        all_matches =  self.sub_query_generator.parse_questions(sub_query_response)
        questions = [match[-1] for match in all_matches]
        sources = [" ".join(match[:-1]) for match in all_matches]

        # answer the broken down questions
        all_streams = []
        for match in all_matches:
            all_streams.append(self.answer_chain.stream(match))

        return list(zip(all_streams, questions, sources))

    def stream_final_response(self, question, sub_queries, responses):
        """
        Streams final response.

        Args:
            question (str): The user's original question.
            sub_queries (List[str]): List of sub-queries.
            responses (List[str]): List of sub-query responses.

        Returns:
            Generator: Final response stream.
        """
        if sub_queries == None or len(sub_queries) == 0:
            print("Oops! something went wrong")
            return "N/A"
        st.markdown(f"<b style='display: block;text-align: center;width: 100%;'> {'='*5} FINAL ANSWER {'='*5}</b>\n\n<i>QUESTION: {question}</i>\n\n", unsafe_allow_html=True)
        temp_mem = [f"Q: {sub_queries[x]}\n\nA: {responses[x]}" for x in range(len(sub_queries))]
        context = "\n\n".join(temp_mem)

        return self.result_chain.stream({"question": question, "context": context})

    """ Render stepback llm chain response """
    def st_render(self, question, skip=0):
        """
        Renders stepback LLN chain response.

        Args:
            question (str): The user's question.
            skip (int, optional): Number of skipped sub-queries. Defaults to 0.

        Returns:
            Tuple[str, str]: Full response and context.
        """
        all_responses = []

        # generate sub queries
        sub_query_gen_stream = self.stream_sub_query_gen(question)
        sub_query_response = Gmisc.write_stream(sub_query_gen_stream)

        all_responses.append(sub_query_response)

        # answer each sub query
        self.retriever_strategy.set_skip(skip)
        sub_query_responses_streams_question = self.stream_sub_query_responses(sub_query_response)

        sub_queries = [q for _, q, _ in sub_query_responses_streams_question]
        sub_query_answers = []

        for idx, (stream, sub_query, source) in enumerate(sub_query_responses_streams_question):
            st.markdown(f"<b>{idx + 1}. {source} - {sub_query}:</b>", unsafe_allow_html=True)
            sub_query_answer = Gmisc.write_stream(stream)
            sub_query_answers.append(f"{sub_query_answer}")
            all_responses.append(f"<b>{idx + 1}. {source} - {sub_query}</b>\n\n{sub_query_answer}")

        # get final result
        if len(sub_queries) == 1:
            context = self.retriever_strategy.get_recent_context()
            self.retriever_strategy.clear_recent_context()
            # only one sub query, treat is as the final result
            return "\n\n".join(all_responses), context
        final_stream = self.stream_final_response(question, sub_queries, sub_query_answers)
        final_response = Gmisc.write_stream(final_stream)
        all_responses.append(f"<b style='display: block;text-align: center;width: 100%;'>===== FINAL ANSWER =====</b>\n\n<i>QUESTION: {question}</i>")
        all_responses.append(final_response)

        full_response = "\n\n".join(all_responses)
        context = self.retriever_strategy.get_recent_context()
        self.retriever_strategy.clear_recent_context()
        return full_response, context





""" Simple stepback """
class SimpleStepbackChatbot(Chatbot):
    """
    A simple financial investment advisor chatbot that answers questions about shareholder reports.

    Attributes:
        simple_system_message (str): A system message describing the role of the chatbot.
        simple_instruction (str): Instructions for the chatbot to answer questions based on context.

    Args:
        retriever_strategy (RetrieverStrategy): The strategy for retrieving context.
        llm (LanguageModel): The language model for generating responses.
        vectorstores (dict): A dictionary of vector stores containing financial data.
        max_k (int): The maximum number of retrieved contexts to consider.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Methods:
        __init__(self, retriever_strategy, llm, vectorstores={}, max_k=8, *args, **kwargs):
            Initializes the SimpleStepbackChatbot instance.
        get_multi_context(self, question):
            Retrieves multiple contexts based on the question.
        update_chain(self):
            Updates the processing chain for the chatbot.
        invoke(self, question):
            Invokes the chatbot to answer a question.
        stream(self, question):
            Streams the chatbot's response.
        st_render(self, question, skip=0):
            Renders the chatbot's response with skip option.
    """

    simple_system_message = """
You are a financial investment advisor who answers questions
about shareholder reports. You will be given a context and will answer the question using that context.
                """.strip()
    simple_instruction = """
Given the context: "{context}"
\n\n
{question}

\n\n
Make sure to source where you got the information from. This source should \
include the company, year, the report type, the quarter if possible, and page \
number as reported at the start of the Excerpt. Do NOT provide any URLs (i.e., https://...). 
                """.strip()

    """
    clist = {
        "company": {
            "year": {
                "report type": {
                    "10K": <report>,
                    "10Q": {
                        "1": <report>,
                        "2": <report>,
                        "3": <report>,
                    }
                }
            }
        }
        ...
    }
    """
    def __init__(self, retriever_strategy, llm, vectorstores={}, max_k=8, *args, **kwargs):
        """
        Initializes the SimpleStepbackChatbot instance.

        Args:
            retriever_strategy (RetrieverStrategy): The strategy for retrieving context.
            llm (LanguageModel): The language model for generating responses.
            vectorstores (dict): A dictionary of vector stores containing financial data.
            max_k (int): The maximum number of retrieved contexts to consider.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(retriever_strategy, llm, *args, **kwargs)
        self.vectorstores = vectorstores
        self.parser = StrOutputParser()
        self.max_k = max_k
        self.tools = []
        self.companies_years = []
        self.num_stores = 0
        for company, company_data in vectorstores.items():
            for year, year_data in company_data.items():
                for report_type, report_type_data in year_data.items():
                    if report_type == "10K":
                        self.tools.append(f"{company}_{year}_{report_type}: Useful for finding information about {company}'s {report_type} in {year}.")
                        self.companies_years.append(f"{company}'s {report_type} on the year {year}")
                        self.num_stores += 1
                    elif report_type == "10Q":
                        for quarter, quarter_data in report_type_data.items():
                            self.tools.append(f"{company}_{year}_{report_type}_{quarter}: Useful for finding information about {company}'s {report_type} for the quarter {quarter} in {year}.")
                            self.companies_years.append(f"{company}'s {report_type} in the quarter {quarter} in the year {year}")
                            self.num_stores += 1
        self.k = math.ceil(max_k/self.num_stores)

        self.simple_template = self.template_formatter.init_template(
            system_message=self.simple_system_message,
            instruction=self.simple_instruction
        )

        self.input_variables = ["question", "context"]

        self.update_chain()

        self.memory = []
        self.mem_length = 0


    def get_multi_context(self, question):
        """
        Retrieves multiple contexts based on the question.

        Args:
            question (str): The user's question.

        Returns:
            str: A concatenated string of retrieved contexts.
        """
        context_list = []
        k_left = self.max_k
        for company, company_data in self.vectorstores.items():
            k = self.k
            if k_left <= 0: # cant do anything more...
                break
            if k_left <= k:
                k = k_left

            for year, year_data in company_data.items():
                for report_type, report_type_data in year_data.items():
                    if report_type == "10K":
                        context = self.retriever_strategy.retrieve_context(question, vectorstore=report_type_data, k=k)
                        context_list.append(context)
                    elif report_type == "10Q":
                        for quarter, quarter_data in report_type_data.items():
                            context = self.retriever_strategy.retrieve_context(question, vectorstore=quarter_data, k=k)
                            context_list.append(context)
        return "\n\n".join(context_list)


    def update_chain(self):
        """
        Updates the processing chain for the chatbot.
        """
        self.simple_prompt = PromptTemplate(template=self.simple_template, input_variables=["question", "context"])


        self.simple_chain = (
             { "context": RunnableLambda(self.get_multi_context),
              "question": RunnablePassthrough()}
             | self.simple_prompt
             | self.llm
             | self.parser
            )


    def invoke(self, question):
        """
        Invokes the chatbot to answer a question.

        Args:
            question (str): The user's question.

        Returns:
            str: The chatbot's response.
        """
        response = self.simple_chain.invoke(question)
        if self.memory_active:
            self.update_memory(question, response)
        return response

    def stream(self, question):
        """
        Streams the chatbot's response.

        Args:
            question (str): The user's question.

        Returns:
            generator: A generator of response tokens.
        """
        return self.simple_chain.stream(question)

    """ render llm st response """
    def st_render(self, question, skip=0):
        """
        Renders the chatbot's response with skip option.

        Args:
            question (str): The user's question.
            skip (int): The number of retrieved contexts to skip.

        Returns:
            tuple: A tuple containing the response (str) and context (str).
        """
        self.retriever_strategy.set_skip(skip)
        stream = self.stream(question)
        response = Gmisc.write_stream(stream)
        context = self.retriever_strategy.get_recent_context()
        self.retriever_strategy.clear_recent_context()
        return response, context
