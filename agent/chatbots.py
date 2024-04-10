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

    def __init__(self, retriever_strategy, llm, template_formatter=NoTemplateFormatter(), memory_limit=3000, progressive_memory=True, memory_active=False):
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
        pass

    @abstractmethod
    def st_render(self, question, skip=0):
        pass

    def render_st_with_score(self, question, expected, cross_encoder=None):
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
number as reported at the start of the Excerpt. Do NOT provide any URLs (i.e., https://...). If the context is empty, say 'No context was given'.
                """.strip()

    def __init__(self, retriever_strategy, llm, vectorstore, *args, **kwargs):
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
        super().update_memory(question, answer)

        # update template
        self.template = self.formatter.init_template_from_memory(system_message=self.system_message,
            instruction=self.instruction,
            memory=self.memory)

        self.update_chain()


    def invoke(self, question):
        response = self.chain.invoke(question)
        if self.memory_active:
            self.update_memory(question, response)
        return response

    def stream(self, question):
        return self.chain.stream(question)

    """ render llm st response """
    def st_render(self, question, skip=0):
        self.retriever_strategy.set_skip(skip)
        stream = self.stream(question)
        response = Gmisc.write_stream(stream)
        context = self.retriever_strategy.get_recent_context()
        self.retriever_strategy.clear_recent_context()
        return response, context


""" simple """
class AgentChatbot(Chatbot):
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
        super().update_memory(question, answer)

         # update template
        self.template = self.formatter.init_template_from_memory(system_message=self.system_message,
             instruction=self.instruction,
             memory=self.memory)

        self.update_chain()


    def invoke(self, question):
        response = self.agent.run(question)
        return response

    def stream(self, question):
        yield self.invoke(question)



    def st_render(self, question, skip=0):
        self.retriever_strategy.set_skip(skip)
        stream = self.stream(question)
        response = Gmisc.write_stream(stream)
        context = self.retriever_strategy.get_recent_context()
        self.retriever_strategy.clear_recent_context()
        return response, context




""" FusionChatbot """
class FusionChatbot(Chatbot):

    result_system_message = """
You are a helpful assistant. Answer questions given the context. \
Make sure to source where you got information from (given in the context). \
This source should include the company, year, the report type, (quarter if \
possible) and page number. Do NOT provide any URLs (i.e., https://...). If the context is empty, say 'No context was given'.
        """

    result_instruction = """
Given the context: '{context}'\n\n
{question}\n\n
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

        self.input_variables = ["question", "context"]

        self.update_chain()

        self.memory = []
        self.mem_length = 0

    def update_chain(self):
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
        return self.sub_query_generator.stream(question)

    def get_context(self, sub_query_response):
        context = self.sub_query_generator.retriever_multicontext(sub_query_response)
        return context

    def stream_result_response(self, question, context):
        return self.result_chain.stream({"context": context, "question": question})

    """ render llm st response """
    def st_render(self, question, skip=0):
        all_responses = []

        # generate sub queries
        sub_query_stream = self.stream_sub_query_response(question)

        sub_query_response = Gmisc.write_stream(sub_query_stream)
        all_responses.append(sub_query_response)

        # check context
        self.retriever_strategy.set_skip(skip)
        context = self.get_context(sub_query_response)

        if context == None:
            all_responses.append("Failed to retrieve context, we might not have been able to parse the LLM's sub queries, pleast try again.")
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

    result_system_message = """
You are a helpful assistant. You will be given a list of questions and their \
answers. References to the source where those answers were taken are embedded \
within the answers themselves. Answer the original question given the list of \
questions and answers. \n\n \
Make sure to source where you got the information from. This source should \
include the company, year, the report type, the quarter if possible, and page \
number as reported in the answer. Do NOT provide any URLs (i.e., https://...). If there are no previous questions and answers, say 'No context was given'.
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
Excerpt. Do NOT provide any URLs (i.e., https://...). If the context is empty, say 'No context was given'.
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
        return self.sub_query_generator.stream(question)

    def stream_sub_query_responses(self, sub_query_response):
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
        if sub_queries == None or len(sub_queries) == 0:
            print("Oops! something went wrong")
            return "N/A"
        st.markdown(f"<b style='display: block;text-align: center;width: 100%;'> {'='*5} FINAL ANSWER {'='*5}</b>\n\n<i>QUESTION: {question}</i>\n\n", unsafe_allow_html=True)
        temp_mem = [f"Q: {sub_queries[x]}\n\nA: {responses[x]}" for x in range(len(sub_queries))]
        context = "\n\n".join(temp_mem)

        return self.result_chain.stream({"question": question, "context": context})

    """ Render stepback llm chain response """
    def st_render(self, question, skip=0):
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
number as reported at the start of the Excerpt. Do NOT provide any URLs (i.e., https://...). If the context is empty, say 'No context was given'.
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
        self.simple_prompt = PromptTemplate(template=self.simple_template, input_variables=["question", "context"])


        self.simple_chain = (
             { "context": RunnableLambda(self.get_multi_context),
              "question": RunnablePassthrough()}
             | self.simple_prompt
             | self.llm
             | self.parser
            )


    def invoke(self, question):
        response = self.simple_chain.invoke(question)
        if self.memory_active:
            self.update_memory(question, response)
        return response

    def stream(self, question):
        return self.simple_chain.stream(question)

    """ render llm st response """
    def st_render(self, question, skip=0):
        self.retriever_strategy.set_skip(skip)
        stream = self.stream(question)
        response = Gmisc.write_stream(stream)
        context = self.retriever_strategy.get_recent_context()
        self.retriever_strategy.clear_recent_context()
        return response, context
