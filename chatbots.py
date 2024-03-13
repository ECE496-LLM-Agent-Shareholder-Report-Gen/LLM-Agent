# Langchain imports
import math
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter

from abc import ABC, abstractmethod

from subquery_generator import SubQueryGenerator
from template_formatter import LlamaTemplateFormatter

class Chatbot(ABC):

    def __init__(self, retriever_strategy, llm, template_formatter=LlamaTemplateFormatter(), memory_limit=3000, progressive_memory=True, memory_active=False):
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
                """
    instruction = """
               Context: "{context}"
                \n\n
                Question: "{question}"

                \n\n
                Make sure to source where you got the information from. 
                This source should include the company, year, the report type, and page number.
                """

    def __init__(self, retriever_strategy, llm, *args, **kwargs):
        super().__init__(retriever_strategy, llm, *args, **kwargs)
        self.parser = StrOutputParser()
        self.template = self.template_formatter.init_template(
            system_message= self.system_message,
            instruction=self.instruction
        )

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
        response = self.chain.invoke({"question": question})
        if self.memory_active:
            self.update_memory(question, response)
        return response
    
    def stream(self, question):
        response = self.chain.stream({"question": question})

        return response
    
""" FusionChatbot """
class FusionChatbot(Chatbot):

    result_system_message = """
        You are a helpful assistant. Answer questions given the context. \
        Make sure to source where you got information from (given in the context). \
        This source should include the company, year, the report type, (quarter if \
        possible) and page number.
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


""" StepbackChatbot """
class StepbackChatbot(Chatbot):

    result_system_message = """
        You are a helpful assistant. Use your previous answers to answer questions. \
        Be sure to reference the source of the original information as you have done \
        in your previous answers.
        """.strip()

    result_instruction = """
        {question}
        """.strip()
    
    simple_system_message = """
        You are a helpful assistant. You will be given a context and will answer the \
        question using that context. Make sure to source where you got the \
        information from. This source should include the company, year, the report \
        type, and page number as reported at the start of the Excerpt.
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


    def invoke(self, question):
        # get the broken down question
        sub_query_response = self.sub_query_generator.invoke(question)
        
        # get the matches
        all_matches =  self.sub_query_generator.parse_questions(sub_query_response)

        # answer the broken down questions
        all_responses = self.answer_chain.batch(all_matches)

        temp_mem = [(all_matches[x][-1],all_responses[x]) for x in range(len(all_matches))]


        if temp_mem == None or len(temp_mem) == 0:
            print("Oops! something went wrong")
            return "N/A"
        result_template = self.template_formatter.init_template_from_memory(
            system_message=self.result_system_message,
            instruction=self.result_instruction,
            memory=temp_mem
        )
        result_prompt =  PromptTemplate(template=result_template, input_variables=["question"])

        result_chain = (
            { 
             "question": RunnablePassthrough()
             }
            | result_prompt
            | self.llm
            | self.parser
        )

        response = result_chain.invoke(question)
        if self.memory_active:
            self.update_memory(question, response)
        return response



""" Simple stepback """
class PreMultiChatbot(Chatbot):    
    simple_system_message = """
                You are a financial investment advisor who answers questions
                about shareholder reports. You will be given a context and will answer the question using that context.
                """
    simple_instruction = """
               Given the context: "{context}"
                \n\n
                {question}

                \n\n
                Make sure to source where you got the information from. 
                This source should include the company, year, the report type, and page number.
                """

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

        for company, company_data in self.vectorstores.items():
            for year, year_data in company_data.items():
                for report_type, report_type_data in year_data.items():
                    if report_type == "10K":
                        context = self.retriever_strategy.retrieve_context(question, vectorstore=report_type_data, k=self.k)
                        context_list.append(context)
                    elif report_type == "10Q":
                        for quarter, quarter_data in report_type_data.items():
                            context = self.retriever_strategy.retrieve_context(question, vectorstore=quarter_data, k=self.k)
                            context_list.append(context)
        return "\n\n".join(context_list)
        

    def update_chain(self):
        self.simple_prompt = PromptTemplate(template=self.simple_template, input_variables=["question", "context"])

     
        self.simple_chain = (
             { "context": itemgetter("context"), 
              "question": itemgetter("question")}
             | self.simple_prompt
             | self.llm
             | self.parser
            )


    def invoke(self, question):
        context = self.get_multi_context(question)
        response = self.simple_chain.invoke({"question": question, "context": context})
        if self.memory_active:
            self.update_memory(question, response)
        return response
