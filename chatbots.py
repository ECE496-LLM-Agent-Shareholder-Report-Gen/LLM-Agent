# Langchain imports
from math import floor
import math
from langchain.prompts import PromptTemplate, BaseChatPromptTemplate, ChatPromptTemplate
from langchain.tools import Tool
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from operator import itemgetter

from abc import ABC, abstractmethod
import os
import re

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
             { "context": {"question": itemgetter("question")} 
              | RunnableLambda(self.retriever_strategy.retrieve_context), 
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
    

class MultiChatbot(Chatbot):

    fusion_system_message = """
        You are a helpful assistant who breaks questions down into smaller questions.
        Be concise with your questions. ONLY state the questions. Do not give reasons for why you chose the questions. 
        """
    
    fusion_few_shot_prompts = [("What was AMD's revenue in 2022?", "1. AMD_2022_10K -- The company's revenue in 2022?"),
        ("How does AMD's revenue compare to INTC's revenue in 2022?", "1. AMD_2022_10K -- The company's revenue in 2022?\n2. INTC_2022_10K -- The company's revenue in 2022?"),
        ("What was AMD's distribution strategy in 2022?", "1. AMD_2022_10K -- The company's distribution strategy?"),
        ("What was AMD's revenue in 2022 Q2?", "1. AMD_2022_10Q_Q2 -- The company's revenue in 2022?"),]

    fusion_instruction_incomplete = """ You have the following sources available to you. \n
        {tools}\n\n
        You must use ONLY the above sources for which you believe your questions can be asked on, every other source is INCORRECT.
        Every smaller question you think of, it must ONLY relate to one company, one year, and if applicable, one quarter.\n
        Format your smaller questions like this:\n
        company_year_reportType -- question. Replace company_year_reportType with one of the sources, and replace question with the smaller question you generated.\n
        If you believe that the question should be asked about the quarter, then use the following format instead:\n
        company_year_reportType_quarter -- question. Replace company_year_reportType_quarter with one of the sources, and replace question with the smaller question you generated.\n
        Generate between one and four smaller questions.
        Break the following question down into smaller questions: {question}"""

    result_system_message = """
        You are a financial investment advisor who answers questions
        about shareholder reports. 
        """

    result_instruction_incomplete = """
        Given the context: '{context}'\n\n
        {question}
        """
   
    result_system_message_incomplete = """
        You are a financial investment advisor who answers questions
        about shareholder reports. You have context about the following companies and years:\n
        {companies_years}\n\n
        Answer questions given the context which holds information about the above companies and years:
        """

    result_instruction = """
        Given the context: '{context}'\n\n
        {question}\n\n
        Make sure to source where you got information from (given in the context). 
        This source should include the company, year, the report type, (quarter is possible) and page number.
        """

    def __init__(self, retriever_strategy, llm, vectorstores={}, max_k=8, *args, **kwargs):
        super().__init__(retriever_strategy, llm, *args, **kwargs)
        self.vectorstores = vectorstores
        self.parser = StrOutputParser()
        self.max_k = max_k
        self.tools = []
        self.companies_years = []
        for company, company_data in vectorstores.items():
            for year, year_data in company_data.items():
                for report_type, report_type_data in year_data.items():
                    if report_type == "10K":
                        self.tools.append(f"{company}_{year}_{report_type}: Useful for finding information about {company}'s {report_type} in {year}.")
                        self.companies_years.append(f"{company}'s {report_type} on the year {year}")
                    elif report_type == "10Q":
                        for quarter, quarter_data in report_type_data.items():
                            self.tools.append(f"{company}_{year}_{report_type}_{quarter}: Useful for finding information about {company}'s {report_type} for the quarter {quarter} in {year}.")
                            self.companies_years.append(f"{company}'s {report_type} in the quarter {quarter} in the year {year}")


        self.fusion_instruction = self.fusion_instruction_incomplete.replace("{tools}", "\n".join(self.tools))

        self.fusion_template = self.template_formatter.init_template_from_memory(
            system_message= self.fusion_system_message,
            instruction=self.fusion_instruction,
            memory=self.fusion_few_shot_prompts
        )

        self.result_system_message = self.result_system_message_incomplete.replace("{companies_years}","\n".join(self.companies_years) )

        self.result_template = self.template_formatter.init_template(
            system_message=self.result_system_message,
            instruction=self.result_instruction
        )

        self.input_variables = ["question", "context"]
        
        self.update_chain()

        self.memory = []
        self.mem_length = 0

    def get_multi_context(self, llm_output):
        self.fusion_output = llm_output
        quarter_pattern = r"(\w+)[\\_]+(\d+)[\\_]+(10Q|10q)[\\_]+(1|2|3)\s--\s(.+)\?"
        yearly_pattern = r"(\w+)[\\_]+(\d+)[\\_]+(10K|10k)\s--\s(.+)\?"
        quarter_matches = re.findall(quarter_pattern, llm_output)
        yearly_matches = re.findall(yearly_pattern, llm_output)

        matches = len(quarter_matches) + len(yearly_matches)
        if matches == 0:
            return None
        
        context_list = []
        fetch_k = matches*20

        k = floor(self.max_k/matches)

        # DO QUARTERS
        if len(quarter_matches)!=0:
            
            for match in quarter_matches:
                if not match[0] in self.vectorstores:
                    print(f"ERROR: unable to find {match[0]} in vectore dict!")   
                    continue
                if not match[1] in self.vectorstores[match[0]]:
                    print(f"ERROR: unable to find {match[1]} in vectore[{match[0]}] dict!") 
                    continue  
                if not match[2] in self.vectorstores[match[0]][match[1]]:
                    print(f"ERROR: unable to find {match[2]} in vectore[{match[0]}][{match[1]}] dict!") 
                    continue  
                if not match[3] in self.vectorstores[match[0]][match[1]][match[2]]:
                    print(f"ERROR: unable to find {match[3]} in vectore[{match[0]}][{match[1]}][{match[2]}] dict!") 
                
                context = self.retriever_strategy.retrieve_context(match[4], vectorstore=self.vectorstores[match[0]][match[1]][match[2]][match[3]], k=k, fetch_k=fetch_k)
                
                
                context_list.append(context)

        # DO YEARS
        if len(yearly_matches)!=0:
            for match in yearly_matches:
                if not match[0] in self.vectorstores:
                    print(f"ERROR: unable to find {match[0]} in vectore dict!")   
                    continue
                if not match[1] in self.vectorstores[match[0]]:
                    print(f"ERROR: unable to find {match[1]} in vectore[{match[0]}] dict!") 
                    continue  
                if not match[2] in self.vectorstores[match[0]][match[1]]:
                    print(f"ERROR: unable to find {match[2]} in vectore[{match[0]}][{match[1]}] dict!") 
                    continue  
                
                
                context = self.retriever_strategy.retrieve_context(match[3], vectorstore=self.vectorstores[match[0]][match[1]][match[2]], k=k, fetch_k=fetch_k)
                
                context_list.append(context)

        return "\n\n".join(context_list)



    def update_chain(self):
        self.fusion_prompt = PromptTemplate(template=self.fusion_template, input_variables=self.input_variables)
        self.result_prompt = PromptTemplate(template=self.result_template, input_variables=self.input_variables)

     
        self.fusion_chain = (
            {"question": RunnablePassthrough()}
            | self.fusion_prompt
            | self.llm
            | self.parser
            | RunnableLambda(self.get_multi_context)
        )

        self.result_chain = (
            { "context": itemgetter("context"),
             "question": itemgetter("question")
             }
            | self.result_prompt
            | self.llm
            | self.parser
        )

        # self.chain = (
        #     { "context": {"question": itemgetter("question")} | self.fusion_chain, "question": RunnablePassthrough()}
        #     | self.result_chain
        # )


    def invoke(self, question):
        fusion_response = self.fusion_chain.invoke({"question": question})
        if fusion_response == None:
            print("Oops! something went wrong")
            return "N/A"
        result_response = self.result_chain.invoke({"context": fusion_response, "question": question})
        if self.memory_active:
            self.update_memory(question, result_response)
        return result_response



class FusionChatbot(Chatbot):
    fusion_system_message = """
        You are a helpful assistant who breaks questions down into smaller questions.
        Be concise with your questions. ONLY state the questions. Do not give reasons for why you chose the questions. 
        """
    
    fusion_few_shot_prompts = [("What was AMD's revenue in 2022?", "1. AMD_2022 -- The company's revenue in 2022?"),
        ("How does AMD's revenue compare to INTC's revenue in 2022?", "1. AMD_2022 -- The company's revenue in 2022?\n2. INTC_2022 -- The company's revenue in 2022?"),
        ("What was AMD's distribution strategy in 2022?", "1. AMD_2022 -- The company's distribution strategy?")]

    fusion_instruction_incomplete = """ You have the following sources available to you. \n
        {tools}\n\n
        You must use ONLY the above sources for which you believe your questions can be asked on, every other source is INCORRECT.
        Every smaller question you think of, it must ONLY relate to one company, one year, and if applicable, one quarter.\n
        With all this in mind, break the question down into smaller questions. \n\n
        Format your smaller questions like this:\n
        company_year_reportType -- question. Replace company_year_reportType with one of the sources, and replace question with the smaller question you generated.\n
        If you believe that the question should be asked about the quarter, then use the following format instead:\n
        company_year_reportType_quarter -- question. Replace company_year_reportType_quarter with one of the sources, and replace question with the smaller question you generated.\n
        Please generate as few questions necessary that you believe can adequately answer the original question.
        Break the following question down into smaller questions: {question}"""

    result_system_message = """
        You are a financial investment advisor who answers questions
        about shareholder reports. 
        """

    result_instruction_incomplete = """
        Use your previous answers to answer the question: {question}
        """
    
    simple_system_message = """
                You are a financial investment advisor who answers questions
                about shareholder reports. You will be given a context and will answer the question using that context.
                """
    simple_instruction = """
               Context: "{context}"
                \n\n
                Question: "{question}"

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
    def __init__(self, retriever_strategy, llm, vectorstores={}, max_k=6, *args, **kwargs):
        super().__init__(retriever_strategy, llm, *args, **kwargs)
        self.vectorstores = vectorstores
        self.parser = StrOutputParser()
        self.max_k = max_k
        self.tools = []
        self.companies_years = []
        for company, company_data in vectorstores.items():
            for year, year_data in company_data.items():
                for report_type, report_type_data in year_data.items():
                    if report_type == "10K":
                        self.tools.append(f"{company}_{year}_{report_type}: Useful for finding information about {company}'s {report_type} in {year}.")
                        self.companies_years.append(f"{company}'s {report_type} on the year {year}")
                    elif report_type == "10Q":
                        for quarter, quarter_data in report_type_data.items():
                            self.tools.append(f"{company}_{year}_{report_type}_{quarter}: Useful for finding information about {company}'s {report_type} for the quarter {quarter} in {year}.")
                            self.companies_years.append(f"{company}'s {report_type} in the quarter {quarter} in the year {year}")


        self.fusion_instruction = self.fusion_instruction_incomplete.replace("{tools}", "\n".join(self.tools))

        self.fusion_template = self.template_formatter.init_template(
            system_message= self.fusion_system_message,
            instruction=self.fusion_instruction,
            # memory=self.fusion_few_shot_prompts
        )

        self.result_instruction = self.result_instruction_incomplete.replace("{companies_years}","\n".join(self.companies_years) )

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


    def get_multi_context(self, llm_output):
        self.fusion_output = llm_output
        quarter_pattern = r"(\w+)[\\_]+(\d+)[\\_]+(10Q|10q)[\\_]+(1|2|3)\s--\s(.+)\?"
        yearly_pattern = r"(\w+)[\\_]+(\d+)[\\_]+(10K|10k)\s--\s(.+)\?"
        quarter_matches = re.findall(quarter_pattern, llm_output)
        yearly_matches = re.findall(yearly_pattern, llm_output)

        qa_list = []

        # DO QUARTERS
        if len(quarter_matches)!=0:
            
            for match in quarter_matches:
                if not match[0] in self.vectorstores:
                    print(f"ERROR: unable to find {match[0]} in vectore dict!")   
                    continue
                if not match[1] in self.vectorstores[match[0]]:
                    print(f"ERROR: unable to find {match[1]} in vectore[{match[0]}] dict!") 
                    continue  
                if not match[2] in self.vectorstores[match[0]][match[1]]:
                    print(f"ERROR: unable to find {match[2]} in vectore[{match[0]}][{match[1]}] dict!") 
                    continue  
                if not match[3] in self.vectorstores[match[0]][match[1]][match[2]]:
                    print(f"ERROR: unable to find {match[3]} in vectore[{match[0]}][{match[1]}][{match[2]}] dict!") 
                
                context = self.retriever_strategy.retrieve_context(match[4], vectorstore=self.vectorstores[match[0]][match[1]][match[2]][match[3]], k=self.max_k)
                
                print(f"\n\n{match[4]}")

                result = self.simple_chain.invoke({"question": match[4], "context": context})

                
                qa_list.append((match[4], result))

        # DO YEARS
        if len(yearly_matches)!=0:
            for match in yearly_matches:
                if not match[0] in self.vectorstores:
                    print(f"ERROR: unable to find {match[0]} in vectore dict!")   
                    continue
                if not match[1] in self.vectorstores[match[0]]:
                    print(f"ERROR: unable to find {match[1]} in vectore[{match[0]}] dict!") 
                    continue  
                if not match[2] in self.vectorstores[match[0]][match[1]]:
                    print(f"ERROR: unable to find {match[2]} in vectore[{match[0]}][{match[1]}] dict!") 
                    continue  
                
                
                context = self.retriever_strategy.retrieve_context(match[3], vectorstore=self.vectorstores[match[0]][match[1]][match[2]], k=self.max_k)
                
                print(f"\n\n{match[3]}")

                result = self.simple_chain.invoke({"question": match[3], "context": context})

                
                qa_list.append((match[3], result))

        return qa_list
        

    def update_chain(self):
        self.fusion_prompt = PromptTemplate(template=self.fusion_template, input_variables=["question"])
        self.simple_prompt = PromptTemplate(template=self.simple_template, input_variables=["question", "context"])

     
        self.fusion_chain = (
            {"question": RunnablePassthrough()}
            | self.fusion_prompt
            | self.llm
            | self.parser
            | RunnableLambda(self.get_multi_context)
        )

       
        self.simple_chain = (
             { "context": itemgetter("context"), 
              "question": itemgetter("question")}
             | self.simple_prompt
             | self.llm
             | self.parser
            )



    def invoke(self, question):
        fusion_response = self.fusion_chain.invoke({"question": question})
        if fusion_response == None:
            print("Oops! something went wrong")
            return "N/A"
        result_template = self.template_formatter.init_template_from_memory(
            system_message=self.result_system_message,
            instruction=self.result_instruction,
            memory=fusion_response
        )
        result_prompt =  PromptTemplate(template=result_template, input_variables=["question"])

        result_chain = (
            { 
             "question": itemgetter("question")
             }
            | result_prompt
            | self.llm
            | self.parser
        )

        response = result_chain.invoke({"question": question})
        if self.memory_active:
            self.update_memory(question, response)
        final_response = ""
        for qa in fusion_response:
            final_response+= f"{qa[0]}\n\n{qa[1]}\n\n"
        final_response += response
        return final_response




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
