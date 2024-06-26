# Langchain imports
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

import math
import re

from agent.template_formatter import LlamaTemplateFormatter

""" Generates sub queries given a query """
class SubQueryGenerator:
    """
    Generates sub queries given a query.

    Attributes:
        sub_query_system_message_incomplete (str): Incomplete system message for step-back guidelines.
        sub_query_instruction (str): Instruction for sub queries.

    Methods:
        __init__(self, retriever_strategy, llm, vectorstores={}, max_k=8, template_formatter=LlamaTemplateFormatter(), *args, **kwargs):
            Initializes a SubQueryGenerator instance.
        
        get_chain(self):
            Returns the sub query processing chain.
        
        invoke(self, question):
            Invokes the sub query processing chain on a given question.
        
        stream(self, question):
            Streams the sub query processing chain on a given question.
        
        retrieve_context_from_year_match(self, match, k):
            Retrieves context from a yearly report match.
        
        retrieve_context_from_quarter_match(self, match, k):
            Retrieves context from a quarterly report match.
        
        retrieve_context_question_dict(self, match):
            Given a single match, returns a dictionary with context and question.
        
        retrieve_contexts(self, matches):
            Given a list of matches, returns a string of contexts.
        
        parse_questions(self, unparsed_questions):
            Parses questions to extract matches.
        
        retriever_multicontext(self, llm_output):
            Given an LLM output, returns a string of contexts.
    """

    sub_query_system_message_incomplete = """
You are an expert at financial advice. \
Your task is to step back and paraphrase a question to more generic \
step-back questions, which are easier to answer. Here are some guidelines:\n \
* If the question is already easy to answer, the step-back questions should just be the original question.\n
* The step-back questions must be derived from the original question such that \
answering these step-back questions would help answer the original question.\n \
* Each step-back question must be about a particular report, out of the \
following report titles:\n \
{report_titles}\n \
* answer each step-back question in the format: <report_title> -- <step-back question>\n \
Replace <step-back question> with the step-back question generate, and <report_title> \
with the report title the step-back question can be answered with.\n
* Do NOT put a report title any where except beside the step-back question.
        """.strip()
    
    sub_query_instruction = """{question}"""

    def __init__(self, retriever_strategy, llm, vectorstores={}, max_k=8, template_formatter=LlamaTemplateFormatter(), *args, **kwargs):
        """
        Initializes a SubQueryGenerator instance.

        Args:
            retriever_strategy: Retrieval strategy.
            llm: Language model.
            vectorstores (dict, optional): Dictionary of vector stores. Defaults to {}.
            max_k (int, optional): Maximum number of retrieved documents. Defaults to 8.
            template_formatter: Template formatter. Defaults to LlamaTemplateFormatter().
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.retriever_strategy = retriever_strategy
        self.llm = llm
        self.vectorstores = vectorstores
        self.template_formatter = template_formatter
        self.max_k = max_k

        self.report_names = []
        for company, company_data in vectorstores.items():
            for year, year_data in company_data.items():
                for report_type, report_type_data in year_data.items():
                    if report_type == "10K":
                        self.report_names.append(f"{company}_{year}_{report_type}")
                    elif report_type == "10Q":
                        for quarter, quarter_data in report_type_data.items():
                            self.report_names.append(f"{company}_{year}_{report_type}_{quarter}")

        self.sub_query_system_message = self.sub_query_system_message_incomplete.replace("{report_titles}", "\n".join(self.report_names))
        self.sub_query_template = self.template_formatter.init_template(
            system_message= self.sub_query_system_message,
            instruction=self.sub_query_instruction
        )

        self.sub_query_prompt = PromptTemplate(template=self.sub_query_template, input_variables=["question"])

        self.parser = StrOutputParser()

        self.sub_query_chain = (
            {"question": RunnablePassthrough()}
            | self.sub_query_prompt
            | self.llm
            | self.parser
        )
    
    def get_chain(self):
        """
        Returns the sub query processing chain.

        Returns:
            Chain: Sub query processing chain.
        """
        return self.sub_query_chain
    
    def invoke(self, question):
        """
        Invokes the sub query processing chain on a given question.

        Args:
            question (str): Input question.

        Returns:
            str: Processed output.
        """
        return self.sub_query_chain.invoke(question)
    
    def stream(self, question):
        """
        Streams the sub query processing chain on a given question.

        Args:
            question (str): Input question.

        Returns:
            str: Streamed output.
        """
        return self.sub_query_chain.stream(question)

    """ given a yearly report match, get the context """
    def retrieve_context_from_year_match(self, match, k):
        """
        Retrieves context from a yearly report match.

        Args:
            match: Yearly report match.
            k (int): Number of retrieved documents.

        Returns:
            str: Retrieved context.
        """
        if not match[0] in self.vectorstores:
            print(f"ERROR: unable to find {match[0]} in vectore dict!")   
            return ""
        if not match[1] in self.vectorstores[match[0]]:
            print(f"ERROR: unable to find {match[1]} in vectore[{match[0]}] dict!") 
            return ""
  
        if not match[2] in self.vectorstores[match[0]][match[1]]:
            print(f"ERROR: unable to find {match[2]} in vectore[{match[0]}][{match[1]}] dict!") 
            return ""
        
        context = self.retriever_strategy.retrieve_context(match[3], vectorstore=self.vectorstores[match[0]][match[1]][match[2]], k=k)
        return context

    """ given a quarterly report match, get the context """
    def retrieve_context_from_quarter_match(self, match, k):
        """
        Retrieves context from a quarterly report match.

        Args:
            match: Quarterly report match.
            k (int): Number of retrieved documents.

        Returns:
            str: Retrieved context.
        """
        if not match[0] in self.vectorstores:
            print(f"ERROR: unable to find {match[0]} in vectore dict!")   
            return ""  

        if not match[1] in self.vectorstores[match[0]]:
            print(f"ERROR: unable to find {match[1]} in vectore[{match[0]}] dict!") 
            return ""  
  
        if not match[2] in self.vectorstores[match[0]][match[1]]:
            print(f"ERROR: unable to find {match[2]} in vectore[{match[0]}][{match[1]}] dict!") 
            return ""  
        if not match[3] in self.vectorstores[match[0]][match[1]][match[2]]:
            print(f"ERROR: unable to find {match[3]} in vectore[{match[0]}][{match[1]}][{match[2]}] dict!") 
            return ""
        
        context = self.retriever_strategy.retrieve_context(match[4], vectorstore=self.vectorstores[match[0]][match[1]][match[2]][match[3]], k=k)

        return context


    """ Given a single match, return a question, context dict """
    def retrieve_context_question_dict(self, match):
        """
        Given a single match, returns a dictionary with context and question.

        Args:
            match: Matched report information.

        Returns:
            dict: Dictionary with context and question.
        """
        context = ""
        question = ""
       
        if len(match) == 5:
            question = match[4]
            context = self.retrieve_context_from_quarter_match(match, self.max_k)
        elif len(match) == 4:
            question = match[3]
            context = self.retrieve_context_from_year_match(match, self.max_k)
        return {"context": context, "question": question}
    
    """ Given a list of matches, return a string of contexts """
    def retrieve_contexts(self, matches):
        """
        Given a list of matches, returns a string of contexts.

        Args:
            matches: List of matched report information.

        Returns:
            str: Concatenated contexts.
        """
        contexts = []
        if len(matches) == 0: 
            return None
        
        k = math.ceil(self.max_k/len(matches))
        docs_left = self.max_k

        for match in matches:
            if docs_left < k:
                k = docs_left
            if k <= 0:
                break

            if len(match) == 5:
                context = self.retrieve_context_from_quarter_match(match, k)
            elif len(match) == 4:
                context = self.retrieve_context_from_year_match(match, k)
            docs_left -= k
            contexts.append(context)
        return "\n\n".join(contexts)

    """ Parse questions like so:
        company_year_reportType -- question
        OR
        company_year_10Q_quarter -- question

        if neither of the above work, result to looking up company_year_reportType
        or company_year_10Q_quarter within each line
        """
    def parse_questions(self, unparsed_questions):
        """
        Parses questions to extract matches.

        Args:
            unparsed_questions (str): Unparsed questions.

        Returns:
            list: List of matches.
        """
        self.fusion_output = unparsed_questions
        quarter_pattern = r"(\w+)[\\_]+(\d+)[\\_]+(10Q|10q)[\\_]+(Q1|Q2|Q3)\s--\s(.+)\?"
        yearly_pattern = r"(\w+)[\\_]+(\d+)[\\_]+([\da-zA-Z]+)\s--\s(.+)\?"
        quarter_matches = re.findall(quarter_pattern, unparsed_questions)
        yearly_matches = re.findall(yearly_pattern, unparsed_questions)
        # in case the above parsing doesn't work...
        # create the matches ourselves (company, year, reportType, quarter, question)
        leftovers = []
        if len(yearly_matches) == 0 and len(quarter_matches) == 0:
            # check by checking each line if the report is there
            lines = unparsed_questions.split("\n")
            for line in lines:
                for report in self.report_names:
                    if report in line:
                        if "sure" in line.lower():
                            continue
                        r_match = [r_part for r_part in report.split("_")]
                        question = line
                        r_match.append(question)
                        leftovers.append(tuple(r_match))
            return leftovers


        return quarter_matches + yearly_matches

    """ Given an llm output, return a string of contexts """
    def retriever_multicontext(self, llm_output):
        """
        Given an LLM output, returns a string of contexts.

        Args:
            llm_output (str): LLM-generated output.

        Returns:
            str: Concatenated contexts.
        """
        matches = self.parse_questions(llm_output)
        if len(matches) == 0: 
            return None
        return self.retrieve_contexts( matches)

    
