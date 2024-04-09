import sys
sys.path.append('/groups/acmogrp/Large-Language-Model-Agent/llm_evaluator/model_testing/llm_agents')

import datetime
import re

from pydantic import BaseModel
from typing import List, Dict, Tuple, Any
#from llm_agents.llm_llama import Chatllama_LLM
#from llm_agents.llm import ChatLLM
#from llm_agents.llm_wrapper import LLM_Wrapper

from llm_agents.llm_wrap import LLM_Wrapper

from llm_agents.tools.base import ToolInterface
from llm_agents.tools.python_repl import PythonREPLTool
#from llm_agents.tools.retriever import RetrieverTool
from llm_agents.tools.retriever_simple import RetrieverTool,createSimpleVectorStoreFromFiles,createBasicRetriever,createRAGChain

from langchain_community.llms import LlamaCpp

FINAL_ANSWER_TOKEN = "Final Answer:"
OBSERVATION_TOKEN = "Observation:"
THOUGHT_TOKEN = "Thought:"
PROMPT_TEMPLATE = """Today is {today} and you can use tools to get new information. Answer the question as best as you can using the following tools:

{tool_description}

Use the following format:

Question: the input question you must answer
Thought: comment on what you want to do next
Action: the action to take, exactly one element of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation repeats N times, use it until you are sure of the answer)
Thought: I now know the final answer
Final Answer: your final answer to the original input question

Begin!

Question: {question}
Thought: {previous_responses}
"""


class Agent(BaseModel):
    llm: Any
    #llm:Chatllama_LLM
    tools: List[ToolInterface]
    prompt_template: str = PROMPT_TEMPLATE
    max_loops: int = 15
    # The stop pattern is used, so the LLM does not hallucinate until the end
    stop_pattern: List[str] = [f'\n{OBSERVATION_TOKEN}', f'\n\t{OBSERVATION_TOKEN}']

    @property
    def tool_description(self) -> str:
        return "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])

    @property
    def tool_names(self) -> str:
        return ",".join([tool.name for tool in self.tools])

    @property
    def tool_by_names(self) -> Dict[str, ToolInterface]:
        return {tool.name: tool for tool in self.tools}

    def run(self, question: str):
        previous_responses = []
        num_loops = 0
        prompt = self.prompt_template.format(
                today = datetime.date.today(),
                tool_description=self.tool_description,
                tool_names=self.tool_names,
                question=question,
                previous_responses='{previous_responses}'
        )
        print(prompt.format(previous_responses=''))
        while num_loops < self.max_loops:
            num_loops += 1
            curr_prompt = prompt.format(previous_responses='\n'.join(previous_responses))
            generated, tool, tool_input = self.decide_next_action(curr_prompt)
            if tool == 'Final Answer':
                return tool_input
            if tool not in self.tool_by_names:
                raise ValueError(f"Unknown tool: {tool}")
            tool_result = self.tool_by_names[tool].use(tool_input)
            generated += f"\n{OBSERVATION_TOKEN} {tool_result}\n{THOUGHT_TOKEN}"
            print(generated)
            previous_responses.append(generated)

    def decide_next_action(self, prompt: str) -> str:
        generated = self.llm.generate(prompt, stop=self.stop_pattern)
        #print(f"The value of the generated is: {generated}")
        tool, tool_input = self._parse(generated)
        return generated, tool, tool_input

    def _parse(self, generated: str) -> Tuple[str, str]:
        if FINAL_ANSWER_TOKEN in generated:
            return "Final Answer", generated.split(FINAL_ANSWER_TOKEN)[-1].strip()
        regex = r"Action: [\[]?(.*?)[\]]?[\n]*Action Input:[\s]*(.*)"
        match = re.search(regex, generated, re.DOTALL)
        if not match:
            raise ValueError(f"Output of LLM is not parsable for next tool use: `{generated}`")
        tool = match.group(1).strip()
        tool_input = match.group(2)
        return tool, tool_input.strip(" ").strip('"')


if __name__ == '__main__':

    files = [
    {
        "company": "AMD",
        "year": "2022",
        "report type": "10Q",
        "quarter": "2",
        "path": "/groups/acmogrp/Large-Language-Model-Agent/app/content/test-set/AMD_2022_10Q_Q2.pdf",
        "index": "AMD_2022_10Q_Q2_index"
    }
    ,{
        "company": "AMD",
        "year": "2022",
        "report type": "10Q",
        "quarter": "3",
        "path": "/groups/acmogrp/Large-Language-Model-Agent/app/content/test-set/AMD_2022_10Q_Q3.pdf",
        "index": "AMD_2022_10Q_Q3_index"
    },
    ]

    llama_llm = LlamaCpp(
      model_path="/groups/acmogrp/Large-Language-Model-Agent/language_models/llama-2-13b/llama-2-13b.gguf.q8_0.bin",
      n_batch=512,
      n_gpu_layers=40,
      n_ctx=3000,
      temperature=0.1,
      verbose=False,  # Verbose is required to pass to the callback manager)
    )

    llm=LLM_Wrapper(llama_llm)
    vectorstore = createSimpleVectorStoreFromFiles(files)
    retriever=createBasicRetriever(vectorstore)
    rag_chain = createRAGChain(retriever,llama_llm)


    agent = Agent(llm=llm, tools=[PythonREPLTool(), RetrieverTool(rag_chain=rag_chain)])
    #agent = Agent(llm=Chatllama_LLM(), tools=[PythonREPLTool()])
    #result = agent.run("what is 5x5 in Python?")
    #result = agent.run("print current directory using ls in Python?")
    #result = agent.run("print current directory using ls in Python?")

    while True:
        user_input = input("Question: ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        result= agent.run(user_input)
        print(f"Final answer is {result}")
