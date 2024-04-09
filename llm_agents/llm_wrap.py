from typing import List,Optional, Any

class LLM_Wrapper():
  def __init__(self, llm):
      # Instance attribute
      self.llm = llm

  def generate(self, prompt: str, stop: List[str] = []):
    response = self.llm.invoke(prompt,stop=stop)
    return response.content

