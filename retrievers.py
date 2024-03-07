from multiprocessing import context
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.vectorstores import FAISS, Chroma
from langchain.agents.agent_toolkits import (
    VectorStoreInfo,
    VectorStoreRouterToolkit,
    create_vectorstore_router_agent,
)
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import ParentDocumentRetriever
from langchain.chains import RetrievalQA

from model_loader import EmbeddingsLoader

import os

class MultiCompanyYearRetriever:
    prefix = """Answer the following questions as best you can. You have tools at your disposal.
    Use tools for a specific company and year if answering the question requires information about that company and year."""

    def __init__(self, companies, years, base_path, llm, embeddings):
        self.companies = companies
        self.years = years
        vectorestores = []

        for c in companies:
            for y in years:
                vectorestores.append(VectorStoreInfo(
                    name=f"{c}_{y}",
                    description=f"Information about {c} in {y}, Action should be: '{c}_{y}'",
                    vectorstore=FAISS.load_local(os.path.join(base_path, c, y, 'index'), embeddings),
                ))

        router_toolkit = VectorStoreRouterToolkit(
            vectorstores=vectorestores, llm=llm
        )
        self.agent_executor = create_vectorstore_router_agent(
            llm=llm, toolkit=router_toolkit, verbose=False, prefix=self.prefix
        )

    def answer(self, question):
        return self.agent_executor.run(question)



class BasicRetriever:
    # memory = ConversationBufferMemory(memory_key="chat_history")

    template = """[INST] You are a financial investment advisor who answers questions
        about shareholder reports. If you are unsure about an answer, truthfully say
        "I don't know".

        You have the following context, use it to answer questions: '{context}'.

        Question: '{question}'
        [/INST]"""


    def __init__(self, llm, index, embeddings):
        self.vector_store = FAISS.load_local(index, embeddings)
        prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])
        self.llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            # memory=self.memory
        )


    def answer(self, question):
        # Create LLM chain
        filter = {
            "report_type": "10K"
        }
        # Retrieve matched documents from index
        matched_docs = self.vector_store.similarity_search(question, k=10, filter=filter)
        context = ""
        for doc in matched_docs:
            context += f"'{doc.page_content}'- source: {doc.metadata['source']} \n\n"


        # Answer the question
        response = self.llm_chain.run(context=context, question=question)
        return response

class PDR:
    def __init__(self, llm, docs):
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
        store = InMemoryStore()
        self.llm = llm
        embeddings_loader=EmbeddingsLoader
        bge_embeddings=embeddings_loader.load_bge
        vectorstore=Chroma(
            collection_name="parents",
            embedding_function=bge_embeddings
        )

        self.retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )
    
    def answer(self, question):
        qa = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever
        )
        response = qa.run(question)
        return response