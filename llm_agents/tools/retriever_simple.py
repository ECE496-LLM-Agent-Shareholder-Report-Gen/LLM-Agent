import sys
sys.path.append('/groups/acmogrp/Large-Language-Model-Agent/llm_evaluator/model_testing/llm_agents')

sys.path.append('/groups/acmogrp/Large-Language-Model-Agent')

from pydantic import BaseModel, Field
from llm_agents.tools.base import ToolInterface

from typing import Any

from langchain.globals import set_debug
from langchain.globals import set_verbose

from retriever_pipeline.WordEmbeddingsLoader import WordEmbeddingsLoader

from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains import RetrievalQA

from langchain.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.retrievers.self_query.chroma import ChromaTranslator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

#from langchain_openai import ChatOpenAI

#not being used now...just using openAI
#from llm_agents.llm import ChatLLM

from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)

from langchain_core.prompts import PromptTemplate

#not neccesary if not using chroma
#__import__('pysqlite3')
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#llm = ChatOpenAI(model="gpt-3.5-turbo")

#embeddings_loader = WordEmbeddingsLoader()
#embeddings = embeddings_loader.load()

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

def generateEmbeddings():
    embeddings_loader = WordEmbeddingsLoader()
    embeddings = embeddings_loader.load()
    return embeddings

def createMetaDocs(path,ticker,year,quarter):
    loader = PyPDFLoader(path)
    docs = loader.load()

    for doc in docs:
        doc.metadata['ticker'] = ticker
        doc.metadata['year'] = year
        doc.metadata['quarter'] = quarter

def createSimpleDocs(path):
    loader = PyPDFLoader(path)
    docs = loader.load()
    return docs

def createMetadataVectorStoreFromFiles(files):
    pass

def createSimpleVectorStoreFromFiles(files):
    all_docs = []
    for file in files:
        docs = createSimpleDocs(file["path"])
        all_docs.extend(docs)

    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    split_docs = text_splitter.split_documents(all_docs)
    embeddings = generateEmbeddings()
    vectorstore = Chroma.from_documents(split_docs, embeddings)

    return vectorstore


def createVectorStore():

    # Load AMD
    AMD_loader = PyPDFLoader("/groups/acmogrp/Large-Language-Model-Agent/content/companies/AMD/2022/AMD_2022_10K.pdf")
    AMD_docs = AMD_loader.load()

    for doc in AMD_docs:
        doc.metadata['ticker'] = 'AMD'
        doc.metadata['year'] = 2022

    # Load Intel
    Intel_Loader = PyPDFLoader("/groups/acmogrp/Large-Language-Model-Agent/content/companies/INTC/2022/INTC_2022_10K.pdf")
    Intel_docs = Intel_Loader.load()

    for doc in Intel_docs:
        doc.metadata['ticker'] = 'INTC'
        doc.metadata['year'] = 2022

    NVDIA_Loader = PyPDFLoader("/groups/acmogrp/Large-Language-Model-Agent/content/companies/NVDA/2022/NVDA_2022_10K.pdf")
    NVDIA_docs = NVDIA_Loader.load()

    for doc in NVDIA_docs:
        doc.metadata['ticker'] = 'NVDA'
        doc.metadata['year'] = 2022

    # Merge the lists
    all_docs = []
    all_docs.extend(AMD_docs)
    all_docs.extend(Intel_docs)
    all_docs.extend(NVDIA_docs)

    text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    split_docs = text_splitter.split_documents(all_docs)

    vectorstore = Chroma.from_documents(split_docs, embeddings)

    return vectorstore

def createSelfQueryRetriever(vectorstore):
    document_content_description = "Excerpts from company financial reports"

    metadata_field_info = [
        AttributeInfo(
            name="ticker",
            description="The ticker of the company",
            type="string",
        ),
        AttributeInfo(
            name="year",
            description="The year of the financial report",
            type="integer",
        ),
        AttributeInfo(
            name="quarter",
            description="The quarter of the financial report (1,2,3 or 4)",
            type="integer",
        ),
    ]

    examples = [
        (
            "What is intel's revenue in 2022",
            {
                "query": "Revenue",
                "filter": 'and(eq("ticker", "INTC"), eq("year", 2022))',
            },
        ),
        (
            "How many employees does AMD have in the first quarter of 2021?",
            {
                "query": "AMD exmployees",
                "filter": 'and(eq("ticker", "AMD"), eq("year", 2021), eq("quarter", 1))',
            },
        ),
    ]

    prompt = get_query_constructor_prompt(
        document_content_description,
        metadata_field_info,
        examples=examples
    )


    output_parser = StructuredQueryOutputParser.from_components()
    query_constructor = prompt | llm | output_parser

    print(prompt.format(query="{query}"))

    retriever = SelfQueryRetriever(
        vectorstore=vectorstore,  # Use the corrected variable here
        query_constructor=query_constructor,
        structured_query_translator=ChromaTranslator(),
        verbose=True
    )

    return retriever

def createBasicRetriever(vectorstore):
    retriever = vectorstore.as_retriever()
    return retriever

def createRAGChain(retriever,llm):
    template = """Use the following pieces of context from company financial reports to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Use three sentences maximum and keep the answer as concise as possible.

    {context}

    Question: {question}

    Helpful Answer:"""
    custom_rag_prompt = PromptTemplate.from_template(template)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )

    #qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever, verbose=False)  # Use llm directly
    return rag_chain

#def retrieve(query: str) -> str:
    #return rag_chain.invoke(query)

class RetrieverTool(ToolInterface):
    """A tool for retrieving context from vector stores."""
    rag_chain: Any
    name: str = "Company Retriever"
    description: str = (
        "Useful to answer questions about companies "
        "Retrieves excerpts from financial reports. "
        "Input should be the information you're looking for "
        "ex. Employees at Apple"
    )

    def use(self, input_text: str) -> str:
        return self.rag_chain.invoke(input_text)


if __name__ == '__main__':
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
    #vectorstore = createVectorStore()
    vectorstore = createSimpleVectorStoreFromFiles(files)
    #retriever = createSelfQueryRetriever(vectorstore)
    retriever=createBasicRetriever(vectorstore)

    from langchain_community.llms import LlamaCpp

    llama_llm = LlamaCpp(
      model_path="/groups/acmogrp/Large-Language-Model-Agent/language_models/llama-2-13b/llama-2-13b.gguf.q8_0.bin",
      n_batch=512,
      n_gpu_layers=40,
      n_ctx=5000,
      temperature=0.1,
      verbose=False,  # Verbose is required to pass to the callback manager)
    )



    rag_chain = createRAGChain(retriever,llama_llm)
    tool = RetrieverTool(rag_chain=rag_chain)
    print(tool.name)
    result = tool.use("what is AMD's revenue")
    print(result)
