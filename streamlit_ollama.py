import streamlit as st
from langchain_community.chat_models import ChatOllama

from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

search_internet = st.checkbox("check internet?", value=False, key="internet")
question = st.text_input("question", value="", key="question")

@st.cache_data
def load_model():
    return ChatOllama(model="llama2") # 👈 stef default

@st.cache_data
def load_retriever():
    embeddings_path = "/groups/acmogrp/Large-Language-Model-Agent/language_models/word_embeddings/BAAI_bge-large-en-v1.5"
    embeddings_args = {
            "model_kwargs": {'device': 'cuda'},
            "encode_kwargs": {'normalize_embeddings': True},
            "model_name": embeddings_path
    }
    path = "/groups/acmogrp/Large-Language-Model-Agent/app/content/test-set/"
    embeddings = HuggingFaceEmbeddings(**embeddings_args)   
    vector_store = FAISS.load_local(path, embeddings, index_name="AMD_2021_10k_index", allow_dangerous_deserialization=True)

    return vector_store.as_retriever()



llm = load_model()

retriever = load_retriever()
template = """[INST] You are a financial investment advisor who answers questions
        about shareholder reports. If you are unsure about an answer, truthfully say
        "I don't know".

        You have the following context, use it to answer questions: '{context}'.

        Question: '{question}'
        [/INST]"""
if question:
    print(question)
    prompt = ChatPromptTemplate.from_template(template)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    try:
        stream = chain.stream(question)
        st.write_stream(stream)
    except:
        print("llm: ", llm)
        print("prompt: ", prompt)
        print("retriever: ", retriever)
        print("chain: ", chain)