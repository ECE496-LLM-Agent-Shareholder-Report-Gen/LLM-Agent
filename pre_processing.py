from langchain.document_loaders.pdf import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
import os

""" IndexGenerator
class for generating the vector stores
"""
class IndexGenerator:
    embeddings = None


    def __init__(self, **kwargs):
        embeddings_path = "/groups/acmogrp/Large-Language-Model-Agent/language_models/word_embeddings/BAAI_bge-large-en-v1.5-ft"
       
        real_embeddings_args = {
            "model_kwargs": {'device': 'cuda'},
            "encode_kwargs": {'normalize_embeddings': True},
            "model_name": embeddings_path
        }

        real_embeddings_args.update(kwargs)
        self.load_bge(**real_embeddings_args)

    # loads embeddings
    def load_bge(self, **kwargs):
        self.embeddings = HuggingFaceEmbeddings(**kwargs)  
     

    # pre process files into chunks. Apply processing techniques into the individual
    def pre_precess(self, files):
        chunk_size = 500
        chunk_overlap = 50
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap  = chunk_overlap,
            length_function = len,
            add_start_index = True
            )
        documents = text_splitter.split_documents(files)
        return documents


    # generate vector store for the given files
    def generate_embeddings(self, files):
        documents = self.pre_precess(files)

        for i, document in enumerate(documents):
            source = document.metadata['source']
            file_name = os.path.basename(source)
            file_name = file_name.split(".")[0]
            if file_name.find("10Q") != -1:
                company, year, report_type, quarter = file_name.split("_")
                documents[i].metadata["quarter"] = quarter
            else:
                company, year, report_type = file_name.split("_")
            documents[i].metadata["company"] = company
            documents[i].metadata["year"] = year
            documents[i].metadata["report_type"] = report_type
        vector_store = FAISS.from_documents(documents, self.embeddings)
        return vector_store


    # PDF FILE: generate a vector store and return it
    def generate_vector_store_pdf_file(self, file):
        loader = PyPDFLoader(file)
        files = loader.load()
        return self.generate_embeddings(files)


    # PDF DIRECTORY: generate a vector store and return it
    def generate_vector_store_pdf_dir(self, directory):
        loader = PyPDFDirectoryLoader(directory)
        files = loader.load()
        return self.generate_embeddings(files)


    # generate embeddings for pdfs in directory, save to index_path
    def save_vector_store(self, vector_store, index_path):
        vector_store.save_local(index_path)


    # merge file into an existing vector store
    def merge_vector_stores(self, vectore_store1, vector_store2):
        vectore_store1.merge_from(vector_store2)
        return vectore_store1

    def load_vector_store(self, path):
        return FAISS.load_local(path, self.embeddings)




