from langchain.document_loaders.pdf import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
import os

""" IndexGenerator
class for generating the vector stores
"""
class IndexGenerator:
    
    def __init__(self, index_name=None):
        if index_name == None:
            self.index_name = "index"
        else:
            self.index_name = index_name.replace("/", "_")

    # pre process files into chunks. Apply processing techniques into the individual
    def pre_process(self, pages):
        chunk_size = 500
        chunk_overlap = 50
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap  = chunk_overlap,
            length_function = len,
            add_start_index = True
            )
        documents = text_splitter.split_documents(pages)
        return documents


    # generate vector store for the given files
    def generate_embeddings(self, embeddings, pages, company, year, report_type, quarter=None):
        documents = self.pre_process(pages)

        for i, document in enumerate(documents):
            documents[i].metadata["company"] = company
            documents[i].metadata["year"] = year
            documents[i].metadata["report type"] = report_type
            if quarter:
                documents[i].metadata["quarter"] = quarter
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store


    # PDF FILE: generate a vector store and return it
    def generate_vector_store_pdf_file(self, embeddings, file, company, year, report_type, quarter=None):
        loader = PyPDFLoader(file)
        pages = loader.load()
        return self.generate_embeddings(embeddings, pages, company, year, report_type, quarter=None)


    # PDF DIRECTORY: generate a vector store and return it
    def generate_vector_store_pdf_dir(self, directory):
        loader = PyPDFDirectoryLoader(directory)
        files = loader.load()
        return self.generate_embeddings(files)


    # generate embeddings for pdfs in directory, save to index_path
    def save_vector_store(self, vector_store, index_path):
        vector_store.save_local(index_path, index_name=self.index_name)


    # merge file into an existing vector store
    def merge_vector_stores(self, vectore_store1, vector_store2):
        vectore_store1.merge_from(vector_store2)
        return vectore_store1

    def load_vector_store(self, path, embeddings):
        return FAISS.load_local(folder_path=path,
                                 embeddings=embeddings,
                                 allow_dangerous_deserialization=True, 
                                 index_name=self.index_name)




