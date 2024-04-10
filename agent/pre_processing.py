from langchain.document_loaders.pdf import PyPDFDirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS

""" IndexGenerator
class for generating the vector stores
"""
class IndexGenerator:
    """
    Class for generating and managing vector stores for text documents.

    Args:
        index_name (str, optional): Name of the index. Defaults to "index".

    Attributes:
        index_name (str): Name of the index.

    Methods:
        pre_process(pages: List[str]) -> List[str]:
            Pre-processes a list of text pages into chunks. Applies processing techniques to individual pages.
            Args:
                pages (List[str]): List of text pages.
            Returns:
                List[str]: List of pre-processed documents.

        generate_embeddings(embeddings, pages, company, year, report_type, quarter=None) -> FAISS:
            Generates a vector store for the given files.
            Args:
                embeddings: Embeddings model.
                pages (List[str]): List of text pages.
                company (str): Company name.
                year (int): Year of the report.
                report_type (str): Type of report.
                quarter (int, optional): Quarter of the report. Defaults to None.
            Returns:
                FAISS: Vector store.

        generate_vector_store_pdf_file(embeddings, file, company, year, report_type, quarter=None) -> FAISS:
            Generates a vector store for a PDF file.
            Args:
                embeddings: Embeddings model.
                file (str): Path to the PDF file.
                company (str): Company name.
                year (int): Year of the report.
                report_type (str): Type of report.
                quarter (int, optional): Quarter of the report. Defaults to None.
            Returns:
                FAISS: Vector store.

        generate_vector_store_pdf_dir(directory) -> FAISS:
            Generates a vector store for a directory of PDF files.
            Args:
                directory (str): Path to the directory containing PDF files.
            Returns:
                FAISS: Vector store.

        save_vector_store(vector_store, index_path):
            Saves the vector store to a local file.
            Args:
                vector_store (FAISS): Vector store.
                index_path (str): Path to save the vector store.

        merge_vector_stores(vector_store1, vector_store2) -> FAISS:
            Merges two vector stores.
            Args:
                vector_store1 (FAISS): First vector store.
                vector_store2 (FAISS): Second vector store.
            Returns:
                FAISS: Merged vector store.

        load_vector_store(path, embeddings) -> FAISS:
            Loads a vector store from a local file.
            Args:
                path (str): Path to the saved vector store.
                embeddings: Embeddings model.
            Returns:
                FAISS: Loaded vector store.
    """
    
    def __init__(self, index_name=None):
        """
        Initializes a IndexGenerator instance.

        Args:
            index_name (str, optional): Name of the vector store index. Defaults to "index".
        """
        if index_name == None:
            self.index_name = "index"
        else:
            self.index_name = index_name.replace("/", "_")

    # pre process files into chunks. Apply processing techniques into the individual
    def pre_process(self, pages):
        """
        Pre-processes files into chunks and applies processing techniques to individual chunks.

        Args:
            pages (list): List of document pages.

        Returns:
            list: Pre-processed documents.
        """
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
        """
        Generates a vector store for the given files.

        Args:
            embeddings: Embeddings model.
            pages (list): List of document pages.
            company (str): Company name.
            year (int): Year of the report.
            report_type (str): Type of report.
            quarter (int, optional): Quarter of the report. Defaults to None.

        Returns:
            FAISS: Vector store.
        """
        documents = self.pre_process(pages)

        for i, document in enumerate(documents):
            documents[i].metadata["company"] = company
            documents[i].metadata["year"] = year
            documents[i].metadata["report type"] = report_type
            if "page" in documents[i].metadata:
                documents[i].metadata["page"] += 1
            if quarter:
                documents[i].metadata["quarter"] = quarter
            
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store


    # PDF FILE: generate a vector store and return it
    def generate_vector_store_pdf_file(self, embeddings, file, company, year, report_type, quarter=None):
        """
        Generates a vector store from a PDF file.

        Args:
            embeddings: Embeddings model.
            file (str): Path to the PDF file.
            company (str): Company name.
            year (int): Year of the report.
            report_type (str): Type of report.
            quarter (int, optional): Quarter of the report. Defaults to None.

        Returns:
            FAISS: Vector store.
        """
        loader = PyPDFLoader(file)
        pages = loader.load()
        return self.generate_embeddings(embeddings, pages, company, year, report_type, quarter=None)


    # PDF DIRECTORY: generate a vector store and return it
    def generate_vector_store_pdf_dir(self, directory):
        """
        Generates a vector store from a directory of PDF files.

        Args:
            directory (str): Path to the directory containing PDF files.

        Returns:
            FAISS: Vector store.
        """
        loader = PyPDFDirectoryLoader(directory)
        files = loader.load()
        return self.generate_embeddings(files)


    # generate embeddings for pdfs in directory, save to index_path
    def save_vector_store(self, vector_store, index_path):
        """
        Saves the vector store to a local file.

        Args:
            vector_store (FAISS): Vector store.
            index_path (str): Path to save the vector store.

        Returns:
            None
        """
        vector_store.save_local(index_path, index_name=self.index_name)


    # merge file into an existing vector store
    def merge_vector_stores(self, vectore_store1, vector_store2):
        """
        Merges two vector stores into one.

        Args:
            vector_store1 (FAISS): First vector store.
            vector_store2 (FAISS): Second vector store.

        Returns:
            FAISS: Merged vector store.
        """
        vectore_store1.merge_from(vector_store2)
        return vectore_store1

    def load_vector_store(self, path, embeddings):
        """
        Loads a vector store from a local file.

        Args:
            path (str): Path to the saved vector store.
            embeddings: Embeddings model.

        Returns:
            FAISS: Loaded vector store.
        """
        return FAISS.load_local(folder_path=path,
                                 embeddings=embeddings,
                                 allow_dangerous_deserialization=True, 
                                 index_name=self.index_name)

