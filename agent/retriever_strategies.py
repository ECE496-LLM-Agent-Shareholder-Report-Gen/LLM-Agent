from abc import ABC, abstractmethod
import random

class RetrieverStrategy(ABC):
    """
    Abstract base class for retriever strategies.

    Attributes:
        vectorstore: The vectorstore used for similarity search.
        skip: The number of documents to skip during retrieval.
    """

    @abstractmethod
    def retrieve_context(self, *args,  **kwargs):
        """
        Retrieve context based on the given input.

        Args:
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Document]: List of relevant documents.
        """
        pass

    def set_vectorstore(self, vectorstore):
        """
        Set the vectorstore for similarity search.

        Args:
            vectorstore: The vectorstore to set.
        """
        self.vectorstore = vectorstore

    def set_skip(self, skip):
        """
        Set the number of documents to skip during retrieval.

        Args:
            skip: The number of documents to skip.
        """
        self.skip = skip

    

class CompositeRetrieverStrategy(RetrieverStrategy):
    """
    Composite retriever strategy that combines multiple retriever strategies.

    Attributes:
        strategies: List of retriever strategies to combine.
        metadata: List of metadata keys to include in the context.
        recent_context: The combined context from all strategies.
    """
    def __init__(self, strategies, metadata=["source"]):
        """
        Initialize the CompositeRetrieverStrategy.

        Args:
            strategies: List of retriever strategies.
            metadata: List of metadata keys to include in the context.
        """
        self.strategies = strategies
        self.metadata = metadata
        self.recent_context = ""

    def retrieve_context(self, input, *args,**kwargs):
        """
        Retrieve context by combining results from multiple strategies.

        Args:
            input: The input (question or dictionary with "question" key).
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Combined context.
        """
        if isinstance(input, dict):
            question = input["question"]
        else:
            question = input
        relevant_documents = None
        for strategy in self.strategies:
            relevant_documents = strategy.retrieve_context(question=question, relevant_documents=relevant_documents, *args, **kwargs)
        self.recent_context += f"<b><i>Question: {question}</i></b>\n\n"
        return self.combine_context(relevant_documents)
    
    def combine_context(self, documents):
        """
        Combine the context from each Langchain document.

        Args:
            documents (List[Document]): List of Langchain documents

        Returns:
            the combined context, with metadata
        """
        context = ""
        for doc in documents:
            show_metadata = ", ".join([f"{metadata_key} {doc.metadata[metadata_key]}" for metadata_key in self.metadata if metadata_key in doc.metadata])
            context += f"Excerpt from {show_metadata}:\n\n[{doc.page_content}]\n\n\n\n"
        self.recent_context += context
        self.recent_context += "=" * 83 
        self.recent_context += "\n\n"
        return context
    
    def get_recent_context(self):
        """
        Get the combined context from all strategies.

        Returns:
            str: Combined context.
        """
        return self.recent_context

    def clear_recent_context(self):
        """
        Clear the combined context.
        """
        self.recent_context = ""

    def set_vectorstore(self, vectorstore):
        """
        Set the vectorstore for all strategies.

        Args:
            vectorstore: The vectorstore to set.
        """
        for strategy in self.strategies:
            strategy.set_vectorstore(vectorstore)

    def set_skip(self, skip):
        """
        Set the number of documents to skip for all strategies.

        Args:
            skip: The number of documents to skip.
        """
        for strategy in self.strategies:
            strategy.set_skip(skip)


class SimpleRetrieverStrategy(RetrieverStrategy):
    """
    Simple retriever strategy based on similarity search.

    Attributes:
        vectorstore: The vectorstore used for similarity search.
        filters: Additional filters for retrieval.
        k: Number of relevant documents to retrieve.
        fetch_k: Number of documents to fetch from similarity search.
    """

    def __init__(self, vectorstore=None, filters={}, k=8, fetch_k=20):
        """
        Initialize the SimpleRetrieverStrategy.

        Args:
            vectorstore: The vectorstore for similarity search.
            filters: Additional filters for retrieval.
            k: Number of relevant documents to retrieve.
            fetch_k: Number of documents to fetch from similarity search.
        """
        self.vectorstore = vectorstore
        self.filters = filters
        self.k = k
        self.fetch_k = fetch_k
        self.skip = 0

    def retrieve_context(self, question,  relevant_documents=None,  vectorstore=None, k=None, *args, **kwargs):
        """
        Retrieve context using similarity search.

        Args:
            question: The input question.
            relevant_documents: List of previously retrieved relevant documents.
            vectorstore: The vectorstore for similarity search.
            k: Number of relevant documents to retrieve.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            List[Document]: List of relevant documents.
        """
        vs = self.vectorstore
        if vectorstore != None:
            vs = vectorstore
        if vs == None:
            return []
        if k == None:
            k = self.k
        if relevant_documents == None:
            relevant_documents = vs.similarity_search(question, k=k+self.skip*k, *args,**kwargs)

        return relevant_documents[self.skip*k:]

    

class ReRankerRetrieverStrategy(RetrieverStrategy):
    """
    Re-ranker retriever strategy using a cross-encoder model.

    Attributes:
        vectorstore: The vectorstore used for similarity search.
        cross_encoder: The cross-encoder model for re-ranking.
        filters: Additional filters for retrieval.
        k: Number of relevant documents to retrieve.
        init_k: Initial number of documents to fetch from similarity search.
    """
    def __init__(self,  cross_encoder,vectorstore=None, filters={}, k=8, init_k=100):
        """
        Initialize the ReRankerRetrieverStrategy.

        Args:
            cross_encoder (CrossEncoder): The cross-encoder model for re-ranking.
            vectorstore (VectorStore, optional): The vectorstore used for similarity search. Defaults to None.
            filters (dict, optional): Additional filters for retrieval. Defaults to {}.
            k (int, optional): Number of relevant documents to retrieve. Defaults to 8.
            init_k (int, optional): Initial number of documents to fetch from similarity search. Defaults to 100.
        """
        self.vectorstore = vectorstore
        self.cross_encoder = cross_encoder
        self.filters = filters
        self.k = k
        self.init_k = init_k
        self.skip = 0


    def retrieve_context(self, question,  relevant_documents=None,  vectorstore=None, k=None, *args, **kwargs):
        """
        Retrieve relevant documents using the re-ranker strategy.

        Args:
            question (str): The input question.
            relevant_documents (list, optional): List of relevant documents. Defaults to None.
            vectorstore (VectorStore, optional): The vectorstore used for retrieval. Defaults to None.
            k (int, optional): Number of relevant documents to retrieve. Defaults to None.

        Returns:
            list: List of relevant documents.
        """
        vs = self.vectorstore
        if vectorstore != None:
            vs = vectorstore
        if k == None:
            k = self.k
        if vs == None:
            return []
        if relevant_documents == None:
            # get relevant documents
            relevant_documents = vs.similarity_search(question,  k=self.init_k+self.skip*k, *args, **kwargs)
            relevant_documents = relevant_documents[self.skip*k:]
        scores = self.cross_encoder.predict([[question, document.page_content] for document in relevant_documents])

        for x in range(len(scores)):
            relevant_documents[x].metadata["ce_score"] = scores[x]
        
        relevant_documents.sort(key=lambda x: x.metadata["ce_score"], reverse=True)

        # use this functions k arg
       

        if k > len(scores):
            k = len(scores)

        relevant_documents = relevant_documents[:k]

        return relevant_documents



class NextRetrieverStrategy(RetrieverStrategy):

    def __init__(self, vectorstore=None, filters={}):
        self.vectorstore = vectorstore
        self.filters = filters
        self.QCTable = {}


    def retrieve_context(self, question,  relevant_documents=None,  vectorstore=None, k=8, *args, **kwargs):
        # set vectorstore
        vs = self.vectorstore
        if vectorstore != None:
            vs = vectorstore
        if vs == None:
            return []
        

        # get question count and update it
        qc = 0
        if question in self.QCTable:
            qc = self.QCTable[question]
            self.QCTable[question] += 1
        else:
            self.QCTable[question] = 1

        skip = qc * k

        fetch_k_real = skip + k
        if relevant_documents == None:
            relevant_documents = vs.similarity_search(question, k=fetch_k_real)

        return relevant_documents[skip:fetch_k_real]
    

class StochasticRetrieverStrategy(RetrieverStrategy):

    def __init__(self, vectorstore=None, filters={}, k=8, fetch_k=20):
        self.vectorstore = vectorstore
        self.filters = filters
        self.k = k
        self.fetch_k = fetch_k
        self.skip = 0

    def retrieve_context(self, question,  relevant_documents=None,  vectorstore=None, k=None, fetch_k=None, *args, **kwargs):
        # set vectorstore
        vs = self.vectorstore
        if vectorstore != None:
            vs = vectorstore
        if vs == None:
            return []
        
        # set k
        if k == None:
            k = self.k
        
         # set fetch_k
        if fetch_k == None:
            fetch_k = self.fetch_k

        if k > fetch_k:
            k = fetch_k # can't choose more than the list

        # get question count and update it

        if relevant_documents == None:
            relevant_documents = vs.similarity_search(question, k=fetch_k+self.skip*k)
            relevant_documents = relevant_documents[self.skip*k:]

        relevant_documents = random.sample(relevant_documents, k)
        return relevant_documents
    
