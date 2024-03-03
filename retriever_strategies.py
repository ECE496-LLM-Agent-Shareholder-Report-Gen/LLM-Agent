
from abc import ABC, abstractmethod
import random

class RetrieverStrategy(ABC):

    @abstractmethod
    def retrieve_context(self, *args,  **kwargs):
        pass

    

class CompositeRetrieverStrategy(RetrieverStrategy):
    def __init__(self, strategies, metadata=["source"]):
        self.strategies = strategies
        self.metadata = metadata

    def retrieve_context(self, input, *args,**kwargs):
        if isinstance(input, dict):
            question = input["question"]
        else:
            question = input

        relevant_documents = None
        for strategy in self.strategies:
            relevant_documents = strategy.retrieve_context(question=question, relevant_documents=relevant_documents, *args, **kwargs)
        
        return self.combine_context(relevant_documents)
    
    def combine_context(self, documents):
        context = ""
        for doc in documents:
            show_metadata = ", ".join([f"{metadata_key} {doc.metadata[metadata_key]}" for metadata_key in self.metadata if metadata_key in doc.metadata])
            context += f"Excerpt from {show_metadata}:\n{doc.page_content}\n\n"
        
        return context


class SimpleRetrieverStrategy(RetrieverStrategy):

    def __init__(self, vectorstore=None, filters={}, k=8, fetch_k=20):
        self.vectorstore = vectorstore
        self.filters = filters
        self.k = k
        self.fetch_k = fetch_k

    def retrieve_context(self, question,  relevant_documents=None,  vectorstore=None, *args, **kwargs):
        vs = self.vectorstore
        if vectorstore != None:
            vs = vectorstore
        if vs == None:
            raise Exception("Error: No vectorstore given!")
        if relevant_documents == None:
            relevant_documents = vs.similarity_search(question, *args,**kwargs)

        return relevant_documents

    

class ReRankerRetrieverStrategy(RetrieverStrategy):
    def __init__(self, vectorstore, cross_encoder, filters={}, init_k=100):
        self.vectorstore = vectorstore
        self.cross_encoder = cross_encoder
        self.filters = filters
        self.init_k = init_k


    def retrieve_context(self, question,  relevant_documents=None,  vectorstore=None, k=8, *args, **kwargs):
        vs = self.vectorstore
        if vectorstore != None:
            vs = vectorstore
        if vs == None:
            raise Exception("Error: No vectorstore given!")
        if relevant_documents == None:
            # get relevant documents
            relevant_documents = vs.similarity_search(question,  k=self.init_k *args, **kwargs)

        scores = self.cross_encoder.predict([[question, document.page_content] for document in relevant_documents])

        for x in range(len(scores)):
            relevant_documents[x].metadata["ce_score"] = scores[x]
        
        relevant_documents.sort(key=lambda x: x.metadata["ce_score"], reverse=True)

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
            print("STRATEGY: Using self inserted vectorstore!")
        if vs == None:
            raise Exception("Error: No vectorstore given!")
        

        # get question count and update it
        qc = 0
        if question in self.QCTable:
            qc = self.QCTable[question]
            print("STRATEGY: Found in QCTable: ", question )
            self.QCTable[question] += 1
        else:
            self.QCTable[question] = 1

        skip = qc * k

        fetch_k_real = skip + k
        if relevant_documents == None:
            relevant_documents = vs.similarity_search(question, k=fetch_k_real)

        return relevant_documents[skip:fetch_k_real]
    

class StochasticRetrieverStrategy(RetrieverStrategy):

    def __init__(self, vectorstore=None, filters={}, k=6, fetch_k=20):
        self.vectorstore = vectorstore
        self.filters = filters
        self.k = k
        self.fetch_k = fetch_k
        self.QCTable = {}

    def retrieve_context(self, question,  relevant_documents=None,  vectorstore=None, k=None, fetch_k=None, *args, **kwargs):
        # set vectorstore
        vs = self.vectorstore
        if vectorstore != None:
            vs = vectorstore
            print("STRATEGY: Using self inserted vectorstore!")
        if vs == None:
            raise Exception("Error: No vectorstore given!")
        
        # set k
        k_real = self.k
        if k!=None:
            k_real = k
        
         # set fetch_k
        fetch_k_real = self.fetch_k
        if fetch_k!=None:
            fetch_k_real = fetch_k

        if k_real > fetch_k_real:
            k_real = fetch_k_real # can't choose more than the list

        # get question count and update it

        if relevant_documents == None:
            relevant_documents = vs.similarity_search(question, k=fetch_k_real)

        relevant_documents = random.sample(relevant_documents, k_real)
        return relevant_documents
    
