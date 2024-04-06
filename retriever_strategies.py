
from abc import ABC, abstractmethod
import random

class RetrieverStrategy(ABC):

    @abstractmethod
    def retrieve_context(self, *args,  **kwargs):
        pass

    def set_vectorstore(self, vectorstore):
        self.vectorstore = vectorstore

    def set_skip(self, skip):
        self.skip = skip

    

class CompositeRetrieverStrategy(RetrieverStrategy):
    def __init__(self, strategies, metadata=["source"]):
        self.strategies = strategies
        self.metadata = metadata
        self.recent_context = ""

    def retrieve_context(self, input, *args,**kwargs):
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
        context = ""
        for doc in documents:
            show_metadata = ", ".join([f"{metadata_key} {doc.metadata[metadata_key]}" for metadata_key in self.metadata if metadata_key in doc.metadata])
            context += f"Excerpt from {show_metadata}:\n\n{doc.page_content}\n\n\n"
        self.recent_context += context
        self.recent_context += "=" * 90 
        self.recent_context += "\n\n"
        return context
    
    def get_recent_context(self):
        return self.recent_context

    def clear_recent_context(self):
        self.recent_context = ""

    def set_vectorstore(self, vectorstore):
        for strategy in self.strategies:
            strategy.set_vectorstore(vectorstore)

    def set_skip(self, skip):
        for strategy in self.strategies:
            strategy.set_skip(skip)


class SimpleRetrieverStrategy(RetrieverStrategy):

    def __init__(self, vectorstore=None, filters={}, k=8, fetch_k=20):
        self.vectorstore = vectorstore
        self.filters = filters
        self.k = k
        self.fetch_k = fetch_k
        self.skip = 0

    def retrieve_context(self, question,  relevant_documents=None,  vectorstore=None, k=None, *args, **kwargs):
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
    def __init__(self,  cross_encoder,vectorstore=None, filters={}, k=8, init_k=100):
        self.vectorstore = vectorstore
        self.cross_encoder = cross_encoder
        self.filters = filters
        self.k = k
        self.init_k = init_k
        self.skip = 0


    def retrieve_context(self, question,  relevant_documents=None,  vectorstore=None, k=None, *args, **kwargs):
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
    
