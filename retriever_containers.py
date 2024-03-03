
from abc import ABC, abstractmethod

class RetrieverContainer(ABC):

    @abstractmethod
    def context_retriever(self, **kwargs):
        pass

    def combine_context(self, documents, metadata):
        context = ""
        for doc in documents:
            show_metadata = ", ".join([f"{metadata_key} {doc.metadata[metadata_key]}" for metadata_key in metadata])
            context += f"Excerpt from {show_metadata}:\n{doc.page_content}\n\n"
        
        return context


class SimpleRetrieverContainer(RetrieverContainer):

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def context_retriever(self, filters={}, k=8, metadata=["source"]):
        def wrapper_func(input):
            question = input["question"]
            relevant_documents = self.vectorstore.similarity_search(question, filters=filters, k=k)

            return self.combine_context(relevant_documents, metadata)
        return wrapper_func
    

class ReRankerRetrieverContainer(RetrieverContainer):
    def __init__(self, vectorstore, cross_encoder):
        self.vectorstore = vectorstore
        self.cross_encoder = cross_encoder

    def context_retriever(self, filters={}, k=8, init_k=100, metadata=["source"], **kwargs):
        if k > init_k:
            k = init_k
        def wrapper_func(input):
            # get relevant documents
            question = input["question"]
            relevant_documents = self.vectorstore.similarity_search(question, filters=filters, k=init_k)

            scores = self.cross_encoder.predict([[question, document.page_content] for document in relevant_documents])

            for x in range(len(scores)):
                relevant_documents[x].metadata["ce_score"] = scores[x]
            
            relevant_documents.sort(key=lambda x: x.metadata["ce_score"], reverse=True)

            relevant_documents = relevant_documents[:k]

            return self.combine_context(relevant_documents, metadata)
        return wrapper_func

