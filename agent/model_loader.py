from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

import os
import sys

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

class EmbeddingsLoader:
    """
    A class for loading embeddings using Hugging Face models.

    Attributes:
        model_name (str): Default model name for embeddings.

    Methods:
        __init__(self, **kwargs):
            Initializes an EmbeddingsLoader instance.
        
        load_bge(self, **kwargs):
            Loads embeddings using the specified model.

    Example:
        loader = EmbeddingsLoader()
        embeddings = loader.load_bge(model_name="BAAI/bge-large-en-v1.5")
    """

    model_name = "BAAI/bge-large-en-v1.5"
    def __init__(self, **kwargs):
        """
        Initializes an EmbeddingsLoader instance.

        Args:
            **kwargs: Additional keyword arguments for customization.
        """
        
        embeddings_path = config.DEFAULT_EMBEDDINGS_PATH
       
        self.real_embeddings_args = {
            "model_kwargs": {'device': 'cuda'},
            "encode_kwargs": {'normalize_embeddings': True},
            "model_name": embeddings_path
        }

        self.real_embeddings_args.update(kwargs)

    # loads embeddings
    def load_bge(self, **kwargs):
        """
        Loads embeddings using the specified model.

        Args:
            **kwargs: Additional keyword arguments for customization.

        Returns:
            HuggingFaceEmbeddings: Loaded embeddings.
        """
        self.real_embeddings_args.update(kwargs)
        return HuggingFaceEmbeddings(**self.real_embeddings_args)  
