from langchain_community.llms import LlamaCpp
from langchain_community.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.embeddings import HuggingFaceEmbeddings
import torch

# class for loading in the language model and embeddings model
class ModelLoader:
    AVAILABLE_MODELS = {"llama-2-13b": {
                        "model_path": "/groups/acmogrp/Large-Language-Model-Agent/language_models/llama-2-13b/llama-2-13b.gguf.q8_0.bin",
                        "n_gpu_layers": 50,
                        "n_ctx": 5000
                    },
                    "llama-2-13b-chat": {
                        "model_path": "/groups/acmogrp/Large-Language-Model-Agent/language_models/llama-2-13b/llama-2-13b-chat.gguf.q8_0.bin",
                        "n_gpu_layers": 80,
                        "n_ctx": 4000
                    },
                    "mixtral-instruct": {
                        "model_path": "/groups/acmogrp/Large-Language-Model-Agent/language_models/mistral/mixtral-8x7b-instruct-v0.1.gguf.Q5_0.bin",
                        "n_gpu_layers": 15,
                        "n_ctx": 32000
                    },
                    "guanaco-33b.Q2_K": {
                        "model_path": "/groups/acmogrp/Large-Language-Model-Agent/language_models/test_models/guanaco-33b.Q2_K.gguf.q8_0.bin",
                        "n_gpu_layers": 60,
                        "n_ctx": 5000
                    },
                }
    DEFAULT_MODEL = "llama-2-13b-chat"


    language_model = None

    def __init__(self, model_name=DEFAULT_MODEL, **kwargs):
        if (self.AVAILABLE_MODELS[model_name] == None):
            raise ValueError(
                "model_name is not in the list of available models. Available models are: ", self.AVAILABLE_MODELS.keys())
        self.model_dir = self.AVAILABLE_MODELS[model_name]
        self.model_name = model_name
        self.language_model = self.load(**kwargs)

    def load(self, streaming=True, **kwargs):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return LlamaCpp(
            **self.AVAILABLE_MODELS[self.model_name],
            n_batch=512,
            max_tokens=1200,
            streaming=streaming,
            #f16_kv = True,
            callback_manager=callback_manager,
            verbose=False, # Verbose is required to pass to the callback manager
            **kwargs
        )

    
class LLMModelLoader:
    AVAILABLE_MODELS = {"llama-2-13b": {
                        "model_path": "/groups/acmogrp/Large-Language-Model-Agent/language_models/llama-2-13b/llama-2-13b.gguf.q8_0.bin",
                        "n_gpu_layers": 50,
                        "n_ctx": 5000
                    },
                    "llama-2-13b-chat": {
                        "model_path": "/groups/acmogrp/Large-Language-Model-Agent/language_models/llama-2-13b/llama-2-13b-chat.gguf.q8_0.bin",
                        "n_gpu_layers": 80,
                        "n_ctx": 4000
                    },
                    "mixtral-instruct": {
                        "model_path": "/groups/acmogrp/Large-Language-Model-Agent/language_models/mistral/mixtral-8x7b-instruct-v0.1.gguf.Q5_0.bin",
                        "n_gpu_layers": 15,
                        "n_ctx": 32000
                    },
                    "guanaco-33b.Q2_K": {
                        "model_path": "/groups/acmogrp/Large-Language-Model-Agent/language_models/test_models/guanaco-33b.Q2_K.gguf.q8_0.bin",
                        "n_gpu_layers": 60,
                        "n_ctx": 5000
                    },
                }
    DEFAULT_MODEL = "llama-2-13b-chat"


    language_model = None

    def __init__(self, model_name=DEFAULT_MODEL, **kwargs):
        if (self.AVAILABLE_MODELS[model_name] == None):
            raise ValueError(
                "model_name is not in the list of available models. Available models are: ", self.AVAILABLE_MODELS.keys())
        self.model_dir = self.AVAILABLE_MODELS[model_name]
        self.model_name = model_name

    def load(self, streaming=True, **kwargs):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        return LlamaCpp(
            **self.AVAILABLE_MODELS[self.model_name],
            n_batch=512,
            max_tokens=1200,
            streaming=streaming,
            #f16_kv = True,
            callback_manager=callback_manager,
            verbose=False, # Verbose is required to pass to the callback manager
            **kwargs
        )

    def load_ollama(self,model="llama2", **kwargs):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        return ChatOllama(model=model, 
                        #   callback_manager=callback_manager, 
                          **kwargs)
    """
    def load_ollama(self,model="llama-2-13b-chat", **kwargs):
        callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

        return ChatOllama(model=model, callback_manager=callback_manager, **kwargs)
"""
   

class EmbeddingsLoader:
    def __init__(self, **kwargs):
        embeddings_path = "/groups/acmogrp/Large-Language-Model-Agent/language_models/word_embeddings/BAAI_bge-large-en-v1.5"
       
        real_embeddings_args = {
            "model_kwargs": {'device': 'cuda'},
            "encode_kwargs": {'normalize_embeddings': True},
            "model_name": embeddings_path
        }

        real_embeddings_args.update(kwargs)

    # loads embeddings
    def load_bge(self, **kwargs):
        return HuggingFaceEmbeddings(**kwargs)  
