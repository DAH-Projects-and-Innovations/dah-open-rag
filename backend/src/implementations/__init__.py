from src.core.factory import RAGPipelineFactory
from src.Loaders.text_loader import UnifiedDocumentLoader
from src.retrieval.retrieval_strategy import RetrievalStrategy
from src.Chunkers.basic_chunker import ConfigurableChunker
from src.Embedders.dummy_embedder import LocalSentenceEmbedder
from src.vectorstores.chroma_store import ChromaVectorStore
from src.retrieval.reranker import CrossEncoderReranker
from src.llm.llm_factory import LLMAdapterFactory
from src.llm.prompt_manager import create_default_prompt_manager


def register_all_components():
    f = RAGPipelineFactory

    # Loaders & chunkers
    f.register_component("loaders",   "text_loader",     UnifiedDocumentLoader)
    f.register_component("chunkers",  "overlap_chunker",  ConfigurableChunker)

    # Embedders
    f.register_component("embedders", "sentence_transformers", LocalSentenceEmbedder)

    # Vector stores
    f.register_component("vector_stores", "chroma", ChromaVectorStore)

    # Retrievers
    f.register_component("retrievers", "vector_retriever", RetrievalStrategy)

    # Rerankers
    f.register_component("rerankers", "cross_encoder", CrossEncoderReranker)

    # LLMs — tous mappés sur LLMAdapterFactory (le provider est lu depuis le YAML)
    f.register_component("llms", "ollama",      LLMAdapterFactory)   # config free
    f.register_component("llms", "mistral",     LLMAdapterFactory)   # config hybrid
    f.register_component("llms", "gemini",      LLMAdapterFactory)   # config hybrid (alt)
    f.register_component("llms", "openai",      LLMAdapterFactory)   # optionnel
    f.register_component("llms", "anthropic",   LLMAdapterFactory)   # optionnel
    f.register_component("llms", "huggingface", LLMAdapterFactory)   # optionnel

    # Prompt managers
    f.register_component("prompt_managers", "default",
                         lambda **_: create_default_prompt_manager())
