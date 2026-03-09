import os
import functools
from src.core.factory import RAGPipelineFactory
from src.implementations import register_all_components


@functools.lru_cache()
def get_pipeline():
    register_all_components()

    # Choisir la configuration via la variable d'environnement RAG_CONFIG.
    # Valeurs acceptées : "free" (Ollama local) | "hybrid" (Mistral/Gemini)
    config_name = os.getenv("RAG_CONFIG", "free").strip().lower()
    config_path = f"configs/{config_name}.yaml"

    config = RAGPipelineFactory.load_config(config_path)
    return RAGPipelineFactory.create_from_config(config)
