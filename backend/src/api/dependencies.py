from src.core.factory import RAGPipelineFactory
from src.implementations import register_all_components
import functools

@functools.lru_cache()
def get_pipeline():
    register_all_components()
    config = RAGPipelineFactory.load_config("configs/free.yaml")
    return RAGPipelineFactory.create_from_config(config)
