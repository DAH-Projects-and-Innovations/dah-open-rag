# from ..core.factory import RAGPipelineFactory
# from .openai_llm import OpenAILLM
# from .ollama_llm import LocalLLM
# from .mistral_llm import MistralLLM

# Enregistrement automatique des composants LLM
# RAGPipelineFactory.register_component('llms', 'openai', OpenAILLM)
# RAGPipelineFactory.register_component('llms', 'local', LocalLLM)
# RAGPipelineFactory.register_component('llms', 'mistral', MistralLLM)

#__all__ = ['OpenAILLM', 'LocalLLM', 'MistralLLM']


"""
Module LLM - Abstraction pour différents providers.
Support d'OpenAI, Anthropic, Ollama, HuggingFace.
"""

from .base_llm import (
    BaseLLM,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMProvider,
    OpenAILLM,
    AnthropicLLM,
    OllamaLLM,
    HuggingFaceLLM,
    create_llm
)

from .llm_adapter import LLMAdapter, create_llm_adapter

from .prompt_manager import (
    PromptTemplate,
    PromptManager,
    create_default_prompt_manager,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
    RAG_WITH_CITATIONS_SYSTEM,
    RAG_WITH_CITATIONS_USER,
    RAG_SAFE_SYSTEM
)

__all__ = [
    # Base LLM
    'BaseLLM',
    'LLMConfig',
    'LLMMessage',
    'LLMResponse',
    'LLMProvider',
    
    # Providers
    'OpenAILLM',
    'AnthropicLLM',
    'OllamaLLM',
    'HuggingFaceLLM',
    
    # Factory
    'create_llm',
    
    # Prompt Management
    'PromptTemplate',
    'PromptManager',
    'create_default_prompt_manager',
    
    # Default Prompts
    'RAG_SYSTEM_PROMPT',
    'RAG_USER_PROMPT',
    'RAG_WITH_CITATIONS_SYSTEM',
    'RAG_WITH_CITATIONS_USER',
    'RAG_SAFE_SYSTEM',

    'LLMAdapter',
    'create_llm_adapter'
]