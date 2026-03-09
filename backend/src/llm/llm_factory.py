"""
Factory function pour créer LLMAdapter depuis la configuration.
Compatible avec RAGPipelineFactory.
"""

from src.llm.base_llm import create_llm
from src.llm.prompt_manager import PromptManager
from src.llm.llm_adapter import LLMAdapter
from src.rag.models import RAGConfig
from typing import Optional


def create_llm_adapter_from_config(
    provider: str,
    model: str,
    prompt_manager: PromptManager,
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    top_p: float = 1.0,
    # Paramètres RAG config
    enable_citations: bool = False,
    system_prompt_template: str = "rag_system",
    user_prompt_template: str = "rag_user",
    prevent_hallucinations: bool = True,
    # Paramètres additionnels Ollama/autres
    **kwargs
):
    """
    Crée un LLMAdapter configuré depuis des paramètres de config YAML.
    
    Args:
        provider: Provider LLM ('ollama', 'openai', 'anthropic', 'huggingface')
        model: Nom du modèle
        prompt_manager: Gestionnaire de prompts
        base_url: URL de base pour l'API (Ollama)
        temperature: Température de génération
        max_tokens: Nombre max de tokens
        top_p: Top-p sampling
        enable_citations: Activer les citations
        system_prompt_template: Template système
        user_prompt_template: Template utilisateur
        prevent_hallucinations: Protection anti-hallucination
        **kwargs: Paramètres additionnels (num_ctx, repeat_penalty, etc.)
        
    Returns:
        LLMAdapter configuré
    """
    # Créer la config RAG
    rag_config = RAGConfig(
        enable_citations=enable_citations,
        system_prompt_template=system_prompt_template,
        user_prompt_template=user_prompt_template,
        prevent_hallucinations=prevent_hallucinations,
        default_temperature=temperature,
        default_max_tokens=max_tokens
    )
    
    # Préparer les kwargs pour create_llm
    llm_kwargs = {
        'temperature': temperature,
        'max_tokens': max_tokens,
        'top_p': top_p
    }
    
    # Ajouter base_url si fourni (pour Ollama)
    if base_url:
        llm_kwargs['api_base'] = base_url
    
    # Ajouter les paramètres additionnels
    llm_kwargs.update(kwargs)
    
    # Créer le LLM de base
    llm = create_llm(
        provider=provider,
        model=model,
        **llm_kwargs
    )
    
    # Créer l'adaptateur
    return LLMAdapter(
        llm=llm,
        prompt_manager=prompt_manager,
        config=rag_config
    )


class LLMAdapterFactory:
    """
    Classe factory pour créer LLMAdapter.
    Compatible avec le système de registre de RAGPipelineFactory.
    """
    
    def __init__(
        self,
        provider: str,
        model: str,
        prompt_manager: PromptManager,
        **kwargs
    ):
        """
        Initialise et crée un LLMAdapter.
        
        Args:
            provider: Provider LLM
            model: Nom du modèle
            prompt_manager: Gestionnaire de prompts
            **kwargs: Autres paramètres
        """
        self.adapter = create_llm_adapter_from_config(
            provider=provider,
            model=model,
            prompt_manager=prompt_manager,
            **kwargs
        )
    
    def __getattr__(self, name):
        """Délègue tous les appels à l'adaptateur."""
        return getattr(self.adapter, name)
    
    # Implémenter l'interface ILLM explicitement
    def generate(self, prompt: str, **kwargs) -> str:
        """Génère une réponse."""
        return self.adapter.generate(prompt, **kwargs)
    
    def generate_with_context(self, query: str, context, **kwargs) -> str:
        """Génère une réponse avec contexte."""
        return self.adapter.generate_with_context(query, context, **kwargs)
    
    def get_config(self):
        """Retourne la configuration."""
        return self.adapter.get_config()