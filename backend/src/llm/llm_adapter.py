"""
Intégration de l'abstraction LLM (Tâche 4) avec l'interface ILLM (Tâche 1).
Adaptateur pour connecter les deux systèmes.
"""

from typing import List, Dict, Any, Optional
from ..core.interfaces import ILLM
from ..core.models import Document
from ..llm.base_llm import BaseLLM, LLMMessage
from ..llm.prompt_manager import PromptManager
from ..rag.models import RAGConfig


class LLMAdapter(ILLM):
    """
    Adaptateur pour connecter BaseLLM (Tâche 4) à l'interface ILLM (Tâche 1).
    
    Permet d'utiliser tous les providers LLM (OpenAI, Anthropic, Ollama, HF)
    dans le RAGPipeline de la Tâche 1.
    """
    
    def __init__(
        self,
        llm: BaseLLM,
        prompt_manager: PromptManager,
        config: Optional[RAGConfig] = None
    ):
        """
        Initialise l'adaptateur LLM.
        
        Args:
            llm: Instance de BaseLLM (OpenAI, Anthropic, Ollama, HF)
            prompt_manager: Gestionnaire de prompts
            config: Configuration RAG optionnelle
        """
        self.llm = llm
        self.prompt_manager = prompt_manager
        self.config = config or RAGConfig()
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Génère une réponse à partir d'un prompt (interface ILLM).
        
        Args:
            prompt: Prompt d'entrée
            **kwargs: Paramètres de génération
            
        Returns:
            Texte généré
        """
        messages = [LLMMessage(role="user", content=prompt)]
        
        response = self.llm.generate(
            messages=messages,
            temperature=kwargs.get('temperature', self.config.default_temperature),
            max_tokens=kwargs.get('max_tokens', self.config.default_max_tokens)
        )
        
        return response.content
    
    def generate_with_context(
        self,
        query: str,
        context: List[Document],
        **kwargs
    ) -> str:
        """
        Génère une réponse avec contexte (interface ILLM).
        
        Args:
            query: Question de l'utilisateur
            context: Documents de contexte
            **kwargs: Paramètres de génération
            
        Returns:
            Réponse générée avec citations si configuré
        """
        # Préparer les sources pour le prompt
        sources = self._format_sources(context, kwargs.get('include_citations', self.config.enable_citations))
        
        # Choisir le template de prompt approprié
        if kwargs.get('include_citations', self.config.enable_citations):
            system_template = self.config.system_prompt_template
            user_template = self.config.user_prompt_template
        else:
            system_template = "rag_system"
            user_template = "rag_user"
        
        # Rendre les prompts
        try:
            if kwargs.get('include_citations', self.config.enable_citations):
                system_prompt = self.prompt_manager.render_template(
                    system_template,
                    sources=sources
                )
            else:
                system_prompt = self.prompt_manager.render_template(
                    system_template,
                    context=sources
                )
            
            user_prompt = self.prompt_manager.render_template(
                user_template,
                question=query
            )
        except Exception as e:
            # Fallback sur des prompts simples
            system_prompt = f"Answer the question based on this context:\n\n{sources}"
            user_prompt = f"Question: {query}"
        
        # Créer les messages
        messages: List[LLMMessage] = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))

        # Insérer l'historique de conversation si fourni (ex: chat_history envoyé par le frontend)
        conv_history = kwargs.get('conversation_history') or kwargs.get('chat_history')
        if conv_history and isinstance(conv_history, list):
            for m in conv_history:
                if isinstance(m, LLMMessage):
                    messages.append(m)
                else:
                    role = m.get('role', 'user') if isinstance(m, dict) else 'user'
                    content = m.get('content', '') if isinstance(m, dict) else str(m)
                    messages.append(LLMMessage(role=role, content=content))

        # Enfin, ajouter le prompt utilisateur actuel
        messages.append(LLMMessage(role="user", content=user_prompt))
        
        # Générer la réponse
        response = self.llm.generate(
            messages=messages,
            temperature=kwargs.get('temperature', self.config.default_temperature),
            max_tokens=kwargs.get('max_tokens', self.config.default_max_tokens)
        )
        
        return response.content
    
    def _format_sources(self, documents: List[Document], include_citations: bool = True) -> str:
        """
        Formate les documents sources pour le prompt.
        
        Args:
            documents: Liste de documents
            include_citations: Si True, numéroter les sources
            
        Returns:
            Sources formatées
        """
        if not documents:
            return "No context available."
        
        if include_citations:
            # Format avec numéros de citation [1], [2], etc.
            formatted = []
            for i, doc in enumerate(documents, start=1):
                header = f"[{i}]"
                if 'source' in doc.metadata:
                    header += f" {doc.metadata['source']}"
                if 'title' in doc.metadata:
                    header += f" - {doc.metadata['title']}"
                
                formatted.append(f"{header}\n{doc.content}")
            
            return "\n\n".join(formatted)
        else:
            # Format simple sans citations
            return "\n\n".join(doc.content for doc in documents)
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration actuelle."""
        return {
            'llm': self.llm.get_config(),
            'rag_config': self.config.to_dict() if hasattr(self.config, 'to_dict') else {}
        }


def create_llm_adapter(
    provider: str,
    model: str,
    prompt_manager: PromptManager,
    config: Optional[RAGConfig] = None,
    **llm_kwargs
) -> LLMAdapter:
    """
    Factory pour créer un LLMAdapter configuré.
    
    Args:
        provider: Provider LLM ('openai', 'anthropic', 'ollama', 'huggingface')
        model: Nom du modèle
        prompt_manager: Gestionnaire de prompts
        config: Configuration RAG
        **llm_kwargs: Arguments pour le LLM (api_key, temperature, etc.)
        
    Returns:
        LLMAdapter configuré
    """
    from ..llm.base_llm import create_llm
    
    # Créer le LLM de base
    llm = create_llm(provider=provider, model=model, **llm_kwargs)
    
    # Créer l'adaptateur
    return LLMAdapter(
        llm=llm,
        prompt_manager=prompt_manager,
        config=config
    )