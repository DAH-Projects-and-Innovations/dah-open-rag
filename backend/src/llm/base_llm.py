"""
Abstraction LLM pour le moteur RAG.
Support d'APIs (OpenAI, Anthropic) et modèles open-source (Ollama, HuggingFace).
Pass
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import os
from dotenv import load_dotenv

load_dotenv()



class LLMProvider(Enum):
    """Providers LLM supportés."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"
    MISTRAL = "mistral"


@dataclass
class LLMConfig:
    """Configuration pour un LLM."""
    provider: LLMProvider
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    
    # Paramètres spécifiques
    stream: bool = False
    timeout: int = 60

    num_ctx: int = 4096
    repeat_penalty: float = 1.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire."""
        return {
            'provider': self.provider.value,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'top_p': self.top_p,
            'frequency_penalty': self.frequency_penalty,
            'presence_penalty': self.presence_penalty,
            'stream': self.stream,
            'timeout': self.timeout
        }


@dataclass
class LLMMessage:
    """Message pour le LLM."""
    role: str  # 'system', 'user', 'assistant'
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        """Convertit en dictionnaire."""
        return {'role': self.role, 'content': self.content}


@dataclass
class LLMResponse:
    """Réponse du LLM."""
    content: str
    model: str
    usage: Dict[str, int]  # tokens utilisés
    finish_reason: str
    metadata: Dict[str, Any]
    
    def __str__(self) -> str:
        return self.content


class BaseLLM(ABC):
    """
    Classe abstraite pour tous les LLMs.
    
    Fournit une interface unifiée pour différents providers.
    """
    
    def __init__(self, config: LLMConfig):
        """
        Initialise le LLM.
        
        Args:
            config: Configuration du LLM
        """
        self.config = config
        self._client = None
    
    @abstractmethod
    def _initialize_client(self) -> None:
        """Initialise le client du provider (lazy loading)."""
        pass
    
    @abstractmethod
    def generate(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """
        Génère une réponse.
        
        Args:
            messages: Liste de messages (conversation)
            **kwargs: Paramètres additionnels (override config)
            
        Returns:
            Réponse du LLM
        """
        pass
    
    def chat(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[LLMMessage]] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Interface simplifiée pour chat.
        
        Args:
            user_message: Message de l'utilisateur
            system_prompt: Prompt système optionnel
            conversation_history: Historique de conversation
            **kwargs: Paramètres additionnels
            
        Returns:
            Réponse du LLM
        """
        messages = []
        
        # Ajouter le system prompt
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        
        # Ajouter l'historique
        if conversation_history:
            messages.extend(conversation_history)
        
        # Ajouter le message utilisateur
        messages.append(LLMMessage(role="user", content=user_message))
        
        return self.generate(messages, **kwargs)
    
    def get_config(self) -> Dict[str, Any]:
        """Retourne la configuration actuelle."""
        return self.config.to_dict()
    
    def update_config(self, **kwargs) -> None:
        """
        Met à jour la configuration.
        
        Args:
            **kwargs: Paramètres à mettre à jour
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


class OpenAILLM(BaseLLM):
    """LLM pour OpenAI (GPT-3.5, GPT-4, etc.)."""
    
    def _initialize_client(self) -> None:
        """Initialise le client OpenAI."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.config.api_key,
                    base_url=self.config.api_base,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ImportError(
                    "openai package required. Install with: pip install openai"
                )
    
    def generate(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """Génère une réponse via OpenAI."""
        self._initialize_client()
        
        # Merger les kwargs avec la config
        params = {
            'model': kwargs.get('model', self.config.model),
            'temperature': kwargs.get('temperature', self.config.temperature),
            'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            'top_p': kwargs.get('top_p', self.config.top_p),
            'frequency_penalty': kwargs.get('frequency_penalty', self.config.frequency_penalty),
            'presence_penalty': kwargs.get('presence_penalty', self.config.presence_penalty),
        }
        
        # Convertir les messages
        messages_dict = [msg.to_dict() for msg in messages]
        
        # Appeler l'API
        response = self._client.chat.completions.create(
            messages=messages_dict,
            **params
        )
        
        # Extraire la réponse
        choice = response.choices[0]
        
        return LLMResponse(
            content=choice.message.content,
            model=response.model,
            usage={
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            },
            finish_reason=choice.finish_reason,
            metadata={'response_id': response.id}
        )


class AnthropicLLM(BaseLLM):
    """LLM pour Anthropic (Claude)."""
    
    def _initialize_client(self) -> None:
        """Initialise le client Anthropic."""
        if self._client is None:
            try:
                from anthropic import Anthropic
                self._client = Anthropic(
                    api_key=self.config.api_key,
                    timeout=self.config.timeout
                )
            except ImportError:
                raise ImportError(
                    "anthropic package required. Install with: pip install anthropic"
                )
    
    def generate(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """Génère une réponse via Anthropic."""
        self._initialize_client()
        
        # Séparer system prompt et messages
        system_prompt = None
        conversation_messages = []
        
        for msg in messages:
            if msg.role == "system":
                system_prompt = msg.content
            else:
                conversation_messages.append(msg.to_dict())
        
        # Paramètres
        params = {
            'model': kwargs.get('model', self.config.model),
            'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            'temperature': kwargs.get('temperature', self.config.temperature),
            'top_p': kwargs.get('top_p', self.config.top_p),
        }
        
        if system_prompt:
            params['system'] = system_prompt
        
        # Appeler l'API
        response = self._client.messages.create(
            messages=conversation_messages,
            **params
        )
        
        return LLMResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                'prompt_tokens': response.usage.input_tokens,
                'completion_tokens': response.usage.output_tokens,
                'total_tokens': response.usage.input_tokens + response.usage.output_tokens
            },
            finish_reason=response.stop_reason,
            metadata={'response_id': response.id}
        )


class OllamaLLM(BaseLLM):
    """LLM pour Ollama (modèles open-source locaux)."""
    
    def _initialize_client(self) -> None:
        """Initialise le client Ollama."""
        if self._client is None:
            try:
                from ollama import Client
                base_url = self.config.api_base or "http://localhost:11434"
                self._client = Client(host=base_url)
            except ImportError:
                raise ImportError(
                    "ollama package required. Install with: pip install ollama"
                )
    
    def generate(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """Génère une réponse via Ollama."""
        self._initialize_client()
        
        # Convertir les messages
        messages_dict = [msg.to_dict() for msg in messages]
        
        # Paramètres
        params = {
            'model': kwargs.get('model', self.config.model),
            'messages': messages_dict,
            'options': {
                'temperature': kwargs.get('temperature', self.config.temperature),
                'num_predict': kwargs.get('max_tokens', self.config.max_tokens),
                'top_p': kwargs.get('top_p', self.config.top_p),
            }
        }
        
        # Appeler l'API
        response = self._client.chat(**params)
        
        return LLMResponse(
            content=response['message']['content'],
            model=response.get('model', self.config.model),
            usage={
                'prompt_tokens': response.get('prompt_eval_count', 0),
                'completion_tokens': response.get('eval_count', 0),
                'total_tokens': response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
            },
            finish_reason='stop',
            metadata={}
        )


class HuggingFaceLLM(BaseLLM):
    """LLM pour HuggingFace (modèles open-source)."""
    
    def _initialize_client(self) -> None:
        """Initialise le pipeline HuggingFace."""
        if self._client is None:
            try:
                from transformers import pipeline
                self._client = pipeline(
                    "text-generation",
                    model=self.config.model,
                    device_map="auto"
                )
            except ImportError:
                raise ImportError(
                    "transformers package required. Install with: pip install transformers torch"
                )
    
    def generate(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """Génère une réponse via HuggingFace."""
        self._initialize_client()
        
        # Formatter les messages en prompt
        prompt = self._format_messages_to_prompt(messages)
        
        # Paramètres
        max_new_tokens = kwargs.get('max_tokens', self.config.max_tokens)
        
        # Générer
        outputs = self._client(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=kwargs.get('temperature', self.config.temperature),
            top_p=kwargs.get('top_p', self.config.top_p),
            do_sample=True,
            return_full_text=False
        )
        
        generated_text = outputs[0]['generated_text']
        
        return LLMResponse(
            content=generated_text,
            model=self.config.model,
            usage={
                'prompt_tokens': len(prompt.split()),
                'completion_tokens': len(generated_text.split()),
                'total_tokens': len(prompt.split()) + len(generated_text.split())
            },
            finish_reason='stop',
            metadata={}
        )
    
    def _format_messages_to_prompt(self, messages: List[LLMMessage]) -> str:
        """Formate les messages en prompt pour HuggingFace."""
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        prompt_parts.append("Assistant:")
        return "\n\n".join(prompt_parts)

class MistralLLM(BaseLLM):
    """LLM pour Mistral AI."""
    
    def _initialize_client(self) -> None:
        """Initialise le client Mistral."""
        if self._client is None:
            try:
                from mistralai import Mistral
                self._client = Mistral(api_key=self.config.api_key)
            except ImportError:
                raise ImportError(
                    "mistralai package required. Install with: pip install mistralai"
                )
    
    def generate(
        self,
        messages: List[LLMMessage],
        **kwargs
    ) -> LLMResponse:
        """Génère une réponse via Mistral AI."""
        self._initialize_client()
        
        # Préparation des paramètres
        params = {
            'model': kwargs.get('model', self.config.model),
            'temperature': kwargs.get('temperature', self.config.temperature),
            'max_tokens': kwargs.get('max_tokens', self.config.max_tokens),
            'top_p': kwargs.get('top_p', self.config.top_p),
        }
        
        # Conversion des messages au format attendu par Mistral
        messages_dict = [msg.to_dict() for msg in messages]
        
        # Appel API
        response = self._client.chat.complete(
            messages=messages_dict,
            **params
        )
        
        choice = response.choices[0]
        
        return LLMResponse(
            content=choice.message.content,
            model=response.model,
            usage={
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            },
            finish_reason=choice.finish_reason,
            metadata={'response_id': response.id}
        )

def create_llm(
    provider: Union[str, LLMProvider],
    model: str,
    **kwargs
) -> BaseLLM:
    """
    Factory pour créer un LLM.
    
    Args:
        provider: Provider du LLM
        model: Nom du modèle
        **kwargs: Configuration additionnelle
        
    Returns:
        Instance de LLM
    """
    if isinstance(provider, str):
        provider = LLMProvider(provider)
    
    config = LLMConfig(provider=provider, model=model, **kwargs)
    
    if provider == LLMProvider.OPENAI:
        return OpenAILLM(config)
    elif provider == LLMProvider.ANTHROPIC:
        return AnthropicLLM(config)
    elif provider == LLMProvider.OLLAMA:
        return OllamaLLM(config)
    elif provider == LLMProvider.HUGGINGFACE:
        return HuggingFaceLLM(config)
    elif provider == LLMProvider.MISTRAL:  # Ajout
        return MistralLLM(config)
    else:
        raise ValueError(f"Unknown provider: {provider}")