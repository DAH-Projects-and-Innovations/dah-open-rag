
from typing import List, Optional, Any, Tuple
from sentence_transformers import SentenceTransformer
from src.core.interfaces import IEmbedder


class LocalSentenceEmbedder(IEmbedder):
    # Cache partagé pour éviter de recharger le même modèle plusieurs fois
    _models_cache: dict[Tuple[str, Optional[str]], SentenceTransformer] = {}

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        **kwargs  # Capture normalize_embeddings, batch_size, etc.
    ):
        # Utiliser le cache pour ne charger le modèle qu'une seule fois par process
        key = (model_name, device)
        if key in LocalSentenceEmbedder._models_cache:
            self.model = LocalSentenceEmbedder._models_cache[key]
        else:
            self.model = SentenceTransformer(model_name, device=device)
            LocalSentenceEmbedder._models_cache[key] = self.model

        # Stockage des paramètres par défaut pour l'encodage
        self.default_kwargs = kwargs

    def embed_texts(self, texts: List[str], **kwargs) -> List[List[float]]:
        # On fusionne les paramètres de la config avec d'éventuels paramètres à l'appel
        params = {**self.default_kwargs, **kwargs}
        embeddings = self.model.encode(texts, **params)
        # SentenceTransformer renvoie déjà un numpy array; on convertit en listes
        return embeddings.tolist()

    def embed_query(self, query: str, **kwargs) -> List[float]:
        params = {**self.default_kwargs, **kwargs}
        return self.model.encode(query, **params).tolist()

    def get_dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()