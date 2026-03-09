import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
from src.core.interfaces import IVectorStore
from src.core.models import Chunk, Document

class ChromaVectorStore(IVectorStore):
    def __init__(
        self, 
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "documents",
        distance_metric: str = "cosine", # Chroma utilise 'cosine', 'l2' ou 'ip'
        **kwargs
    ):
        # Initialisation du client persistant
        self.client = chromadb.PersistentClient(path=persist_directory)
        
        # Mappage de la métrique de distance (Chroma utilise "hnsw:space")
        # Les valeurs possibles sont "l2", "ip" ou "cosine"
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric}
        )

    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return

        # Chroma attend des listes séparées pour les IDs, embeddings, documents et métadonnées
        self.collection.add(
            ids=[getattr(c, 'chunk_id', f"id_{i}_{hash(c.content)}") for i, c in enumerate(chunks)],
            embeddings=[c.embedding for c in chunks],
            metadatas=[c.metadata if hasattr(c, 'metadata') else {} for c in chunks],
            documents=[c.content for c in chunks]
        )

    def search(self, query_embedding: List[float], top_k: int = 5, **kwargs) -> List[Document]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        # Reconstruction des objets Document à partir des résultats
        docs: List[Document] = []
        if results.get('documents'):
            docs_list = results.get('documents', [[]])[0]
            metadatas = results.get('metadatas', [[]])[0]
            embeddings = results.get('embeddings', [[]])[0] if results.get('embeddings') else [None]*len(docs_list)
            distances = results.get('distances', [[]])[0] if results.get('distances') else [None]*len(docs_list)

            for i in range(len(docs_list)):
                meta = metadatas[i] if i < len(metadatas) else {}
                # assigner le score depuis distances si disponible
                if distances and i < len(distances) and distances[i] is not None:
                    meta = {**meta, 'score': float(distances[i])}

                doc = Document(
                    content=docs_list[i],
                    metadata=meta
                )
                # si embedding fourni, on peut l'attacher comme attribut custom
                if embeddings and i < len(embeddings):
                    setattr(doc, 'embedding', embeddings[i])

                docs.append(doc)

        return docs

    def delete_collection(self, collection_name: str) -> None:
        try:
            self.client.delete_collection(name=collection_name)
        except Exception as e:
            print(f"Erreur lors de la suppression : {e}")

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        return {"total_vectors": self.collection.count()}