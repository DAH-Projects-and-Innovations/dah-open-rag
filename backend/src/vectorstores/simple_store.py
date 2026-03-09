import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Optional
from src.core.interfaces import IVectorStore
from src.core.models import Chunk

class FAISSVectorStore(IVectorStore):
    def __init__(
        self, 
        dimension: int = 1024, # BGE-large-en-v1.5 utilise 1024
        persist_directory: str = "./data/faiss_db", # Mappé depuis votre config
        collection_name: str = "documents", # Accepté mais optionnel pour FAISS
        distance_metric: str = "cosine",
        **kwargs
    ):
        # On utilise persist_directory s'il est fourni, sinon storage
        self.folder_path = persist_directory
        self.index_path = os.path.join(self.folder_path, f"{collection_name}_index.faiss")
        self.map_path = os.path.join(self.folder_path, f"{collection_name}_chunks.pkl")
        
        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)

        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.map_path, 'rb') as f:
                self.chunks_map = pickle.load(f)
        else:
            # Gestion de la métrique de distance
            if distance_metric == "cosine":
                # Pour le cosine, on utilise Inner Product sur des vecteurs normalisés
                self.index = faiss.IndexFlatIP(dimension)
            else:
                self.index = faiss.IndexFlatL2(dimension)
            self.chunks_map = {}

    def add_chunks(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
            
        embeddings = np.array([c.embedding for c in chunks]).astype('float32')
        
        # Si on est en cosine, il faut normaliser les vecteurs avant l'ajout
        faiss.normalize_L2(embeddings)
        
        start_idx = self.index.ntotal
        self.index.add(embeddings)
        
        for i, chunk in enumerate(chunks):
            self.chunks_map[start_idx + i] = chunk
            
        faiss.write_index(self.index, self.index_path)
        with open(self.map_path, 'wb') as f:
            pickle.dump(self.chunks_map, f)

    def search(self, query_embedding: List[float], top_k: int = 5, **kwargs) -> List[Chunk]:
        query_vec = np.array([query_embedding]).astype('float32')
        # Normalisation de la requête pour la similarité cosine
        faiss.normalize_L2(query_vec)
        
        distances, indices = self.index.search(query_vec, top_k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx in self.chunks_map: # -1 signifie aucun résultat trouvé
                results.append(self.chunks_map[idx])
        return results

    def delete_collection(self, collection_name: str) -> None:
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        if os.path.exists(self.map_path):
            os.remove(self.map_path)

    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        return {"total_vectors": self.index.ntotal}