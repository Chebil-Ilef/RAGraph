import pickle
import hashlib
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

class LocalEmbeddingManager:
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L12-v2", 
                 cache_dir: str = "embedding_cache",
                 use_gpu: bool = True):  
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
                
        if use_gpu and torch.cuda.is_available():
            self.device = "cuda"
            logger.info(f"Using GPU for embeddings: {torch.cuda.get_device_name()}")
        else:
            self.device = "cpu"
            logger.info("Using CPU for embeddings")
        
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name, device=self.device)
            self.dimension = self.model.get_sentence_embedding_dimension()
            logger.info(f"Model loaded successfully. Embedding dimension: {self.dimension}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {e}")
            raise
        
        # cache for session (in-memory)
        self._session_cache: Dict[str, np.ndarray] = {}
        
    def _get_text_hash(self, text: str) -> str:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    def _get_cache_path(self, text_hash: str) -> Path:
        return self.cache_dir / f"{text_hash}.pkl"
    
    def _load_from_cache(self, text_hash: str) -> Optional[np.ndarray]:
        
        if text_hash in self._session_cache:
            return self._session_cache[text_hash]
        
        cache_path = self._get_cache_path(text_hash)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    embedding = pickle.load(f)
                self._session_cache[text_hash] = embedding
                return embedding
            
            except Exception as e:
                logger.warning(f"Failed to load cached embedding {text_hash}: {e}")
                try:
                    cache_path.unlink()
                except:
                    pass
        
        return None
    
    def _save_to_cache(self, text_hash: str, embedding: np.ndarray):

        try:
            self._session_cache[text_hash] = embedding
            
            cache_path = self._get_cache_path(text_hash)
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
                
        except Exception as e:
            logger.warning(f"Failed to save embedding to cache {text_hash}: {e}")
    
    def embed_single(self, text: str) -> np.ndarray:

        text_hash = self._get_text_hash(text)
        
        cached_embedding = self._load_from_cache(text_hash)
        if cached_embedding is not None:
            return cached_embedding
        
        try:
            embedding = self.model.encode([text], convert_to_numpy=True)[0]
            embedding = embedding.astype(np.float32)
            
            # normalize for cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            self._save_to_cache(text_hash, embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to create embedding for text: {e}")
            raise
    
    def embed_documents(self, texts: List[str], batch_size: int = 32) -> np.ndarray:

        if not texts:
            return np.array([])
        
        logger.info(f"Creating embeddings for {len(texts)} texts using {self.model_name}")
        
        embeddings = []
        cache_hits = 0
        new_embeddings_needed = []
        new_embedding_indices = []
        
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text)
            cached_embedding = self._load_from_cache(text_hash)
            
            if cached_embedding is not None:
                embeddings.append(cached_embedding)
                cache_hits += 1
            else:
                embeddings.append(None)  # Placeholder
                new_embeddings_needed.append(text)
                new_embedding_indices.append(i)
        
        logger.info(f"Cache hits: {cache_hits}/{len(texts)} ({cache_hits/len(texts)*100:.1f}%)")
        
        if new_embeddings_needed:
            logger.info(f"Generating {len(new_embeddings_needed)} new embeddings")
            
            try:
                new_embeddings = []
                for i in range(0, len(new_embeddings_needed), batch_size):
                    batch_texts = new_embeddings_needed[i:i + batch_size]
                    batch_embeddings = self.model.encode(
                        batch_texts, 
                        convert_to_numpy=True,
                        show_progress_bar=len(batch_texts) > 10
                    )
                    new_embeddings.extend(batch_embeddings)
                
                new_embeddings = np.array(new_embeddings, dtype=np.float32)
                norms = np.linalg.norm(new_embeddings, axis=1, keepdims=True)
                new_embeddings = new_embeddings / np.where(norms > 0, norms, 1)
                
                for i, (text, embedding) in enumerate(zip(new_embeddings_needed, new_embeddings)):
                    text_hash = self._get_text_hash(text)
                    self._save_to_cache(text_hash, embedding)
                    
                    original_index = new_embedding_indices[i]
                    embeddings[original_index] = embedding
                
            except Exception as e:
                logger.error(f"Failed to generate new embeddings: {e}")
                raise
        
        result = np.array(embeddings, dtype=np.float32)
        logger.info(f"Created embeddings: shape {result.shape}")
        
        return result
    
    def clear_cache(self, keep_session: bool = False):

        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            
            if not keep_session:
                self._session_cache.clear()
            
            logger.info("Embedding cache cleared")
            
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:

        try:
            disk_cache_files = list(self.cache_dir.glob("*.pkl"))
            disk_cache_size = sum(f.stat().st_size for f in disk_cache_files)
            
            return {
                "model_name": self.model_name,
                "device": self.device,
                "dimension": self.dimension,
                "cache_directory": str(self.cache_dir),
                "disk_cache_entries": len(disk_cache_files),
                "disk_cache_size_mb": disk_cache_size / (1024 * 1024),
                "session_cache_entries": len(self._session_cache),
                "gpu_available": torch.cuda.is_available(),
                "gpu_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"error": str(e)}


# global instance for reuse
_embedding_manager = None

def get_embedding_manager(model_name: str = "all-MiniLM-L6-v2", 
                         cache_dir: str = "embedding_cache",
                         use_gpu: bool = True) -> LocalEmbeddingManager:

    global _embedding_manager
    
    if _embedding_manager is None:
        _embedding_manager = LocalEmbeddingManager(
            model_name=model_name,
            cache_dir=cache_dir,
            use_gpu=use_gpu
        )
    
    return _embedding_manager
