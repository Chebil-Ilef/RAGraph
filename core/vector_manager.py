import os
import pickle
import logging
from typing import List, Dict, Any
from datetime import datetime
import hashlib
import json
import tempfile
import faiss
import numpy as np
from fastapi import UploadFile, HTTPException
from utils.document_processor import DocumentProcessor, reset_docling_converter, clear_gpu_memory
from utils.embeddings import get_embedding_manager
from utils.document_cache import SystemCache
from config.embedding_config import EMBEDDING_MODEL, EMBEDDING_CACHE_DIR, USE_GPU
import shutil
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class VectorManager:
    
    def __init__(self, base_path: str = "vector_store"):
        self.base_path = base_path
        self.document_processor = DocumentProcessor()
        
        # Initialize shared document cache for vector system
        self.cache_manager = SystemCache("VECTOR")
        
        # Initialize local embedding manager
        self.embedding_manager = get_embedding_manager(
            model_name=EMBEDDING_MODEL,
            cache_dir=EMBEDDING_CACHE_DIR,
            use_gpu=USE_GPU
        )
        self.dimension = self.embedding_manager.dimension
        
        self.use_gpu = False
        logger.info("FAISS using CPU")
        logger.info(f"Using local embeddings with dimension {self.dimension}")
        
    def _get_user_path(self, user_id: str) -> str:
        return os.path.join(self.base_path, user_id)
    
    def _get_index_path(self, user_id: str) -> str:
        return os.path.join(self._get_user_path(user_id), "faiss.index")
    
    def _get_metadata_path(self, user_id: str) -> str:
        return os.path.join(self._get_user_path(user_id), "metadata.pkl")
    
    def _get_documents_path(self, user_id: str) -> str:
        return os.path.join(self._get_user_path(user_id), "documents.json")
    
    def _ensure_user_directory(self, user_id: str):
        user_path = self._get_user_path(user_id)
        os.makedirs(user_path, exist_ok=True)
        return user_path
    
    def _user_store_exists(self, user_id: str) -> bool:
        user_path = self._get_user_path(user_id)
        index_path = self._get_index_path(user_id)
        metadata_path = self._get_metadata_path(user_id)
        
        return (
            os.path.exists(user_path) and 
            os.path.exists(index_path) and 
            os.path.exists(metadata_path)
        )
    
    def _load_user_index(self, user_id: str):
        if not self._user_store_exists(user_id):
            return None, []
        
        try:
            index_path = self._get_index_path(user_id)
            metadata_path = self._get_metadata_path(user_id)
            
            # CPU index from disk
            index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index for user {user_id}: {index.ntotal} vectors (CPU)")
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            return index, metadata
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index for user {user_id}: {e}")
            return None, []
    
    def _create_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        try:
            index = faiss.IndexFlatIP(self.dimension)
            logger.info("Created FAISS CPU index")
            
            index.add(embeddings)
            return index
            
        except Exception as e:
            logger.error(f"Failed to create FAISS index: {e}")
            raise
    
    def _save_user_index(self, user_id: str, index, metadata: List[Dict]):
        try:
            self._ensure_user_directory(user_id)
            
            index_path = self._get_index_path(user_id)
            metadata_path = self._get_metadata_path(user_id)
            
            faiss.write_index(index, index_path)
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            logger.info(f"Saved FAISS index for user {user_id}: {index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to save FAISS index for user {user_id}: {e}")
            raise
    
    def _load_documents_info(self, user_id: str) -> Dict[str, Dict]:

        documents_path = self._get_documents_path(user_id)
        if os.path.exists(documents_path):
            try:
                with open(documents_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load documents info for user {user_id}: {e}")
        return {}
    
    def _save_documents_info(self, user_id: str, documents_info: Dict[str, Dict]):

        try:
            self._ensure_user_directory(user_id)
            documents_path = self._get_documents_path(user_id)
            
            with open(documents_path, 'w') as f:
                json.dump(documents_info, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save documents info for user {user_id}: {e}")
            raise
    
    def _compute_file_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()
    
    def _create_embeddings(self, text_chunks: List[str]) -> np.ndarray:

        try:
            logger.info(f"Creating embeddings for {len(text_chunks)} chunks using local model")
            
            # local embedding manager with caching
            embeddings_array = self.embedding_manager.embed_documents(
                text_chunks, 
                batch_size=32  # in batches for efficiency
            )
            
            logger.info(f"Created {len(embeddings_array)} embeddings with dimension {embeddings_array.shape[1]}")
            return embeddings_array
            
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise HTTPException(status_code=500, detail=f"Embedding creation failed: {e}")
    
    def _extract_text_from_pdf(self, file_content: bytes, filename: str) -> List[str]:

        try:
            logger.info(f"Extracting text from PDF: {filename}")
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(file_content)
                tmp_file_path = tmp_file.name
            
            try:
                # docling (with pdfplumber fallback)
                raw_text = self.document_processor.extract_text_docling(tmp_file_path)
                
                if not raw_text or not raw_text.strip():
                    logger.warning(f"No text extracted from {filename}")
                    return []
                
                logger.info(f"Extracted {len(raw_text)} characters from {filename}")
                
                # SEMANTIC chunking with document processor
                chunks = self.document_processor.semantic_chunk(raw_text)
                logger.info(f"Created {len(chunks)} semantic chunks from {filename}")
                
                return chunks
                
            finally:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Failed to extract text from {filename}: {e}")
            return []
    
    async def build_vector_store_from_uploads(self, user_id: str, uploaded_files: List[UploadFile], force_rebuild: bool = False) -> Dict[str, Any]:
        
        logger.info(f"Building vector store for user {user_id} from {len(uploaded_files)} uploaded files")
        
        start_time = datetime.now()
        
        # existing data
        existing_index, existing_metadata = self._load_user_index(user_id)
        documents_info = self._load_documents_info(user_id)
        
        all_chunks = []
        all_metadata = existing_metadata.copy() if existing_metadata else []
        processed_files = 0
        failed_files = 0
        new_chunks_count = 0
        
        for file in uploaded_files:
            try:
                logger.info(f"Processing file: {file.filename}")
                
                content = await file.read()
                content_hash = self.cache_manager.doc_cache.compute_file_hash(content)
                
                # check if already processed (unless force rebuild)
                if self.cache_manager.is_processed(documents_info, content_hash, force_rebuild):
                    processed_files += 1
                    continue
                
                text_chunks = self._extract_text_from_pdf(content, file.filename)
                
                if not text_chunks:
                    logger.warning(f"No text extracted from {file.filename}")
                    failed_files += 1
                    continue
                
                embeddings = self._create_embeddings(text_chunks)
                
                # add to collection
                for i, (chunk, embedding) in enumerate(zip(text_chunks, embeddings)):
                    all_chunks.append(embedding)
                    all_metadata.append({
                        "text": chunk,
                        "source_file": file.filename,
                        "chunk_index": i,
                        "content_hash": content_hash,
                        "processed_at": datetime.now().isoformat()
                    })
                    new_chunks_count += 1
                
                # Mark document as processed using cache manager
                self.cache_manager.mark_processed(
                    documents_info,
                    file.filename,
                    content,
                    {
                        "chunks_count": len(text_chunks),
                        "vector_dimension": self.dimension
                    },
                    status="processed"
                )
                
                processed_files += 1
                
            except Exception as e:
                logger.error(f"Failed to process file {file.filename}: {e}")
                failed_files += 1
                continue
        
        # build/update FAISS index
        if all_chunks:
            new_embeddings = np.array(all_chunks[-new_chunks_count:], dtype=np.float32) if new_chunks_count > 0 else None
            
            if existing_index is not None and new_embeddings is not None:
                existing_index.add(new_embeddings)
                final_index = existing_index
                logger.info(f"Added {new_chunks_count} new vectors to existing index")
                
            elif new_embeddings is not None:
                if existing_index is not None:
                    all_embeddings = np.array(all_chunks, dtype=np.float32)
                else:
                    all_embeddings = new_embeddings
                
                final_index = self._create_faiss_index(all_embeddings)
                logger.info(f"Created new FAISS index with {final_index.ntotal} vectors")
                
            else:
                final_index = existing_index 
            
            if final_index is not None:
                self._save_user_index(user_id, final_index, all_metadata)
            
        else:
            final_index = existing_index
        
        self._save_documents_info(user_id, documents_info)
        
        total_chunks = len(all_metadata)
        processing_time = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Vector store build complete for user {user_id}: {total_chunks} total chunks")
        
        return {
            "user_id": user_id,
            "status": "success" if failed_files == 0 else "partial",
            "chunks_created": new_chunks_count,
            "documents_processed": processed_files,
            "documents_failed": failed_files,
            "vector_dimension": self.dimension,
            "processing_time_seconds": processing_time,
            "message": f"Vector store built with {total_chunks} total chunks ({new_chunks_count} new)",
            "created_at": datetime.now()
        }
    
    async def get_status(self, user_id: str) -> Dict[str, Any]:
        
        if not self._user_store_exists(user_id):
            return {
                "user_id": user_id,
                "exists": False,
                "last_updated": None,
                "document_count": 0,
                "total_size": 0,
                "health_status": "unavailable",
                "details": {"error": "No vector store found for user"}
            }
        
        try:
            index, metadata = self._load_user_index(user_id)
            documents_info = self._load_documents_info(user_id)
            
            if index is None:
                raise Exception("Failed to load index")
            
            last_updated = None
            if documents_info:
                timestamps = [doc.get("processed_at") for doc in documents_info.values() if doc.get("processed_at")]
                if timestamps:
                    last_updated = max(timestamps)
            
            return {
                "user_id": user_id,
                "exists": True,
                "last_updated": last_updated,
                "document_count": len(documents_info),
                "total_size": len(metadata),
                "health_status": "healthy",
                "details": {
                    "chunks": len(metadata),
                    "dimension": self.dimension,
                    "index_size": index.ntotal if index else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get status for user {user_id}: {e}")
            return {
                "user_id": user_id,
                "exists": True,
                "last_updated": None,
                "document_count": 0,
                "total_size": 0,
                "health_status": "degraded",
                "details": {"error": str(e)}
            }
    
    async def clear_vector_store(self, user_id: str) -> Dict[str, Any]:
        
        logger.info(f"Clearing vector store for user {user_id}")
        
        try:
            user_path = self._get_user_path(user_id)
            
            if not os.path.exists(user_path):
                return {
                    "user_id": user_id,
                    "deleted": False,
                    "items_removed": 0,
                    "message": "No vector store found for user",
                    "deleted_at": datetime.now()
                }
            
            items_removed = 0
            if self._user_store_exists(user_id):
                _, metadata = self._load_user_index(user_id)
                items_removed = len(metadata) if metadata else 0
            
            shutil.rmtree(user_path, ignore_errors=True)
            
            logger.info(f"Removed vector store directory for user {user_id}: {items_removed} items")
            
            return {
                "user_id": user_id,
                "deleted": True,
                "items_removed": items_removed,
                "message": f"Vector store cleared - removed {items_removed} chunks",
                "deleted_at": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to clear vector store for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to clear vector store: {e}")
    
    def cleanup_memory(self):
        """Cleanup GPU memory and reset docling converter if needed."""
        try:
            logger.info("Performing memory cleanup...")
            clear_gpu_memory()
            reset_docling_converter()
            logger.info("Memory cleanup completed")
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")

