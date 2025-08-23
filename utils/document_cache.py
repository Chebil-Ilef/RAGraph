import hashlib
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentCache:
    
    @staticmethod
    def compute_file_hash(content: bytes) -> str:

        return hashlib.sha256(content).hexdigest()
    
    @staticmethod
    def create_document_entry(
        filename: str,
        content_hash: str,
        content_size: int,
        processing_result: Dict[str, Any],
        status: str = "processed"
    ) -> Dict[str, Any]:

        return {
            "filename": filename,
            "content_hash": content_hash,
            "size_bytes": content_size,
            "processed_at": datetime.now().isoformat(),
            "status": status,
            **processing_result
        }
    
    @staticmethod
    def is_document_processed(
        documents_cache: Dict[str, Dict],
        content_hash: str,
        force_rebuild: bool = False
    ) -> bool:

        if force_rebuild:
            return False
        
        if content_hash not in documents_cache:
            return False
        
        doc_info = documents_cache[content_hash]
        return doc_info.get("status") == "processed"
    
    @staticmethod
    def get_processing_stats(documents_cache: Dict[str, Dict]) -> Dict[str, Any]:

        if not documents_cache:
            return {
                "total_documents": 0,
                "processed_documents": 0,
                "failed_documents": 0,
                "total_size_bytes": 0,
                "last_processed": None
            }
        
        processed_docs = [doc for doc in documents_cache.values() if doc.get("status") == "processed"]
        failed_docs = [doc for doc in documents_cache.values() if doc.get("status") == "failed"]
        
        # last processing time
        timestamps = [doc.get("processed_at") for doc in documents_cache.values() if doc.get("processed_at")]
        last_processed = max(timestamps) if timestamps else None
        
        total_size = sum(doc.get("size_bytes", 0) for doc in documents_cache.values())
        
        return {
            "total_documents": len(documents_cache),
            "processed_documents": len(processed_docs),
            "failed_documents": len(failed_docs),
            "total_size_bytes": total_size,
            "last_processed": last_processed
        }
    
    @staticmethod
    def log_cache_hit(filename: str, content_hash: str, system: str):

        logger.info(f"[{system}] Cache hit for {filename} (hash: {content_hash[:8]}...)")
    
    @staticmethod
    def log_cache_miss(filename: str, content_hash: str, system: str):

        logger.info(f"[{system}] Cache miss for {filename} (hash: {content_hash[:8]}...) - processing")
    
    @staticmethod
    def validate_content_hash(content: bytes, expected_hash: str) -> bool:

        actual_hash = DocumentCache.compute_file_hash(content)
        return actual_hash == expected_hash


class SystemCache:
    
    def __init__(self, system_name: str):
        self.system_name = system_name
        self.doc_cache = DocumentCache()
    
    def is_processed(self, documents_cache: Dict, content_hash: str, force_rebuild: bool = False) -> bool:

        is_processed = self.doc_cache.is_document_processed(documents_cache, content_hash, force_rebuild)
        
        if is_processed:

            doc_info = documents_cache.get(content_hash, {})
            filename = doc_info.get("filename", "unknown")
            self.doc_cache.log_cache_hit(filename, content_hash, self.system_name)
        
        return is_processed
    
    def mark_processed(
        self, 
        documents_cache: Dict, 
        filename: str, 
        content: bytes, 
        processing_result: Dict[str, Any],
        status: str = "processed"
    ) -> str:

        content_hash = self.doc_cache.compute_file_hash(content)
        
        if status == "processed":
            self.doc_cache.log_cache_miss(filename, content_hash, self.system_name)
        
        documents_cache[content_hash] = self.doc_cache.create_document_entry(
            filename=filename,
            content_hash=content_hash,
            content_size=len(content),
            processing_result=processing_result,
            status=status
        )
        
        return content_hash
    
    def get_stats(self, documents_cache: Dict) -> Dict[str, Any]:

        base_stats = self.doc_cache.get_processing_stats(documents_cache)
        base_stats["system"] = self.system_name
        return base_stats


class CacheManager:
    
    def __init__(self):
        self.vector_cache = SystemCache("VECTOR")
        self.kg_cache = SystemCache("KG")
    
    def get_cross_system_status(
        self, 
        vector_cache: Dict, 
        kg_cache: Dict
    ) -> Dict[str, Any]:

        all_hashes = set(vector_cache.keys()) | set(kg_cache.keys())
        
        vector_only = set(vector_cache.keys()) - set(kg_cache.keys())
        kg_only = set(kg_cache.keys()) - set(vector_cache.keys())
        both_systems = set(vector_cache.keys()) & set(kg_cache.keys())
        
        return {
            "total_unique_documents": len(all_hashes),
            "vector_only": len(vector_only),
            "kg_only": len(kg_only),
            "both_systems": len(both_systems),
            "vector_stats": self.vector_cache.get_stats(vector_cache),
            "kg_stats": self.kg_cache.get_stats(kg_cache)
        }
