import logging
import asyncio
from typing import List, Dict, Any
from datetime import datetime

from core.vector_manager import VectorManager
from core.kg_manager import KnowledgeGraphManager
from utils.document_cache import CacheManager

logger = logging.getLogger(__name__)

class HybridSearch:

    
    def __init__(self, vector_manager: VectorManager, kg_manager: KnowledgeGraphManager):
        self.vector_manager = vector_manager
        self.kg_manager = kg_manager
        self.cache_manager = CacheManager()
        
        logger.info("Hybrid search initialized with vector and KG managers")
    
    async def query(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        include_kg: bool = True,
        include_vector: bool = True,
        rerank: bool = True
    ) -> Dict[str, Any]:
     

        logger.info(f"Hybrid query for user {user_id}: '{query}' (kg={include_kg}, vector={include_vector})")
        
        start_time = datetime.now()
        
        try:
            # // execution of both searches
            tasks = []
            
            if include_vector:
                tasks.append(self._search_vector(user_id, query, limit))
            
            if include_kg:
                tasks.append(self._search_kg(user_id, query, limit))
            
            if not tasks:
                return self._empty_response(user_id, query, "No search systems enabled")
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            vector_results = []
            kg_results = []
            fallback_used = None
            
            if include_vector:
                vector_result = results[0]
                if isinstance(vector_result, Exception):
                    logger.error(f"Vector search failed for user {user_id}: {vector_result}")
                    fallback_used = "vector_failed"
                else:
                    vector_results = vector_result
            
            if include_kg:
                kg_result = results[-1] if include_vector else results[0]
                if isinstance(kg_result, Exception):
                    logger.error(f"KG search failed for user {user_id}: {kg_result}")
                    fallback_used = "kg_failed" if not fallback_used else "both_failed"
                else:
                    kg_results = kg_result
            
            # combine and optionally rerank results
            combined_results = self._combine_results(
                vector_results, 
                kg_results, 
                limit, 
                rerank
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "user_id": user_id,
                "query": query,
                "total_results": len(combined_results),
                "kg_results": len(kg_results),
                "vector_results": len(vector_results),
                "results": combined_results,
                "processing_time_seconds": processing_time,
                "fallback_used": fallback_used,
                "message": f"Hybrid search completed with {len(combined_results)} results"
            }
            
        except Exception as e:
            logger.error(f"Hybrid search failed for user {user_id}: {e}")
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "user_id": user_id,
                "query": query,
                "total_results": 0,
                "kg_results": 0,
                "vector_results": 0,
                "results": [],
                "processing_time_seconds": processing_time,
                "fallback_used": "system_error",
                "message": f"Hybrid search failed: {str(e)}"
            }
    
    async def _search_vector(self, user_id: str, query: str, limit: int) -> List[Dict[str, Any]]:

        try:
            return await self.vector_manager.search_vector_store(user_id, query, limit)
            
        except Exception as e:
            logger.error(f"Vector search error for user {user_id}: {e}")
            raise
    
    async def _search_kg(self, user_id: str, query: str, limit: int) -> List[Dict[str, Any]]:

        try:
            return await self.kg_manager.search_graph(user_id, query, limit)
            
        except Exception as e:
            logger.error(f"KG search error for user {user_id}: {e}")
            raise
    
    def _combine_results(
        self, 
        vector_results: List[Dict], 
        kg_results: List[Dict], 
        limit: int, 
        rerank: bool
    ) -> List[Dict[str, Any]]:
      
        all_results = []
        
        for result in vector_results:
            all_results.append({
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "source": "vector",
                "metadata": result.get("metadata", {}),
                "uuid": result.get("uuid")
            })
        
        for result in kg_results:
            all_results.append({
                "content": result.get("content", ""),
                "score": result.get("score", 0.0),
                "source": "kg",
                "metadata": result.get("metadata", {}),
                "uuid": result.get("uuid")
            })
        
        if not all_results:
            return []
        
        if rerank:
            # simple reranking: sort by score, but boost KG results slightly
            # we can use a proper reranking model but that would add latency
            for result in all_results:
                if result["source"] == "kg":
                    result["score"] *= 1.1  # Slight boost for structured knowledge
            
            # sort by score (descending)
            all_results.sort(key=lambda x: x["score"], reverse=True)
        
        return all_results[:limit]
    
    def _empty_response(self, user_id: str, query: str, message: str) -> Dict[str, Any]:

        return {
            "user_id": user_id,
            "query": query,
            "total_results": 0,
            "kg_results": 0,
            "vector_results": 0,
            "results": [],
            "processing_time_seconds": 0.0,
            "message": message
        }
    
    async def get_cross_system_status(self, user_id: str) -> Dict[str, Any]:
    

        try:
            vector_status = await self.vector_manager.get_status(user_id)
            kg_status = await self.kg_manager.get_status(user_id)
            
            # caches
            vector_cache = self.vector_manager._load_documents_info(user_id)
            kg_cache = self.kg_manager._load_documents_info(user_id)
            
            cross_status = self.cache_manager.get_cross_system_status(vector_cache, kg_cache)
            
            return {
                "user_id": user_id,
                "vector_system": vector_status,
                "kg_system": kg_status,
                "cross_system_analysis": cross_status,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get cross-system status for user {user_id}: {e}")
            return {
                "user_id": user_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
