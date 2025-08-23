"""
Hybrid Query System for Graph-RAG

Query flow:
1. Vector search over chunks (fast candidate retrieval)
2. Graph traversal for context expansion
3. Merge and rerank results
4. Generate final answer
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from graphiti_core import Graphiti
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from utils.hybrid_ingestion import HybridIngestionPipeline

logger = logging.getLogger(__name__)

class HybridQueryEngine:
    def __init__(self, graphiti: Graphiti, vector_store_path: str = "vector_store"):
        self.graphiti = graphiti
        self.pipeline = HybridIngestionPipeline(graphiti, vector_store_path)
    
    async def query(self, question: str, top_k_chunks: int = 10, top_k_graph: int = 5) -> Dict[str, Any]:
        """
        Hybrid query that combines vector search + graph traversal.
        
        Args:
            question: User's question
            top_k_chunks: Number of chunks to retrieve from vector store
            top_k_graph: Number of graph results to retrieve
        
        Returns:
            Dict with answer, sources, and metadata
        """
        try:
            # Step 1: Vector search over chunks
            logger.info(f"Vector search for: {question}")
            vector_results = await self._vector_search(question, top_k_chunks)
            
            # Step 2: Graph search for structured knowledge
            logger.info(f"Graph search for: {question}")
            graph_results = await self._graph_search(question, top_k_graph)
            
            # Step 3: Merge and contextualize results
            logger.info("Merging vector and graph results...")
            merged_context = self._merge_results(vector_results, graph_results)
            
            # Step 4: Generate answer (you'd typically use an LLM here)
            answer = self._generate_answer(question, merged_context)
            
            return {
                'question': question,
                'answer': answer,
                'chunk_sources': vector_results,
                'graph_sources': graph_results,
                'merged_context': merged_context,
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Query failed: {e}", exc_info=True)
            return {
                'question': question,
                'answer': f"Query failed: {str(e)}",
                'chunk_sources': [],
                'graph_sources': [],
                'merged_context': "",
                'status': 'failed'
            }
    
    async def _vector_search(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """Search chunks in vector store."""
        try:
            # Generate embedding for question
            question_embedding = await self.pipeline._get_embedding(question)
            
            # Search vector store
            results = self.pipeline.search_chunks(question_embedding, top_k)
            
            logger.info(f"Found {len(results)} chunk candidates")
            return results
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    async def _graph_search(self, question: str, top_k: int) -> List[Dict[str, Any]]:
        """Search structured knowledge in graph."""
        try:
            # Use Graphiti's built-in search (without search_config parameter)
            graph_results = await self.graphiti.search(question)
            
            # Convert to consistent format
            formatted_results = []
            for result in graph_results[:top_k]:
                formatted_results.append({
                    'content': getattr(result, 'fact', str(result)),
                    'source': getattr(result, 'source_name', 'Graph'),
                    'uuid': getattr(result, 'uuid', None),
                    'type': 'graph_entity'
                })
            
            logger.info(f"Found {len(formatted_results)} graph entities")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            return []
    
    def _merge_results(self, vector_results: List[Dict], graph_results: List[Dict]) -> str:
        """Merge vector chunks and graph entities into coherent context."""
        context_parts = []
        
        # Add graph entities first (higher-level concepts)
        if graph_results:
            context_parts.append("=== STRUCTURED KNOWLEDGE ===")
            for result in graph_results:
                context_parts.append(f"• {result['content']}")
            context_parts.append("")
        
        # Add relevant chunks
        if vector_results:
            context_parts.append("=== DOCUMENT EXCERPTS ===")
            for i, result in enumerate(vector_results, 1):
                doc_name = result.get('doc_name', 'Unknown')
                chunk_content = result.get('content', '')[:500]  # Truncate
                score = result.get('similarity_score', 0)
                
                context_parts.append(f"[{i}] From {doc_name} (score: {score:.3f}):")
                context_parts.append(chunk_content)
                context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer based on merged context."""
        # In a real implementation, you'd use an LLM to generate the answer
        # For now, return a simple summary
        
        if not context.strip():
            return "No relevant information found in the knowledge base."
        
        return f"""Based on the available information:

{context[:1000]}...

[This is a placeholder answer. In production, an LLM would generate a proper response based on the context above.]
"""

# Example usage function
async def demo_hybrid_query(graphiti: Graphiti, questions: List[str]):
    """Demonstrate hybrid querying."""
    query_engine = HybridQueryEngine(graphiti)
    
    results = []
    for question in questions:
        print(f"\n🔍 QUESTION: {question}")
        print("=" * 60)
        
        result = await query_engine.query(question)
        
        print(f"📊 CHUNK SOURCES: {len(result['chunk_sources'])}")
        print(f"🕸️  GRAPH SOURCES: {len(result['graph_sources'])}")
        print(f"💬 ANSWER:\n{result['answer']}")
        
        results.append(result)
    
    return results
