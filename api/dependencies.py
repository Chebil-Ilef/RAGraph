from fastapi import HTTPException, Depends
from typing import Optional
import logging
import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv()

from core.vector_manager import VectorManager

logger = logging.getLogger(__name__)


# DATABASE CONNECTIONS

class Neo4jConnection:
    
    def __init__(self):
        self.uri = None
        self.user = None
        self.password = None
        self.driver = None
        self._initialized = False
    
    async def initialize(self):

        if self._initialized:
            return
            
        try:
            self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
            self.user = os.getenv("NEO4J_USER", "neo4j")
            self.password = os.getenv("NEO4J_PASSWORD", "testpass")
            

            logger.info(f"Connecting to Neo4j at {self.uri}")
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            
            self._initialized = True
            logger.info("Neo4j connection established")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise HTTPException(
                status_code=503,
                detail="Database connection failed"
            )
    
    async def health_check(self) -> bool:

        try:
            if not self.driver:
                logger.error("Neo4j driver not initialized")
                return False
                
            # simple query to check cnx
            with self.driver.session() as session:
                logger.info("Neo4j session created, testing query...")
                result = session.run("RETURN 1 AS test")
                record = result.single()
                test_value = record["test"]
                logger.info(f"Neo4j health check query returned: {test_value}")
                return test_value == 1
                
        except Exception as e:
            logger.error(f"Neo4j health check failed: {type(e).__name__}: {e}")
            return False

class FAISSConnection:
    
    def __init__(self):
        self.vector_store_path = "vector_store"
        self._initialized = False
    
    async def initialize(self):

        if self._initialized:
            return
            
        try:
            # ensure base vector store directory exists
            os.makedirs(self.vector_store_path, exist_ok=True)

            logger.info(f"FAISS vector store base path: {self.vector_store_path}")
            self._initialized = True
            logger.info("FAISS connection ready")
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {e}")
            raise HTTPException(
                status_code=503,
                detail="Vector store initialization failed"
            )
    

# global connection instances
_neo4j_connection: Optional[Neo4jConnection] = None
_faiss_connection: Optional[FAISSConnection] = None

async def get_neo4j_connection() -> Neo4jConnection:

    global _neo4j_connection
    
    if _neo4j_connection is None:
        _neo4j_connection = Neo4jConnection()
        await _neo4j_connection.initialize()
    
    return _neo4j_connection

async def get_faiss_connection() -> FAISSConnection:

    global _faiss_connection
    
    if _faiss_connection is None:
        _faiss_connection = FAISSConnection()
        await _faiss_connection.initialize()
    
    return _faiss_connection

# MANAGER DEPENDENCIES WILL BE DELETED LATER AND DEVELOPED OUTSIDE IN CORE

class MockKGManager:
    """Mock KG Manager for Phase 1 - will be replaced with real implementation."""
    
    def __init__(self, neo4j_conn: Neo4jConnection):
        self.neo4j_conn = neo4j_conn
    
    async def build_graph_from_uploads(self, user_id: str, uploaded_files: list, force_rebuild: bool = False) -> dict:
        """Mock graph building from uploaded files."""
        file_names = [f.filename for f in uploaded_files]
        logger.info(f"🔨 [MOCK] Building KG for user {user_id} from uploads: {file_names}")
        return {
            "user_id": user_id,
            "status": "success", 
            "entities_created": 50 * len(uploaded_files),
            "relationships_created": 25 * len(uploaded_files),
            "documents_processed": len(uploaded_files),
            "documents_failed": 0,
            "processing_time_seconds": 2.5,
            "message": f"Mock graph built from {len(uploaded_files)} uploaded files",
            "created_at": "2025-08-23T10:00:00Z"
        }
    
    async def build_graph_from_paths(self, user_id: str, document_paths: list, force_rebuild: bool = False) -> dict:
        """Mock graph building from document paths."""
        logger.info(f"🔨 [MOCK] Building KG for user {user_id} from paths: {document_paths}")
        return {
            "user_id": user_id,
            "status": "success", 
            "entities_created": 50 * len(document_paths),
            "relationships_created": 25 * len(document_paths),
            "documents_processed": len(document_paths),
            "documents_failed": 0,
            "processing_time_seconds": 2.5,
            "message": f"Mock graph built from {len(document_paths)} document paths",
            "created_at": "2025-08-23T10:00:00Z"
        }
    
    async def get_status(self, user_id: str) -> dict:
        """Mock status check."""
        return {
            "user_id": user_id,
            "exists": True,
            "last_updated": "2025-08-23T10:00:00Z",
            "document_count": 5,
            "total_size": 75,
            "health_status": "healthy",
            "details": {"nodes": 50, "relationships": 25}
        }
    
    async def clear_graph(self, user_id: str) -> dict:
        """Mock graph clearing."""
        logger.info(f"[MOCK] Clearing KG for user {user_id}")
        return {
            "user_id": user_id,
            "deleted": True,
            "items_removed": 75,
            "message": "Mock graph cleared",
            "deleted_at": "2025-08-23T10:00:00Z"
        }

class MockHybridSearch:
    """Mock Hybrid Search for Phase 1 - will be replaced with real implementation."""
    
    async def query(self, user_id: str, query: str, limit: int = 10, 
                   include_kg: bool = True, include_vector: bool = True) -> dict:
        """Mock hybrid search."""
        logger.info(f"🔍 [MOCK] Hybrid search for user {user_id}: {query}")
        
        # Mock results
        results = [
            {
                "content": f"Mock KG result for: {query}",
                "score": 0.95,
                "source": "kg",
                "metadata": {"type": "entity", "document": "doc1.pdf"},
                "uuid": "kg-uuid-123"
            },
            {
                "content": f"Mock vector result for: {query}",
                "score": 0.87,
                "source": "vector", 
                "metadata": {"chunk_id": 42, "document": "doc2.pdf"},
                "uuid": None
            }
        ]
        
        return {
            "user_id": user_id,
            "query": query,
            "total_results": len(results),
            "kg_results": 1,
            "vector_results": 1,
            "results": results[:limit],
            "processing_time_seconds": 0.8,
            "fallback_used": None,
            "message": "Mock hybrid search completed"
        }

async def get_kg_manager(
    neo4j_conn: Neo4jConnection = Depends(get_neo4j_connection)
) -> MockKGManager:

    return MockKGManager(neo4j_conn)

async def get_vector_manager(
    faiss_conn: FAISSConnection = Depends(get_faiss_connection)
) -> VectorManager:

    return VectorManager(base_path=faiss_conn.vector_store_path)

async def get_hybrid_search() -> MockHybridSearch:

    return MockHybridSearch()
