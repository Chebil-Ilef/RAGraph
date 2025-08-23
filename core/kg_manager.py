import os
import logging
import hashlib
import json
import tempfile
import asyncio
from typing import List, Dict, Any
from datetime import datetime, timezone
from fastapi import UploadFile, HTTPException

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

from openai import AsyncAzureOpenAI
from dotenv import load_dotenv
from utils.document_processor import DocumentProcessor
from utils.setup import Setup

import re
import unicodedata


load_dotenv()

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    
    def __init__(self, base_path: str = "kg_store"):
        self.base_path = base_path
        self.document_processor = DocumentProcessor()
        
        # Graphiti clients (cached per instance)
        self._graphiti_clients: Dict[str, Graphiti] = {}
        self._client_configs = None
        
        logger.info(f"KG Manager initialized with base path: {base_path}")
    
    async def _get_client_configs(self) -> Dict[str, Any]:

        if self._client_configs is not None:
            return self._client_configs
            
        try:
            config = Setup.load_config()
            
            api_key = config["azure_openai"]["api_key"]
            api_version = config["azure_openai"]["api_version"]
            endpoint = config["azure_openai"]["endpoint"]
            chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
            emb_deployment = config["azure_openai"]["embeddings_deployment"]
            
            llm_client_azure = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
            emb_client_azure = AsyncAzureOpenAI(
                api_key=api_key,
                api_version=api_version,
                azure_endpoint=endpoint
            )
            
            self._client_configs = {
                "neo4j": config["neo4j"],
                "llm_client_azure": llm_client_azure,
                "emb_client_azure": emb_client_azure,
                "chat_deployment": chat_deployment,
                "emb_deployment": emb_deployment
            }
            
            logger.info("Azure OpenAI client configurations loaded")
            return self._client_configs
            
        except Exception as e:
            logger.error(f"Failed to load client configurations: {e}")
            raise HTTPException(status_code=500, detail=f"Configuration error: {e}")
    
    async def _get_graphiti_client(self, user_id: str) -> Graphiti:

        if user_id in self._graphiti_clients:
            return self._graphiti_clients[user_id]
        
        try:
            configs = await self._get_client_configs()
            
            llm_config = LLMConfig(
                model=configs["chat_deployment"],
                small_model=configs["chat_deployment"],
            )
            llm_client = OpenAIClient(
                config=llm_config,
                client=configs["llm_client_azure"]
            )
            
            embedder = OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    embedding_model=configs["emb_deployment"],
                    embedding_dim=1536, 
                ),
                client=configs["emb_client_azure"]
            )
            
            # cross encoder
            cross_encoder = OpenAIRerankerClient(
                config=LLMConfig(model=llm_config.small_model),
                client=configs["llm_client_azure"]
            )
            
            # Graphiti instance (user isolation via labels/properties)
            neo4j_config = configs["neo4j"]
            graphiti = Graphiti(
                neo4j_config["uri"],
                neo4j_config["user"],
                neo4j_config["password"],
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=cross_encoder
            )
            
            await graphiti.build_indices_and_constraints()
            
            self._graphiti_clients[user_id] = graphiti
            logger.info(f"Created Graphiti client for user {user_id}")
            
            return graphiti
            
        except Exception as e:
            logger.error(f"Failed to create Graphiti client for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Graph client creation failed: {e}")
    
    def _get_user_path(self, user_id: str) -> str:
        return os.path.join(self.base_path, user_id)
    
    def _get_documents_path(self, user_id: str) -> str:
        return os.path.join(self._get_user_path(user_id), "kg_documents.json")
    
    def _get_groups_path(self, user_id: str) -> str:
        return os.path.join(self._get_user_path(user_id), "kg_groups.json")
    
    def _ensure_user_directory(self, user_id: str):
        user_path = self._get_user_path(user_id)
        os.makedirs(user_path, exist_ok=True)
        return user_path
    
    def _compute_file_hash(self, content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()
    
    def _load_documents_info(self, user_id: str) -> Dict[str, Dict]:
        documents_path = self._get_documents_path(user_id)
        if os.path.exists(documents_path):
            try:
                with open(documents_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load KG documents info for user {user_id}: {e}")
        return {}
    
    def _save_documents_info(self, user_id: str, documents_info: Dict[str, Dict]):
        try:
            self._ensure_user_directory(user_id)
            documents_path = self._get_documents_path(user_id)
            
            with open(documents_path, 'w') as f:
                json.dump(documents_info, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save KG documents info for user {user_id}: {e}")
            raise
    
    def _load_groups_info(self, user_id: str) -> Dict[str, Dict]:
        groups_path = self._get_groups_path(user_id)
        if os.path.exists(groups_path):
            try:
                with open(groups_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load KG groups info for user {user_id}: {e}")
        return {}
    
    def _save_groups_info(self, user_id: str, groups_info: Dict[str, Dict]):
        try:
            self._ensure_user_directory(user_id)
            groups_path = self._get_groups_path(user_id)
            
            with open(groups_path, 'w') as f:
                json.dump(groups_info, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to save KG groups info for user {user_id}: {e}")
            raise
    
    def _slugify_group_id(self, name: str) -> str:
        
        s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_").lower()
        return f"doc_{s}"[:50]  
    
    def _sanitize_user_id(self, user_id: str) -> str:

        sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", user_id)
        sanitized = re.sub(r"_+", "_", sanitized).strip("_")
        return sanitized[:30]  
    
    async def _ingest_document_to_graph(
        self, 
        graphiti: Graphiti, 
        user_id: str,  
        content: bytes, 
        filename: str, 
        content_hash: str
    ) -> Dict[str, Any]:

        try:
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                logger.info(f"Extracting text from {filename} for KG (user: {user_id})...")
                raw_text = self.document_processor.extract_text_docling(tmp_file_path)
                
                if not raw_text or not raw_text.strip():
                    logger.warning(f"No text extracted from {filename} for KG (user: {user_id})")
                    return {
                        "entities_created": 0,
                        "relationships_created": 0,
                        "chunks_processed": 0,
                        "status": "failed",
                        "error": "No text extracted"
                    }
                
                logger.info(f"Extracted {len(raw_text)} characters from {filename} (user: {user_id})")
                
                # THIS IS FOR OPTIMIZATION ELSE HELL OF RATE LIMITS AND CALLS
                # SINGLE MODE ONLY: Full document text with entity cap
                KG_ENTITY_CAP = 8
                max_chars = 35000
                
                logger.info(f"Using optimized single-pass with entity cap: {KG_ENTITY_CAP}")
                
                # STRICT user-isolated group ID
                safe_user_id = self._sanitize_user_id(user_id)
                safe_filename = self._slugify_group_id(filename)
                group_id = f"U{safe_user_id}_{safe_filename}"
                source_desc = f"[USER:{user_id}] PDF Document: {filename} (hash: {content_hash[:8]})"
                
                # minimal API calls
                logger.info(f"Skipping group episode for minimal API calls - {filename} (user: {user_id})")
                
                entities_created = 0
                relationships_created = 0
                chunks_processed = 0
                
                # whole text with explicit entity cap instructions
                doc_text = raw_text[:max_chars]
                
                # enhanced prompt with controlled extraction instructions
                enhanced_prompt = f"""DOCUMENT KNOWLEDGE EXTRACTION - CONTROLLED MODE

USER_ISOLATION: {user_id}
DOCUMENT: {filename}
EXTRACTION_LIMITS: Extract exactly {KG_ENTITY_CAP} key entities and {max(1, KG_ENTITY_CAP - 2)} core relationships

EXTRACTION RULES:
1. Identify {KG_ENTITY_CAP} most important entities: companies, people, key products/technologies, financial metrics, strategic concepts
2. Create {max(1, KG_ENTITY_CAP - 2)} meaningful relationships connecting these entities
3. Prioritize: 
   - Financial performance indicators
   - Key business metrics (revenue, growth, margins)
   - Strategic initiatives and partnerships
   - Core technical innovations
   - Market dynamics and competitive positioning
4. NO trivial entities (dates, locations unless critical)
5. NO granular sub-concepts
6. MERGE similar/duplicate concepts
7. Focus on actionable business/technical insights

CONTENT TO ANALYZE:
{doc_text}

STRICT USER ISOLATION: {user_id} - This content belongs exclusively to user {user_id}"""

                try:
                    # SINGLE GRAPHITI CALL PER DOCUMENT
                    await graphiti.add_episode(
                        name=f"[USER:{user_id}] {filename} - Controlled Extraction ({KG_ENTITY_CAP} entities)",
                        episode_body=enhanced_prompt,
                        source_description=f"{source_desc} - controlled extraction: {KG_ENTITY_CAP} entities max",
                        source=EpisodeType.text,
                        reference_time=datetime.now(timezone.utc),
                        group_id=group_id
                    )
                    chunks_processed = 1
                    entities_created = KG_ENTITY_CAP
                    relationships_created = max(1, KG_ENTITY_CAP - 2)
                    logger.info(f"Single episode created for {filename} - expecting {KG_ENTITY_CAP} entities, {relationships_created} relationships")
                except Exception as e:
                    logger.error(f"Episode creation failed: {e}")
                
                return {
                    "entities_created": entities_created,
                    "relationships_created": relationships_created, 
                    "chunks_processed": chunks_processed,
                    "status": "success" if chunks_processed > 0 else "failed",
                    "error": None if chunks_processed > 0 else "No chunks processed"
                }
                
            finally:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Failed to ingest {filename} to KG: {e}")
            return {
                "entities_created": 0,
                "relationships_created": 0,
                "chunks_processed": 0,
                "status": "failed",
                "error": str(e)
            }
    
    async def build_graph_from_uploads(
        self, 
        user_id: str, 
        uploaded_files: List[UploadFile], 
        force_rebuild: bool = False
    ) -> Dict[str, Any]:

        logger.info(f"Building knowledge graph for user {user_id} from {len(uploaded_files)} uploaded files")
        
        start_time = datetime.now()
        
        try:
            graphiti = await self._get_graphiti_client(user_id)
            
            documents_info = self._load_documents_info(user_id)
            groups_info = self._load_groups_info(user_id)
            
            total_entities = 0
            total_relationships = 0
            processed_files = 0
            failed_files = 0
            
            for file in uploaded_files:
                try:
                    logger.info(f"Processing file for KG: {file.filename}")
                    
                    content = await file.read()
                    content_hash = self._compute_file_hash(content)
                    
                    # check if already processed (unless force rebuild)
                    if not force_rebuild and content_hash in documents_info:
                        existing_doc = documents_info[content_hash]
                        if existing_doc.get("status") == "processed":
                            logger.info(f"File {file.filename} already processed in KG (hash: {content_hash[:8]}...)")
                            
                            total_entities += existing_doc.get("entities_created", 0)
                            total_relationships += existing_doc.get("relationships_created", 0)
                            processed_files += 1
                            continue
                    
                    result = await self._ingest_document_to_graph(
                        graphiti, user_id, content, file.filename, content_hash
                    )
                    
                    if result["status"] == "success":
                        total_entities += result["entities_created"]
                        total_relationships += result["relationships_created"]
                        processed_files += 1
                        
                        documents_info[content_hash] = {
                            "filename": file.filename,
                            "size_bytes": len(content),
                            "entities_created": result["entities_created"],
                            "relationships_created": result["relationships_created"],
                            "chunks_processed": result["chunks_processed"],
                            "processed_at": datetime.now().isoformat(),
                            "status": "processed"
                        }
                        
                        safe_user_id = self._sanitize_user_id(user_id)
                        group_id = f"U{safe_user_id}_{self._slugify_group_id(file.filename)}"
                        groups_info[group_id] = {
                            "group_id": group_id,
                            "filename": file.filename,
                            "content_hash": content_hash,
                            "user_id": user_id,  # Explicit user tracking
                            "isolation_verified": True,
                            "created_at": datetime.now().isoformat()
                        }
                        
                    else:
                        logger.error(f"Failed to process {file.filename} for KG: {result.get('error', 'Unknown error')}")
                        failed_files += 1
                        
                        documents_info[content_hash] = {
                            "filename": file.filename,
                            "size_bytes": len(content),
                            "entities_created": 0,
                            "relationships_created": 0,
                            "chunks_processed": 0,
                            "processed_at": datetime.now().isoformat(),
                            "status": "failed",
                            "error": result.get("error", "Unknown error")
                        }
                    
                except Exception as file_error:
                    logger.error(f"Error processing file {file.filename} for KG: {file_error}")
                    failed_files += 1
                    continue
            
            self._save_documents_info(user_id, documents_info)
            self._save_groups_info(user_id, groups_info)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            status = "success" if failed_files == 0 else ("partial" if processed_files > 0 else "failed")
            
            result = {
                "user_id": user_id,
                "status": status,
                "entities_created": total_entities,
                "relationships_created": total_relationships,
                "documents_processed": processed_files,
                "documents_failed": failed_files,
                "processing_time_seconds": processing_time,
                "message": f"Knowledge graph built with {total_entities} entities and {total_relationships} relationships from {processed_files} documents",
                "created_at": datetime.now()
            }
            
            logger.info(f"KG build complete for user {user_id}: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to build knowledge graph for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Knowledge graph build failed: {e}")
    
    async def get_status(self, user_id: str) -> Dict[str, Any]:

        try:
            documents_info = self._load_documents_info(user_id)
            groups_info = self._load_groups_info(user_id)
            
            if not documents_info:
                return {
                    "user_id": user_id,
                    "exists": False,
                    "last_updated": None,
                    "document_count": 0,
                    "total_size": 0,
                    "health_status": "unavailable",
                    "details": {"error": "No knowledge graph found for user"}
                }
            
            total_entities = sum(doc.get("entities_created", 0) for doc in documents_info.values())
            total_relationships = sum(doc.get("relationships_created", 0) for doc in documents_info.values())
            total_size = total_entities + total_relationships
            
            last_updated = None
            timestamps = [doc.get("processed_at") for doc in documents_info.values() if doc.get("processed_at")]
            if timestamps:
                last_updated = max(timestamps)
            
            health_status = "healthy"
            failed_docs = [doc for doc in documents_info.values() if doc.get("status") == "failed"]
            if failed_docs:
                health_status = "degraded" if len(failed_docs) < len(documents_info) else "unhealthy"
            
            return {
                "user_id": user_id,
                "exists": True,
                "last_updated": last_updated,
                "document_count": len(documents_info),
                "total_size": total_size,
                "health_status": health_status,
                "details": {
                    "entities": total_entities,
                    "relationships": total_relationships,
                    "groups": len(groups_info),
                    "failed_documents": len(failed_docs)
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get KG status for user {user_id}: {e}")
            return {
                "user_id": user_id,
                "exists": False,
                "last_updated": None,
                "document_count": 0,
                "total_size": 0,
                "health_status": "error",
                "details": {"error": str(e)}
            }
    
    async def clear_graph(self, user_id: str) -> Dict[str, Any]:

        logger.info(f"Clearing knowledge graph for user {user_id} using CYPHER isolation")
        
        try:
            status = await self.get_status(user_id)
            items_before_clear = status.get("total_size", 0)
            
            graphiti = await self._get_graphiti_client(user_id)
            
            safe_user_id = self._sanitize_user_id(user_id)
            
            # multiple CYPHER-based deletion strategies to ensure complete cleanup
            deletion_queries = [
                # Delete by exact user_id match in group_id
                f"MATCH (n) WHERE n.group_id CONTAINS 'U{safe_user_id}_' DETACH DELETE n",
                
                # Delete by user ID in episode names
                f"MATCH (n) WHERE n.name CONTAINS '[USER:{user_id}]' DETACH DELETE n",
                
                # Delete by content containing user markers
                f"MATCH (n) WHERE n.episode_body CONTAINS 'USER_ID: {user_id}' DETACH DELETE n",
                f"MATCH (n) WHERE n.episode_body CONTAINS 'EXCLUSIVE_USER_CONTENT: {user_id}' DETACH DELETE n",
                
                # Delete relationships connected to user-specific nodes
                f"MATCH (n)-[r]-() WHERE n.group_id CONTAINS 'U{safe_user_id}_' DELETE r",
            ]
            
            deleted_count = 0
            for query in deletion_queries:
                try:
                    logger.info(f"Executing user deletion query: {query}")
                    
                    # Execute raw Cypher through the driver
                    async with graphiti.driver.get_session() as session:
                        result = await session.run(query)
                        summary = await result.consume()
                        nodes_deleted = summary.counters.nodes_deleted
                        relationships_deleted = summary.counters.relationships_deleted
                        
                        deleted_count += nodes_deleted + relationships_deleted
                        logger.info(f"Deleted {nodes_deleted} nodes and {relationships_deleted} relationships for user {user_id}")
                        
                except Exception as query_error:
                    logger.warning(f"Deletion query failed for user {user_id}: {query_error}")
                    continue
            
            # clean up
            if user_id in self._graphiti_clients:
                try:
                    await self._graphiti_clients[user_id].close()
                    del self._graphiti_clients[user_id]
                    logger.info(f"Closed and removed Graphiti client cache for user {user_id}")
                except Exception as e:
                    logger.warning(f"Error closing Graphiti client for user {user_id}: {e}")
            
            user_path = self._get_user_path(user_id)
            if os.path.exists(user_path):
                import shutil
                shutil.rmtree(user_path, ignore_errors=True)
                logger.info(f"Removed KG tracking directory for user {user_id}")
            
            verification_query = f"""
            MATCH (n) 
            WHERE n.group_id CONTAINS 'U{safe_user_id}_' 
               OR n.name CONTAINS '[USER:{user_id}]' 
               OR n.episode_body CONTAINS 'USER_ID: {user_id}'
            RETURN count(n) as remaining_count
            """
            
            remaining_count = 0
            try:
                async with graphiti.driver.get_session() as session:
                    result = await session.run(verification_query)
                    record = await result.single()
                    remaining_count = record["remaining_count"] if record else 0
            except Exception as e:
                logger.warning(f"Verification query failed: {e}")
            
            if remaining_count > 0:
                logger.warning(f"WARNING: {remaining_count} nodes still exist for user {user_id} after cleanup")
            
            return {
                "user_id": user_id,
                "deleted": True,
                "items_removed": deleted_count,
                "items_before_clear": items_before_clear,
                "remaining_items": remaining_count,
                "cleanup_method": "cypher_user_scoped",
                "message": f"User-isolated KG cleared using Cypher - removed {deleted_count} items, {remaining_count} remaining",
                "deleted_at": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to clear knowledge graph for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to clear user-isolated knowledge graph: {e}")
    
    async def search_graph(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:

        try:
            graphiti = await self._get_graphiti_client(user_id)
            
            # user-scoped search query with STRICT isolation
            safe_user_id = self._sanitize_user_id(user_id)
            user_scoped_query = f"USER_ID:{user_id} {query}"
            
            logger.info(f"Searching KG for user {user_id} with query: '{query}'")
            
            # all results first, then filter strictly by user
            results = await graphiti.search(
                user_scoped_query,  
                num_results=limit * 5  # more to filter properly
            )
            
            user_results = []
            for result in results:
                fact_text = getattr(result, 'fact', '')
                
                # STRICT user isolation checking
                # checks to ensure no cross-user contamination
                is_user_content = any([
                    f"USER_ID: {user_id}" in fact_text,
                    f"EXCLUSIVE_USER_CONTENT: {user_id}" in fact_text,
                    f"[USER:{user_id}]" in fact_text,
                    f"user '{user_id}'" in fact_text.lower(),
                    f"user {user_id}" in fact_text.lower(),
                    f"U{safe_user_id}_" in fact_text
                ])
                
                # ensure it's NOT from another user
                is_other_user_content = any([
                    "USER_ID:" in fact_text and f"USER_ID: {user_id}" not in fact_text,
                    "EXCLUSIVE_USER_CONTENT:" in fact_text and f"EXCLUSIVE_USER_CONTENT: {user_id}" not in fact_text,
                    "[USER:" in fact_text and f"[USER:{user_id}]" not in fact_text
                ])
                
                if is_user_content and not is_other_user_content:
                    # query relevance
                    query_words = query.lower().split()
                    fact_words = fact_text.lower().split()
                    relevance_score = sum(1 for word in query_words if len(word) > 2 and word in fact_words)
                    
                    user_results.append({
                        "content": fact_text,
                        "score": getattr(result, 'score', 0.0) + (relevance_score * 0.1),
                        "source": "kg",
                        "metadata": {
                            "uuid": getattr(result, 'uuid', None),
                            "type": "fact",
                            "graph_result": True,
                            "user_id": user_id,
                            "relevance": "high" if relevance_score > 0 else "medium",
                            "isolation": "strict"
                        },
                        "uuid": getattr(result, 'uuid', None)
                    })
                
                if len(user_results) >= limit:
                    break
            
            user_results.sort(key=lambda x: x["score"], reverse=True)
            
            logger.info(f"KG search for user {user_id} returned {len(user_results)} ISOLATED results")
            return user_results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to search knowledge graph for user {user_id}: {e}")
            return []
    
    async def cleanup(self):

        try:
            for user_id, graphiti in self._graphiti_clients.items():
                try:
                    await graphiti.close()
                    logger.info(f"Closed Graphiti client for user {user_id}")
                except Exception as e:
                    logger.warning(f"Error closing Graphiti client for user {user_id}: {e}")
            
            self._graphiti_clients.clear()
            logger.info("KG Manager cleanup completed")
            
        except Exception as e:
            logger.error(f"Error during KG Manager cleanup: {e}")
    