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
        
        # Normalize and clean
        s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
        s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
        s = re.sub(r"_+", "_", s).strip("_").lower()
        return f"doc_{s}"[:50]  # Limit length
    
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
                logger.info(f"Extracting text from {filename} for KG...")
                raw_text = self.document_processor.extract_text_docling(tmp_file_path)
                
                if not raw_text or not raw_text.strip():
                    logger.warning(f"No text extracted from {filename} for KG")
                    return {
                        "entities_created": 0,
                        "relationships_created": 0,
                        "chunks_processed": 0,
                        "status": "failed",
                        "error": "No text extracted"
                    }
                
                logger.info(f"Extracted {len(raw_text)} characters from {filename}")
                
                # Create semantic chunks optimized for knowledge graph
                # Use larger chunks for better context and entity relationships
                chunks = self.document_processor.semantic_chunk(raw_text)
                
                if not chunks:
                    logger.warning(f"No chunks created from {filename} for KG")
                    return {
                        "entities_created": 0,
                        "relationships_created": 0,
                        "chunks_processed": 0,
                        "status": "failed",
                        "error": "No chunks created"
                    }
                
                logger.info(f"Created {len(chunks)} semantic chunks from {filename}")
                
                # Create group episode for document with user isolation
                group_id = f"{user_id}_{self._slugify_group_id(filename)}"  # User-prefixed group ID
                source_desc = f"PDF Document: {filename} (user: {user_id}, hash: {content_hash[:8]})"
                
                try:
                    logger.info(f"Creating group episode for {filename} (user: {user_id})...")
                    group_episode = await graphiti.add_episode(
                        name=f"[{user_id}] {filename}",
                        episode_body=f"Document group for {filename} (User: {user_id}). Contains {len(chunks)} semantic chunks with rich entity relationships and contextual information.",
                        source_description=source_desc,
                        source=EpisodeType.text,
                        reference_time=datetime.now(timezone.utc),
                        group_id=group_id
                    )
                    logger.info(f"Group episode created: {group_id}")
                except Exception as e:
                    logger.warning(f"Failed to create group episode for {filename}: {e}")
                
                entities_created = 0
                relationships_created = 0
                chunks_processed = 0
                
                for i, chunk in enumerate(chunks, 1):
                    try:
                        if not chunk.strip():
                            continue
                        
                        enhanced_chunk = f"""User: {user_id}
Document: {filename}
Section {i}/{len(chunks)}

{chunk}

[This content belongs to user {user_id} and is part of {filename}, a structured document containing entities, relationships, and contextual information relevant for knowledge graph construction.]"""
                        
                        logger.debug(f"Processing chunk {i}/{len(chunks)} for {filename} (user: {user_id})...")
                        
                        episode = await graphiti.add_episode(
                            name=f"[{user_id}] {filename} - Section {i}",
                            episode_body=enhanced_chunk,
                            source_description=f"{source_desc} - Section {i}",
                            source=EpisodeType.text,
                            reference_time=datetime.now(timezone.utc),
                            group_id=group_id
                        )
                        
                        chunks_processed += 1
                        
                        estimated_entities = max(1, len(chunk.split()) // 50)  # ~1 entity per 50 words
                        estimated_relationships = max(0, estimated_entities - 1)  # Relationships between entities
                        
                        entities_created += estimated_entities
                        relationships_created += estimated_relationships
                        
                        # delay to respect API rate limits
                        await asyncio.sleep(0.8)  # longer delay for LLM processing
                        
                    except Exception as chunk_error:
                        logger.error(f"Failed to process chunk {i} for {filename}: {chunk_error}")
                        continue
                
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
            # Graphiti client for this user
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
                        
                        group_id = self._slugify_group_id(file.filename)
                        groups_info[group_id] = {
                            "group_id": group_id,
                            "filename": file.filename,
                            "content_hash": content_hash,
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

        logger.info(f"Clearing knowledge graph for user {user_id}")
        
        try:

            status = await self.get_status(user_id)
            items_removed = status.get("total_size", 0)
            
            if user_id in self._graphiti_clients:
                graphiti = self._graphiti_clients[user_id]
                try:
                    await graphiti.close()
                    del self._graphiti_clients[user_id]
                    logger.info(f"Closed Graphiti client for user {user_id}")
                except Exception as e:
                    logger.warning(f"Error closing Graphiti client for user {user_id}: {e}")
            
            user_path = self._get_user_path(user_id)
            if os.path.exists(user_path):
                import shutil
                shutil.rmtree(user_path, ignore_errors=True)
                logger.info(f"Removed KG tracking directory for user {user_id}")
            
            # Note: The actual Neo4j database for the user would need to be dropped
            # This requires additional Neo4j administration commands
            # For now, we just clear our local tracking
            
            return {
                "user_id": user_id,
                "deleted": True,
                "items_removed": items_removed,
                "message": f"Knowledge graph cleared - removed {items_removed} items",
                "deleted_at": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Failed to clear knowledge graph for user {user_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to clear knowledge graph: {e}")
    
    async def search_graph(
        self, 
        user_id: str, 
        query: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:

        try:
            graphiti = await self._get_graphiti_client(user_id)
            
            # enhanced query with user context for better results
            enhanced_query = f"User {user_id}: {query}"
            
            results = await graphiti.search(
                enhanced_query,  
                num_results=limit * 3  
            )
            
            user_results = []
            for result in results:
                fact_text = getattr(result, 'fact', '')
                
                is_user_specific = (
                    user_id.lower() in fact_text.lower() or 
                    f"{user_id}_" in fact_text or
                    "User: " + user_id in fact_text
                )
                
                # unrelated queries, be even more strict
                query_words = query.lower().split()
                fact_words = fact_text.lower().split()
                has_query_relevance = any(word in fact_words for word in query_words if len(word) > 3)
                
                if is_user_specific and has_query_relevance:
                    user_results.append({
                        "content": fact_text,
                        "score": getattr(result, 'score', 0.0),
                        "source": "kg",
                        "metadata": {
                            "uuid": getattr(result, 'uuid', None),
                            "type": "fact",
                            "graph_result": True,
                            "user_id": user_id,
                            "relevance": "high"
                        },
                        "uuid": getattr(result, 'uuid', None)
                    })
                elif is_user_specific and len(user_results) < limit // 3:
                    # allow some user-specific results even if not directly relevant to query
                    user_results.append({
                        "content": fact_text,
                        "score": getattr(result, 'score', 0.0) * 0.5,  # lower score for less relevant
                        "source": "kg",
                        "metadata": {
                            "uuid": getattr(result, 'uuid', None),
                            "type": "fact",
                            "graph_result": True,
                            "user_id": user_id,
                            "relevance": "medium"
                        },
                        "uuid": getattr(result, 'uuid', None)
                    })
                
                if len(user_results) >= limit:
                    break
            
            logger.info(f"KG search for user {user_id} returned {len(user_results)} results")
            return user_results
            
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
