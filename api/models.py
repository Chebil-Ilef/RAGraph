from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# REQUEST MODELS

class QueryRequest(BaseModel):

    user_id: str = Field(
        description="User identifier"
    )
    query: str = Field(
        description="Natural language query string"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )
    include_kg: bool = Field(
        default=True,
        description="Include knowledge graph results"
    )
    include_vector: bool = Field(
        default=True,
        description="Include vector database results"
    )
    rerank: bool = Field(
        default=True,
        description="Apply reranking to results"
    )

class StatusRequest(BaseModel):

    user_id: str = Field(
        description="User identifier"
    )

class DeleteRequest(BaseModel):

    user_id: str = Field(
        description="User identifier"
    )

# RESPONSE MODELS

class BuildGraphResponse(BaseModel):

    user_id: str
    status: str  # "success", "partial", "failed"
    entities_created: int
    relationships_created: int
    documents_processed: int
    documents_failed: int
    processing_time_seconds: float
    message: str
    created_at: datetime

class BuildVectorResponse(BaseModel):

    user_id: str
    status: str  # "success", "partial", "failed"
    chunks_created: int
    documents_processed: int
    documents_failed: int
    vector_dimension: int
    processing_time_seconds: float
    message: str
    created_at: datetime

class StatusResponse(BaseModel):

    user_id: str
    exists: bool
    last_updated: Optional[datetime]
    document_count: int
    total_size: int  # entities/relationships for KG, chunks for vector
    health_status: str  # "healthy", "degraded", "unavailable"
    details: Dict[str, Any]

class DeleteResponse(BaseModel):

    user_id: str
    deleted: bool
    items_removed: int
    message: str
    deleted_at: datetime

class QueryResult(BaseModel):

    content: str = Field(description="Result content/text")
    score: float = Field(description="Relevance score")
    source: str = Field(description="Source type: 'kg' or 'vector'")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (document, page, etc.)"
    )
    uuid: Optional[str] = Field(
        default=None,
        description="Unique identifier for KG results"
    )

class QueryResponse(BaseModel):

    user_id: str
    query: str
    total_results: int
    kg_results: int
    vector_results: int
    results: List[QueryResult]
    processing_time_seconds: float
    fallback_used: Optional[str] = Field(
        default=None,
        description="Fallback mode used if primary method failed"
    )
    message: str

# INTERNAL MODELS

class DocumentInfo(BaseModel):

    file_path: str
    file_name: str
    content_hash: str
    size_bytes: int
    processed_at: datetime
    status: str  # "processed", "failed", "partial"

class UserSession(BaseModel):

    user_id: str
    kg_available: bool
    vector_available: bool
    last_activity: datetime
    document_count: int
