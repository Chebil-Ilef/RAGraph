from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import logging
import uvicorn
from .models import (
    BuildGraphResponse,
    BuildVectorResponse,
    QueryRequest, QueryResponse,
    StatusRequest, StatusResponse, 
    DeleteRequest, DeleteResponse
)
from .dependencies import (
    get_kg_manager, get_vector_manager, get_hybrid_search,
    get_neo4j_connection, get_faiss_connection
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HUGO API",
    description="Hybrid Graph-RAG system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### GENERAL APP ENDPOINTS

@app.on_event("startup")
async def startup_event():

    logger.info("**HUGO API starting up...**")
    logger.info("HUGO API ready!")

@app.on_event("shutdown")
async def shutdown_event():

    logger.info("!! HUGO API shutting down...")

@app.get("/health")
async def health_check(
    neo4j_conn = Depends(get_neo4j_connection),
):
    health_status = {
        "status": "healthy",
        "service": "HUGO API",
        "checks": {
            "neo4j": {"status": "unknown", "details": ""},
        }
    }
    
    # check Neo4j connection
    try:
        neo4j_healthy = await neo4j_conn.health_check()
        if neo4j_healthy:
            health_status["checks"]["neo4j"]["status"] = "healthy"
            health_status["checks"]["neo4j"]["details"] = f"Connected to {neo4j_conn.uri}"
        else:
            health_status["checks"]["neo4j"]["status"] = "unhealthy"
            health_status["checks"]["neo4j"]["details"] = "Connection test failed"
            health_status["status"] = "degraded"

    except Exception as e:
        health_status["checks"]["neo4j"]["status"] = "error"
        health_status["checks"]["neo4j"]["details"] = str(e)
        health_status["status"] = "degraded"
    
    return health_status


### KNOWLEDGE GRAPH ENDPOINTS

@app.post("/kg/build", response_model=BuildGraphResponse)
async def build_knowledge_graph(
    user_id: str = Form(..., description="User identifier"),
    force_rebuild: bool = Form(False, description="Force rebuild even if documents already processed"),
    files: List[UploadFile] = File(..., description="PDF documents to process"),
    kg_manager = Depends(get_kg_manager)
):
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
    
    logger.info(f"Building knowledge graph for user: {user_id} with {len(files)} files")
    
    try:
        if not files:
            raise HTTPException(
                status_code=400, 
                detail="At least one file must be uploaded"
            )
        
        validated_files = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not a PDF. Only PDF files are supported."
                )
            
            # file size (limit to 50MB per file)
            if file.size and file.size > 50 * 1024 * 1024:  # 50MB
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is too large. Maximum size is 50MB."
                )
            
            validated_files.append(file)
        
        result = await kg_manager.build_graph_from_uploads(user_id, validated_files, force_rebuild)
        
        logger.info(f"Knowledge graph built for user {user_id}: {result}")
        return BuildGraphResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to build knowledge graph for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/kg/status", response_model=StatusResponse)
async def get_kg_status(
    request: StatusRequest,
    kg_manager = Depends(get_kg_manager)
):
    if not request.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    user_id = request.user_id
    try:
        status = await kg_manager.get_status(user_id)
        return StatusResponse(**status)
    
    except Exception as e:
        logger.error(f"Failed to get KG status for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/kg/clear", response_model=DeleteResponse)
async def clear_knowledge_graph(
    request: DeleteRequest,
    kg_manager = Depends(get_kg_manager)
):
    
    if not request.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    user_id = request.user_id 

    logger.info(f"Clearing knowledge graph for user: {user_id}")
    
    try:
        result = await kg_manager.clear_graph(user_id)
        logger.info(f"Knowledge graph cleared for user {user_id}")

        return DeleteResponse(**result)
    
    except Exception as e:
        logger.error(f"Failed to clear knowledge graph for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

### VECTOR DATABASE ENDPOINTS 

@app.post("/vectordb/build", response_model=BuildVectorResponse)
async def build_vector_database(
    user_id: str = Form(..., description="User identifier"),
    force_rebuild: bool = Form(False, description="Force rebuild even if documents already processed"),
    files: List[UploadFile] = File(..., description="PDF documents to process"),
    vector_manager = Depends(get_vector_manager)
):
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    logger.info(f"--- Building vector database for user: {user_id} with {len(files)} files")
    
    try:
        if not files:
            raise HTTPException(
                status_code=400,
                detail="At least one file must be uploaded"
            )
        
        validated_files = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is not a PDF. Only PDF files are supported."
                )
            
            # limit to 50MB per file
            if file.size and file.size > 50 * 1024 * 1024:  # 50MB
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} is too large. Maximum size is 50MB."
                )
            
            validated_files.append(file)
        
        result = await vector_manager.build_vector_store_from_uploads(user_id, validated_files, force_rebuild)
        
        logger.info(f"Vector database built for user {user_id}: {result}")
        return BuildVectorResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to build vector database for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/vectordb/status", response_model=StatusResponse)
async def get_vector_status(
    request: StatusRequest,
    vector_manager = Depends(get_vector_manager)
):
    if not request.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    user_id = request.user_id
    try:
        status = await vector_manager.get_status(user_id)
        return StatusResponse(**status)
    
    except Exception as e:
        logger.error(f"Failed to get vector DB status for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/vectordb/clear", response_model=DeleteResponse)
async def clear_vector_database(
    request: DeleteRequest,
    vector_manager = Depends(get_vector_manager)
):

    if not request.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    user_id = request.user_id
    logger.info(f"Clearing vector database for user: {user_id}")
    
    try:
        result = await vector_manager.clear_vector_store(user_id)
        logger.info(f"Vector database cleared for user {user_id}")

        return DeleteResponse(**result)
    
    except Exception as e:
        logger.error(f"Failed to clear vector database for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


### HYBRID QUERY ENDPOINTS

@app.post("/query", response_model=QueryResponse)
async def hybrid_query(
    request: QueryRequest,
    hybrid_search = Depends(get_hybrid_search)
):
    if not request.user_id:
        raise HTTPException(status_code=400, detail="User ID is required")

    user_id = request.user_id
    logger.info(f"🔍 Hybrid query for user {user_id}: {request.query}")
    
    try:
        result = await hybrid_search.query(
            user_id=user_id,
            query=request.query,
            limit=request.limit,
            include_kg=request.include_kg,
            include_vector=request.include_vector
        )
        
        logger.info(f"Query completed for user {user_id}")
        return QueryResponse(**result)
        
    except Exception as e:
        logger.error(f"Query failed for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":

    
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
