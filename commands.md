docker run -d --name neo4j-graphiti \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/testpass \
  -v $PWD/neo4j_data:/data \
  neo4j:5.26


docker start neo4j-graphiti

MATCH (n) DETACH DELETE n;


# Try without the resume flag
hf download ds4sd/docling-layout-old --local-dir ~/.cache/huggingface/hub/models--ds4sd--docling-layout-old/


nohup python -m uvicorn api.main:app --host 0.0.0.0 --port 8000

pkill -f uvicorn


# File Upload Methods
POST /kg/build              # Upload files for KG
POST /vectordb/build        # Upload files for Vector DB

# File Path Methods  
POST /kg/build-paths        # Process server files for KG
POST /vectordb/build-paths  # Process server files for Vector DB

# Other endpoints unchanged
POST /kg/status, /kg/clear, /vectordb/status, /vectordb/clear, /query