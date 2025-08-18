docker run -d --name neo4j-graphiti \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/testpass \
  -v $PWD/neo4j_data:/data \
  neo4j:5.26


docker start neo4j-graphiti

MATCH (n) DETACH DELETE n;


# Try without the resume flag
hf download ds4sd/docling-layout-old --local-dir ~/.cache/huggingface/hub/models--ds4sd--docling-layout-old/