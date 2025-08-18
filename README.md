docker run -d --name neo4j-graphiti \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/testpass \
  -v $PWD/neo4j_data:/data \
  neo4j:5.26



MATCH (n) DETACH DELETE n;


