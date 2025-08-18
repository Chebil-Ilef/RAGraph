import logging
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class Setup:

    @staticmethod
    def check_health(uri: str, user: str, password: str) -> bool:

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            # verify connectivity by running a simple query
            with driver.session() as session:
                session.run("MATCH (n) RETURN n LIMIT 1")
            driver.close()
            logger.info(f"Neo4j connection successful: {uri}")
            return True
        
        except AuthError as e:
            logger.error(f"Neo4j authentication failed: {e}")
            raise ValueError(f"Invalid Neo4j credentials: {e}")
        
        except ServiceUnavailable as e:
            logger.error(f"Neo4j service unavailable at {uri}: {e}")
            raise ServiceUnavailable(f"Failed to connect to Neo4j at {uri}: {e}")
        
        except Exception as e:
            logger.error(f"Unexpected error during Neo4j health check: {e}")
            raise


    def load_config():

        # Neo4j
        neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
        neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
        neo4j_password = os.environ.get('NEO4J_PASSWORD', 'testpass')

        if not all([neo4j_uri, neo4j_user, neo4j_password]):
            raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

        # Azure OpenAI
        azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT')
        azure_openai_api_key = os.environ.get('AZURE_OPENAI_API_KEY')
        azure_openai_api_version = os.environ.get('AZURE_OPENAI_API_VERSION')
        azure_openai_embeddings_deployment = os.environ.get('AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT')

        azure_openai_chat_deployment = os.environ.get('AZURE_OPENAI_CHAT_DEPLOYMENT')
        if not all([azure_openai_endpoint, azure_openai_api_key, azure_openai_api_version,
                    azure_openai_embeddings_deployment, azure_openai_chat_deployment]):
            raise ValueError('Azure OpenAI environment variables must be set')

        return {
        'neo4j': {'uri': neo4j_uri, 'user': neo4j_user, 'password': neo4j_password},
        'azure_openai': {
            'endpoint': azure_openai_endpoint,
            'api_key': azure_openai_api_key,
            'api_version': azure_openai_api_version,
            'embeddings_deployment': azure_openai_embeddings_deployment,
            'chat_deployment': azure_openai_chat_deployment,
        }
        }
    
