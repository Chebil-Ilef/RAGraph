import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class Setup:

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
    
