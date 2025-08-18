import asyncio
import json
import logging
import os
from datetime import datetime, timezone
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from chunking import ingest_document

#################################################
# CONFIGURATION
#################################################

logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'testpass')

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')



async def main():
    # INITIALIZATION

    llm_config = LLMConfig(
    api_key=os.environ["OPENAI_API_KEY"],
    model="gpt-4o-mini",      
    small_model="gpt-4o-mini",  
)

    llm_client = OpenAIClient(config=llm_config)

    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=os.environ["OPENAI_API_KEY"],
            embedding_model="text-embedding-3-small", 
            embedding_dim=1536,
        )
    )

    cross_encoder = OpenAIRerankerClient(client=llm_client, config=llm_config)


    graphiti = Graphiti(
        neo4j_uri, 
        neo4j_user, 
        neo4j_password,
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    try:
        await graphiti.build_indices_and_constraints()

        # ADDING EPISODES
        docs = [
            # ("files/oneNDA.pdf",                            "oneNDA v2.0",            "nda")
            ("files/Mutual-NDA-public.pdf",                 "CommonPaper MNDA",       "nda"),
            ("files/Press Release - Agreement to Acquire DS Smith.pdf",  "IP + DS Smith PR", "press_release"),
            ("files/IT_Service_Agreement_Final.pdf",        "EBMUD IT Services",      "contract"),
            ("files/Cooley SaaS Agreement ACC Form.pdf",    "Cooley SaaS Agreement",  "saas_agreement"),
        ]

        for p, name, desc in docs:
            info = await ingest_document(graphiti, doc_path=p, name=name, source_desc=desc)
            print(f"[INGESTED] {info['name']} | chunks={info['chunks']} | group_id={info['group_id']}")

        # DEMO

        print("\n--- DEMO QUERIES ---")
        questions = [
            "What counts as Confidential Information in the NDAs, and what are the exceptions?",
            "Where are Governing Law and Jurisdiction specified in these NDAs?",
            "Define Force Majeure in the IT Services Agreement and note any explicit exclusions.",
            "What is a Security Breach and what is the notification requirement?",
            "What is a Change Order?",
            "List the categories of Protected Information.",
            "How much synergy is projected in the IP + DS Smith deal and what's the breakdown?",
            "What is the share exchange ratio and expected closing timeline for IP + DS Smith?",
            "Where will the combined company's EMEA headquarters be located?",
        ]
        for q in questions:
            print(f"\nQ: {q}")
            res = await graphiti.search(q)
            for r in res[:3]:
                print(f" • {r.fact}  [uuid={getattr(r, 'uuid', None)}]")

    finally:
        #################################################
        # CLEANUP

        # Close the connection
        await graphiti.close()
        print('\nConnection closed')


if __name__ == '__main__':
    asyncio.run(main())
    