import asyncio
import logging
import os
from glob import glob
from typing import List, Tuple
from openai import AsyncAzureOpenAI  

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

from utils.setup import Setup  
from utils.ingestion import ingest_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_pdf_files() -> List[Tuple[str, str, str]]:
    folders = ["files/Finance"]
    pdf_files = []
    for folder in folders:
        folder_path = os.path.join(folder, "*.pdf")
        for file_path in glob(folder_path):
            file_name = os.path.basename(file_path)
            source_desc = f"Document from {folder.split('/')[-1]} folder"
            pdf_files.append((file_path, file_name, source_desc))
    return pdf_files


async def main():
    try:
        config = Setup.load_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return

    api_key = config["azure_openai"]["api_key"]
    api_version = config["azure_openai"]["api_version"]
    llm_endpoint = config["azure_openai"]["endpoint"]
    emb_endpoint = config["azure_openai"]["endpoint"]  

    chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")         
    emb_deployment = config["azure_openai"]["embeddings_deployment"]     

    llm_client_azure = AsyncAzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=llm_endpoint
    )
    emb_client_azure = AsyncAzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=emb_endpoint
    )

    llm_config = LLMConfig(
        model=chat_deployment,
        small_model=chat_deployment,
    )

    llm_client = OpenAIClient(
        config=llm_config,
        client=llm_client_azure
    )

    embedder = OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            embedding_model=emb_deployment,
            embedding_dim=1536,
        ),
        client=emb_client_azure
    )

    cross_encoder = OpenAIRerankerClient(
        config=LLMConfig(model=llm_config.small_model),
        client=llm_client_azure
    )


    graphiti = Graphiti(
        config["neo4j"]["uri"],
        config["neo4j"]["user"],
        config["neo4j"]["password"],
        llm_client=llm_client,
        embedder=embedder,
        cross_encoder=cross_encoder,
    )

    try:
        logger.info("Building indices and constraints...")
        await graphiti.build_indices_and_constraints()

        logger.info("Ingesting PDF documents...")
        pdf_files = get_pdf_files()
        if not pdf_files:
            logger.warning("No PDF files found in specified folders.")
            return

        # sequential ingestion to avoid API rate limits
        ingestion_results = await ingest_documents(graphiti, pdf_files, max_concurrency=1)
        for result in ingestion_results:
            if result['status'] == 'success':
                logger.info(f"[INGESTED] {result['name']} | chunks={result['chunks']} | group_id={result['group_id']}")
            elif result['status'] == 'partial':
                logger.warning(f"[PARTIAL] {result['name']} | chunks={result['chunks']} | group_id={result['group_id']} | error={result['error']}")
            else:
                logger.error(f"[FAILED] {result['name']} | error={result['error']}")        
        # # Demo queries
        # logger.info("\n--- Running Demo Queries ---")
        # questions = [
        #     "What counts as Confidential Information in the NDAs, and what are the exceptions?",
        #     "Where are Governing Law and Jurisdiction specified in these NDAs?",
        #     "Define Force Majeure in the IT Services Agreement and note any explicit exclusions.",
        #     "What is a Security Breach and what is the notification requirement?",
        #     "What is a Change Order?",
        #     "List the categories of Protected Information.",
        # ]

        # for q in questions:
        #     logger.info(f"\nQuery: {q}")
        #     results = await graphiti.search(q, search_config=NODE_HYBRID_SEARCH_RRF)
        #     for r in results[:3]:
        #         logger.info(f" • {r.fact} [uuid={getattr(r, 'uuid', None)}]")

    finally:
        logger.info("Closing Graphiti connection...")
        await graphiti.close()


if __name__ == "__main__":
    asyncio.run(main())
