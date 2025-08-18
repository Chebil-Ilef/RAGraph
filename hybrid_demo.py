"""
Hybrid Graph-RAG Demo

This demonstrates the recommended architecture:
- Chunks → FAISS vector store (cheap, fast retrieval)
- Entities/Relations → Graphiti (structured knowledge)
- Hybrid queries that leverage both systems
"""

import asyncio
import logging
import os
from glob import glob
from typing import List, Tuple
from openai import AsyncAzureOpenAI

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

from utils.setup import Setup
from utils.hybrid_ingestion import hybrid_ingest_documents
from utils.hybrid_query import demo_hybrid_query

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

def get_pdf_files() -> List[Tuple[str, str, str]]:
    """Get list of PDF files to process."""
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
    """Main function demonstrating hybrid graph-RAG."""
    try:
        config = Setup.load_config()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return

    # Setup Azure OpenAI clients
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

    # Setup Graphiti components
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

    # Initialize Graphiti
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

        logger.info("🔄 HYBRID INGESTION: Chunks → Vector Store, Entities → Graph")
        pdf_files = get_pdf_files()
        if not pdf_files:
            logger.warning("No PDF files found in specified folders.")
            return

        # Hybrid ingestion
        ingestion_results = await hybrid_ingest_documents(graphiti, pdf_files)
        
        # Report results
        total_chunks = 0
        total_entities = 0
        for result in ingestion_results:
            if result['status'] == 'success':
                chunks = result['chunks']
                entities = result['entities']
                total_chunks += chunks
                total_entities += entities
                logger.info(f"✅ {result['name']} | chunks: {chunks} (vector) | entities: {entities} (graph)")
            else:
                logger.error(f"❌ {result['name']} | error: {result['error']}")

        logger.info(f"\n📊 INGESTION SUMMARY:")
        logger.info(f"   • Total chunks in vector store: {total_chunks}")
        logger.info(f"   • Total entities in graph: {total_entities}")
        logger.info(f"   • Documents processed: {len([r for r in ingestion_results if r['status'] == 'success'])}")

        # Demo hybrid queries
        logger.info("\n🔍 HYBRID QUERY DEMO")
        questions = [
            "What are the key financial metrics mentioned in the earnings reports?",
            "Which companies are discussed and what are their relationships?",
            "What are the main risks or challenges mentioned?",
            "What growth strategies are outlined in these documents?",
        ]

        await demo_hybrid_query(graphiti, questions)

    finally:
        logger.info("Closing Graphiti connection...")
        await graphiti.close()

if __name__ == "__main__":
    print("🚀 Starting Hybrid Graph-RAG Demo")
    print("📋 Architecture:")
    print("   • Document chunks → FAISS vector store")
    print("   • Entities & relations → Graphiti knowledge graph")
    print("   • Queries leverage both systems for optimal results")
    print("=" * 60)
    
    asyncio.run(main())
