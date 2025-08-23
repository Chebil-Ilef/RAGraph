import asyncio
import os
import time
from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

load_dotenv()

neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD', 'testpass')

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

async def run_demo_and_save(graphiti, questions, output_path="showcase.md"):
    md_lines = ["# Graphiti PDF QA Demo\n"]
    
    for q in questions:
        start_time = time.perf_counter()
        res = await graphiti.search(q)
        elapsed = time.perf_counter() - start_time

        md_lines.append(f"## Question\n{q}\n")
        
        if not res:
            md_lines.append("**Answer**\n_Not found in documents._\n")
        else:
            top_result = res[0]
            fact_text = getattr(top_result, "fact", "").strip()
            source_name = getattr(top_result, "source_name", "Unknown Source")
            md_lines.append(f"**Answer**\n{fact_text}\n\n")

        md_lines.append(f"**Time to Answer**: {elapsed:.2f} seconds\n")
        md_lines.append("\n---\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"[SAVED] Demo results to {output_path}")


async def main():
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
        await run_demo_and_save(graphiti, questions)
    finally:
        await graphiti.close()


if __name__ == '__main__':
    asyncio.run(main())
