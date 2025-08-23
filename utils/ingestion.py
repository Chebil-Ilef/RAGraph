import asyncio
import logging
import re
import unicodedata
from typing import List, Tuple, Dict, Any
from datetime import datetime, timezone
from graphiti_core import Graphiti
import graphiti_core
from graphiti_core.nodes import EpisodeType
from utils.document_processor import DocumentProcessor

logger = logging.getLogger(__name__)

def slugify_group_id(name: str) -> str:
    """Create a slugified group ID from a name."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s

async def ingest_documents(graphiti: Graphiti, pdf_files: List[Tuple[str, str, str]], max_concurrency: int = 1) -> List[Dict[str, Any]]:
    """
    Ingest multiple PDF documents sequentially to avoid API rate limits.
    Args:
        graphiti: Graphiti instance for episode creation.
        pdf_files: List of tuples (file_path, name, source_desc) for PDFs to ingest.
        max_concurrency: Kept for compatibility but now processes sequentially.
    Returns:
        List of dictionaries containing ingestion results: {'name': str, 'chunks': int, 'group_id': str, 'status': str, 'error': str|None}.
    """
    processor = DocumentProcessor()
    results = []

    async def process_document(file_path: str, name: str, source_desc: str) -> Dict[str, Any]:
        try:
            # Validate inputs
            if not file_path or not name or not source_desc:
                raise ValueError(f"Invalid input: file_path={file_path}, name={name}, source_desc={source_desc}")

            # Extract text and create semantic chunks
            logger.info(f"Extracting text from {file_path}...")
            text = processor.extract_text_docling(file_path)
            if not text.strip():
                raise ValueError("Extracted text is empty")

            logger.info(f"Creating semantic chunks for {name}...")
            chunks = processor.semantic_chunk(text)
            if not chunks:
                raise ValueError("No chunks generated")

            # Create a group episode
            group_id = f"doc_{slugify_group_id(name)}"
            try:
                logger.info(f"Creating group episode for {name}...")
                group_episode = await graphiti.add_episode(
                    name=name,
                    episode_body=f"Document group for {name}",
                    source_description=source_desc,
                    source=EpisodeType.text,
                    reference_time=datetime.now(timezone.utc),
                    group_id=group_id
                )
                logger.info(f"Group episode created with ID: {group_id}")
            except Exception as e:
                logger.warning(f"Failed to create group episode for {name}: {e}. Proceeding with independent chunks.")

            # Add chunk episodes sequentially with delay to avoid rate limits
            chunk_count = 0
            for i, chunk in enumerate(chunks, 1):
                try:
                    # Skip empty or whitespace-only chunks
                    if not chunk.strip():
                        logger.debug(f"Skipping empty chunk {i} for {name}")
                        continue
                        
                    logger.debug(f"Adding chunk {i} for {name}...")
                    await graphiti.add_episode(
                        name=f"{name} - Chunk {i}",
                        episode_body=chunk,
                        source_description=source_desc,
                        source=EpisodeType.text,
                        reference_time=datetime.now(timezone.utc),
                        group_id=group_id
                    )
                    chunk_count += 1
                    
                    # Add a small delay between chunks to help with rate limits
                    await asyncio.sleep(0.5)
                    
                except Exception as chunk_error:
                    logger.error(f"Failed to add chunk {i} for {name}: {chunk_error}")
                    continue  # Continue with next chunk

            return {
                'name': name,
                'chunks': chunk_count,
                'group_id': group_id,
                'status': 'success' if chunk_count > 0 else 'partial',
                'error': None if chunk_count > 0 else 'No chunks ingested'
            }
        except Exception as e:
            logger.error(f"Failed to ingest {name}: {e}", exc_info=True)
            return {
                'name': name,
                'chunks': 0,
                'group_id': None,
                'status': 'failed',
                'error': str(e)
            }

    # Process documents sequentially to avoid rate limits
    for file_path, name, source_desc in pdf_files:
        logger.info(f"Processing document {len(results) + 1}/{len(pdf_files)}: {name}")
        result = await process_document(file_path, name, source_desc)
        results.append(result)
        
        # Add delay between documents to help with rate limits
        if len(results) < len(pdf_files):  # Don't delay after the last document
            logger.debug("Waiting between documents to avoid rate limits...")
            await asyncio.sleep(2.0)
    
    return results