import pdfplumber
from typing import List
from datetime import datetime, timezone
import re
import unicodedata
from graphiti_core.nodes import EpisodeType

def chunk_text(text: str, max_len: int = 1200, overlap: int = 150) -> List[str]:
    text = " ".join(text.split())  
    chunks, i = [], 0
    while i < len(text):
        end = min(len(text), i + max_len)
        chunks.append(text[i:end])
        if end == len(text): break
        i = end - overlap
    return chunks

def extract_pdf_text(path: str) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return "\n".join(pages)


def slugify_group_id(name: str) -> str:
    # ascii fold, keep only [A-Za-z0-9_-], collapse runs to "_"
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^A-Za-z0-9_-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_").lower()
    return s

async def ingest_document(graphiti, doc_path: str, name: str, source_desc: str):
    """Ingest a PDF into Graphiti as text episodes, skipping entity extraction."""
    body = extract_pdf_text(doc_path)
    pieces = chunk_text(body, max_len=2200, overlap=200)

    group_id = f"doc_{slugify_group_id(name)}"
    uuids = []
    now = datetime.now(timezone.utc)

    for idx, piece in enumerate(pieces, start=1):
        try:
            episode_kwargs = {
                "name": f"{name} :: chunk {idx:03d}",
                "episode_body": piece,
                "source": EpisodeType.text,
                "source_description": source_desc,
                "reference_time": now,
                "group_id": group_id
            }

            res = await graphiti.add_episode(**episode_kwargs)
            uuids.append(getattr(res, "uuid", None))

        except Exception as e:
            print(f"Failed to add episode for chunk {idx}: {e}")
            continue

    return {
        "name": name,
        "path": doc_path,
        "chunks": len(pieces),
        "group_id": group_id,
        "uuids": uuids
    }
