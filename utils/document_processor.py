import os
import logging
import torch
import gc
from typing import List, Optional
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
import pdfplumber
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

load_dotenv()

os.environ["HF_TOKEN"] = ""

logger = logging.getLogger(__name__)

try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError as e:
    logger.error(f"Docling import failed: {e}")
    DOCLING_AVAILABLE = False

# global singleton DocumentConverter
_docling_converter = None
_converter_lock = False

def get_docling_converter() -> Optional[DocumentConverter]:

    global _docling_converter, _converter_lock
    
    if not DOCLING_AVAILABLE:
        return None
    
    if _docling_converter is not None:
        return _docling_converter
    
    if _converter_lock:
        logger.warning("DocumentConverter creation already in progress, returning None")
        return None
    
    try:
        _converter_lock = True
        logger.info("Creating singleton DocumentConverter with GPU support...")
        
        # Configure pipeline for GPU with memory management
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = False 
        pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
        
        _docling_converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        
        logger.info("DocumentConverter singleton created successfully with GPU support")
        return _docling_converter
        
    except Exception as e:
        logger.error(f"Failed to create DocumentConverter: {e}")
        return None
    finally:
        _converter_lock = False

def clear_gpu_memory():

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            logger.info("GPU memory cleared")
    except Exception as e:
        logger.warning(f"Failed to clear GPU memory: {e}")

def reset_docling_converter():

    global _docling_converter
    if _docling_converter is not None:
        logger.info("Resetting DocumentConverter singleton...")
        clear_gpu_memory()
        _docling_converter = None
        logger.info("DocumentConverter singleton reset")

class DocumentProcessor:

    def __init__(self, breakpoint_type: str = "percentile", threshold: int = 90):

        self.text_splitter = SemanticChunker(
            AzureOpenAIEmbeddings(
                azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),  
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            ),
            breakpoint_threshold_type=breakpoint_type,
            breakpoint_threshold_amount=threshold,
            min_chunk_size=300
        )

    def extract_text_docling(self, path: str) -> str:

        if not DOCLING_AVAILABLE:
            logger.warning("Docling is not available. Falling back to pdfplumber...")
            return self.extract_text_pdfplumber(path)
        
        converter = get_docling_converter()
        if converter is None:
            logger.warning("Could not get DocumentConverter. Falling back to pdfplumber...")
            return self.extract_text_pdfplumber(path)
        
        try:
            logger.info(f"Processing with docling (GPU): {os.path.basename(path)}")
            
            result = converter.convert(source=path)
            doc = result.document
            raw_text = doc.export_to_text()
            
            if not raw_text or not raw_text.strip():
                logger.warning("Docling extracted empty text, falling back to pdfplumber...")
                return self.extract_text_pdfplumber(path)
            
            logger.info(f"Docling extracted {len(raw_text)} characters from {os.path.basename(path)}")
            

            return raw_text
            
        except torch.cuda.OutOfMemoryError as gpu_error:
            logger.error(f"GPU OUT OF MEMORY in docling for {os.path.basename(path)}: {gpu_error}")
            logger.info("Clearing GPU memory and falling back to pdfplumber...")
            
            # clear GPU memory and reset converter !! important for gpu memory
            clear_gpu_memory()
            reset_docling_converter()
            
            return self.extract_text_pdfplumber(path)
            
        except RuntimeError as runtime_error:
            if "CUDA out of memory" in str(runtime_error) or "out of memory" in str(runtime_error).lower():
                logger.error(f"GPU MEMORY ERROR in docling for {os.path.basename(path)}: {runtime_error}")
                logger.info("Clearing GPU memory and falling back to pdfplumber...")
                
                clear_gpu_memory()
                reset_docling_converter()
                
                return self.extract_text_pdfplumber(path)
            else:
                raise runtime_error
                
        except Exception as e:
            logger.error(f"Docling conversion failed for {os.path.basename(path)}: {e}")
            logger.info("Falling back to pdfplumber...")
            return self.extract_text_pdfplumber(path)
    
    def extract_text_pdfplumber(self, path: str) -> str:

        logger.info(f"Processing with pdfplumber (CPU fallback): {os.path.basename(path)}")
        
        all_text = []

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # plain text
                text = page.extract_text() or ""
                all_text.append(f"\n--- Page {page_num} Text ---\n{text}")

                # tables
                tables = page.extract_tables()
                for t_i, table in enumerate(tables, start=1):
                    # table rows to Markdown-style
                    md_table = "\n".join(
                        [" | ".join(map(str, row)) for row in table if row]
                    )
                    header = f"\n--- Page {page_num} Table {t_i} ---\n"
                    all_text.append(header + md_table)

        combined = "\n".join(all_text)
        logger.info(f"pdfplumber extracted {len(combined)} characters from {os.path.basename(path)}")

        return combined


    def semantic_chunk(self, content: str, logs="logs/docling_chunks.txt") -> List[str]:

        docs = self.text_splitter.create_documents([content])
        chunks = [doc.page_content for doc in docs]

        os.makedirs("logs", exist_ok=True)
        with open(logs, "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"--- Chunk {i} ---\n{chunk}\n\n")

        return chunks
    
if __name__ == "__main__":
    processor = DocumentProcessor()

    print("Starting document processing...")
    path="files/Finance/FINANCE_2025q1-alphabet-earnings-release.pdf"
    text = processor.extract_text_docling(path)

    print(f"Extracted text length: {len(text)} characters")
    print("Starting semantic chunking...")
    
    chunks = processor.semantic_chunk(text)
    print(f"Generated {len(chunks)} chunks. See logs/docling_chunks.txt for details.")

    text = processor.extract_text_pdfplumber(path)
    chunks = processor.semantic_chunk(text, "logs/pdfplumber_chunks.txt")
    print(f"Generated {len(chunks)} chunks. See logs/pdfplumber_chunks.txt for details.")


