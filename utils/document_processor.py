import os
from typing import List
from dotenv import load_dotenv
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import AzureOpenAIEmbeddings
import pdfplumber
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode

load_dotenv()

os.environ["HF_TOKEN"] = ""


try:
    from docling.document_converter import DocumentConverter
    DOCLING_AVAILABLE = True
except ImportError as e:
    print(f"Docling import failed: {e}")
    DOCLING_AVAILABLE = False

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
            breakpoint_threshold_amount=threshold
        )

    def extract_text_docling(self, path: str) -> str:
        if not DOCLING_AVAILABLE:
            print("Docling is not available. Falling back to pdfplumber...")
            return self.extract_text_pdfplumber(path)
        
        try:
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_table_structure = False 
            pipeline_options.table_structure_options.mode = TableFormerMode.FAST

            converter = DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )

            result = converter.convert(source=path)
            doc = result.document
            raw_text = doc.export_to_text()
            print("Document conversion completed successfully.")
            return raw_text
        except Exception as e:
            print(f"Docling conversion failed: {e}")
            print("Falling back to pdfplumber...")
            return self.extract_text_pdfplumber(path)
    
    def extract_text_pdfplumber(self, path: str) -> str:

        all_text = []

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract plain text
                text = page.extract_text() or ""
                all_text.append(f"\n--- Page {page_num} Text ---\n{text}")

                # Extract tables
                tables = page.extract_tables()
                for t_i, table in enumerate(tables, start=1):
                    # Convert table rows to Markdown-style
                    md_table = "\n".join(
                        [" | ".join(map(str, row)) for row in table if row]
                    )
                    header = f"\n--- Page {page_num} Table {t_i} ---\n"
                    all_text.append(header + md_table)


        combined = "\n".join(all_text)
        with open("logs/pdfplumber_extracted.txt", "w", encoding="utf-8") as f:
            f.write(combined)

        return combined


    def semantic_chunk(self, content: str) -> List[str]:

        docs = self.text_splitter.create_documents([content])
        chunks = [doc.page_content for doc in docs]

        os.makedirs("logs", exist_ok=True)
        with open("logs/chunks.txt", "w", encoding="utf-8") as f:
            for i, chunk in enumerate(chunks, 1):
                f.write(f"--- Chunk {i} ---\n{chunk}\n\n")

        return chunks
    
if __name__ == "__main__":
    processor = DocumentProcessor()

    print("Starting document processing...")
    
    text = processor.extract_text_docling("files/oneNDA.pdf")
    # text = processor.extract_text_pdfplumber("files/oneNDA.pdf")

    
    print(f"Extracted text length: {len(text)} characters")
    print("Starting semantic chunking...")
    
    chunks = processor.semantic_chunk(text)
    print(f"Generated {len(chunks)} chunks. See logs/chunks.txt for details.")

