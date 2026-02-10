from typing import List
from src.ingestion.loader import Document
import uuid
import re
import time
import logging

logging.basicConfig(level=logging.INFO)


class Chunk:
    def __init__(self, chunk_id: str, source: str, content: str, metadata: dict = None):
        self.chunk_id = chunk_id
        self.source = source
        self.content = content
        self.metadata = metadata or {}


class SmartChunker:
    def __init__(self, chunk_size: int = 500, overlap: int = 80):
        self.chunk_size = chunk_size
        self.overlap = overlap

    # ---------------------------
    # TEXT CHUNKING (Heading-aware + paragraph safe)
    # ---------------------------
    def chunk_text_by_heading(self, text: str) -> List[str]:
        sections = re.split(
            r"\n(?=(?:#{1,6}\s|[A-Z][A-Z\s]{5,}))",
            text
        )

        chunks = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if len(section) <= self.chunk_size:
                chunks.append(section)
            else:
                chunks.extend(self.chunk_by_size(section))

        return chunks

    def chunk_by_size(self, text: str) -> List[str]:
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]

            # Avoid cutting mid-word aggressively
            if end < text_length:
                last_space = chunk.rfind(" ")
                if last_space > self.chunk_size * 0.6:
                    chunk = chunk[:last_space]
                    end = start + last_space

            chunks.append(chunk.strip())
            start = end - self.overlap

        return chunks

    # ---------------------------
    # CSV CHUNKING (Row-based grouping)
    # ---------------------------
    def chunk_csv(self, text: str) -> List[str]:
        rows = [row.strip() for row in text.split("\n") if row.strip()]

        chunks = []
        current_chunk = ""

        for row in rows:
            if len(current_chunk) + len(row) < self.chunk_size:
                current_chunk += row + "\n"
            else:
                chunks.append(current_chunk.strip())
                current_chunk = row + "\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks

    # ---------------------------
    # PDF CHUNKING
    # ---------------------------
    def chunk_pdf(self, text: str) -> List[str]:
        if "=== EXTRACTED TABLE DATA ===" in text:
            main_text, table_text = text.split("=== EXTRACTED TABLE DATA ===", 1)

            text_chunks = self.chunk_text_by_heading(main_text.strip())
            table_chunks = self.chunk_csv(table_text.strip())

            return text_chunks + table_chunks

        return self.chunk_text_by_heading(text)

    # ---------------------------
    # MAIN ENTRY
    # ---------------------------
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        start_time = time.time()
        all_chunks = []

        for doc in documents:
            if doc.doc_type == "csv":
                raw_chunks = self.chunk_csv(doc.content)
            elif doc.doc_type == "pdf":
                raw_chunks = self.chunk_pdf(doc.content)
            else:
                raw_chunks = self.chunk_text_by_heading(doc.content)

            for chunk_text in raw_chunks:
                if len(chunk_text) < 50:
                    continue

                all_chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        source=doc.source,
                        content=chunk_text.strip(),
                        metadata={
                            "doc_type": doc.doc_type
                        }
                    )
                )

        duration = time.time() - start_time
        logging.info(f"[Chunking] Created {len(all_chunks)} chunks in {duration:.2f}s")

        return all_chunks
