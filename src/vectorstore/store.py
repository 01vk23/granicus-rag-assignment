import chromadb
from typing import List, Optional
from src.chunking.chunker import Chunk
from src.vectorstore.embeddings import Embedder
import time
import logging

logging.basicConfig(level=logging.INFO)


class VectorStore:
    def __init__(self, embedder: Embedder, persist_dir: str = "chroma_db"):
        start_time = time.time()

        try:
            self.client = chromadb.PersistentClient(path=persist_dir)

            self.collection = self.client.get_or_create_collection(
                name="granicus_docs",
                metadata={"hnsw:space": "cosine"}
            )

            self.embedder = embedder

            logging.info(
                f"[VectorStore] Initialized in {time.time() - start_time:.2f}s"
            )

        except Exception as e:
            logging.error(f"[VectorStore INIT ERROR] {str(e)}")
            raise e
        
 # ---------------------------
    # Check if Empty
    # ---------------------------
    def is_empty(self) -> bool:
        try:
            return self.collection.count() == 0
        except Exception as e:
            logging.error(f"[VectorStore EMPTY CHECK ERROR] {str(e)}")
            return True

    # ---------------------------
    # Indexing
    # ---------------------------
    def index_chunks(self, chunks: List[Chunk]):
        start_time = time.time()

        

        try:
            if not chunks:
                logging.warning("[VectorStore] No chunks to index.")
                return

            texts = [chunk.content for chunk in chunks]
            ids = [chunk.chunk_id for chunk in chunks]

            metadata = [
                {
                    "source": chunk.source,
                    **getattr(chunk, "metadata", {})
                }
                for chunk in chunks
            ]

            embeddings = self.embedder.embed_texts(texts)

            if not embeddings:
                logging.error("[VectorStore] Embedding generation failed.")
                return

            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadata
            )

            logging.info(
                f"[VectorStore] Indexed {len(chunks)} chunks in {time.time() - start_time:.2f}s"
            )

        except Exception as e:
            logging.error(f"[VectorStore INDEX ERROR] {str(e)}")

    # ---------------------------
    # Query
    # ---------------------------
    def query(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[dict] = None
    ):
        start_time = time.time()

        try:
            query_embedding = self.embedder.embed_query(query)

            if not query_embedding:
                logging.error("[VectorStore] Query embedding failed.")
                return {"documents": [[]], "distances": [[]]}

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filters
            )

            logging.info(
                f"[VectorStore] Query retrieved {top_k} results in {time.time() - start_time:.2f}s"
            )

            return results

        except Exception as e:
            logging.error(f"[VectorStore QUERY ERROR] {str(e)}")
            return {"documents": [[]], "distances": [[]]}
