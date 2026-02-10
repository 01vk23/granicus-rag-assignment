import time
import logging
import numpy as np

from src.vectorstore.store import VectorStore
from src.vectorstore.embeddings import Embedder
from src.llm.context_builder import ContextBuilder
from src.llm.generator import GroundedGenerator
from src.ingestion.loader import DocumentLoader
from src.chunking.chunker import SmartChunker


logging.basicConfig(level=logging.INFO)


class RAGPipeline:
    def __init__(self):
        start_time = time.time()

        try:
            self.embedder = Embedder()
            self.store = VectorStore(embedder=self.embedder)
            self.context_builder = ContextBuilder()
            self.generator = GroundedGenerator()

                        # ---------------------------
            # Auto Index Initialization
            # ---------------------------
            if self.store.is_empty():
                logging.info("[RAGPipeline] Vector store empty. Initializing index...")

                loader = DocumentLoader(data_dir="data")
                documents = loader.load()

                chunker = SmartChunker()
                chunks = chunker.chunk_documents(documents)

                self.store.index_chunks(chunks)

                logging.info("[RAGPipeline] Index initialization complete.")


            # High-threshold cache
            self.cache = {}

            logging.info(
                f"[RAGPipeline] Initialized in {time.time() - start_time:.2f}s"
            )

        except Exception as e:
            logging.error(f"[RAGPipeline INIT ERROR] {str(e)}")
            raise e

    async def ask(self, question: str, top_k: int = 5):
        pipeline_start = time.time()

        try:
            # ---------------------------
            # Cache Check
            # ---------------------------
            if question in self.cache:
                logging.info("[RAG] Cache hit")
                return self.cache[question]

            # ---------------------------
            # Retrieval
            # ---------------------------
            retrieval_start = time.time()
            results = self.store.query(question, top_k=top_k)
            logging.info(
                f"[RAG] Retrieval time: {time.time() - retrieval_start:.2f}s"
            )

            docs = results.get("documents", [[]])[0]
            distances = results.get("distances", [[]])[0]

            if not docs:
                return {
                    "answer": "I do not have enough information to answer this question.",
                    "confidence": 0.0
                }

            # ---------------------------
            # Similarity Threshold Guard
            # ---------------------------
            if distances and min(distances) > 0.35:
                return {
                    "answer": "I do not have enough information to answer this question.",
                    "confidence": 0.0
                }

            # ---------------------------
            # Re-ranking (keep best 2)
            # ---------------------------
            ranked = sorted(zip(docs, distances), key=lambda x: x[1])
            top_docs = [doc for doc, _ in ranked[:3]]

            # ---------------------------
            # Context Build
            # ---------------------------
            context = "\n\n".join(top_docs)

            # ---------------------------
            # Generation
            # ---------------------------
            generation_start = time.time()
            answer = await self.generator.generate(question, context)
            logging.info(
                f"[RAG] Generation time: {time.time() - generation_start:.2f}s"
            )

            # ---------------------------
            # Confidence
            # ---------------------------
            avg_distance = np.mean([d for _, d in ranked[:2]])
            confidence = max(0.0, 1 - avg_distance)

            result = {
                "answer": answer,
                "confidence": round(confidence, 3)
            }

            # ---------------------------
            # High-Confidence Cache (>0.85)
            # ---------------------------
            if confidence > 0.85:
                self.cache[question] = result

            logging.info(
                f"[RAG] Total pipeline time: {time.time() - pipeline_start:.2f}s"
            )

            return result

        except Exception as e:
            logging.error(f"[RAG ERROR] {str(e)}")
            return {
                "answer": "I do not have enough information to answer this question.",
                "confidence": 0.0
            }
