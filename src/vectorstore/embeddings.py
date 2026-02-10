from sentence_transformers import SentenceTransformer
from typing import List
import torch
import logging
import time

logging.basicConfig(level=logging.INFO)


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        start_time = time.time()

        # Auto device detection
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"[Embedder] Using device: {self.device}")

        self.model = SentenceTransformer(model_name, device=self.device)

        logging.info(
            f"[Embedder] Model loaded in {time.time() - start_time:.2f}s"
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        except Exception as e:
            logging.error(f"[Embedder ERROR] {str(e)}")
            return []

    def embed_query(self, query: str) -> List[float]:
        try:
            embedding = self.model.encode(
                query,
                normalize_embeddings=True
            )
            return embedding.tolist()
        except Exception as e:
            logging.error(f"[Embedder ERROR] {str(e)}")
            return []
