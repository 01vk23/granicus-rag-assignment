import chromadb
from chromadb.config import Settings
from typing import List
from src.chunking.chunker import Chunk
from src.vectorstore.embeddings import Embedder


class VectorStore:
    def __init__(self, persist_dir: str = "chroma_db"):

        self.client = chromadb.PersistentClient(path=persist_dir)


        self.collection = self.client.get_or_create_collection(
            name="granicus_docs",
            metadata={"hnsw:space": "cosine"}
        )

        self.embedder = Embedder()

    def index_chunks(self, chunks: List[Chunk]):
        texts = [chunk.content for chunk in chunks]
        ids = [chunk.chunk_id for chunk in chunks]
        metadata = [{"source": chunk.source} for chunk in chunks]

        embeddings = self.embedder.embed_texts(texts)

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadata
        )
        #self.client.persist()


    def query(self, query: str, top_k: int = 5):
        query_embedding = self.embedder.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results
