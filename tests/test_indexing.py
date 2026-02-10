import os
import uuid
from src.ingestion.loader import DocumentLoader
from src.chunking.chunker import SmartChunker
from src.vectorstore.store import VectorStore
from src.vectorstore.embeddings import Embedder


def test_indexing_creates_embeddings():

    # Create unique test DB folder
    test_dir = f"test_chroma_{uuid.uuid4().hex}"
    os.makedirs(test_dir, exist_ok=True)

    loader = DocumentLoader(data_dir="data")
    documents = loader.load()

    chunker = SmartChunker()
    chunks = chunker.chunk_documents(documents)

    embedder = Embedder()
    store = VectorStore(embedder=embedder, persist_dir=test_dir)

    store.index_chunks(chunks)

    count = store.collection.count()

    assert count > 0
    assert count == len(chunks)
