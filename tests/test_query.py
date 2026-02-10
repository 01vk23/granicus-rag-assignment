import os
import uuid
from src.ingestion.loader import DocumentLoader
from src.chunking.chunker import SmartChunker
from src.vectorstore.store import VectorStore
from src.vectorstore.embeddings import Embedder


def test_query_returns_results():

    # Create isolated test DB
    test_dir = f"test_chroma_{uuid.uuid4().hex}"
    os.makedirs(test_dir, exist_ok=True)

    loader = DocumentLoader(data_dir="data")
    documents = loader.load()

    chunker = SmartChunker()
    chunks = chunker.chunk_documents(documents)

    embedder = Embedder()
    store = VectorStore(embedder=embedder, persist_dir=test_dir)

    store.index_chunks(chunks)

    query = "GovDelivery Communications Cloud"
    results = store.query(query, top_k=2)

    documents_result = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # Assertions
    assert len(documents_result) > 0
    assert len(distances) > 0
    assert len(documents_result) == len(distances)
