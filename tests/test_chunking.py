from src.ingestion.loader import DocumentLoader
from src.chunking.chunker import SmartChunker


def test_chunking_creates_chunks():
    loader = DocumentLoader(data_dir="data")
    documents = loader.load()

    chunker = SmartChunker()
    chunks = chunker.chunk_documents(documents)

    # Ensure chunks are created
    assert isinstance(chunks, list)
    assert len(chunks) > 0

    # Validate first chunk structure
    first_chunk = chunks[0]

    assert hasattr(first_chunk, "chunk_id")
    assert hasattr(first_chunk, "source")
    assert hasattr(first_chunk, "content")
    assert hasattr(first_chunk, "metadata")

    assert isinstance(first_chunk.content, str)
    assert len(first_chunk.content) > 50
