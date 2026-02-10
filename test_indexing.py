from src.ingestion.loader import DocumentLoader
from src.chunking.chunker import SmartChunker
from src.vectorstore.store import VectorStore

if __name__ == "__main__":
    print("Loading documents...")
    loader = DocumentLoader(data_dir="data")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")

    print("Chunking...")
    chunker = SmartChunker()
    chunks = chunker.chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Indexing into Chroma...")
    store = VectorStore()
    store.index_chunks(chunks)

    print("Indexing complete.")
    print("Collection count:", store.collection.count())

