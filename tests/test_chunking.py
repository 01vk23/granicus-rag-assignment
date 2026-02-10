from src.ingestion.loader import DocumentLoader
from src.chunking.chunker import SmartChunker

if __name__ == "__main__":
    loader = DocumentLoader(data_dir="data")
    documents = loader.load()

    print(f"\nLoaded {len(documents)} documents")

    chunker = SmartChunker()

    chunks = chunker.chunk_documents(documents)

    print(f"\nCreated {len(chunks)} chunks\n")

    for chunk in chunks[:3]:
        print("----")
        print("Chunk ID:", chunk.chunk_id)
        print("Source:", chunk.source)
        print(chunk.content[:400])
        print()
