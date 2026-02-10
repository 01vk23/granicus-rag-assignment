from src.vectorstore.store import VectorStore
from src.vectorstore.embeddings import Embedder


if __name__ == "__main__":
    print("Initializing embedder...")
    embedder = Embedder()

    print("Initializing vector store...")
    store = VectorStore(embedder=embedder)

    count = store.collection.count()
    print("Collection count:", count)

    query = "What are the key features of GovDelivery Communications Cloud?"

    print("\nRunning query...")
    results = store.query(query, top_k=2)

    print("\nQuery:", query)
    print("\nTop Results:\n")
    print(results)
