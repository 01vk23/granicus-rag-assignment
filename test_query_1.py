from src.vectorstore.store import VectorStore

if __name__ == "__main__":
    store = VectorStore()

    query = "What are the key features of GovDelivery Communications Cloud?"

    results = store.query(query, top_k=3)

    print("\nQuery:", query)
    print("\nTop Results:\n")

    for i, doc in enumerate(results["documents"][0]):
        print(f"Result {i+1}:")
        print(doc[:500])
        print("-" * 50)
