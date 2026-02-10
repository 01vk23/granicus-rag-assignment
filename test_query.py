from src.vectorstore.store import VectorStore

if __name__ == "__main__":
    store = VectorStore()

    count = store.collection.count()
    print("Collection count:", count)

    query = "What are the key features of GovDelivery Communications Cloud?"

    results = store.query(query, top_k=3)

    print("\nQuery:", query)
    print("\nTop Results:\n")

    print(results)
