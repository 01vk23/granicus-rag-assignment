from src.ingestion.loader import DocumentLoader

if __name__ == "__main__":
    loader = DocumentLoader(data_dir="data")
    documents = loader.load()

    print(f"\nLoaded {len(documents)} documents\n")

    for doc in documents:
        print("----")
        print("ID:", doc.doc_id)
        print("Source:", doc.source)
        print("Type:", doc.doc_type)
        print(doc.content[:300])
        print()
