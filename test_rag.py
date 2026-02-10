from src.rag_pipeline import RAGPipeline

if __name__ == "__main__":
    rag = RAGPipeline()

    question = "How is weather today"

    response = rag.ask(question)

    print("\nQuestion:", question)
    print("\nAnswer:\n", response["answer"])
    print("\nConfidence:", response["confidence"])
