import asyncio
from src.rag_pipeline import RAGPipeline


async def main():
    rag = RAGPipeline()

    question = "What are the key features of GovDelivery Communications Cloud?"

    response = await rag.ask(question)

    print("\nQuestion:", question)
    print("\nAnswer:\n", response["answer"])
    print("\nConfidence:", response["confidence"])


if __name__ == "__main__":
    asyncio.run(main())
