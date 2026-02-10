from src.vectorstore.store import VectorStore
from src.llm.context_builder import ContextBuilder
from src.llm.generator import GroundedGenerator


class RAGPipeline:
    def __init__(self):
        self.store = VectorStore()
        self.context_builder = ContextBuilder()
        self.generator = GroundedGenerator()

    def ask(self, question: str, top_k: int = 3):
        # Step 1: Retrieve
        results = self.store.query(question, top_k=top_k)

        # Step 2: Check if retrieval empty
        if not results["documents"][0]:
            return {
                "answer": "I do not have enough information to answer this question.",
                "confidence": 0.0
            }

        # Step 3: Build context
        context = self.context_builder.build_context(results)

        # Step 4: Generate grounded answer
        answer = self.generator.generate(question, context)

        # Step 5: Simple confidence score based on similarity
        distances = results.get("distances", [[]])[0]
        if distances:
            avg_distance = sum(distances) / len(distances)
            confidence = max(0.0, 1 - avg_distance)
        else:
            confidence = 0.0

        return {
            "answer": answer,
            "confidence": round(confidence, 3)
        }
