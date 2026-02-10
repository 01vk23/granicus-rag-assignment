import httpx
import time
import logging

logging.basicConfig(level=logging.INFO)


class GroundedGenerator:
    def __init__(self, model_name="phi3:mini"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434/api/generate"

        logging.info(f"[Generator] Using Ollama model: {self.model_name}")

    async def generate(self, question: str, context: str) -> str:
        start_time = time.time()

        prompt = f"""
You are a government technology assistant specialized in Granicus products.

You MUST answer using ONLY the provided context blocks.

CRITICAL RULES:
1. Do NOT use any external knowledge.
2. Do NOT invent missing details.
3. If the answer is not clearly found in the context, respond:
   "I do not have enough information to answer this question."
4. If the question is ambiguous (e.g., asks about pricing or features without specifying product or plan tier):
   - Identify relevant products or plan tiers mentioned in the context.
   - Briefly list the possible options found.
   - Ask the user to clarify which specific product or plan they mean.
   - Try to be specific rather than being verbose.
   - Ask user for specific details in case of ambiguity or multiple answers.
5. If information differs across context blocks, clearly distinguish them by product or plan tier.
6. Do NOT say "According to the context".
7. Keep answers concise, structured, and factual.
8. Prefer bullet points for multiple features or comparisons.

Important: Frame Answers in a way that user do not think you are reading a document.

Context Blocks:
{context[:1000]}

User Question:
{question}

Final Answer:
"""

        try:
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(
                    self.base_url,
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.0,
                            "num_predict": 75
                        }
                    }
                )

            result = response.json()
            answer = result.get("response", "").strip()

            logging.info(
                f"[Generator] Ollama generation time: {time.time() - start_time:.2f}s"
            )

            return answer if answer else "I do not have enough information to answer this question."

        except Exception as e:
            logging.error(f"[Generator ERROR] {str(e)}")
            return "I do not have enough information to answer this question."
