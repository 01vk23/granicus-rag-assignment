import httpx
import time
import logging
import torch

logging.basicConfig(level=logging.INFO)


class GroundedGenerator:
    def __init__(self, model_name="phi3:mini"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_gpu_llm = torch.cuda.is_available()

        if self.use_gpu_llm:
            logging.info("[Generator] CUDA detected. Using Transformers GPU model.")

            from transformers import AutoTokenizer, AutoModelForCausalLM

            self.hf_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                torch_dtype=torch.float16,
                device_map="auto"
            )

            self.model.eval()

        else:
            logging.info("[Generator] CUDA not available. Using Ollama.")
            self.model_name = model_name
            self.base_url = "http://localhost:11434/api/generate"

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

Important: Frame answers naturally so the user does not feel you are reading a document.

Context Blocks:
{context[:1000]}

User Question:
{question}

Final Answer:
"""

        try:
            # ---------------- GPU PATH ----------------
            if self.use_gpu_llm:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                input_length = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=80,
                        do_sample=False
                    )

                generated_tokens = outputs[0][input_length:]

                answer = self.tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True
                ).strip()

                logging.info(f"[Generator GPU] Question: {question}")
                logging.info(
                    f"[Generator GPU] Generation time: {time.time() - start_time:.2f}s"
                )

                return answer if answer else "I do not have enough information to answer this question."

            # ---------------- CPU PATH (Ollama) ----------------
            else:
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

                logging.info(f"[Generator CPU] Question: {question}")
                logging.info(
                    f"[Generator CPU] Ollama time: {time.time() - start_time:.2f}s"
                )

                return answer if answer else "I do not have enough information to answer this question."

        except Exception as e:
            logging.error(f"[Generator ERROR] {str(e)}")
            return "I do not have enough information to answer this question."
