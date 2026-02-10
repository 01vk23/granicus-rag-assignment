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
4. If the question is ambiguous:
   - Identify relevant products or plan tiers.
   - List possible options.
   - Ask for clarification.
5. Keep answers concise and structured.
6. Prefer bullet points when listing features.

Context:
{context[:1000]}

Question:
{question}

Answer:
"""

        try:
            # ---------------- GPU PATH (Colab) ----------------
            if self.use_gpu_llm:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=80,
                        temperature=0.0,
                        do_sample=False
                    )

                answer = self.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )

                logging.info(
                    f"[Generator GPU] Generation time: {time.time() - start_time:.2f}s"
                )

                return answer.strip()

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

                logging.info(
                    f"[Generator CPU] Ollama time: {time.time() - start_time:.2f}s"
                )

                return answer if answer else "I do not have enough information to answer this question."

        except Exception as e:
            logging.error(f"[Generator ERROR] {str(e)}")
            return "I do not have enough information to answer this question."
