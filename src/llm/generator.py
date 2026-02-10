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
            logging.info("[Generator] CUDA detected. Using Mistral-7B-Instruct (4-bit).")

            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

            self.hf_model_name = "mistralai/Mistral-7B-Instruct-v0.2"

            # 4-bit quantization for Kaggle GPU
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )

            self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_name)

            self.model = AutoModelForCausalLM.from_pretrained(
                self.hf_model_name,
                quantization_config=quant_config,
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
You are a highly disciplined government technology assistant specialized in Granicus products.

STRICT INSTRUCTIONS:

You MUST answer ONLY using the provided context.
You MUST NOT use any external knowledge.
You MUST NOT assume missing details.
You MUST NOT fabricate product names, pricing, features, or integrations.

If the answer is NOT clearly present in the context:
Respond exactly:
"I do not have enough information to answer this question."

----------------------------
HANDLING AMBIGUOUS QUESTIONS
----------------------------

If a question is vague or underspecified, for example:

- "What is the pricing?"
- "Which plan is best?"
- "Tell me about the product."
- "What does it include?"
- "How much does it cost?"

You MUST:

1. Identify possible relevant products or tiers found in the context.
2. Briefly list available options from context.
3. Ask the user to clarify which specific product or plan tier they mean.
4. Do NOT guess.
5. Do NOT recommend.

Example:

If user asks:
"How much does it cost?"

Correct behavior:
"The context mentions multiple plan tiers (Starter, Professional, Enterprise). Please specify which plan tier you are referring to."

----------------------------
OUT-OF-SCOPE QUESTIONS
----------------------------

If the question is unrelated to Granicus documentation (e.g., weather, competitors, CEO phone number, general opinion):

Respond exactly:
"I do not have enough information to answer this question."

Do NOT explain why.
Do NOT apologize excessively.
Keep response minimal and factual.

----------------------------
MULTIPLE CONTEXT BLOCKS
----------------------------

If different blocks contain different information:
- Clearly separate answers by product or plan tier.
- Use bullet points.
- Stay concise.

----------------------------
STYLE REQUIREMENTS
----------------------------

- Be structured and factual.
- Prefer bullet points when listing features.
- Do NOT say "According to the context".
- Keep answer concise.
- Do NOT repeat the question.
- Do NOT be verbose.

----------------------------

Context:
{context[:1500]}

User Question:
{question}

Final Answer:
"""

        try:
            # ---------------- GPU PATH (Mistral) ----------------
            if self.use_gpu_llm:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                input_length = inputs["input_ids"].shape[1]

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=120,
                        do_sample=False,
                        temperature=0.01
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
                                "temperature": 0.01,
                                "num_predict": 100
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
