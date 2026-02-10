from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class GroundedGenerator:
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def generate(self, question: str, context: str) -> str:
        prompt = f"""
You are a government technology assistant.

Answer the question using ONLY the provided context.

Instructions:
- Provide a complete answer framing the sentence from the question itself.
- If multiple features are listed, include all of them.
- Format the answer as bullet points when appropriate.
- Do NOT add any information not found in the context.
- If the answer is not in the context, say:
"I do not have enough information to answer this question."

Context:
{context}

Question:
{question}

Answer:
"""


        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.0
            )

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return answer.strip()
