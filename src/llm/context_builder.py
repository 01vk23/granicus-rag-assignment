from typing import List, Dict


class ContextBuilder:
    def build_context(self, results: Dict) -> str:
        documents = results.get("documents", [[]])[0]

        context_blocks = []

        for i, doc in enumerate(documents):
            context_blocks.append(f"[Source {i+1}]\n{doc.strip()}")

        return "\n\n".join(context_blocks)
