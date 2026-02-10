from typing import Dict


class ContextBuilder:
    def build_context(self, results: Dict) -> str:
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        context_blocks = []

        for i, doc in enumerate(documents):
            metadata = metadatas[i] if i < len(metadatas) else {}

            source = metadata.get("source", "Unknown Source")
            doc_type = metadata.get("doc_type", "Unknown Type")

            block = (
                f"[Context Block {i+1}]\n"
                f"Source File: {source}\n"
                f"Document Type: {doc_type}\n\n"
                f"{doc.strip()}"
            )

            context_blocks.append(block)

        return "\n\n-----------------------------\n\n".join(context_blocks)
