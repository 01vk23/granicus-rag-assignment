import pytest
from src.ingestion.loader import DocumentLoader


def test_loader_loads_documents():
    loader = DocumentLoader(data_dir="data")
    documents = loader.load()

    # Ensure documents were loaded
    assert isinstance(documents, list)

    # Ensure at least one valid document exists
    assert len(documents) > 0

    # Validate structure of first document
    first_doc = documents[0]

    assert hasattr(first_doc, "doc_id")
    assert hasattr(first_doc, "source")
    assert hasattr(first_doc, "doc_type")
    assert hasattr(first_doc, "content")

    assert isinstance(first_doc.content, str)
    assert len(first_doc.content) > 50
