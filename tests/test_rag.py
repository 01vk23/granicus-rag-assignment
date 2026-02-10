import pytest
from src.rag_pipeline import RAGPipeline


@pytest.mark.asyncio
async def test_rag_pipeline_returns_answer():

    rag = RAGPipeline()

    question = "What are the key features of GovDelivery Communications Cloud?"

    response = await rag.ask(question)

    assert isinstance(response, dict)
    assert "answer" in response
    assert "confidence" in response

    assert isinstance(response["answer"], str)
    assert isinstance(response["confidence"], float)

    assert 0.0 <= response["confidence"] <= 1.0
    assert len(response["answer"]) > 0
