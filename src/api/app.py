from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag_pipeline import RAGPipeline
import time
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Granicus RAG Chatbot")

rag_pipeline = RAGPipeline()
request_count = 0


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    latency_seconds: float


@app.get("/health")
async def health():
    try:
        count = rag_pipeline.store.collection.count()
        return {
            "status": "ok",
            "vectorstore_ready": True,
            "indexed_chunks": count
        }
    except Exception:
        return {
            "status": "error",
            "vectorstore_ready": False
        }


@app.get("/stats")
async def stats():
    return {
        "indexed_documents": rag_pipeline.store.collection.count(),
        "total_requests": request_count
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global request_count

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.time()

    response = await rag_pipeline.ask(request.question)

    latency = time.time() - start
    request_count += 1

    logging.info(f"[API] Total latency: {latency:.2f}s")

    return {
        "answer": response["answer"],
        "confidence": response["confidence"],
        "latency_seconds": round(latency, 3)
    }
