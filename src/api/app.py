from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag_pipeline import RAGPipeline
import time

app = FastAPI(title="Granicus RAG Chatbot")

# Initialize pipeline once
rag_pipeline = RAGPipeline()

# Simple runtime metrics
request_count = 0


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    latency_seconds: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    return {
        "indexed_documents": rag_pipeline.store.collection.count(),
        "total_requests": request_count
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    global request_count

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    start = time.time()

    response = rag_pipeline.ask(request.question)

    latency = time.time() - start
    request_count += 1

    return {
        "answer": response["answer"],
        "confidence": response["confidence"],
        "latency_seconds": round(latency, 3)
    }
