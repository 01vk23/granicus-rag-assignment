# Granicus RAG Chatbot System

A production-grade Retrieval-Augmented Generation (RAG) system built to handle government product and pricing queries. This system utilizes structured document ingestion, persistent vector search, and grounded LLM responses to ensure accuracy and data sovereignty.

---

## 🏗️ Architecture Overview

The system follows a modular pipeline designed for local execution and high reliability:



1. **Ingestion:** Processes PDF, CSV, TXT, and HTML.
2. **Indexing:** Smart chunking and local embedding generation.
3. **Storage:** Persistent vector storage via ChromaDB.
4. **Retrieval:** Semantic similarity search with distance-based guardrails.
5. **Generation:** Grounded LLM response generation with confidence scoring.

---

## 🛠️ Tech Stack

| Component          | Technology                                      |
|:-------------------|:------------------------------------------------|
|                                   |
 LLM Engine** | Ollama (phi-mini )  and                   |
| **Embeddings** | Sentence-Transformers (`multi-qa-mpnet-base-dot-v1`) |
| **Vector Database**| ChromaDB (Persistent)                          |
| **API** | FastAPI                                         |
| **Testing** | Pytest                                          |

---

## 📂 Project Structure

```text
├── src/
│   ├── api/                # FastAPI application & endpoints
│   ├── ingestion/          # Document loading & parsing logic
│   ├── chunking/           # Smart recursive & structure-aware splitting
│   ├── vectorstore/        # ChromaDB integration & embedding logic
│   ├── llm/                # Prompt engineering & LLM connectors
│   └── rag_pipeline.py     # Pipeline orchestration
├── tests/                  # Unit and integration test suite
├── evaluations/            # Batch performance & accuracy tools
├── data/                   # Source documents (PDF, CSV, TXT)
├── chroma_db/              # Local persistent vector storage
├── requirements.txt        # Project dependencies
└── README.md

# Granicus RAG Chatbot

## Overview

This project implements a Retrieval-Augmented Generation (RAG) system with:

- Document ingestion (PDF, CSV, TXT, HTML)
- Smart chunking
- ChromaDB vector storage
- Embedding-based semantic search
- Grounded LLM generation
- FastAPI REST API
- Pytest-based test suite

---

## Requirements

- Python 3.9+
- pip
- (Optional) Ollama installed for local LLM

---

## Setup Instructions

### 1. Clone Repository

git clone <repository-url>
cd <repository-folder>

### 2. Create Virtual Environment

Windows:
python -m venv venv
venv\Scripts\activate

Mac/Linux:
python -m venv venv
source venv/bin/activate

### 3. Install Dependencies

pip install -r requirements.txt

---

## Running the API

uvicorn src.api.app:app

Swagger documentation:
http://localhost:8000/docs

---

## API Endpoints

POST /chat

Request:
{
  "question": "What are the key features of GovDelivery Communications Cloud?"
}

Response:
{
  "answer": "...",
  "confidence": 0.91,
  "latency_seconds": 2.4
}

GET /health  
Returns system health and index status.

GET /stats  
Returns indexed chunk count and request statistics.

---

## Running Tests

pytest

---

## Batch Evaluation

Place questions.xlsx inside:
evaluations/

Run:
python evaluations/run_batch_evaluation.py

Results saved as:
evaluations/rag_results.xlsx

---

## Notes

- Documents must be placed inside the data/ folder.
- The vector database (chroma_db/) initializes automatically if empty.
- The system runs on CPU and does not require external API keys.



