import asyncio
import time
import pandas as pd
from pathlib import Path
from src.rag_pipeline import RAGPipeline


# Get current script directory (tests folder)
BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "questions.xlsx"
OUTPUT_FILE = BASE_DIR / "rag_results.xlsx"


async def run_batch():
    # Load Excel
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found.")

    df = pd.read_excel(INPUT_FILE)

    if "Questions" not in df.columns:
        raise ValueError("Excel must contain a column named 'Questions'")

    rag = RAGPipeline()

    answers = []
    confidences = []
    latencies = []

    for question in df["Questions"]:
        print(f"\nProcessing: {question}")

        start_time = time.time()

        response = await rag.ask(question)

        end_time = time.time()

        latency = round(end_time - start_time, 2)

        answers.append(response.get("answer", ""))
        confidences.append(response.get("confidence", ""))
        latencies.append(latency)

        print(f"Latency: {latency}s")

    # Add results
    df["Answer"] = answers
    df["Confidence"] = confidences
    df["Latency_sec"] = latencies

    # Save Excel in same folder
    df.to_excel(OUTPUT_FILE, index=False)

    print(f"\nSaved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(run_batch())
