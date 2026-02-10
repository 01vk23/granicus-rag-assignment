import asyncio
import time
import pandas as pd
from pathlib import Path
from src.rag_pipeline import RAGPipeline


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "questions.xlsx"
OUTPUT_FILE = BASE_DIR / "rag_results.xlsx"


async def run_batch():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"{INPUT_FILE} not found.")

    df = pd.read_excel(INPUT_FILE)

    if "Questions" not in df.columns:
        raise ValueError("Excel must contain a column named 'Questions'")

    rag = RAGPipeline()

    answers = []
    confidences = []
    latencies = []

    print("\n🚀 Starting Batch Evaluation...\n")

    for question in df["Questions"]:
        if not isinstance(question, str) or not question.strip():
            answers.append("")
            confidences.append(0.0)
            latencies.append(0.0)
            continue

        print(f"Processing: {question}")

        start_time = time.time()

        try:
            response = await rag.ask(question)
            answer = response.get("answer", "")
            confidence = response.get("confidence", 0.0)
        except Exception as e:
            answer = f"ERROR: {str(e)}"
            confidence = 0.0

        latency = round(time.time() - start_time, 3)

        answers.append(answer)
        confidences.append(confidence)
        latencies.append(latency)

        print(f"   → Done in {latency}s")

    df["Answer"] = answers
    df["Confidence"] = confidences
    df["Latency_sec"] = latencies

    df.to_excel(OUTPUT_FILE, index=False)

    print(f"\n✅ Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(run_batch())
