import asyncio
import uvicorn
import base64
import json
from fastapi import FastAPI, HTTPException
from typing import Dict, Any

from worker import process_pdf_stream, chunking_pdf
from graph import graph_init
import os
import json
import os
import time
import psutil
from pathlib import Path

app = FastAPI(title="PDF Processing API")

METRICS_DIR = Path(os.environ.get("BLASE_METRICS_DIR", ".")) / "matrics"


def _file() -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    return METRICS_DIR / "data.json"


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(path)

def record_request(
    worker_name: str,
    t0: float,
    *,
    metrics: dict | None = None,
) -> None:
    proc = psutil.Process(os.getpid())

    entry = {
        "timestamp": time.strftime("%H:%M:%S"),
        "ram_mb": round(proc.memory_info().rss / 1_048_576, 2),
        "duration_ms": round((time.time() - t0) * 1000, 2),
    }

    if metrics:
        entry["metrics"] = metrics

    path = _file()
    data = _read(path)

    data.setdefault(worker_name, []).append(entry)

    _write(path, data)

def _structure_result(res, q_counters, questions):
    for qtype in ("mcq", "written", "complete"):
        for item in res.get(qtype, {}).values():
            key = f"q{q_counters[qtype]}"
            questions[qtype][key] = item
            q_counters[qtype] += 1



def _process_pdf(pdf_bytes: bytes) -> Dict[str, Any]:
    t0 = time.time()

    pages = process_pdf_stream(pdf_bytes)
    chunks = chunking_pdf(pages, 9)

    print(f"total_pages={len(pages)} | total_chunks={len(chunks)}")

    pipeline = graph_init()

    q_counters = {"mcq": 1, "written": 1, "complete": 1}
    questions = {"mcq": {}, "written": {}, "complete": {}}

    for idx, chunk in enumerate(chunks):
        chunk_t0 = time.time()

        result = pipeline.invoke({
            "text": chunk,
            "keypoints": [],
            "results": {},
            "token_usage": [],
            "usage_acc": {},
        })

        new = result["results"]

        snap = (
            result["token_usage"][0]
            if result["token_usage"]
            else {
                "tokens_input": 0,
                "tokens_output": 0,
                "api_calls": 0,
            }
        )

        unit = {
            "unit_index": idx + 1,
            "label": f"C{idx + 1}",
            "tokens_input": snap["tokens_input"],
            "tokens_output": snap["tokens_output"],
            "api_calls": snap["api_calls"],
        }

        mcq_count = len(new.get("mcq", {}))
        written_count = len(new.get("written", {}))
        complete_count = len(new.get("complete", {}))

        print(
            f"[Chunk {idx+1}] "
            f"{len(result['keypoints'])} keypoints | "
            f"{mcq_count} MCQ | "
            f"{written_count} Written | "
            f"{complete_count} Complete | "
            f"in={snap['tokens_input']} "
            f"out={snap['tokens_output']} "
            f"calls={snap['api_calls']}"
        )

        _structure_result(new, q_counters, questions)

        payload = {
                    "unit_label": "chunk",
                    "units": [unit],
                }
        record_request(
            "question_generation",
            chunk_t0,
            metrics=payload,
        )
        break 
    return {"questions": questions}

    # print(f"Token usage saved → {USAGE_PATH}")


@app.post("/pdf")
async def submit_pdf(data: dict):
    if not data:
        raise HTTPException(status_code=400, detail="Request body must not be empty.")
    all_results = {}
    for key, value in data.items():
        print(key)
        pdf_bytes = base64.b64decode(value)
        all_results[key] = _process_pdf(pdf_bytes)
    return all_results


def main():
    uvicorn.run("server:app", port=4321)

if __name__ == "__main__":
    main()