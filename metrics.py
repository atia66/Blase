import json
import os
import time
import psutil
from pathlib import Path

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
    tokens: dict | None = None,
) -> None:
    proc = psutil.Process(os.getpid())

    entry = {
        "timestamp":   time.strftime("%H:%M:%S"),
        "ram_mb":      round(proc.memory_info().rss / 1_048_576, 2),
        "duration_ms": round((time.time() - t0) * 1000, 2),
    }

    if tokens:
        entry["tokens"] = tokens

    path = _file()
    data = _read(path)

    data.setdefault(worker_name, []).append(entry)

    _write(path, data)


def log_usage(matrices: dict) -> None:
    """Write a top-level 'matrices' block for the full exam summary."""
    path = _file()
    data = _read(path)
    data["matrices"] = matrices
    _write(path, data)