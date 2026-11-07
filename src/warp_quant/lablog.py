"""Structured event logging for warp-quant runs.

Emits JSONL events into runs/<tool>.<UTC ts>.<pid>/events.jsonl (RUN_START /
RUN_DONE / RUN_FAILED / stage events).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

_TOOL = Path(sys.argv[0]).stem or "warp_quant"
_ROOT = Path(os.environ.get("RUNS_DIR") or (Path.cwd() / "runs"))
_DIR: Path | None = None


def _dir() -> Path:
    global _DIR
    if _DIR is None:
        ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        _DIR = _ROOT / f"{_TOOL}.{ts}.{os.getpid()}"
        _DIR.mkdir(parents=True, exist_ok=True)
    return _DIR


def log_event(etype: str, **kw) -> None:
    obj = {
        "ts": f"{time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime())}.{int(time.time() * 1000) % 1000:03d}Z",
        "type": etype,
        "tool": _TOOL,
        **kw,
    }
    with (_dir() / "events.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj) + "\n")
    fields = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[{obj['ts']}] [{_TOOL}] {etype} {fields}".rstrip(), file=sys.stderr)


def run_done(**kw) -> None:
    log_event("RUN_DONE", status="ok", **kw)


def run_fail(**kw) -> None:
    log_event("RUN_FAILED", status="fail", **kw)
