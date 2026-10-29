"""Observability: record agent events as JSONL trace for replay/analysis."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TraceRecorder:
    """Agent event hook that appends every event to a JSONL file.

    Usage:
        tracer = TraceRecorder("run.jsonl")
        agent.on(tracer)          # record everything
        agent.run("...")
        tracer.close()
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = self.path.open("a", encoding="utf-8")

    def __call__(self, event: str, payload: dict[str, Any]) -> None:
        record = {"ts": round(time.time(), 3), "event": event,
                  "payload": self._safe(payload)}
        self._fh.write(json.dumps(record, ensure_ascii=False,
                                  default=str) + "\n")
        self._fh.flush()

    @staticmethod
    def _safe(payload: dict[str, Any]) -> dict[str, Any]:
        out = {}
        for k, v in payload.items():
            if hasattr(v, "__dict__"):
                out[k] = {kk: vv for kk, vv in vars(v).items()
                          if isinstance(vv, (str, int, float, bool, list, dict,
                                             type(None)))} or repr(v)
            else:
                out[k] = v
        return out

    def close(self) -> None:
        self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSONL trace file back into a list of event records."""
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
