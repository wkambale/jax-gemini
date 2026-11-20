"""Experiment tracking interfaces and local/optional external loggers."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class MetricsTracker(ABC):
    """Protocol for experiment metric logging backends."""

    @abstractmethod
    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        ...

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        ...


class InMemoryTracker(MetricsTracker):
    """Stores logged parameters and metric records in memory."""

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}
        self.records: list[dict[str, Any]] = []

    def log_params(self, params: dict[str, Any]) -> None:
        self.params.update(params)

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self.records.append({"step": step, "timestamp": time.time(), "metrics": dict(metrics)})


class JsonlTracker(MetricsTracker):
    """Appends structured parameter and metric records into a JSONL log file."""

    def __init__(self, log_path: str | Path):
        self.path = Path(log_path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_params(self, params: dict[str, Any]) -> None:
        self._append({"type": "params", "timestamp": time.time(), "data": params})

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        self._append({"type": "metrics", "step": step, "timestamp": time.time(), "data": metrics})

    def _append(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")


def create_tracker(kind: str = "memory", path: str | Path | None = None) -> MetricsTracker:
    """Construct a metrics tracker ('memory' or 'jsonl')."""
    if kind == "memory":
        return InMemoryTracker()
    if kind == "jsonl":
        target = path or Path("jaxforge_runs") / "metrics.jsonl"
        return JsonlTracker(target)
    raise ValueError(f"Unknown tracker kind: '{kind}'. Expected 'memory' or 'jsonl'.")
