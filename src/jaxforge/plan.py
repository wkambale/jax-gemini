"""Structured plan, dataset profiling, and experiment lineage data structures."""

from __future__ import annotations

import html
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_MAX_PROFILE_SAMPLE_ELEMENTS = 65_536
_MAX_SEQ_CONVERT_LEN = 10_000


def _summarize_array(arr: Any) -> dict[str, Any]:
    """Extract bounded, non-sensitive structural metadata without exhausting iterators."""
    if isinstance(arr, Iterator):
        return {"type": type(arr).__name__, "streaming": True}

    np_sample: np.ndarray[Any, Any] | None = None
    full_shape: list[int] | None = None
    dtype_str: str | None = None
    total_size: int | None = None
    sampled = False

    if hasattr(arr, "shape") and hasattr(arr, "dtype") and not isinstance(arr, type):
        try:
            raw_shape = [int(d) for d in arr.shape]
            full_shape = raw_shape
            dtype_str = str(arr.dtype)
            total_size = int(math.prod(raw_shape))
            if total_size > _MAX_PROFILE_SAMPLE_ELEMENTS:
                sampled = True
                flat_view = np.asarray(arr.reshape(-1)[:_MAX_PROFILE_SAMPLE_ELEMENTS])
                np_sample = flat_view
            else:
                np_sample = np.asarray(arr)
        except Exception:
            return {"type": type(arr).__name__}
    elif isinstance(arr, (list, tuple)):
        if len(arr) > _MAX_SEQ_CONVERT_LEN:
            return {"type": type(arr).__name__, "length": len(arr), "sampled": True}
        try:
            np_sample = np.asarray(arr)
            full_shape = list(np_sample.shape)
            dtype_str = str(np_sample.dtype)
            total_size = int(np_sample.size)
        except Exception:
            return {"type": type(arr).__name__, "length": len(arr)}
    else:
        return {"type": type(arr).__name__}

    info: dict[str, Any] = {
        "shape": full_shape,
        "dtype": dtype_str,
        "size": total_size,
    }
    if sampled:
        info["sampled"] = True
    if np_sample is not None and np.issubdtype(np_sample.dtype, np.number) and np_sample.size > 0:
        finite_mask = np.isfinite(np_sample)
        info["nan_or_inf_count"] = int(np_sample.size - int(np.sum(finite_mask)))
        if np.any(finite_mask):
            finite_vals = np_sample[finite_mask]
            info["min"] = float(np.min(finite_vals))
            info["max"] = float(np.max(finite_vals))
        if np.issubdtype(np_sample.dtype, np.integer):
            info["unique_values"] = len(np.unique(np_sample))
    return info


@dataclass(frozen=True)
class DatasetProfile:
    """Privacy-preserving structural summary of a dataset injected into LLM prompts."""

    kind: str
    summary: dict[str, Any]

    @classmethod
    def from_dataset(cls, dataset: Any) -> DatasetProfile:
        """Build a concise profile without leaking raw records or consuming streams."""
        if dataset is None:
            return cls(kind="none", summary={})
        if isinstance(dataset, tuple) and len(dataset) == 2:
            x_part, y_part = dataset
            return cls(
                kind="supervised_pair",
                summary={
                    "features": _summarize_array(x_part),
                    "targets": _summarize_array(y_part),
                },
            )
        if isinstance(dataset, dict):
            return cls(
                kind="mapping",
                summary={
                    str(k): _summarize_array(v) for k, v in list(dataset.items())[:16]
                },
            )
        return cls(kind=type(dataset).__name__, summary={"value": _summarize_array(dataset)})

    def to_prompt_text(self) -> str:
        """Render a compact text block suitable for prompt injection."""
        if not self.summary:
            return ""
        lines = [f"Dataset profile (kind={self.kind}):"]
        for key, stats in self.summary.items():
            lines.append(f"- {key}: {stats}")
        return "\n".join(lines)


@dataclass
class GeneratedPlan:
    """Preview artifact produced before sandbox execution occurs."""

    intent: str
    code: str
    description: str
    provider: str
    model_name: str
    framework: str = "flax"
    validation_passed: bool = True
    unified_diff: str = ""
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def _repr_html_(self) -> str:
        """Render a clean HTML preview card for Jupyter notebooks."""
        escaped_intent = html.escape(self.intent)
        escaped_desc = html.escape(self.description)
        escaped_code = html.escape(self.code)
        meta_bits = [
            f"Provider: {html.escape(self.provider)} ({html.escape(self.model_name)})",
            f"Framework: {html.escape(self.framework)}",
            f"Validated: {'yes' if self.validation_passed else 'no'}",
        ]
        meta_line = " | ".join(meta_bits)
        warnings_block = ""
        if self.warnings:
            items = "".join(f"<li>{html.escape(str(w))}</li>" for w in self.warnings)
            warnings_block = (
                "<ul style='color:#9a3412;font-size:12px;margin:8px 0;padding-left:18px;'>"
                f"{items}</ul>"
            )
        diff_block = ""
        if self.unified_diff:
            diff_block = (
                "<h4>Diff from previous code</h4>"
                f"<pre><code>{html.escape(self.unified_diff)}</code></pre>"
            )
        return (
            "<div style='border:1px solid #dce3ec;border-radius:10px;padding:16px;"
            "font-family:Inter,system-ui,sans-serif;background:#faf8f4;'>"
            f"<h3 style='margin-top:0;'>Jax Forge preview: {escaped_intent}</h3>"
            f"<p style='font-size:12px;color:#5f6f82;'>{meta_line}</p>"
            f"<p style='color:#3e4c5e;'>{escaped_desc}</p>"
            f"{warnings_block}"
            f"{diff_block}"
            f"<pre style='background:#fff;border:1px solid #dce3ec;padding:12px;"
            f"border-radius:8px;overflow:auto;'><code>{escaped_code}</code></pre>"
            "</div>"
        )


@dataclass
class ExperimentCandidate:
    """Single candidate outcome produced during explore() tree search."""

    index: int
    prompt_variant: str
    code: str
    description: str
    metrics: dict[str, Any]
    score: float | None = None
    error: str | None = None
    model: Any = None


@dataclass
class ExperimentResult:
    """Aggregated outcome from metric-guided branching exploration."""

    best_candidate: ExperimentCandidate
    candidates: list[ExperimentCandidate]
    metric: str
    maximize: bool

    @property
    def best_model(self) -> Any:
        return self.best_candidate.model
