"""Structured plan, dataset profiling, and experiment lineage data structures."""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _summarize_array(arr: Any) -> dict[str, Any]:
    """Extract non-sensitive structural metadata from an array-like object."""
    try:
        np_arr = np.asarray(arr)
    except Exception:
        return {"type": type(arr).__name__}

    info: dict[str, Any] = {
        "shape": list(np_arr.shape),
        "dtype": str(np_arr.dtype),
        "size": int(np_arr.size),
    }
    if np.issubdtype(np_arr.dtype, np.number) and np_arr.size > 0:
        finite_mask = np.isfinite(np_arr)
        info["nan_or_inf_count"] = int(np_arr.size - int(np.sum(finite_mask)))
        if np.any(finite_mask):
            finite_vals = np_arr[finite_mask]
            info["min"] = float(np.min(finite_vals))
            info["max"] = float(np.max(finite_vals))
        if np.issubdtype(np_arr.dtype, np.integer):
            unique_count = len(np.unique(np_arr))
            info["unique_values"] = unique_count
    return info


@dataclass(frozen=True)
class DatasetProfile:
    """Privacy-preserving structural summary of a dataset injected into LLM prompts."""

    kind: str
    summary: dict[str, Any]

    @classmethod
    def from_dataset(cls, dataset: Any) -> DatasetProfile:
        """Build a concise profile without leaking raw records."""
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
    estimated_params: int | None = None
    estimated_output_shape: str | None = None
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
        if self.estimated_params is not None:
            meta_bits.append(f"Estimated parameters: {self.estimated_params:,}")
        if self.estimated_output_shape:
            meta_bits.append(f"Output shape: {html.escape(self.estimated_output_shape)}")
        meta_line = " | ".join(meta_bits)
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
