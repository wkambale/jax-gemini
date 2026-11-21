"""Dataset profiling and sandboxed data-pipeline example with Jax Forge."""

import os

import numpy as np

import jaxforge as jf
from jaxforge.plan import DatasetProfile


def main() -> None:
    print("Configuring Jax Forge data pipeline...")
    jf.config.set({"model_name": "gemini-3.1-pro"})

    rng = np.random.default_rng(0)
    features = rng.standard_normal((100, 4), dtype=np.float32)
    labels = rng.integers(0, 2, size=(100,))

    profile = DatasetProfile.from_dataset((features, labels))
    print("Dataset profile:\n" + profile.to_prompt_text())

    if not os.environ.get("GEMINI_API_KEY"):
        print("Set GEMINI_API_KEY to run LLM-generated load_data and analyze_data steps.")
        return

    loaded = jf.load_data("Synthesize a normalized dataset of 100 rows and 4 float32 columns")
    summary = jf.analyze_data("Compute per-column mean and standard deviation", dataset=loaded)
    print("Analysis summary:", summary)


if __name__ == "__main__":
    main()
