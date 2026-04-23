import os

import numpy as np

import jax_gemini as jg


def main():
    jg.config.set({"gemini_api_key": os.environ.get("GEMINI_API_KEY", "dummy")})

    # 1. Build
    jg.build("Build a CNN for MNIST mapping 28x28 grayscale to 10 classes.")

    # Generate dummy MNIST data
    X = np.random.randn(100, 28, 28, 1).astype(np.float32)
    y = np.random.randint(0, 10, size=(100,))

    # 2. Train
    print("Training model...")
    _model, metrics = jg.train("Train for 3 epochs using SGD", dataset=(X, y))

    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
