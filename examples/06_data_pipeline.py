import numpy as np

import jax_gemini as jg


def main():
    print("Setting up Jax Gemini Config...")
    # Using a fake key for local execution testing since actual key relies on user's env
    jg.config.set({"model_name": "gemini-3.1-pro"})

    print("Generating synthetic data...")
    dataset = np.random.randn(100, 2)

    # We will just print the prompts to show end to end capability.
    # Since we don't have the API key in the automated runner environment,
    # we just run pytest for actual verification, but this serves as a documented example.

    print("Building Data Pipeline...")
    try:
        loaded_data = jg.load_data("Load a synthetic dataset of 100 rows and 2 columns from memory")
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Skipping actual API call due to missing key: {e}")


if __name__ == "__main__":
    main()
