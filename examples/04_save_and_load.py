import os

import jax_gemini as jg


def main():
    jg.config.set({"gemini_api_key": os.environ.get("GEMINI_API_KEY", "dummy")})

    jg.build("Build a basic dense model")

    # Save the model
    save_path = jg.save("test_checkpoint")
    print(f"Saved to: {save_path}")

    # Reset and load
    jg.reset()
    jg.load("test_checkpoint")
    print("Model loaded successfully!")


if __name__ == "__main__":
    main()
