import os

import jaxforge as jf


def main():
    jf.config.set({"gemini_api_key": os.environ.get("GEMINI_API_KEY", "dummy")})

    jf.build("Build a basic dense model")

    # Save the model
    save_path = jf.save("test_checkpoint")
    print(f"Saved to: {save_path}")

    # Reset and load
    jf.reset()
    jf.load("test_checkpoint")
    print("Model loaded successfully!")


if __name__ == "__main__":
    main()
