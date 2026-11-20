import os

import jaxforge as jf


def main():
    jf.config.set({"gemini_api_key": os.environ.get("GEMINI_API_KEY", "dummy")})

    print("Building model...")
    jf.build("Build a simple 3-layer MLP with 128 hidden units for 10 classes.")
    print("Done!")

    print("Explanation:", jf.explain())
    print("\nCode:\n", jf.show_code())


if __name__ == "__main__":
    main()
