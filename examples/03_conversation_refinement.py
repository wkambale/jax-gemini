import os

import jax_gemini as jg


def main():
    jg.config.set({"gemini_api_key": os.environ.get("GEMINI_API_KEY", "dummy")})

    print("Initial build...")
    jg.build("Build a 2-layer MLP")

    print("Modifying model...")
    jg.modify("Add a 0.5 dropout layer between the two dense layers")

    print("New Code:\n", jg.show_code())


if __name__ == "__main__":
    main()
