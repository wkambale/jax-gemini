import os

import jax_gemini as jg


def main():
    jg.config.set({"gemini_api_key": os.environ.get("GEMINI_API_KEY", "dummy")})

    jg.build("Build a CNN with 3 layers")

    print(jg.show_code())


if __name__ == "__main__":
    main()
