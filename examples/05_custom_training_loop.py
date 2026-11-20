import os

import jaxforge as jf


def main():
    jf.config.set({"gemini_api_key": os.environ.get("GEMINI_API_KEY", "dummy")})

    jf.build("Build a CNN with 3 layers")

    print(jf.show_code())


if __name__ == "__main__":
    main()
