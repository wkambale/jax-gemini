import os

import jaxforge as jf


def main():
    jf.config.set({"gemini_api_key": os.environ.get("GEMINI_API_KEY", "dummy")})

    print("Initial build...")
    jf.build("Build a 2-layer MLP")

    print("Modifying model...")
    jf.modify("Add a 0.5 dropout layer between the two dense layers")

    print("New Code:\n", jf.show_code())


if __name__ == "__main__":
    main()
