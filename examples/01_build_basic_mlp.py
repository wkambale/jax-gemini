import jax_gemini as jg
import os

def main():
    jg.config.set({"gemini_api_key": os.environ.get("GEMINI_API_KEY", "dummy")})
    
    print("Building model...")
    model = jg.build("Build a simple 3-layer MLP with 128 hidden units for 10 classes.")
    print("Done!")
    
    print("Explanation:", jg.explain())
    print("\nCode:\n", jg.show_code())

if __name__ == "__main__":
    main()
