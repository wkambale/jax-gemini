# LLM providers

Jax Forge (`jaxforge`) works across hosted and local language models through a unified `AbstractLLM` adapter interface. Configure providers globally via `jf.config.set(...)` or per session using isolated `JaxForge(JaxForgeConfig(...))` instances.

## Supported providers

| Provider key | Prefix or example model | API key environment variable | Transport |
| --- | --- | --- | --- |
| `google` | `gemini-3.1-pro` or `gemini:gemini-3.1-flash` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` | `google-genai` SDK with HTTPS REST fallback |
| `openai` | `openai:gpt-5` or `gpt-4.1` | `OPENAI_API_KEY` | OpenAI-compatible Chat Completions |
| `anthropic` | `anthropic:claude-sonnet-4-5` | `ANTHROPIC_API_KEY` | Anthropic Messages API |
| `openrouter` | `openrouter:anthropic/claude-sonnet-4-5` | `OPENROUTER_API_KEY` | OpenRouter Chat Completions endpoint |
| `ollama` | `ollama:qwen2.5-coder:32b` | Optional local key (`"ollama"`) | Local `base_url` (defaults to `http://localhost:11434/v1`) |
| `litellm` | `litellm:bedrock/anthropic.claude-v2` | Provider-specific env vars | `litellm.completion` routing |

=== "Google Gemini"

    ```python
    import jaxforge as jf

    jf.config.set({
        "provider": "google",
        "model_name": "gemini-3.1-pro",
        "fast_model_name": "gemini-3.1-flash",
    })
    ```

=== "OpenAI and OpenRouter"

    ```python
    import jaxforge as jf

    jf.config.set({
        "model_name": "openai:gpt-5",
        "fast_model_name": "openai:gpt-4.1-mini",
    })
    ```

=== "Anthropic Claude"

    ```python
    import jaxforge as jf

    jf.config.set({
        "model_name": "anthropic:claude-sonnet-4-5",
    })
    ```

=== "Local Ollama or vLLM"

    ```python
    import jaxforge as jf

    jf.config.set({
        "model_name": "ollama:qwen2.5-coder:32b",
        "base_url": "http://localhost:11434/v1",
    })
    ```

## Model routing and retry acceleration

Setting `fast_model_name` enables `RouterLLM`, which uses the primary `model_name` for architecture synthesis (`build`, `train`, `explore`) and routes repair attempts (`retry > 1` after validation or sandbox diagnostics) to the lower-latency model automatically.

## Token and latency telemetry

Every LLM call records an `LLMUsage` entry tracking provider, model identifier, prompt tokens, completion tokens, and wall-clock latency:

```python
summary = jf.usage_summary()
print(summary)
# {'calls': 2, 'prompt_tokens': 840, 'completion_tokens': 412, 'total_tokens': 1252, 'total_latency_ms': 920.5}
```
