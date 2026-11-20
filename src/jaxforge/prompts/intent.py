"""Word-boundary intent classifier and framework detection for prompts."""

import re

from jaxforge.exceptions import JaxForgeIntentError

_INTENT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "visualize_data",
        re.compile(r"\b(visuali[zs]e|plot|plotting|graph|chart|draw)\b"),
    ),
    (
        "preprocess_data",
        re.compile(r"\b(preprocess|normali[zs]e|standardi[zs]e|clean|reshape|augment|split)\b"),
    ),
    (
        "analyze_data",
        re.compile(r"\b(analy[zs]e|explore|stats?|statistics|summari[zs]e|summary|inspect)\b"),
    ),
    (
        "load_data",
        re.compile(
            r"\b(load|read|generate|fetch)\s+(the\s+|a\s+)?(data|dataset)\b|\bdataset\b"
        ),
    ),
    (
        "evaluate",
        re.compile(r"\b(eval|evaluate|test|assess|validate|benchmark)\b"),
    ),
    (
        "train",
        re.compile(r"\b(train|fit|finetune|fine-tune)\b"),
    ),
    (
        "save",
        re.compile(r"\b(save|export)\b"),
    ),
    (
        "load",
        re.compile(r"\b(load|restore)\b"),
    ),
    (
        "build",
        re.compile(
            r"\b(build|create|make|construct|design|define|modify|change|add|remove|update)\b"
        ),
    ),
)


class IntentClassifier:
    """Word-boundary rule-based intent classifier for jaxforge prompts."""

    @staticmethod
    def classify(prompt: str) -> str:
        """Classify a user prompt into a supported jaxforge intent."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise JaxForgeIntentError("Cannot classify an empty prompt.")
        normalized = prompt.lower()
        for intent, pattern in _INTENT_RULES:
            if pattern.search(normalized):
                return intent
        raise JaxForgeIntentError(
            "Unable to determine operation intent from prompt. "
            "Try an explicit action such as 'build', 'train', 'evaluate', 'save', 'load', "
            "'load data', 'preprocess', 'analyze', or 'plot'."
        )

    @staticmethod
    def detect_framework(prompt: str, default: str = "flax") -> str:
        """Infer preferred JAX modeling framework ('flax' or 'equinox') from prompt."""
        lower = prompt.lower()
        if re.search(r"\b(equinox|eqx|diffrax)\b", lower):
            return "equinox"
        if re.search(r"\b(flax|nnx)\b", lower):
            return "flax"
        return "flax" if default == "auto" else default
