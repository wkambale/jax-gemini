import ast

from jax_gemini.exceptions import JaxGeminiValidationError

WHITELIST_IMPORTS: frozenset[str] = frozenset(
    {
        "jax",
        "flax",
        "optax",
        "orbax",
        "numpy",
        "functools",
        "math",
        "typing",
        "dataclasses",
        "matplotlib",
    }
)

BANNED_BUILTINS: frozenset[str] = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "__import__",
        "open",
        "input",
        "memoryview",
        "breakpoint",
    }
)

BANNED_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__globals__",
        "__locals__",
        "__builtins__",
        "__code__",
        "__closure__",
        "__reduce__",
    }
)

REQUIRED_FUNCTION_NAMES: dict[str, str] = {
    "build": "build_model",
    "train": "train_model",
    "evaluate": "evaluate_model",
    "save": "save_model",
    "load": "load_model",
    "load_data": "load_data",
    "preprocess_data": "preprocess_data",
    "analyze_data": "analyze_data",
    "visualize_data": "visualize_data",
}

MAX_CODE_LINES = 200


class CodeValidator:
    @staticmethod
    def validate(code: str, intent: str) -> None:
        """
        Validate generated code via AST analysis.

        Raises JaxGeminiValidationError with a descriptive message
        if any security or structural rule is violated.
        """
        # 1. Line limit check
        lines = code.strip().splitlines()
        if len(lines) > MAX_CODE_LINES:
            raise JaxGeminiValidationError(
                f"Generated code is {len(lines)} lines — maximum is {MAX_CODE_LINES}. "
                "This may indicate a hallucination or prompt injection attempt.",
                code=code,
            )

        # 2. Parse to AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise JaxGeminiValidationError(
                f"Generated code has a syntax error: {e}",
                code=code
            ) from e

        # 3. Walk AST and check every node
        for node in ast.walk(tree):
            # Import validation
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in WHITELIST_IMPORTS:
                        raise JaxGeminiValidationError(
                            f"Import '{alias.name}' is not in the allowed list. "
                            f"Allowed: {sorted(WHITELIST_IMPORTS)}",
                            code=code,
                        )

            if isinstance(node, ast.ImportFrom):
                if node.module:
                    root = node.module.split(".")[0]
                    if root not in WHITELIST_IMPORTS:
                        raise JaxGeminiValidationError(
                            f"Import from '{node.module}' is not allowed.", code=code
                        )

            # Banned builtin calls
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BANNED_BUILTINS:
                        raise JaxGeminiValidationError(
                            f"Call to '{node.func.id}' is not allowed in generated code.", code=code
                        )

            # Banned attribute access
            if isinstance(node, ast.Attribute):
                if node.attr in BANNED_ATTRIBUTES:
                    raise JaxGeminiValidationError(
                        f"Access to attribute '{node.attr}' is not allowed.", code=code
                    )

        # 4. Required function check
        required_fn = REQUIRED_FUNCTION_NAMES.get(intent)
        if required_fn:
            defined_functions = {
                node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
            }
            if required_fn not in defined_functions:
                raise JaxGeminiValidationError(
                    f"Generated code must define a function named '{required_fn}'. "
                    f"Found functions: {sorted(defined_functions)}",
                    code=code,
                )
