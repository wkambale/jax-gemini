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
        "getattr",
        "setattr",
        "delattr",
        "globals",
        "locals",
        "vars",
        "dir",
        "type",
    }
)

BANNED_NAMES: frozenset[str] = BANNED_BUILTINS | frozenset(
    {
        "__builtins__",
        "__build_class__",
        "__package__",
        "__spec__",
        "__loader__",
        "__name__",
        "__file__",
        "__doc__",
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
        "__reduce_ex__",
        "__mro__",
        "__dict__",
        "__getattribute__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        "__traceback__",
        "__wrapped__",
        "tb_frame",
        "tb_next",
        "f_back",
        "f_globals",
        "f_builtins",
        "f_locals",
        "f_code",
        "gi_frame",
        "gi_code",
        "cr_frame",
        "cr_code",
        "ag_frame",
        "ag_code",
        "func_globals",
        "func_code",
        "func_closure",
        "im_func",
        "im_self",
        "im_class",
    }
)

ALLOWED_DUNDER_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__init__",
        "__call__",
        "__len__",
        "__iter__",
        "__next__",
        "__getitem__",
        "__setitem__",
        "__repr__",
        "__str__",
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
MAX_CODE_BYTES = 20_000
MAX_AST_NODES = 2_000


def _contains_call(node: ast.AST) -> bool:
    return any(isinstance(child, ast.Call) for child in ast.walk(node))


def _validate_top_level_statement(stmt: ast.stmt, code: str) -> None:
    if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
        return
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(
        stmt.value.value, str
    ):
        return
    if isinstance(stmt, ast.Assign) and not _contains_call(stmt.value):
        return
    if isinstance(stmt, ast.AnnAssign) and (stmt.value is None or not _contains_call(stmt.value)):
        return
    raise JaxGeminiValidationError(
        f"Top-level statement '{type(stmt).__name__}' is not allowed outside functions/classes.",
        code=code,
    )


class CodeValidator:
    @staticmethod
    def validate(code: str, intent: str) -> None:
        """
        Validate generated code via AST analysis.

        Raises JaxGeminiValidationError with a descriptive message
        if any security or structural rule is violated.
        """
        if not isinstance(code, str) or not code.strip():
            raise JaxGeminiValidationError("Generated code is empty.", code=str(code))

        # 1. Size and line limit checks
        if len(code.encode("utf-8")) > MAX_CODE_BYTES:
            raise JaxGeminiValidationError(
                f"Generated code exceeds maximum byte length of {MAX_CODE_BYTES}.",
                code=code,
            )

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
                code=code,
            ) from e

        nodes = list(ast.walk(tree))
        if len(nodes) > MAX_AST_NODES:
            raise JaxGeminiValidationError(
                f"Generated code AST has {len(nodes)} nodes — maximum is {MAX_AST_NODES}.",
                code=code,
            )

        # 3. Top-level structure check (block arbitrary top-level execution side effects)
        for stmt in tree.body:
            _validate_top_level_statement(stmt, code)

        # 4. Walk AST and check every node
        for node in nodes:
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
                if node.level > 0 or not node.module:
                    raise JaxGeminiValidationError(
                        "Relative imports are not allowed.",
                        code=code,
                    )
                root = node.module.split(".")[0]
                if root not in WHITELIST_IMPORTS:
                    raise JaxGeminiValidationError(
                        f"Import from '{node.module}' is not allowed.", code=code
                    )

            # Banned builtin/name references (blocks both direct calls and aliasing)
            if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
                raise JaxGeminiValidationError(
                    f"Use of '{node.id}' is not allowed in generated code.", code=code
                )

            # Banned call checks (belt-and-suspenders)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in BANNED_BUILTINS:
                    raise JaxGeminiValidationError(
                        f"Call to '{node.func.id}' is not allowed in generated code.", code=code
                    )

            # Banned attribute access (explicit blocklist + unapproved dunder attributes)
            if isinstance(node, ast.Attribute):
                is_dunder = node.attr.startswith("__") and node.attr.endswith("__")
                if node.attr in BANNED_ATTRIBUTES or (
                    is_dunder and node.attr not in ALLOWED_DUNDER_ATTRIBUTES
                ):
                    raise JaxGeminiValidationError(
                        f"Access to attribute '{node.attr}' is not allowed.", code=code
                    )

        # 5. Required function check (fail closed on unknown intents)
        required_fn = REQUIRED_FUNCTION_NAMES.get(intent)
        if not required_fn:
            raise JaxGeminiValidationError(f"Unknown intent: '{intent}'", code=code)

        defined_functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        if required_fn not in defined_functions:
            raise JaxGeminiValidationError(
                f"Generated code must define a top-level function named '{required_fn}'. "
                f"Found functions: {sorted(defined_functions)}",
                code=code,
            )
