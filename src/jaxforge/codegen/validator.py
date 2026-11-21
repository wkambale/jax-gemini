"""Static AST validator enforcing sandbox imports, builtins, attributes, and limits."""

import ast

from jaxforge.exceptions import JaxForgeValidationError

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
        "equinox",
        "grain",
        "safetensors",
    }
)

_CORE_ML_IMPORTS: frozenset[str] = frozenset(
    {
        "jax",
        "flax",
        "optax",
        "numpy",
        "functools",
        "math",
        "typing",
        "dataclasses",
        "equinox",
        "grain",
    }
)

INTENT_ALLOWED_IMPORTS: dict[str, frozenset[str]] = {
    "build": _CORE_ML_IMPORTS,
    "train": _CORE_ML_IMPORTS,
    "evaluate": _CORE_ML_IMPORTS,
    "save": _CORE_ML_IMPORTS | {"orbax"},
    "load": _CORE_ML_IMPORTS | {"orbax", "safetensors"},
    "load_data": _CORE_ML_IMPORTS,
    "preprocess_data": _CORE_ML_IMPORTS,
    "analyze_data": _CORE_ML_IMPORTS,
    "visualize_data": _CORE_ML_IMPORTS | {"matplotlib"},
}

BANNED_NUMPY_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "fromfile",
        "memmap",
        "DataSource",
        "ctypeslib",
        "f2py",
        "distutils",
        "testing",
        "lib",
    }
)

BANNED_MODULE_PREFIXES: frozenset[str] = frozenset(
    {
        "numpy.ctypeslib",
        "numpy.f2py",
        "numpy.distutils",
        "numpy.testing",
        "numpy.lib",
        "numpy.compat",
        "jax.profiler",
        "jax.experimental.compilation_cache",
        "flax.io",
        "flax.training.checkpoints",
        "orbax.checkpoint.test_utils",
        "matplotlib._api",
        "matplotlib.backend_bases",
        "matplotlib.backends",
        "safetensors._safetensors_rust",
        "safetensors.torch",
        "safetensors.tensorflow",
        "safetensors.paddle",
        "grain.multiprocessing",
    }
)


def is_banned_module_path(module_path: str) -> bool:
    """Return True if a dotted module path or any parent matches BANNED_MODULE_PREFIXES."""
    return any(
        module_path == prefix or module_path.startswith(f"{prefix}.")
        for prefix in BANNED_MODULE_PREFIXES
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
        "BaseException",
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
        "__self__",
        "__func__",
        "format",
        "format_map",
        "_target",
        "_overrides",
        "_blocked",
        "_inner",
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
    raise JaxForgeValidationError(
        f"Top-level statement '{type(stmt).__name__}' is not allowed outside functions/classes.",
        code=code,
    )


class CodeValidator:
    @staticmethod
    def validate(code: str, intent: str) -> None:
        """Validate generated code via AST analysis, raising JaxForgeValidationError on breach."""
        if not isinstance(code, str) or not code.strip():
            raise JaxForgeValidationError("Generated code is empty.", code=str(code))

        if len(code.encode("utf-8")) > MAX_CODE_BYTES:
            raise JaxForgeValidationError(
                f"Generated code exceeds maximum byte length of {MAX_CODE_BYTES}.",
                code=code,
            )

        lines = code.strip().splitlines()
        if len(lines) > MAX_CODE_LINES:
            raise JaxForgeValidationError(
                f"Generated code is {len(lines)} lines; maximum is {MAX_CODE_LINES}. "
                "This may indicate a hallucination or prompt injection attempt.",
                code=code,
            )

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            raise JaxForgeValidationError(
                f"Generated code has a syntax error: {e}",
                code=code,
            ) from e

        nodes = list(ast.walk(tree))
        if len(nodes) > MAX_AST_NODES:
            raise JaxForgeValidationError(
                f"Generated code AST has {len(nodes)} nodes; maximum is {MAX_AST_NODES}.",
                code=code,
            )

        required_fn = REQUIRED_FUNCTION_NAMES.get(intent)
        if not required_fn:
            raise JaxForgeValidationError(f"Unknown intent: '{intent}'", code=code)
        allowed_imports = INTENT_ALLOWED_IMPORTS.get(intent, WHITELIST_IMPORTS)

        for stmt in tree.body:
            _validate_top_level_statement(stmt, code)

        for node in nodes:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root not in allowed_imports:
                        raise JaxForgeValidationError(
                            f"Import '{alias.name}' is not allowed for intent '{intent}'. "
                            f"Allowed: {sorted(allowed_imports)}",
                            code=code,
                        )
                    if is_banned_module_path(alias.name) or any(
                        part in BANNED_ATTRIBUTES
                        or (root == "numpy" and part in BANNED_NUMPY_ATTRIBUTES)
                        for part in alias.name.split(".")[1:]
                    ):
                        raise JaxForgeValidationError(
                            f"Import '{alias.name}' is not allowed for intent '{intent}'.",
                            code=code,
                        )

            if isinstance(node, ast.ImportFrom):
                if node.level > 0 or not node.module:
                    raise JaxForgeValidationError(
                        "Relative imports are not allowed.",
                        code=code,
                    )
                root = node.module.split(".")[0]
                if root not in allowed_imports:
                    raise JaxForgeValidationError(
                        f"Import from '{node.module}' is not allowed for intent '{intent}'.",
                        code=code,
                    )
                if is_banned_module_path(node.module) or any(
                    part in BANNED_ATTRIBUTES
                    or (root == "numpy" and part in BANNED_NUMPY_ATTRIBUTES)
                    for part in node.module.split(".")[1:]
                ):
                    raise JaxForgeValidationError(
                        f"Import from '{node.module}' is not allowed for intent '{intent}'.",
                        code=code,
                    )
                for alias in node.names:
                    imported = alias.name
                    if (
                        imported == "*"
                        or is_banned_module_path(f"{node.module}.{imported}")
                        or imported in BANNED_ATTRIBUTES
                        or imported in BANNED_NAMES
                        or (root == "numpy" and imported in BANNED_NUMPY_ATTRIBUTES)
                    ):
                        raise JaxForgeValidationError(
                            f"Import of '{imported}' from '{node.module}' is not allowed.",
                            code=code,
                        )

            if isinstance(node, ast.ExceptHandler) and node.type is None:
                raise JaxForgeValidationError(
                    "Bare 'except:' clauses are not allowed; catch Exception or a specific type.",
                    code=code,
                )

            if isinstance(node, ast.Name) and node.id in BANNED_NAMES:
                raise JaxForgeValidationError(
                    f"Use of '{node.id}' is not allowed in generated code.", code=code
                )

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in BANNED_BUILTINS:
                    raise JaxForgeValidationError(
                        f"Call to '{node.func.id}' is not allowed in generated code.", code=code
                    )
                for kw in node.keywords:
                    if kw.arg == "allow_pickle" and not (
                        isinstance(kw.value, ast.Constant) and kw.value.value is False
                    ):
                        raise JaxForgeValidationError(
                            "allow_pickle=True is not allowed in generated code.", code=code
                        )

            if isinstance(node, ast.Attribute):
                is_dunder = node.attr.startswith("__") and node.attr.endswith("__")
                if (
                    node.attr in BANNED_ATTRIBUTES
                    or node.attr in BANNED_NUMPY_ATTRIBUTES
                    or (is_dunder and node.attr not in ALLOWED_DUNDER_ATTRIBUTES)
                ):
                    raise JaxForgeValidationError(
                        f"Access to attribute '{node.attr}' is not allowed.", code=code
                    )

        defined_functions = {
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        }
        if required_fn not in defined_functions:
            raise JaxForgeValidationError(
                f"Generated code must define a top-level function named '{required_fn}'. "
                f"Found functions: {sorted(defined_functions)}",
                code=code,
            )
