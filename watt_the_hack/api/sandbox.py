"""Restricted execution of user-provided controller source code."""

from __future__ import annotations

import builtins
import math
import threading
from typing import Any, Callable


_SAFE_BUILTIN_NAMES = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "divmod",
    "enumerate",
    "filter",
    "float",
    "int",
    "len",
    "list",
    "map",
    "max",
    "min",
    "pow",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
)


SAFE_BUILTINS: dict[str, Any] = {
    name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES
}


DISALLOWED_TOKENS = (
    "__import__",
    "__builtins__",
    "__class__",
    "__bases__",
    "__subclasses__",
    "__globals__",
    "open(",
    "exec(",
    "eval(",
    "compile(",
    "globals(",
    "locals(",
    "getattr(",
    "setattr(",
    "delattr(",
)


class ControllerCompileError(ValueError):
    """Raised when user-provided controller source fails to compile or execute."""


def compile_controller_source(source: str) -> Callable[[dict], dict]:
    """Compile a user-provided controller source string into a callable.

    The returned callable expects `state` and returns an `action` dict. The
    caller is responsible for catching runtime errors that occur inside the
    compiled controller.
    """

    if not isinstance(source, str) or not source.strip():
        raise ControllerCompileError("Controller source must be a non-empty string.")

    if len(source.encode("utf-8")) > 50 * 1024:
        raise ControllerCompileError("Controller source exceeds 50 KB size limit.")

    for token in DISALLOWED_TOKENS:
        if token in source:
            raise ControllerCompileError(
                f"Disallowed token in controller source: '{token}'."
            )

    if _contains_import_statement(source):
        raise ControllerCompileError("Imports are not allowed in controller source.")

    namespace: dict[str, Any] = {"__builtins__": SAFE_BUILTINS}
    try:
        compiled = compile(source, "<user-controller>", "exec")
        exec(compiled, namespace)
    except SyntaxError as exc:
        raise ControllerCompileError(
            f"Syntax error: {exc.msg} at line {exc.lineno}"
        ) from exc
    except Exception as exc:
        raise ControllerCompileError(f"Compilation failed: {exc}") from exc

    controller = namespace.get("controller")
    if not callable(controller):
        raise ControllerCompileError(
            "Controller source must define a `controller(state)` function."
        )

    def safe_controller(state: dict) -> dict:
        result = {}
        exc_info = []

        def worker():
            try:
                action = controller(state)
                if not isinstance(action, dict):
                    exc_info.append(ValueError("Controller must return a dict."))
                else:
                    result.update(action)
            except Exception as e:
                exc_info.append(e)

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout=0.1)  # 100ms time limit

        if t.is_alive():
            raise TimeoutError("Controller execution exceeded 100ms time limit.")

        if exc_info:
            raise exc_info[0]

        for k, v in result.items():
            if isinstance(v, (int, float)):
                if math.isnan(v) or math.isinf(v):
                    raise ValueError(f"Action value for '{k}' is NaN or infinity.")

        return result

    return safe_controller


def _contains_import_statement(source: str) -> bool:
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if line.startswith("import ") or line.startswith("from "):
            return True
    return False
