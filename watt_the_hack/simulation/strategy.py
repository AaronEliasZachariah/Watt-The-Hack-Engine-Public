"""Strategy lifecycle adapter.

Bridges the *several* controller shapes the platform accepts (class with
``plan``/``replan``/``step``, bare callable, instance, or module file) to
the single lifecycle the engine expects:

    plan(state) called once before step 0,
    replan(state, alerts) called when state["alerts"] is non-empty,
    step(state) called every timestep.

All four harnesses (local CLI, reference-tier scripts, FastAPI playground,
admin eval container) used to reimplement this resolution. They are now
expected to call :func:`resolve_strategy` once and drive the resulting
:class:`ResolvedStrategy` through :func:`watt_the_hack.simulation.runner.run_strategy`.

Adding a new controller shape (e.g. an async LLM strategy) only requires
extending this module — every consumer picks up the new shape for free.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


# Controllers that raise are treated as if they returned this action.
# Identical to the fallback the admin eval container uses on per-step error.
ZERO_ACTION: dict[str, float] = {
    "battery_flow_mw": 0.0,
    "emergency_generator": 0.0,
    "curtail_solar": 0.0,
    "fcas_reserve_mw": 0.0,
}


@dataclass
class ResolvedStrategy:
    """Resolved controller handles. The runner calls ``step`` every tick,
    ``plan`` once at boot (if not None), ``replan`` when alerts fire (if
    not None).

    ``kind`` is for logging only ("class", "callable", "instance"). The
    runner doesn't branch on it.
    """

    step: Callable[[dict], dict]
    plan: Callable[[dict], dict] | None = None
    replan: Callable[[dict, list], dict] | None = None
    kind: str = "callable"
    instance: Any = None
    name: str = ""


def resolve_strategy(target: Any, *, name: str = "") -> ResolvedStrategy:
    """Accept any of the supported controller shapes and return a
    :class:`ResolvedStrategy`.

    Shapes (checked in this order):

    1. An existing :class:`ResolvedStrategy` — returned unchanged.
    2. A module object — looked up for ``Strategy`` class first, then
       a top-level ``controller`` callable.
    3. A class — instantiated, then treated as case (4).
    4. An instance with a ``step`` method — bound methods are extracted.
    5. A bare callable — used as ``step``, no plan/replan.

    Raises ``TypeError`` if none of the above match.
    """
    if isinstance(target, ResolvedStrategy):
        return target

    # Module
    if hasattr(target, "__name__") and hasattr(target, "__file__"):
        cls = getattr(target, "Strategy", None)
        if isinstance(cls, type):
            return _from_class(cls, name=name or target.__name__)
        func = getattr(target, "controller", None)
        if callable(func):
            return ResolvedStrategy(
                step=func, kind="callable", name=name or target.__name__
            )
        raise AttributeError(
            f"Module {target.__name__!r} exposes neither `Strategy` class "
            f"nor `controller` function."
        )

    # Class
    if isinstance(target, type):
        return _from_class(target, name=name or target.__name__)

    # Instance with a step method
    step = getattr(target, "step", None)
    if callable(step):
        return _from_instance(target, name=name)

    # Bare callable
    if callable(target):
        return ResolvedStrategy(
            step=target, kind="callable", name=name or getattr(target, "__name__", "")
        )

    raise TypeError(
        f"Cannot resolve {target!r} into a strategy. "
        f"Expected a Strategy class, an instance with .step, a controller(state) callable, "
        f"or a module exposing one of the above."
    )


def resolve_strategy_from_path(
    path: str | Path,
    *,
    class_name: str | None = None,
    function_name: str | None = None,
    name: str | None = None,
) -> ResolvedStrategy:
    """Load a .py file by absolute path, then resolve the strategy.

    If ``class_name`` or ``function_name`` is given, the lookup is
    explicit (cloud submission contract: ``metadata.json`` specifies
    one of these). Otherwise falls back to the auto-detection in
    :func:`resolve_strategy`.

    The file's parent directory is added to ``sys.path`` so co-located
    helper modules import cleanly.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)

    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    label = name or path.stem

    if class_name:
        cls = getattr(module, class_name, None)
        if not isinstance(cls, type):
            raise ImportError(
                f"Class {class_name!r} not found (or not a class) in {path}."
            )
        return _from_class(cls, name=label)

    if function_name:
        func = getattr(module, function_name, None)
        if not callable(func):
            raise ImportError(
                f"Function {function_name!r} not found (or not callable) in {path}."
            )
        return ResolvedStrategy(step=func, kind="callable", name=label)

    return resolve_strategy(module, name=label)


def _from_class(cls: type, *, name: str) -> ResolvedStrategy:
    try:
        instance = cls()
    except Exception as exc:  # noqa: BLE001
        raise TypeError(
            f"Strategy class {cls.__name__} must be instantiable with no args."
        ) from exc
    return _from_instance(instance, name=name, kind="class")


def _from_instance(
    instance: Any, *, name: str, kind: str = "instance"
) -> ResolvedStrategy:
    step = getattr(instance, "step", None)
    if not callable(step):
        raise TypeError(
            f"Strategy instance {instance!r} must expose a `step(state)` method."
        )
    plan = getattr(instance, "plan", None)
    if not callable(plan):
        plan = None
    replan = getattr(instance, "replan", None)
    if not callable(replan):
        replan = None
    return ResolvedStrategy(
        step=step, plan=plan, replan=replan, kind=kind, instance=instance, name=name
    )
