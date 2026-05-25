"""Local playtest CLI for Watt The Hack controllers.

Run a controller against any scenario without touching the cloud admin
server. Mirrors the cloud evaluator's contract (``Strategy`` class with
``plan``/``replan``/``step``, or a ``controller`` function) so a local
green run translates directly to a cloud submission.

Usage::

    python -m watt_the_hack.playtest path/to/my_controller.py --scenario duck_curve
    python -m watt_the_hack.playtest --list-scenarios

Per-run outputs land in ``runs/<scenario_id>_<UTC timestamp>/``:
    metrics.json   final cost summary + breakdown by component
    steps.csv      one row per simulation step (15 minutes each)
    meta.json      scenario + controller info (for reproducibility)
    soc.png        SOC trajectory (skipped if matplotlib missing)
    cost.png       cumulative cost over time
    action.png     dispatch overlay (battery, diesel, curtailment)

Controllers may live anywhere on disk; the file's parent directory is
added to ``sys.path`` so co-located helpers (``utils.py``, etc.) import
cleanly.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from watt_the_hack.data_loaders.scenarios import (
    config_overrides,
    find_scenario_by_id,
    list_scenarios,
    load_scenario,
)
from watt_the_hack.engine.engine import Engine, SimulationConfig
from watt_the_hack.metrics.metrics import Metrics


ZERO_ACTION: dict[str, float] = {
    "battery_flow_kw": 0.0,
    "emergency_generator": 0.0,
    "curtail_solar": 0.0,
    "fcas_reserve_kw": 0.0,
}


def load_controller_module(path: Path) -> Any:
    """Import a .py file by absolute path and return the module object.

    The file's parent dir is added to ``sys.path`` first so co-located
    helper modules import normally.
    """
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Controller file not found: {path}")
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not build import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_strategy(
    module: Any,
) -> tuple[str, Callable, Callable | None, Callable | None, Any]:
    """Inspect a loaded module and return (kind, step_fn, plan_fn, replan_fn, instance).

    Accepts the same shapes the cloud admin server accepts:
      * class ``Strategy`` with at least a ``step(state)`` method, optionally
        ``plan(state)`` and ``replan(state, alerts)``;
      * top-level callable ``controller(state)``.
    """
    cls = getattr(module, "Strategy", None)
    if isinstance(cls, type):
        instance = cls()
        step = getattr(instance, "step", None)
        if not callable(step):
            raise TypeError(
                "`Strategy` class must define a `step(self, state)` method."
            )
        return (
            "class",
            step,
            getattr(instance, "plan", None) if callable(getattr(instance, "plan", None)) else None,
            getattr(instance, "replan", None) if callable(getattr(instance, "replan", None)) else None,
            instance,
        )
    func = getattr(module, "controller", None)
    if callable(func):
        return ("function", func, None, None, None)
    raise AttributeError(
        f"Controller module {module.__name__!r} must expose either a "
        f"`Strategy` class or a `controller(state)` function."
    )


def run_playtest(
    controller_path: Path,
    scenario_id: str,
    out_dir: Path | None = None,
    plots: bool = True,
    max_steps: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run one scenario end-to-end against a controller file. Returns a
    result dict and writes artifacts to ``out_dir`` (created if missing).

    Set ``plots=False`` to skip PNG generation (no matplotlib needed).
    """
    controller_path = Path(controller_path).resolve()
    spec_path = find_scenario_by_id(scenario_id)
    if spec_path is None:
        raise SystemExit(
            f"Unknown scenario_id: {scenario_id!r}. "
            f"Run with --list-scenarios to see available options."
        )

    spec, state = load_scenario(spec_path)
    overrides = config_overrides(spec)
    engine = (
        Engine(config=SimulationConfig(**overrides)) if overrides else Engine()
    )
    engine.add_forecast_to_state(state)

    module = load_controller_module(controller_path)
    kind, step_fn, plan_fn, replan_fn, _instance = _resolve_strategy(module)

    metrics = Metrics(dt_hours=engine.config.dt_hours)
    breakdown: dict[str, float] = {}

    agent_plan: dict[str, Any] = {}
    if plan_fn is not None:
        try:
            r = plan_fn(Engine.controller_view(state))
            if isinstance(r, dict):
                if "agent_plan" in r and isinstance(r["agent_plan"], dict):
                    agent_plan.update(r["agent_plan"])
                else:
                    agent_plan.update(r)
        except Exception as exc:  # noqa: BLE001
            print(f"  [plan] raised: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
    state["agent_plan"] = agent_plan

    total_steps = len(state["_profiles_full"]["demand"])
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))

    rows: list[dict[str, Any]] = []
    controller_errors = 0
    started = time.perf_counter()

    if verbose:
        print(
            f"Scenario : {spec.get('id')} - {spec.get('title', '')}".rstrip()
        )
        print(f"Steps    : {total_steps}  (dt = {engine.config.dt_hours}h)")
        print(f"Controller: {controller_path}  (kind={kind})")

    for i in range(total_steps):
        view = Engine.controller_view(state)
        alerts = view.get("alerts") or []
        if alerts and replan_fn is not None:
            try:
                u = replan_fn(view, alerts)
                if isinstance(u, dict):
                    if "agent_plan" in u and isinstance(u["agent_plan"], dict):
                        agent_plan = {**agent_plan, **u["agent_plan"]}
                    else:
                        agent_plan = {**agent_plan, **u}
                    state["agent_plan"] = agent_plan
            except Exception as exc:  # noqa: BLE001
                print(f"  [replan @ step {i}] raised: {exc}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

        view = Engine.controller_view(state)
        try:
            action = step_fn(view)
            if not isinstance(action, dict):
                raise TypeError(
                    f"step() must return a dict, got {type(action).__name__}"
                )
        except Exception as exc:  # noqa: BLE001
            controller_errors += 1
            print(f"  [step @ {i}] raised: {exc}", file=sys.stderr)
            action = dict(ZERO_ACTION)

        state, outputs = engine.step(state, action)
        metrics.update(state, outputs)
        for k, v in outputs.get("cost_breakdown", {}).items():
            breakdown[k] = breakdown.get(k, 0.0) + float(v)

        rows.append(_per_step_row(i, engine, view, action, outputs, state, breakdown))

    wall = time.perf_counter() - started
    summary = metrics.summary()
    summary["controller_errors"] = controller_errors

    if out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path("runs") / f"{spec.get('id', 'scenario')}_{ts}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_metrics(out_dir / "metrics.json", summary, breakdown)
    _write_steps_csv(out_dir / "steps.csv", rows)
    _write_meta(
        out_dir / "meta.json",
        scenario=spec,
        controller_path=controller_path,
        kind=kind,
        wall_seconds=wall,
        total_steps=total_steps,
    )
    if plots:
        _maybe_write_plots(out_dir, rows, breakdown)

    if verbose:
        _print_summary(summary, breakdown, wall, out_dir)

    return {
        "metrics": summary,
        "breakdown": breakdown,
        "rows": rows,
        "out_dir": str(out_dir),
        "wall_seconds": wall,
    }


def _per_step_row(
    step: int,
    engine: Engine,
    view: dict,
    action: dict,
    outputs: dict,
    state: dict,
    breakdown_cum: dict,
) -> dict:
    return {
        "step": step,
        "time_hours": round(step * engine.config.dt_hours, 4),
        "demand": float(view.get("demand", 0.0)),
        "solar": float(view.get("solar", 0.0)),
        "price": float(view.get("price", 0.0)),
        "soc": float(state.get("soc", 0.0)),
        "battery_flow_kw": float(action.get("battery_flow_kw", 0.0)),
        "emergency_generator_kw": float(action.get("emergency_generator", 0.0)),
        "curtail_solar_kw": float(action.get("curtail_solar", 0.0)),
        "fcas_reserve_kw": float(action.get("fcas_reserve_kw", 0.0)),
        "net_grid_power_kw": float(outputs.get("net_grid_power", 0.0)),
        "unmet_demand_kw": float(outputs.get("unmet_demand", 0.0)),
        "overvoltage_kw": float(outputs.get("overvoltage_kw", 0.0)),
        "step_cost": float(outputs.get("cost", 0.0)),
        "cum_cost": float(sum(breakdown_cum.values())),
    }


def _write_metrics(path: Path, summary: dict, breakdown: dict) -> None:
    payload = {
        "summary": summary,
        "cost_breakdown": {k: round(v, 4) for k, v in breakdown.items()},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_steps_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_meta(
    path: Path,
    *,
    scenario: dict,
    controller_path: Path,
    kind: str,
    wall_seconds: float,
    total_steps: int,
) -> None:
    meta = {
        "scenario_id": scenario.get("id"),
        "scenario_title": scenario.get("title"),
        "scenario_mechanics": [m for m in scenario.get("features", {}).items()] if scenario.get("features") else "all enabled",
        "controller_path": str(controller_path),
        "controller_kind": kind,
        "total_steps": total_steps,
        "wall_seconds": round(wall_seconds, 3),
        "engine_version": "0.1.0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def _maybe_write_plots(out_dir: Path, rows: list[dict], breakdown: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
    except ImportError:
        print(
            "  [plots] matplotlib not installed; skipping PNGs. "
            "`pip install matplotlib` to enable.",
            file=sys.stderr,
        )
        return
    if not rows:
        return

    t = [r["time_hours"] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, [r["soc"] for r in rows], color="tab:blue")
    ax.set_xlabel("hours")
    ax.set_ylabel("SOC (0–1)")
    ax.set_title("Battery state of charge")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "soc.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t, [r["cum_cost"] for r in rows], color="tab:red")
    ax.set_xlabel("hours")
    ax.set_ylabel("cumulative cost ($)")
    ax.set_title(f"Total cost = ${rows[-1]['cum_cost']:.2f}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "cost.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax = axes[0]
    ax.plot(t, [r["demand"] for r in rows], label="demand", color="tab:orange")
    ax.plot(t, [r["solar"] for r in rows], label="solar", color="tab:green")
    ax.plot(t, [r["net_grid_power_kw"] for r in rows], label="net grid (+import / -export)", color="tab:gray")
    ax.set_ylabel("kW")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax = axes[1]
    ax.plot(t, [r["battery_flow_kw"] for r in rows], label="battery flow (+disch/-ch)", color="tab:blue")
    ax.plot(t, [r["emergency_generator_kw"] for r in rows], label="diesel", color="tab:red")
    ax.plot(t, [r["curtail_solar_kw"] for r in rows], label="curtail", color="tab:green", linestyle="--")
    ax.set_xlabel("hours")
    ax.set_ylabel("kW")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.suptitle("Dispatch overview")
    fig.tight_layout()
    fig.savefig(out_dir / "action.png", dpi=120)
    plt.close(fig)


def _print_summary(
    summary: dict, breakdown: dict, wall_seconds: float, out_dir: Path
) -> None:
    print()
    print(f"final_score          ${summary['final_score']:>12.2f}   (lower wins)")
    print(f"renewable_ratio       {summary['renewable_ratio']:>12.3f}")
    print(f"unmet_demand_total    {summary['unmet_demand_total']:>12.3f} kWh")
    if summary.get("controller_errors"):
        print(f"controller_errors     {summary['controller_errors']:>12d}")
    print(f"wall_clock            {wall_seconds:>12.2f} s")
    if breakdown:
        print()
        print("cost breakdown:")
        for k, v in sorted(breakdown.items(), key=lambda kv: -abs(kv[1])):
            if k == "total":
                continue  # duplicates final_score
            print(f"  {k:<28s} ${v:>10.2f}")
    print()
    print(f"artifacts: {out_dir}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m watt_the_hack.playtest",
        description="Run a controller against a Watt The Hack scenario locally.",
    )
    p.add_argument(
        "controller",
        nargs="?",
        type=Path,
        help="Path to a .py file containing a `Strategy` class or `controller` function.",
    )
    p.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario id (e.g. 'duck_curve'). See --list-scenarios.",
    )
    p.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: runs/<scenario>_<timestamp>/",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG generation (matplotlib not required).",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Truncate the run after N steps (for fast iteration).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-run console summary; just write artifacts.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list_scenarios:
        rows = list_scenarios(include_judging=False)
        if not rows:
            print("No scenarios found.")
            return 0
        width = max(len(s["id"]) for s in rows)
        for s in rows:
            print(f"  {s['id']:<{width}s}  {s['pool']:<10s} {s['title']}")
        return 0

    if args.controller is None or args.scenario is None:
        print(
            "Usage: python -m watt_the_hack.playtest <controller.py> --scenario <id>\n"
            "       python -m watt_the_hack.playtest --list-scenarios",
            file=sys.stderr,
        )
        return 2

    run_playtest(
        controller_path=args.controller,
        scenario_id=args.scenario,
        out_dir=args.out,
        plots=not args.no_plots,
        max_steps=args.steps,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
