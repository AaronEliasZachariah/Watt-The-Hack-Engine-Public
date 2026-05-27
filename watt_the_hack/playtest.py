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
import html
import json
import sys
import time
import traceback
import webbrowser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from watt_the_hack.data_loaders.scenarios import list_scenarios
from watt_the_hack.engine.engine import Engine
from watt_the_hack.simulation.boot import (
    ScenarioNotFound,
    boot_scenario,
    scenario_steps,
)
from watt_the_hack.simulation.runner import run_strategy
from watt_the_hack.simulation.strategy import (
    ZERO_ACTION,
    resolve_strategy_from_path,
)


def run_playtest(
    controller_path: Path,
    scenario_id: str,
    out_dir: Path | None = None,
    plots: bool = True,
    max_steps: int | None = None,
    verbose: bool = True,
    open_report: bool = False,
) -> dict:
    """Run one scenario end-to-end against a controller file. Returns a
    result dict and writes artifacts to ``out_dir`` (created if missing).

    Set ``plots=False`` to skip PNG generation (no matplotlib needed).
    """
    controller_path = Path(controller_path).resolve()

    try:
        engine, state, spec = boot_scenario(scenario_id)
    except ScenarioNotFound as exc:
        raise SystemExit(
            f"{exc} Run with --list-scenarios to see available options."
        ) from exc

    strategy = resolve_strategy_from_path(controller_path, name=controller_path.stem)

    total_steps = scenario_steps(state)
    if max_steps is not None:
        total_steps = min(total_steps, int(max_steps))

    rows: list[dict[str, Any]] = []

    def _on_step(i, view, action, outputs, state_after):
        rows.append(_per_step_row(i, engine, view, action, outputs, state_after, _breakdown))

    def _on_error(phase, step, exc):
        prefix = f"[{phase}" + (f" @ step {step}]" if step >= 0 else "]")
        print(f"  {prefix} raised: {exc}", file=sys.stderr)
        if phase != "step":
            traceback.print_exc(file=sys.stderr)

    if verbose:
        print(
            f"Scenario : {spec.get('id')} - {spec.get('title', '')}".rstrip()
        )
        print(f"Steps    : {total_steps}  (dt = {engine.config.dt_hours}h)")
        print(f"Controller: {controller_path}  (kind={strategy.kind})")

    # _breakdown is populated incrementally by the engine's per-step cost
    # breakdown so each CSV row carries the running cumulative cost.
    _breakdown: dict[str, float] = {}
    started = time.perf_counter()

    def _accumulate_breakdown(i, view, action, outputs, state_after):
        for k, v in outputs.get("cost_breakdown", {}).items():
            _breakdown[k] = _breakdown.get(k, 0.0) + float(v)
        _on_step(i, view, action, outputs, state_after)

    result = run_strategy(
        engine,
        state,
        strategy,
        total_steps,
        on_step=_accumulate_breakdown,
        on_error=_on_error,
    )
    wall = time.perf_counter() - started

    summary = result["metrics"]
    breakdown = result["cost_breakdown"]

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
        kind=strategy.kind,
        wall_seconds=wall,
        total_steps=total_steps,
    )
    if plots:
        _maybe_write_plots(out_dir, rows, breakdown)
    report_path = _write_report_html(
        out_dir / "report.html",
        summary=summary,
        breakdown=breakdown,
        rows=rows,
        scenario=spec,
        controller_path=controller_path,
    )

    if verbose:
        _print_summary(summary, breakdown, wall, out_dir, report_path)

    if open_report:
        _open_report(report_path)

    return {
        "metrics": summary,
        "breakdown": breakdown,
        "rows": rows,
        "out_dir": str(out_dir),
        "report_path": str(report_path),
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


def _fmt_money(value: float) -> str:
    return f"${value:,.2f}"


def _fmt_num(value: float, digits: int = 2) -> str:
    return f"{value:,.{digits}f}"


def _td(value: Any) -> str:
    return f"<td>{html.escape(str(value))}</td>"


def _diagnostic_hints(breakdown: dict) -> list[str]:
    costs = {k: v for k, v in breakdown.items() if k != "total"}
    top = [k for k, v in sorted(costs.items(), key=lambda kv: -abs(kv[1]))[:4] if abs(v) > 1e-9]
    hints: list[str] = []
    if "overvoltage_penalty" in top:
        hints.append("Overvoltage is expensive here: inspect high-solar periods where net grid power is negative. Better controllers should charge the battery or curtail solar before exporting too much.")
    if "tariff_import" in top or "demand_charge" in top:
        hints.append("Import and demand charges are driving cost: look for evening or morning peaks where the battery could discharge to reduce grid import.")
    if "ramp_charge" in top:
        hints.append("Ramp charges are material: avoid abrupt battery or diesel changes unless they prevent a larger penalty.")
    if "battery_wear" in top:
        hints.append("Battery wear is material: cycling needs to be reserved for high-value periods, not every small price movement.")
    if "blackout_penalty" in top:
        hints.append("Blackout penalties dominate: the controller is failing the physical supply constraint before it is optimising cost.")
    if "fcas_revenue" in top:
        hints.append("FCAS revenue is affecting score: check whether reserve commitments are crowding out battery dispatch when the grid needs energy.")
    return hints or ["No single diagnostic dominates. Compare the worst timesteps and action stats against the scenario intent."]


def _write_report_html(
    path: Path,
    *,
    summary: dict,
    breakdown: dict,
    rows: list[dict],
    scenario: dict,
    controller_path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dt_hours = rows[1]["time_hours"] - rows[0]["time_hours"] if len(rows) > 1 else 0.25
    worst_steps = sorted(rows, key=lambda r: r["step_cost"], reverse=True)[:10]
    overvoltage_steps = [r for r in rows if abs(r["overvoltage_kw"]) > 1e-9]
    unmet_steps = [r for r in rows if abs(r["unmet_demand_kw"]) > 1e-9]
    imports = [max(0.0, r["net_grid_power_kw"]) for r in rows]
    exports = [max(0.0, -r["net_grid_power_kw"]) for r in rows]
    battery = [r["battery_flow_kw"] for r in rows]
    diesel = [r["emergency_generator_kw"] for r in rows]
    curtail = [r["curtail_solar_kw"] for r in rows]

    cost_rows = "\n".join(
        "<tr>"
        + _td(k)
        + _td(_fmt_money(v))
        + _td(f"{(v / summary['final_score'] * 100):.1f}%" if summary["final_score"] else "n/a")
        + "</tr>"
        for k, v in sorted(breakdown.items(), key=lambda kv: -abs(kv[1]))
        if k != "total"
    )
    worst_rows = "\n".join(
        "<tr>"
        + _td(r["step"])
        + _td(_fmt_num(r["time_hours"], 2))
        + _td(_fmt_money(r["step_cost"]))
        + _td(_fmt_money(r["cum_cost"]))
        + _td(_fmt_num(r["demand"], 1))
        + _td(_fmt_num(r["solar"], 1))
        + _td(_fmt_num(r["net_grid_power_kw"], 1))
        + _td(_fmt_num(r["soc"], 2))
        + _td(_fmt_num(r["battery_flow_kw"], 1))
        + _td(_fmt_num(r["overvoltage_kw"], 1))
        + _td(_fmt_num(r["unmet_demand_kw"], 1))
        + "</tr>"
        for r in worst_steps
    )
    hint_items = "\n".join(f"<li>{html.escape(h)}</li>" for h in _diagnostic_hints(breakdown))
    plot_cards = []
    for filename, title in [
        ("action.png", "Dispatch Overview"),
        ("cost.png", "Cumulative Cost"),
        ("soc.png", "Battery SOC"),
    ]:
        if (path.parent / filename).exists():
            plot_cards.append(
                f'<section class="card"><h2>{title}</h2><img src="{filename}" alt="{title}"></section>'
            )
    if not plot_cards:
        plot_cards.append(
            '<section class="card"><h2>Plots</h2><p>No PNG plots were generated. Install matplotlib or run without <code>--no-plots</code>.</p></section>'
        )

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Watt The Hack Playtest Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #172033; background: #f6f7fb; }}
    h1, h2 {{ margin: 0 0 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; }}
    .card {{ background: white; border: 1px solid #dde2ee; border-radius: 10px; padding: 16px; margin: 0 0 16px; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
    .metric {{ font-size: 28px; font-weight: 700; }}
    .label {{ color: #5b6475; font-size: 13px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e6e9f2; padding: 7px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #e6e9f2; border-radius: 8px; background: white; }}
    code {{ background: #eef1f7; padding: 2px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Watt The Hack Playtest Report</h1>
  <p>{html.escape(str(scenario.get("id")))} - {html.escape(str(scenario.get("title", "")))}<br>
  Controller: <code>{html.escape(str(controller_path))}</code></p>

  <section class="grid">
    <div class="card"><div class="label">Final Score</div><div class="metric">{_fmt_money(summary["final_score"])}</div><div class="label">lower wins</div></div>
    <div class="card"><div class="label">Renewable Ratio</div><div class="metric">{summary["renewable_ratio"]:.3f}</div></div>
    <div class="card"><div class="label">Unmet Demand</div><div class="metric">{summary["unmet_demand_total"]:.2f} kWh</div></div>
    <div class="card"><div class="label">Steps</div><div class="metric">{len(rows)}</div><div class="label">{dt_hours:.2f} h timestep</div></div>
  </section>

  <section class="card">
    <h2>Diagnostic Hints</h2>
    <ul>{hint_items}</ul>
  </section>

  {"".join(plot_cards)}

  <section class="grid">
    <div class="card"><h2>Grid Violations</h2>
      <table>
        <tr><th>Signal</th><th>Steps</th><th>Max kW</th></tr>
        <tr>{_td("overvoltage")}{_td(len(overvoltage_steps))}{_td(_fmt_num(max([r["overvoltage_kw"] for r in rows], default=0.0), 1))}</tr>
        <tr>{_td("unmet demand")}{_td(len(unmet_steps))}{_td(_fmt_num(max([r["unmet_demand_kw"] for r in rows], default=0.0), 1))}</tr>
      </table>
    </div>
    <div class="card"><h2>Action Stats</h2>
      <table>
        <tr><th>Signal</th><th>Min</th><th>Max</th><th>Total</th></tr>
        <tr>{_td("battery kW")}{_td(_fmt_num(min(battery, default=0.0), 1))}{_td(_fmt_num(max(battery, default=0.0), 1))}{_td(_fmt_num(sum(abs(v) for v in battery) * dt_hours, 1) + " kWh throughput")}</tr>
        <tr>{_td("diesel kW")}{_td(_fmt_num(min(diesel, default=0.0), 1))}{_td(_fmt_num(max(diesel, default=0.0), 1))}{_td(_fmt_num(sum(diesel) * dt_hours, 1) + " kWh")}</tr>
        <tr>{_td("curtail kW")}{_td(_fmt_num(min(curtail, default=0.0), 1))}{_td(_fmt_num(max(curtail, default=0.0), 1))}{_td(_fmt_num(sum(curtail) * dt_hours, 1) + " kWh")}</tr>
        <tr>{_td("grid import/export")}{_td(_fmt_num(max(exports, default=0.0), 1) + " export")}{_td(_fmt_num(max(imports, default=0.0), 1) + " import")}{_td(_fmt_num(sum(imports) * dt_hours, 1) + " kWh import")}</tr>
      </table>
    </div>
  </section>

  <section class="card">
    <h2>Cost Breakdown</h2>
    <table><tr><th>Component</th><th>Cost</th><th>Share</th></tr>{cost_rows}</table>
  </section>

  <section class="card">
    <h2>Worst Timesteps</h2>
    <table>
      <tr><th>Step</th><th>Hour</th><th>Step Cost</th><th>Cumulative</th><th>Demand</th><th>Solar</th><th>Net Grid</th><th>SOC</th><th>Battery</th><th>Overvoltage</th><th>Unmet</th></tr>
      {worst_rows}
    </table>
  </section>

  <section class="card">
    <h2>Artifacts</h2>
    <p><a href="steps.csv">steps.csv</a> | <a href="metrics.json">metrics.json</a> | <a href="meta.json">meta.json</a></p>
  </section>
</body>
</html>
"""
    path.write_text(body, encoding="utf-8")
    return path


def _open_report(path: Path) -> None:
    try:
        webbrowser.open(path.resolve().as_uri())
        print(f"opened report: {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"  [report] could not open browser: {exc}", file=sys.stderr)


def _print_summary(
    summary: dict, breakdown: dict, wall_seconds: float, out_dir: Path, report_path: Path
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
    print(f"report:    {report_path}")


def _disambiguate_names(paths: list[Path]) -> list[str]:
    """Pick a short unique name for each controller path. Falls back to
    ``<parent>__<stem>`` for stem collisions (e.g. two ``ctrl.py`` files
    in different folders)."""
    stems = [p.stem for p in paths]
    if len(set(stems)) == len(stems):
        return stems
    return [f"{p.parent.name}__{p.stem}" for p in paths]


def run_sweep(
    controller_paths: list[Path],
    scenario_id: str,
    out_dir: Path | None = None,
    plots: bool = True,
    max_steps: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run several controllers against the same scenario and produce a
    side-by-side comparison.

    Writes ``runs/sweep_<scenario>_<ts>/`` containing:
      * one ``<controller_name>/`` per controller (same artifacts
        ``run_playtest`` writes for solo runs)
      * ``comparison.csv`` — long-format rows for pandas pivots
      * ``comparison.png`` — overlay of SOC, cumulative cost, net grid
      * ``per_controller.png`` — small multiples for forensic inspection
      * ``summary.json`` — ranked controller scores + breakdowns
    """
    if not controller_paths:
        raise ValueError("controller_paths must be non-empty")

    controller_paths = [Path(p).resolve() for p in controller_paths]
    names = _disambiguate_names(controller_paths)

    if out_dir is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir = Path("runs") / f"sweep_{scenario_id}_{ts}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sweep: list[dict] = []
    for name, path in zip(names, controller_paths):
        if verbose:
            print(f"[{name}] running... ", end="", flush=True)
        sub_out = out_dir / name
        try:
            res = run_playtest(
                controller_path=path,
                scenario_id=scenario_id,
                out_dir=sub_out,
                plots=plots,
                max_steps=max_steps,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"FAILED: {exc}")
            sweep.append({"name": name, "path": str(path), "error": str(exc)})
            continue
        score = res["metrics"]["final_score"]
        if verbose:
            print(f"score=${score:.2f}")
        sweep.append(
            {
                "name": name,
                "path": str(path),
                "metrics": res["metrics"],
                "breakdown": res["breakdown"],
                "rows": res["rows"],
                "out_dir": res["out_dir"],
            }
        )

    successful = [s for s in sweep if "rows" in s]
    if successful:
        _write_comparison_csv(out_dir / "comparison.csv", successful)
        _write_sweep_summary_json(out_dir / "summary.json", sweep)
        if plots:
            _maybe_write_sweep_plots(out_dir, successful)

    if verbose:
        _print_ranked_table(sweep, scenario_id, out_dir)

    return {"sweep": sweep, "out_dir": str(out_dir)}


def _write_comparison_csv(path: Path, sweep: list[dict]) -> None:
    if not sweep:
        return
    fieldnames = ["controller"] + list(sweep[0]["rows"][0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for entry in sweep:
            for row in entry["rows"]:
                writer.writerow({"controller": entry["name"], **row})


def _write_sweep_summary_json(path: Path, sweep: list[dict]) -> None:
    payload = []
    for s in sweep:
        if "error" in s:
            payload.append({"controller": s["name"], "error": s["error"]})
            continue
        payload.append(
            {
                "controller": s["name"],
                "final_score": s["metrics"]["final_score"],
                "renewable_ratio": s["metrics"]["renewable_ratio"],
                "unmet_demand_total": s["metrics"]["unmet_demand_total"],
                "controller_errors": s["metrics"].get("controller_errors", 0),
                "cost_breakdown": {
                    k: round(v, 4) for k, v in s["breakdown"].items() if k != "total"
                },
            }
        )
    payload.sort(key=lambda d: d.get("final_score", float("inf")))
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _maybe_write_sweep_plots(out_dir: Path, sweep: list[dict]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: E402
    except ImportError:
        print(
            "  [plots] matplotlib not installed; skipping sweep PNGs.",
            file=sys.stderr,
        )
        return
    if not sweep:
        return

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(sweep))]

    # 1. comparison.png — three stacked overlays
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for entry, color in zip(sweep, colors):
        rows = entry["rows"]
        t = [r["time_hours"] for r in rows]
        label = f"{entry['name']}  (${entry['metrics']['final_score']:.0f})"
        axes[0].plot(t, [r["soc"] for r in rows], label=label, color=color)
        axes[1].plot(t, [r["cum_cost"] for r in rows], label=label, color=color)
        axes[2].plot(t, [r["net_grid_power_kw"] for r in rows], label=label, color=color)
    axes[0].set_ylabel("SOC")
    axes[0].set_ylim(0, 1)
    axes[1].set_ylabel("cumulative cost ($)")
    axes[2].set_ylabel("net grid kW (+imp / -exp)")
    axes[2].set_xlabel("hours")
    for ax in axes:
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8, ncol=min(2, len(sweep)))
    fig.suptitle("Sweep comparison")
    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=120)
    plt.close(fig)

    # 2. per_controller.png — small multiples, one row per controller
    n = len(sweep)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.6 * n), sharex=True, squeeze=False)
    for i, (entry, color) in enumerate(zip(sweep, colors)):
        rows = entry["rows"]
        t = [r["time_hours"] for r in rows]
        score = entry["metrics"]["final_score"]

        ax = axes[i, 0]
        ax.plot(t, [r["soc"] for r in rows], color=color)
        ax.set_ylim(0, 1)
        ax.set_ylabel("SOC")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{entry['name']}  -  ${score:.2f}", loc="left", fontsize=10)

        ax = axes[i, 1]
        ax.plot(t, [r["battery_flow_kw"] for r in rows], label="battery", color="tab:blue")
        ax.plot(t, [r["emergency_generator_kw"] for r in rows], label="diesel", color="tab:red")
        ax.plot(t, [r["curtail_solar_kw"] for r in rows], label="curtail", color="tab:green", linestyle="--")
        ax.set_ylabel("kW")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1, 0].set_xlabel("hours")
    axes[-1, 1].set_xlabel("hours")
    fig.suptitle("Per-controller forensics")
    fig.tight_layout()
    fig.savefig(out_dir / "per_controller.png", dpi=120)
    plt.close(fig)


def _print_ranked_table(sweep: list[dict], scenario_id: str, out_dir: Path) -> None:
    print()
    print(f"Sweep results: {scenario_id}")
    print("-" * 78)
    print(f"  {'rank':>4}  {'controller':<28s}  {'final_score':>12s}  top cost components")
    print("-" * 78)

    ok = [s for s in sweep if "metrics" in s]
    bad = [s for s in sweep if "error" in s]
    ok.sort(key=lambda s: s["metrics"]["final_score"])

    for rank, s in enumerate(ok, 1):
        score = s["metrics"]["final_score"]
        bd = {k: v for k, v in s["breakdown"].items() if k != "total"}
        top = sorted(bd.items(), key=lambda kv: -abs(kv[1]))[:3]
        top_str = ", ".join(f"{k}=${v:.0f}" for k, v in top)
        print(f"  {rank:>4}  {s['name']:<28s}  ${score:>10.2f}    {top_str}")
    for s in bad:
        print(f"  {'!':>4}  {s['name']:<28s}  {'ERROR':>12s}    {s['error']}")
    print("-" * 78)
    print(f"artifacts: {out_dir}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m watt_the_hack.playtest",
        description="Run one or more controllers against a Watt The Hack scenario locally.",
    )
    p.add_argument(
        "controller",
        nargs="*",
        type=Path,
        help="Path(s) to .py files containing a `Strategy` class or `controller` function. "
             "Pass multiple paths (or a shell glob like `probes/*.py`) for a side-by-side sweep.",
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
    p.add_argument(
        "--open-report",
        action="store_true",
        help="Open the generated report.html in your default browser after the run.",
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

    if not args.controller or args.scenario is None:
        print(
            "Usage: python -m watt_the_hack.playtest <controller.py> [more.py ...] --scenario <id>\n"
            "       python -m watt_the_hack.playtest --list-scenarios",
            file=sys.stderr,
        )
        return 2

    if len(args.controller) == 1:
        run_playtest(
            controller_path=args.controller[0],
            scenario_id=args.scenario,
            out_dir=args.out,
            plots=not args.no_plots,
            max_steps=args.steps,
            verbose=not args.quiet,
            open_report=args.open_report,
        )
    else:
        if args.open_report:
            print(
                "  [report] --open-report is currently only supported for single-controller runs.",
                file=sys.stderr,
            )
        run_sweep(
            controller_paths=args.controller,
            scenario_id=args.scenario,
            out_dir=args.out,
            plots=not args.no_plots,
            max_steps=args.steps,
            verbose=not args.quiet,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
