"""Compute do-nothing baselines for a scenario and write them to its
`scoring.baselines` block.

Run after adding a new scenario JSON. Without baselines, the metric
layer falls back to global defaults that won't reflect the scenario's
actual difficulty — so a brilliant controller and a passive one can end
up with the same final score.

Usage:
    python -m watt_the_hack.scripts.freeze_baselines duck_curve datacenter_burst
    python -m watt_the_hack.scripts.freeze_baselines --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from watt_the_hack.controllers.parametric import (
    ParametricControllerParams,
    make_parametric_controller,
)
from watt_the_hack.data_loaders.scenarios import (
    find_scenario_by_id,
    list_scenarios,
    load_scenario,
)
from watt_the_hack.engine.engine import Engine
from watt_the_hack.metrics.metrics import Metrics


# Floors so reliability/stability scores aren't degenerate when the
# do-nothing run happens to be perfect on a given dimension.
MIN_BASELINES = {"cost": 50.0, "stability_abs": 100.0, "unmet": 5.0}


def freeze(scenario_id: str) -> dict:
    path = find_scenario_by_id(scenario_id)
    if path is None:
        raise SystemExit(f"Unknown scenario_id: {scenario_id!r}")

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    spec, state = load_scenario(path)

    engine = Engine()
    engine.add_forecast_to_state(state)

    metrics = Metrics(dt_hours=engine.config.dt_hours)
    controller = make_parametric_controller(ParametricControllerParams())
    steps = len(state["profiles"]["demand"])

    # Sum each cost component across the do-nothing run. Persisted as
    # `baseline_breakdown` in the scenario JSON for any future tooling or
    # post-hoc analysis — the live UI no longer displays it.
    breakdown_totals: dict[str, float] = {}
    for _ in range(steps):
        action = controller(state)
        state, outputs = engine.step(state, action)
        metrics.update(state, outputs)
        for key, value in outputs["cost_breakdown"].items():
            if key == "total":
                continue
            breakdown_totals[key] = breakdown_totals.get(key, 0.0) + value

    s = metrics.summary()
    baselines = {
        "cost": round(max(s["cost"], MIN_BASELINES["cost"]), 2),
        "stability_abs": round(
            max(abs(s["grid_stability"]), MIN_BASELINES["stability_abs"]), 2
        ),
        "unmet": round(max(s["unmet_demand_total"], MIN_BASELINES["unmet"]), 2),
        "renewable": round(s["renewable_ratio"], 4),
    }
    baseline_breakdown = {k: round(v, 2) for k, v in breakdown_totals.items()}

    raw.setdefault("scoring", {})
    raw["scoring"]["baselines"] = baselines
    raw["scoring"]["baseline_breakdown"] = baseline_breakdown

    Path(path).write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return {
        "scenario_id": scenario_id,
        "baselines": baselines,
        "baseline_breakdown": baseline_breakdown,
        "path": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze do-nothing baselines into scenarios."
    )
    parser.add_argument("scenario_ids", nargs="*", help="Scenario ids to freeze.")
    parser.add_argument(
        "--all", action="store_true", help="Freeze every discoverable scenario."
    )
    args = parser.parse_args()

    if args.all:
        ids = [s["id"] for s in list_scenarios()]
    else:
        ids = args.scenario_ids
    if not ids:
        parser.print_help()
        return 1

    for sid in ids:
        result = freeze(sid)
        print(f"  {result['scenario_id']:30s}  {result['baselines']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
