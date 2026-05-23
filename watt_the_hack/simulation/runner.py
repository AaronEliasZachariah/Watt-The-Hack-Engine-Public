"""Simulation runner — thin orchestration layer.

Wires together an engine, a controller, and the metrics accumulator to
produce a complete simulation run.  Profile generation and scenario
loading live in ``watt_the_hack.data_loaders``; this module just loops.
"""

from watt_the_hack.constants import DEFAULT_STEPS, DT_HOURS
from watt_the_hack.controllers.rule_based import rule_based_controller

from watt_the_hack.engine.base_engine import SimulationEngine
from watt_the_hack.engine.engine import Engine
from watt_the_hack.metrics.metrics import Metrics


def run_simulation(
    engine: SimulationEngine | None = None,
    controller=rule_based_controller,
    initial_state: dict | None = None,
    steps: int = DEFAULT_STEPS,
) -> dict:
    """Run a headless 24-hour energy simulation."""
    engine = engine or Engine()
    metrics = Metrics(dt_hours=getattr(engine.config, "dt_hours", DT_HOURS))
    if initial_state is None:
        raise ValueError("initial_state must be provided.")
    state = initial_state

    states = []
    outputs_history = []

    for _ in range(steps):
        action = controller(state)
        state, outputs = engine.step(state, action)
        metrics.update(state, outputs)
        states.append(dict(state))
        outputs_history.append(dict(outputs))

    return {
        "final_state": state,
        "states": states,
        "outputs": outputs_history,
        "metrics": metrics.summary(),
    }


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from watt_the_hack.data_loaders.scenarios import (
        find_scenario_by_id,
        list_scenarios,
        load_scenario,
    )

    parser = argparse.ArgumentParser(
        description="Run a headless Watt The Hack simulation."
    )
    parser.add_argument(
        "--scenario",
        type=str,
        default=None,
        help="Scenario id (e.g. 'datacenter_burst') OR path to a scenario JSON file.",
    )
    parser.add_argument(
        "--list-scenarios",
        action="store_true",
        help="List available scenarios and exit.",
    )
    args = parser.parse_args()

    if args.list_scenarios:
        for s in list_scenarios():
            print(f"  {s['id']:30s} {s['pool']:10s} {s['title']}")
        raise SystemExit(0)

    initial_state = None
    scenario_label = "default"
    if args.scenario:
        # Accept either an id ("datacenter_burst") or a path
        if Path(args.scenario).is_file():
            scenario_path = Path(args.scenario)
        else:
            scenario_path = find_scenario_by_id(args.scenario)
            if scenario_path is None:
                raise SystemExit(
                    f"Could not find scenario {args.scenario!r} as either a path or an id. "
                    f"Try --list-scenarios."
                )
        spec, initial_state = load_scenario(scenario_path)
        scenario_label = spec.get("title", spec.get("id", str(scenario_path)))

    if not args.scenario:
        parser.error(
            "--scenario is required. Use --list-scenarios to see available options."
        )

    print(f"Running scenario: {scenario_label}")
    result = run_simulation(initial_state=initial_state)
    print(result["metrics"])
