"""Local agent evaluation example.

Thin programmatic example that uses the local playtest harness. For
interactive use prefer the CLI directly::

    python -m watt_the_hack.playtest playtester/my_controller.py --scenario duck_curve

The CLI is also driveable from Python, which is what this script does
when you need to script a sweep or wire it into a larger experiment.
"""

from __future__ import annotations

from pathlib import Path

from watt_the_hack.playtest import run_playtest


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    controller = repo_root / "playtester" / "my_controller.py"

    result = run_playtest(
        controller_path=controller,
        scenario_id="duck_curve",
        plots=False,
    )

    print(f"\nFinal score: ${result['metrics']['final_score']:.2f}")
    print(f"Artifacts:   {result['out_dir']}")


if __name__ == "__main__":
    main()
