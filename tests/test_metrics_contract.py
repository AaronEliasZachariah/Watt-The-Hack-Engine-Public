"""Contract test pinning the exact scoring-formula behaviour.

This file freezes ``component_scores`` (used by ad-hoc tooling) and the
fact that ``Metrics.summary()["final_score"]`` is the accumulated cost.
The frontend has no equivalent of ``component_scores`` after the
scoring simplification, so this test is the sole source of truth.

If you intentionally change the formula:
    1. Update component_scores / Metrics.summary in watt_the_hack/metrics/metrics.py.
    2. Re-derive the expected outputs in this file and update the assertions.
"""

import math

import pytest

from watt_the_hack.metrics.metrics import Metrics, component_scores


# ---------------------------------------------------------------------------
# Canonical test vectors: source of truth for the scoring contract.
# Both Python and TypeScript implementations must produce these outputs.
# ---------------------------------------------------------------------------

DEFAULT_BASELINES_FOR_TESTS = {
    "cost": 1000.0,
    "stability_abs": 5000.0,
    "unmet": 50.0,
    "renewable": 0.30,
}


COMPONENT_VECTORS = [
    {
        "name": "perfect",
        "inputs": {
            "renewable_ratio": 1.0,
            "grid_stability": 0.0,
            "cost": 0.0,
            "unmet_demand_total": 0.0,
        },
        "expected": {
            "cost": 1.0,
            "renewable": 1.0,
            "stability": 1.0,
            "reliability": 1.0,
        },
    },
    {
        "name": "matches_baseline",
        "inputs": {
            "renewable_ratio": 0.30,
            "grid_stability": -5000.0,
            "cost": 1000.0,
            "unmet_demand_total": 50.0,
        },
        "expected": {
            "cost": 0.0,
            "renewable": 0.0,
            "stability": 0.0,
            "reliability": 0.0,
        },
    },
    {
        "name": "half_baseline",
        "inputs": {
            "renewable_ratio": 0.65,
            "grid_stability": -2500.0,
            "cost": 500.0,
            "unmet_demand_total": 25.0,
        },
        "expected": {
            "cost": 0.5,
            "renewable": 0.5,
            "stability": 0.5,
            "reliability": 0.5,
        },
    },
    {
        "name": "worse_than_baseline_clamps_to_zero",
        "inputs": {
            "renewable_ratio": 0.10,
            "grid_stability": -10000.0,
            "cost": 2000.0,
            "unmet_demand_total": 100.0,
        },
        "expected": {
            "cost": 0.0,
            "renewable": 0.0,
            "stability": 0.0,
            "reliability": 0.0,
        },
    },
    {
        "name": "profit_caps_at_one",
        "inputs": {
            "renewable_ratio": 0.30,
            "grid_stability": -5000.0,
            "cost": -200.0,
            "unmet_demand_total": 50.0,
        },
        "expected": {
            "cost": 1.0,
            "renewable": 0.0,
            "stability": 0.0,
            "reliability": 0.0,
        },
    },
    {
        "name": "abs_stability",
        "inputs": {
            "renewable_ratio": 0.30,
            "grid_stability": 2500.0,
            "cost": 1000.0,
            "unmet_demand_total": 50.0,
        },
        "expected": {
            "cost": 0.0,
            "renewable": 0.0,
            "stability": 0.5,
            "reliability": 0.0,
        },
    },
]


FINAL_SCORE_VECTORS = [
    {
        "name": "zero_cost",
        "step_costs": [0.0],
        "expected_final": 0.0,
    },
    {
        "name": "positive_cost",
        "step_costs": [12.5, 7.25],
        "expected_final": 19.75,
    },
    {
        "name": "revenue",
        "step_costs": [-4.0, 1.5],
        "expected_final": -2.5,
    },
]


@pytest.mark.parametrize("vec", COMPONENT_VECTORS, ids=lambda v: v["name"])
def test_component_scores_pin(vec):
    actual = component_scores(
        baselines=DEFAULT_BASELINES_FOR_TESTS,
        **vec["inputs"],
    )
    for key, expected in vec["expected"].items():
        assert math.isclose(
            actual[key], expected, abs_tol=1e-6
        ), f"vector {vec['name']!r}: {key} expected {expected}, got {actual[key]}"


@pytest.mark.parametrize("vec", FINAL_SCORE_VECTORS, ids=lambda v: v["name"])
def test_summary_final_score_is_cost_sum(vec):
    metrics = Metrics(baselines=DEFAULT_BASELINES_FOR_TESTS)
    for step_cost in vec["step_costs"]:
        metrics.update(
            {"demand": 0.0},
            {
                "net_grid_power": 0.0,
                "emergency_generator": 0.0,
                "unmet_demand": 0.0,
                "step_cost": step_cost,
            },
        )

    actual = metrics.summary()["final_score"]
    assert math.isclose(
        actual, vec["expected_final"], abs_tol=1e-6
    ), f"vector {vec['name']!r}: expected {vec['expected_final']}, got {actual}"
