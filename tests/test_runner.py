"""Tests for the simulation runner."""

import pytest

from watt_the_hack.simulation.runner import run_simulation
from watt_the_hack.engine.engine import Engine


def _dummy_state(steps: int = 10) -> dict:
    return {
        "time": 0,
        "demand": 10.0,
        "solar": 5.0,
        "soc": 0.5,
        "price": 0.1,
        "profiles": {
            "demand": [10.0] * steps,
            "solar": [5.0] * steps,
        },
        "price_profile": [0.1] * steps,
        "events": [],
        "forecast_config": {},
    }


class TestRunSimulation:
    def test_returns_expected_shape(self):
        result = run_simulation(initial_state=_dummy_state(4), steps=4)
        assert "final_state" in result
        assert "states" in result
        assert "outputs" in result
        assert "metrics" in result

    def test_correct_number_of_steps(self):
        result = run_simulation(initial_state=_dummy_state(10), steps=10)
        assert len(result["states"]) == 10
        assert len(result["outputs"]) == 10

    def test_metrics_populated(self):
        result = run_simulation(initial_state=_dummy_state(10), steps=10)
        m = result["metrics"]
        assert "final_score" in m
        assert "renewable_ratio" in m
        assert "cost" in m

    def test_final_state_time_advanced(self):
        result = run_simulation(initial_state=_dummy_state(10), steps=10)
        assert result["final_state"]["time"] == 10

    def test_custom_initial_state(self):
        state = _dummy_state(steps=8)
        result = run_simulation(initial_state=state, steps=8)
        assert len(result["states"]) == 8

    def test_custom_engine(self):
        engine = Engine()
        result = run_simulation(engine=engine, initial_state=_dummy_state(4), steps=4)
        assert len(result["states"]) == 4

    def test_do_nothing_controller(self):
        """A controller that returns empty dicts should still complete."""
        result = run_simulation(
            controller=lambda _: {},
            initial_state=_dummy_state(4),
            steps=4,
        )
        assert len(result["states"]) == 4
