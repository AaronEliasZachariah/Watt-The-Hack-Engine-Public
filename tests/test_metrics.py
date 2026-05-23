"""Tests for the scoring / metrics layer.

Exercises component_scores and the Metrics accumulator to ensure
formulas match expectations and edge cases (zero demand, NaN inputs,
etc.) are handled. The final score is the raw cost sum — there is no
weighted-component aggregator anymore.
"""

import math
import pytest

from watt_the_hack.metrics.metrics import (
    DEFAULT_BASELINES,
    Metrics,
    _clamp01,
    component_scores,
)


# ---------------------------------------------------------------------------
# clamp01
# ---------------------------------------------------------------------------


class TestClamp01:
    def test_within_range(self):
        assert _clamp01(0.5) == 0.5

    def test_below_zero(self):
        assert _clamp01(-0.1) == 0.0

    def test_above_one(self):
        assert _clamp01(1.5) == 1.0

    def test_nan_returns_zero(self):
        assert _clamp01(float("nan")) == 0.0

    def test_exact_boundaries(self):
        assert _clamp01(0.0) == 0.0
        assert _clamp01(1.0) == 1.0


# ---------------------------------------------------------------------------
# component_scores
# ---------------------------------------------------------------------------


class TestComponentScores:
    def test_perfect_scores(self):
        """Zero cost, full renewable, no instability, no unmet → all 1.0."""
        scores = component_scores(
            renewable_ratio=1.0,
            grid_stability=0.0,
            cost=0.0,
            unmet_demand_total=0.0,
            baselines=DEFAULT_BASELINES,
        )
        assert scores["cost"] == pytest.approx(1.0)
        assert scores["stability"] == pytest.approx(1.0)
        assert scores["reliability"] == pytest.approx(1.0)
        # Renewable is relative to baseline
        assert scores["renewable"] == pytest.approx(1.0)

    def test_baseline_scores(self):
        """Matching the do-nothing baseline → all 0.0."""
        scores = component_scores(
            renewable_ratio=DEFAULT_BASELINES["renewable"],
            grid_stability=-DEFAULT_BASELINES["stability_abs"],
            cost=DEFAULT_BASELINES["cost"],
            unmet_demand_total=DEFAULT_BASELINES["unmet"],
            baselines=DEFAULT_BASELINES,
        )
        assert scores["cost"] == pytest.approx(0.0)
        assert scores["stability"] == pytest.approx(0.0)
        assert scores["reliability"] == pytest.approx(0.0)
        assert scores["renewable"] == pytest.approx(0.0)

    def test_custom_baselines(self):
        custom = {
            "cost": 100.0,
            "stability_abs": 500.0,
            "unmet": 10.0,
            "renewable": 0.0,
        }
        scores = component_scores(
            renewable_ratio=0.5,
            grid_stability=-250.0,
            cost=50.0,
            unmet_demand_total=5.0,
            baselines=custom,
        )
        assert scores["cost"] == pytest.approx(0.5)
        assert scores["stability"] == pytest.approx(0.5)
        assert scores["reliability"] == pytest.approx(0.5)
        assert scores["renewable"] == pytest.approx(0.5)

    def test_scores_clamped_to_01(self):
        """Costs exceeding baselines should clamp to 0, not go negative."""
        scores = component_scores(
            renewable_ratio=0.0,
            grid_stability=-99999.0,
            cost=99999.0,
            unmet_demand_total=99999.0,
            baselines=DEFAULT_BASELINES,
        )
        for v in scores.values():
            assert 0.0 <= v <= 1.0


# ---------------------------------------------------------------------------
# Metrics accumulator
# ---------------------------------------------------------------------------


class TestMetricsAccumulator:
    def _make_state(self, demand: float = 100.0, **kwargs) -> dict:
        return {"demand": demand, **kwargs}

    def _make_outputs(
        self,
        net_grid_power: float = 0.0,
        emergency_generator: float = 0.0,
        unmet_demand: float = 0.0,
        step_cost: float = 0.0,
    ) -> dict:
        return {
            "net_grid_power": net_grid_power,
            "emergency_generator": emergency_generator,
            "unmet_demand": unmet_demand,
            "step_cost": step_cost,
        }

    def test_single_step_cost(self):
        m = Metrics(dt_hours=0.25)
        m.update(self._make_state(), self._make_outputs(step_cost=10.0))
        assert m.cost_sum == 10.0

    def test_unmet_accumulation(self):
        m = Metrics(dt_hours=0.25)
        m.update(self._make_state(), self._make_outputs(unmet_demand=40.0))
        assert m.unmet_demand_sum == pytest.approx(40.0 * 0.25)

    def test_renewable_ratio(self):
        """100kW demand, 0kW grid import, 0kW diesel → all local = 100% renewable."""
        m = Metrics(dt_hours=0.25)
        m.update(
            self._make_state(demand=100.0),
            self._make_outputs(net_grid_power=0.0, emergency_generator=0.0),
        )
        s = m.summary()
        assert s["renewable_ratio"] == pytest.approx(1.0)

    def test_zero_demand(self):
        """Zero demand should produce renewable_ratio = 0, not a division error."""
        m = Metrics(dt_hours=0.25)
        m.update(self._make_state(demand=0.0), self._make_outputs())
        s = m.summary()
        assert s["renewable_ratio"] == 0.0

    def test_grid_stability_ramp(self):
        """Two steps with differing grid power → stability penalised."""
        m = Metrics(dt_hours=0.25)
        m.update(self._make_state(), self._make_outputs(net_grid_power=10.0))
        m.update(self._make_state(), self._make_outputs(net_grid_power=50.0))
        # ramp = 40, penalty = -40^2 = -1600
        assert m.grid_stability_sum == pytest.approx(-1600.0)

    def test_summary_shape(self):
        """summary() returns all expected keys."""
        m = Metrics()
        m.update(self._make_state(), self._make_outputs(step_cost=1.0))
        s = m.summary()
        expected_keys = {
            "renewable_ratio",
            "grid_stability",
            "cost",
            "unmet_demand_total",
            "final_score",
        }
        assert expected_keys == set(s.keys())

    def test_final_score_is_cost_sum(self):
        """final_score is the accumulated step cost — nothing else."""
        m = Metrics()
        m.update(
            self._make_state(), self._make_outputs(step_cost=500.0, unmet_demand=20.0)
        )
        s = m.summary()
        assert s["final_score"] == pytest.approx(500.0)
        assert s["cost"] == pytest.approx(500.0)
