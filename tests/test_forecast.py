"""Tests for forecast injection."""

import pytest

from watt_the_hack.engine.engine import Engine, SimulationConfig


def _dummy_state(steps: int = 96) -> dict:
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


class TestAddForecastToState:
    def test_adds_forecast_key(self):
        engine = Engine()
        state = _dummy_state(steps=96)
        engine.add_forecast_to_state(state)
        assert "forecast" in state

    def test_forecast_has_expected_keys(self):
        engine = Engine()
        state = _dummy_state(steps=96)
        engine.add_forecast_to_state(state)
        fc = state["forecast"]
        assert "demand" in fc
        assert "solar" in fc
        assert "price" in fc

    def test_forecast_length_matches_horizon(self):
        engine = Engine()
        state = _dummy_state(steps=96)
        engine.add_forecast_to_state(state)
        fc = state["forecast"]
        horizon = engine.config.forecast_horizon
        assert len(fc["demand"]) == horizon
        assert len(fc["solar"]) == horizon
        assert len(fc["price"]) == horizon

    def test_aligns_price_field(self):
        engine = Engine()
        state = _dummy_state(steps=96)
        original_price_profile = state["price_profile"]
        engine.add_forecast_to_state(state)
        assert state["price"] == pytest.approx(original_price_profile[0])

    def test_deterministic_with_seed(self):
        """Same seed → same forecast."""
        config = SimulationConfig(forecast_seed=42)
        engine = Engine(config=config)

        state1 = _dummy_state(steps=96)
        engine.add_forecast_to_state(state1)

        state2 = _dummy_state(steps=96)
        engine.add_forecast_to_state(state2)

        assert state1["forecast"]["demand"] == state2["forecast"]["demand"]
        assert state1["forecast"]["solar"] == state2["forecast"]["solar"]

    def test_returns_state(self):
        engine = Engine()
        state = _dummy_state(steps=96)
        result = engine.add_forecast_to_state(state)
        assert result is state  # mutates in place and returns


class TestForecastBias:
    """`forecast_bias` events deterministically shift the forecast view of
    a single channel during a window. Distinct from `weather_anomaly`
    (random sigma multiplier) and `attack_windows` (cybersecurity).
    """

    def _state(self, events: list[dict]) -> dict:
        steps = 96
        return {
            "time": 0,
            "demand": 50.0,
            "solar": 30.0,
            "soc": 0.5,
            "price": 0.20,
            "profiles": {
                "demand": [50.0] * steps,
                "solar": [30.0] * steps,
            },
            "price_profile": [0.20] * steps,
            "events": events,
            "forecast_config": {"horizon_steps": 16},
        }

    def test_solar_bias_is_multiplicative(self):
        """`bias = 1.0` on solar should roughly double the forecast inside
        the window, while values outside the window remain near the
        unbiased baseline.
        """
        # Zero-noise config so the bias signal is exactly observable.
        config = SimulationConfig(
            forecast_seed=0,
            forecast_sigma_demand=0.0,
            forecast_sigma_price=0.0,
            forecast_solar_noise_pct=0.0,
        )
        engine = Engine(config=config)

        baseline = engine._build_forecast(self._state([]), time=0)
        biased = engine._build_forecast(
            self._state(
                [
                    {
                        "type": "forecast_bias",
                        "channel": "solar",
                        "bias": 1.0,
                        "at_step": 4,
                        "end_step": 10,
                    }
                ]
            ),
            time=0,
        )

        # Outside window — values match baseline.
        assert biased["solar"][0] == pytest.approx(baseline["solar"][0])
        # Inside window — forecast roughly 2× the baseline (1 + 1.0).
        for h in range(4, 11):
            assert biased["solar"][h] == pytest.approx(2.0 * baseline["solar"][h])

    def test_demand_bias_is_additive(self):
        config = SimulationConfig(
            forecast_seed=0,
            forecast_sigma_demand=0.0,
            forecast_sigma_price=0.0,
            forecast_solar_noise_pct=0.0,
        )
        engine = Engine(config=config)

        baseline = engine._build_forecast(self._state([]), time=0)
        biased = engine._build_forecast(
            self._state(
                [
                    {
                        "type": "forecast_bias",
                        "channel": "demand",
                        "bias": -15.0,
                        "at_step": 2,
                        "end_step": 5,
                    }
                ]
            ),
            time=0,
        )
        assert biased["demand"][0] == pytest.approx(baseline["demand"][0])
        for h in range(2, 6):
            assert biased["demand"][h] == pytest.approx(baseline["demand"][h] - 15.0)
        # Engine clamps demand to >= 0 — verify bias-driven negatives don't leak.
        very_negative = engine._build_forecast(
            self._state(
                [
                    {
                        "type": "forecast_bias",
                        "channel": "demand",
                        "bias": -1000.0,
                        "at_step": 0,
                        "end_step": 15,
                    }
                ]
            ),
            time=0,
        )
        assert all(v >= 0.0 for v in very_negative["demand"])

    def test_price_bias_is_additive(self):
        config = SimulationConfig(
            forecast_seed=0,
            forecast_sigma_demand=0.0,
            forecast_sigma_price=0.0,
            forecast_solar_noise_pct=0.0,
        )
        engine = Engine(config=config)
        biased = engine._build_forecast(
            self._state(
                [
                    {
                        "type": "forecast_bias",
                        "channel": "price",
                        "bias": 0.10,
                        "at_step": 0,
                        "end_step": 15,
                    }
                ]
            ),
            time=0,
        )
        # Baseline price was 0.20; all 16 horizon points should be 0.30.
        for v in biased["price"]:
            assert v == pytest.approx(0.30)

    def test_normaliser_preserves_extras(self):
        """Regression: `_normalise_events` previously dropped any field
        outside the core set, silently nuking `channel` / `bias`.
        """
        from watt_the_hack.data_loaders.scenarios import _normalise_events

        normalised = _normalise_events(
            [
                {
                    "id": "bias_window",
                    "type": "forecast_bias",
                    "severity": "high",
                    "channel": "solar",
                    "bias": 0.5,
                    "at_step": 10,
                    "end_step": 20,
                    "title": "",
                    "description": "",
                }
            ]
        )
        assert normalised[0]["channel"] == "solar"
        assert normalised[0]["bias"] == 0.5

    def test_scenario_mechanics_surfaces_forecast_bias(self):
        """When a scenario declares a `forecast_bias` event, the mechanics
        list should include `forecast_bias` so the frontend briefing card
        can render the chip + the solar divergence chart.
        """
        from watt_the_hack.data_loaders.scenarios import scenario_mechanics

        spec = {
            "events": [
                {
                    "type": "forecast_bias",
                    "channel": "solar",
                    "bias": 1.0,
                    "at_step": 10,
                    "end_step": 20,
                }
            ]
        }
        ids = [m["id"] for m in scenario_mechanics(spec)]
        assert "forecast_bias" in ids
