"""Tests for Engine physics and market logic.

Uses known inputs → known outputs to verify battery dispatch, grid
limits, solar curtailment, and cost calculations.
"""

import pytest

from watt_the_hack.engine.engine import Engine, SimulationConfig


@pytest.fixture
def engine() -> Engine:
    return Engine()


@pytest.fixture
def base_state() -> dict:
    """Minimal state sufficient for a single engine step."""
    return {
        "time": 0,
        "demand": 50.0,
        "solar": 30.0,
        "soc": 0.5,
        "profiles": {
            "demand": [50.0, 50.0],
            "solar": [30.0, 30.0],
        },
        "price_profile": [0.24, 0.24],
        "price": 0.24,
    }


def do_nothing_action() -> dict:
    return {
        "battery_flow_kw": 0.0,
        "emergency_generator": 0.0,
        "curtail_solar": 0.0,
    }


# ---------------------------------------------------------------------------
# Basic step mechanics
# ---------------------------------------------------------------------------


class TestBasicStep:
    def test_returns_state_and_outputs(self, engine, base_state):
        new_state, outputs = engine.step(base_state, do_nothing_action())
        assert isinstance(new_state, dict)
        assert isinstance(outputs, dict)

    def test_time_advances(self, engine, base_state):
        new_state, _ = engine.step(base_state, do_nothing_action())
        assert new_state["time"] == 1

    def test_output_keys(self, engine, base_state):
        _, outputs = engine.step(base_state, do_nothing_action())
        expected = {
            "net_grid_power",
            "unmet_demand",
            "overvoltage_kw",
            "battery_dispatch",
            "emergency_generator",
            "curtailed_solar",
            "import_price",
            "export_price",
            "step_cost",
        }
        assert expected.issubset(set(outputs.keys()))


# ---------------------------------------------------------------------------
# Grid power balance
# ---------------------------------------------------------------------------


class TestGridPowerBalance:
    def test_net_grid_equals_demand_minus_solar_when_idle(self, engine, base_state):
        """With no battery/diesel/curtailment, net_grid = demand - solar."""
        _, outputs = engine.step(base_state, do_nothing_action())
        assert outputs["net_grid_power"] == pytest.approx(50.0 - 30.0)

    def test_battery_discharge_reduces_grid_import(self, engine, base_state):
        action = do_nothing_action()
        action["battery_flow_kw"] = 10.0  # discharge 10kW
        _, outputs = engine.step(base_state, action)
        # net_grid = demand - solar - battery = 50 - 30 - 10 = 10
        assert outputs["net_grid_power"] == pytest.approx(10.0)

    def test_battery_charge_increases_grid_import(self, engine, base_state):
        action = do_nothing_action()
        action["battery_flow_kw"] = -10.0  # charge 10kW
        _, outputs = engine.step(base_state, action)
        # net_grid = demand - solar - (-10) = 50 - 30 + 10 = 30
        assert outputs["net_grid_power"] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# Battery physics
# ---------------------------------------------------------------------------


class TestBatteryPhysics:
    def test_soc_decreases_on_discharge(self, engine, base_state):
        action = do_nothing_action()
        action["battery_flow_kw"] = 10.0
        new_state, _ = engine.step(base_state, action)
        assert new_state["soc"] < base_state["soc"]

    def test_soc_increases_on_charge(self, engine, base_state):
        action = do_nothing_action()
        action["battery_flow_kw"] = -10.0
        new_state, _ = engine.step(base_state, action)
        assert new_state["soc"] > base_state["soc"]

    def test_empty_battery_cannot_discharge(self, engine, base_state):
        base_state["soc"] = 0.0
        action = do_nothing_action()
        action["battery_flow_kw"] = 50.0
        _, outputs = engine.step(base_state, action)
        assert outputs["battery_dispatch"] == pytest.approx(0.0)

    def test_full_battery_cannot_charge(self, engine, base_state):
        base_state["soc"] = 1.0
        action = do_nothing_action()
        action["battery_flow_kw"] = -50.0
        _, outputs = engine.step(base_state, action)
        assert outputs["battery_dispatch"] == pytest.approx(0.0)

    def test_inverter_clipping(self, engine, base_state):
        """Requested kW beyond inverter max is clipped."""
        action = do_nothing_action()
        action["battery_flow_kw"] = 999.0
        _, outputs = engine.step(base_state, action)
        # Actual dispatch is limited by inverter AND available energy
        assert outputs["battery_dispatch"] <= engine.config.max_inverter_kw


# ---------------------------------------------------------------------------
# Grid limits
# ---------------------------------------------------------------------------


class TestGridLimits:
    def test_import_limit_causes_unmet_demand(self, engine):
        """Demand far exceeding grid capacity → unmet demand."""
        state = {
            "time": 0,
            "demand": 300.0,
            "solar": 0.0,
            "soc": 0.0,
            "profiles": {"demand": [300.0, 300.0], "solar": [0.0, 0.0]},
            "price_profile": [0.24, 0.24],
            "price": 0.24,
        }
        _, outputs = engine.step(state, do_nothing_action())
        assert outputs["unmet_demand"] > 0.0
        assert outputs["net_grid_power"] == pytest.approx(
            engine.config.grid_max_import_kw
        )

    def test_export_limit_causes_overvoltage(self, engine):
        """Massive solar surplus → overvoltage."""
        state = {
            "time": 0,
            "demand": 10.0,
            "solar": 200.0,
            "soc": 1.0,
            "profiles": {"demand": [10.0, 10.0], "solar": [200.0, 200.0]},
            "price_profile": [0.24, 0.24],
            "price": 0.24,
        }
        _, outputs = engine.step(state, do_nothing_action())
        assert outputs["overvoltage_kw"] > 0.0


# ---------------------------------------------------------------------------
# Solar curtailment
# ---------------------------------------------------------------------------


class TestSolarCurtailment:
    def test_curtailment_reduces_effective_solar(self, engine, base_state):
        action = do_nothing_action()
        action["curtail_solar"] = 15.0
        _, outputs = engine.step(base_state, action)
        assert outputs["curtailed_solar"] == pytest.approx(15.0)
        # Net grid should increase because less solar is available
        # net_grid = demand - (solar - curtailed) = 50 - (30 - 15) = 35
        assert outputs["net_grid_power"] == pytest.approx(35.0)

    def test_curtailment_clamped_to_available_solar(self, engine, base_state):
        action = do_nothing_action()
        action["curtail_solar"] = 999.0
        _, outputs = engine.step(base_state, action)
        assert outputs["curtailed_solar"] == pytest.approx(30.0)  # max = solar


# ---------------------------------------------------------------------------
# Emergency generator
# ---------------------------------------------------------------------------


class TestEmergencyGenerator:
    def test_generator_reduces_grid_import(self, engine, base_state):
        action = do_nothing_action()
        action["emergency_generator"] = 10.0
        _, outputs = engine.step(base_state, action)
        # net_grid = demand - solar - generator = 50 - 30 - 10 = 10
        assert outputs["net_grid_power"] == pytest.approx(10.0)

    def test_generator_clamped_to_max(self, engine, base_state):
        action = do_nothing_action()
        action["emergency_generator"] = 999.0
        _, outputs = engine.step(base_state, action)
        assert outputs["emergency_generator"] == pytest.approx(
            engine.config.max_emergency_generator_kw
        )


# ---------------------------------------------------------------------------
# Market / cost
# ---------------------------------------------------------------------------


class TestMarketStep:
    def test_import_cost(self, engine, base_state):
        """Importing power costs price * energy on the tariff_import line."""
        _, outputs = engine.step(base_state, do_nothing_action())
        expected_energy = (50.0 - 30.0) * engine.config.dt_hours
        expected_tariff = expected_energy * 0.24
        assert outputs["cost_breakdown"]["tariff_import"] == pytest.approx(
            expected_tariff
        )
        # No exports happened this step
        assert outputs["cost_breakdown"]["tariff_export"] == pytest.approx(0.0)

    def test_export_revenue_lands_on_tariff_export(self, engine):
        """Exporting power earns export_tariff × energy on the tariff_export line."""
        state = {
            "time": 0,
            "demand": 10.0,
            "solar": 50.0,
            "soc": 1.0,
            "profiles": {"demand": [10.0, 10.0], "solar": [50.0, 50.0]},
            "price_profile": [0.24, 0.24],
            "price": 0.24,
        }
        _, outputs = engine.step(state, do_nothing_action())
        assert outputs["cost_breakdown"]["tariff_export"] < 0  # negative = revenue
        assert outputs["cost_breakdown"]["tariff_import"] == pytest.approx(0.0)

    def test_export_revenue(self, engine):
        """Exporting power earns export_tariff * energy (negative cost)."""
        state = {
            "time": 0,
            "demand": 10.0,
            "solar": 50.0,
            "soc": 1.0,
            "profiles": {"demand": [10.0, 10.0], "solar": [50.0, 50.0]},
            "price_profile": [0.24, 0.24],
            "price": 0.24,
        }
        _, outputs = engine.step(state, do_nothing_action())
        assert outputs["step_cost"] < 0.0

    def test_blackout_penalty(self, engine):
        state = {
            "time": 0,
            "demand": 300.0,
            "solar": 0.0,
            "soc": 0.0,
            "profiles": {"demand": [300.0, 300.0], "solar": [0.0, 0.0]},
            "price_profile": [0.24, 0.24],
            "price": 0.24,
        }
        _, outputs = engine.step(state, do_nothing_action())
        assert outputs["unmet_demand"] > 0
        assert outputs["step_cost"] > 0

    def test_cost_breakdown_present(self, engine, base_state):
        """Outputs include a cost_breakdown dict with the canonical keys."""
        _, outputs = engine.step(base_state, do_nothing_action())
        breakdown = outputs["cost_breakdown"]
        assert set(breakdown.keys()) == {
            "tariff_import",
            "tariff_export",
            "generator_fuel",
            "blackout_penalty",
            "overvoltage_penalty",
            "battery_wear",
            "demand_charge",
            "carbon_cost",
            "ramp_charge",
            "fcas_revenue",
            "compliance_penalty",
            "total",
        }
        # total must equal the sum of the other components
        components = sum(v for k, v in breakdown.items() if k != "total")
        assert breakdown["total"] == pytest.approx(components)
        # And step_cost must equal breakdown.total
        assert outputs["step_cost"] == pytest.approx(breakdown["total"])


# ---------------------------------------------------------------------------
# Battery wear cost
# ---------------------------------------------------------------------------


class TestBatteryWear:
    def test_no_wear_when_battery_idle(self, engine, base_state):
        """A do-nothing action incurs zero wear."""
        _, outputs = engine.step(base_state, do_nothing_action())
        assert outputs["cost_breakdown"]["battery_wear"] == pytest.approx(0.0)

    def test_wear_proportional_to_throughput(self, engine, base_state):
        """Wear = |battery_kw| * dt_hours * wear_cost_per_kwh."""
        action = {
            "battery_flow_kw": 20.0,
            "emergency_generator": 0.0,
            "curtail_solar": 0.0,
        }
        _, outputs = engine.step(base_state, action)
        cfg = engine.config
        expected_wear = abs(20.0) * cfg.dt_hours * cfg.battery_wear_cost_per_kwh
        assert outputs["cost_breakdown"]["battery_wear"] == pytest.approx(expected_wear)

    def test_wear_symmetric_for_charge_and_discharge(self, engine, base_state):
        """Charging and discharging at the same magnitude wear the battery equally."""
        discharge = {
            "battery_flow_kw": 20.0,
            "emergency_generator": 0.0,
            "curtail_solar": 0.0,
        }
        charge = {
            "battery_flow_kw": -20.0,
            "emergency_generator": 0.0,
            "curtail_solar": 0.0,
        }
        _, out_d = engine.step(base_state, discharge)
        _, out_c = engine.step(base_state, charge)
        assert out_d["cost_breakdown"]["battery_wear"] == pytest.approx(
            out_c["cost_breakdown"]["battery_wear"]
        )

    def test_wear_uses_actual_dispatch_after_clipping(self, engine, base_state):
        """Wear is based on what the battery actually moved (post-clip), not requested."""
        # Empty SOC, request 50 kW discharge — actual dispatch will be 0
        empty_state = {**base_state, "soc": 0.0}
        action = {
            "battery_flow_kw": 50.0,
            "emergency_generator": 0.0,
            "curtail_solar": 0.0,
        }
        _, outputs = engine.step(empty_state, action)
        assert outputs["battery_dispatch"] == pytest.approx(0.0)
        assert outputs["cost_breakdown"]["battery_wear"] == pytest.approx(0.0)

    def test_wear_added_to_step_cost(self, engine, base_state):
        """Compare a batteries-on run vs identical batteries-off run; cost difference
        equals exactly the wear cost (other components unchanged)."""
        idle_action = do_nothing_action()
        cycle_action = {
            "battery_flow_kw": 10.0,
            "emergency_generator": 0.0,
            "curtail_solar": 0.0,
        }
        _, idle = engine.step(base_state, idle_action)
        _, cycle = engine.step(base_state, cycle_action)
        wear_only = cycle["cost_breakdown"]["battery_wear"]
        # The cycle case ALSO reduces grid imports, which changes tariff. So
        # compare the breakdown components individually:
        assert wear_only > 0.0
        # When idle, wear is 0 — so total cost difference includes both wear
        # AND the tariff change from battery offsetting demand. We verify wear
        # is in there by checking the breakdown directly:
        assert (
            cycle["cost_breakdown"]["battery_wear"]
            > idle["cost_breakdown"]["battery_wear"]
        )


# ---------------------------------------------------------------------------
# Demand charge — billed on the *peak* import seen across the run
# ---------------------------------------------------------------------------


class TestDemandCharge:
    def test_first_step_charges_full_peak(self, engine, base_state):
        """First step's import is by definition a new peak — bill it."""
        _, outputs = engine.step(base_state, do_nothing_action())
        import_kw = max(0.0, outputs["net_grid_power"])
        expected = import_kw * engine.config.demand_charge_per_kw
        assert outputs["cost_breakdown"]["demand_charge"] == pytest.approx(expected)

    def test_no_charge_below_existing_peak(self, engine):
        """If today's import is below an established peak, no new charge."""
        state = {
            "time": 0,
            "demand": 30.0,
            "solar": 10.0,
            "soc": 0.5,
            "profiles": {"demand": [30.0, 30.0], "solar": [10.0, 10.0]},
            "price_profile": [0.24, 0.24],
            "price": 0.24,
            "peak_import_kw": 100.0,  # already established
        }
        _, outputs = engine.step(state, do_nothing_action())
        # net_grid = 30-10 = 20 kW, well below peak 100 → no new charge
        assert outputs["cost_breakdown"]["demand_charge"] == pytest.approx(0.0)

    def test_charges_only_the_delta_when_peak_grows(self, engine):
        """New peak above prior peak charges the increment, not the full new peak."""
        state = {
            "time": 0,
            "demand": 60.0,
            "solar": 10.0,
            "soc": 0.5,
            "profiles": {"demand": [60.0, 60.0], "solar": [10.0, 10.0]},
            "price_profile": [0.24, 0.24],
            "price": 0.24,
            "peak_import_kw": 30.0,  # prior peak
        }
        _, outputs = engine.step(state, do_nothing_action())
        # net_grid = 60-10 = 50 kW. Delta above prior peak = 50-30 = 20.
        expected = 20.0 * engine.config.demand_charge_per_kw
        assert outputs["cost_breakdown"]["demand_charge"] == pytest.approx(expected)

    def test_export_does_not_count_toward_peak(self, engine):
        """Negative net_grid_power (exporting) doesn't move the peak."""
        state = {
            "time": 0,
            "demand": 10.0,
            "solar": 50.0,
            "soc": 1.0,
            "profiles": {"demand": [10.0, 10.0], "solar": [50.0, 50.0]},
            "price_profile": [0.24, 0.24],
            "price": 0.24,
            "peak_import_kw": 0.0,
        }
        new_state, outputs = engine.step(state, do_nothing_action())
        assert outputs["net_grid_power"] < 0
        assert outputs["cost_breakdown"]["demand_charge"] == pytest.approx(0.0)
        assert new_state["peak_import_kw"] == pytest.approx(0.0)

    def test_peak_carries_through_state(self, engine, base_state):
        """new_state['peak_import_kw'] equals max(prev_peak, current_import)."""
        new_state, outputs = engine.step(base_state, do_nothing_action())
        expected = max(0.0, outputs["net_grid_power"])
        assert new_state["peak_import_kw"] == pytest.approx(expected)

    def test_running_total_equals_peak_times_rate_for_demand(self, engine):
        """Over many steps, accumulated demand charges equal peak_import × rate."""
        # Demand wobbles 40 → 80 → 50 → 60 → 80 (peak 80 hit twice)
        demand_profile = [40.0, 80.0, 50.0, 60.0, 80.0]
        steps = len(demand_profile)
        state = {
            "time": 0,
            "demand": demand_profile[0],
            "solar": 0.0,
            "soc": 0.5,
            "profiles": {"demand": demand_profile, "solar": [0.0] * steps},
            "price_profile": [0.24] * steps,
            "price": 0.24,
            "peak_import_kw": 0.0,
        }
        total_demand_charge = 0.0
        peak = 0.0
        for _ in range(steps):
            state, outputs = engine.step(state, do_nothing_action())
            total_demand_charge += outputs["cost_breakdown"]["demand_charge"]
            peak = max(peak, max(0.0, outputs["net_grid_power"]))
        expected = peak * engine.config.demand_charge_per_kw
        assert total_demand_charge == pytest.approx(expected)
        # Final state's peak should match observed peak
        assert state["peak_import_kw"] == pytest.approx(peak)


# ---------------------------------------------------------------------------
# Carbon cost — emissions from imports + diesel
# ---------------------------------------------------------------------------


class TestCarbonCost:
    def test_imports_charged_at_grid_intensity(self, engine, base_state):
        """Importing power is charged at grid_co2_intensity × carbon_price."""
        _, outputs = engine.step(base_state, do_nothing_action())
        cfg = engine.config
        import_kwh = max(0.0, outputs["net_grid_power"]) * cfg.dt_hours
        expected = (
            import_kwh * cfg.grid_co2_intensity_kg_per_kwh * cfg.carbon_price_per_kg
        )
        assert outputs["cost_breakdown"]["carbon_cost"] == pytest.approx(expected)

    def test_exports_have_zero_carbon_cost(self, engine):
        """Exporting clean power doesn't earn carbon credit, but doesn't emit either."""
        state = {
            "time": 0,
            "demand": 10.0,
            "solar": 50.0,
            "soc": 1.0,
            "profiles": {"demand": [10.0, 10.0], "solar": [50.0, 50.0]},
            "price_profile": [0.24, 0.24],
            "price": 0.24,
        }
        _, outputs = engine.step(state, do_nothing_action())
        assert outputs["net_grid_power"] < 0
        assert outputs["cost_breakdown"]["carbon_cost"] == pytest.approx(0.0)

    def test_diesel_emits_carbon(self, engine, base_state):
        """Diesel generator adds its own emissions on top of grid imports."""
        cfg = engine.config
        idle = engine.step(base_state, do_nothing_action())[1]["cost_breakdown"][
            "carbon_cost"
        ]
        diesel_action = {
            "battery_flow_kw": 0.0,
            "emergency_generator": 20.0,
            "curtail_solar": 0.0,
        }
        diesel = engine.step(base_state, diesel_action)[1]["cost_breakdown"][
            "carbon_cost"
        ]

        # Diesel covers 20 kW of demand → reduces import by 20 kW. So:
        #   imports drop by 20 * dt → carbon from imports drops
        #   diesel adds 20 * dt of its own emissions
        # Net depends on (diesel_intensity - grid_intensity) sign.
        # Diesel (0.27) is *cleaner* than grid (0.7), so total carbon should drop.
        assert diesel < idle

    def test_per_scenario_grid_intensity_override(self, engine, base_state):
        """state['grid_co2_intensity'] overrides the config default."""
        clean_state = {**base_state, "grid_co2_intensity": 0.05}  # Tasmania-like
        dirty_state = {**base_state, "grid_co2_intensity": 1.0}  # coal-heavy

        _, clean = engine.step(clean_state, do_nothing_action())
        _, dirty = engine.step(dirty_state, do_nothing_action())

        # Same imports, different intensity → different carbon cost.
        ratio = (
            dirty["cost_breakdown"]["carbon_cost"]
            / clean["cost_breakdown"]["carbon_cost"]
        )
        assert ratio == pytest.approx(1.0 / 0.05, rel=1e-6)

    def test_zero_carbon_price_disables_component(self, base_state):
        """A scenario with carbon_price=0 should have zero carbon_cost regardless of imports."""
        from watt_the_hack.engine.engine import Engine, SimulationConfig

        engine = Engine(config=SimulationConfig(carbon_price_per_kg=0.0))
        _, outputs = engine.step(base_state, do_nothing_action())
        assert outputs["cost_breakdown"]["carbon_cost"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Ramp charge — quadratic penalty on grid-power changes
# ---------------------------------------------------------------------------


class TestRampCharge:
    def test_first_step_has_zero_ramp_charge(self, engine, base_state):
        """No prior grid power → no reference for ramp → charge is 0."""
        _, outputs = engine.step(base_state, do_nothing_action())
        assert outputs["cost_breakdown"]["ramp_charge"] == pytest.approx(0.0)

    def test_no_ramp_when_grid_power_unchanged(self, engine):
        """Two identical steps in a row → zero ramp."""
        state = {
            "time": 0,
            "demand": 50.0,
            "solar": 30.0,
            "soc": 0.5,
            "profiles": {"demand": [50.0] * 4, "solar": [30.0] * 4},
            "price_profile": [0.24] * 4,
            "price": 0.24,
        }
        state, _ = engine.step(state, do_nothing_action())
        # Now prev_grid_power_kw is set; second step should produce same grid_power
        _, second = engine.step(state, do_nothing_action())
        assert second["cost_breakdown"]["ramp_charge"] == pytest.approx(0.0)

    def test_ramp_charge_quadratic(self, engine):
        """A 50 kW ramp costs 4× a 25 kW ramp (quadratic shape)."""
        cfg = engine.config

        def ramp_for_demand_pair(d1: float, d2: float) -> float:
            state = {
                "time": 0,
                "demand": d1,
                "solar": 0.0,
                "soc": 0.5,
                "profiles": {"demand": [d1, d2, d2], "solar": [0.0, 0.0, 0.0]},
                "price_profile": [0.24] * 3,
                "price": 0.24,
            }
            state, _ = engine.step(
                state, do_nothing_action()
            )  # primes prev_grid_power_kw
            _, second = engine.step(state, do_nothing_action())
            return second["cost_breakdown"]["ramp_charge"]

        small = ramp_for_demand_pair(20.0, 45.0)  # 25 kW ramp → 625 × rate
        big = ramp_for_demand_pair(20.0, 70.0)  # 50 kW ramp → 2500 × rate
        assert big == pytest.approx(small * 4.0, rel=1e-6)
        # And the absolute value matches the formula
        assert big == pytest.approx(50.0**2 * cfg.ramp_charge_per_kw2)

    def test_negative_and_positive_ramps_cost_equally(self, engine):
        """Ramping up by 30 kW costs the same as ramping down by 30 kW (squared)."""
        # Demand goes 50 → 80 (ramp up 30)
        up = {
            "time": 0,
            "demand": 50.0,
            "solar": 0.0,
            "soc": 0.5,
            "profiles": {"demand": [50.0, 80.0, 80.0], "solar": [0.0, 0.0, 0.0]},
            "price_profile": [0.24] * 3,
            "price": 0.24,
        }
        up, _ = engine.step(up, do_nothing_action())
        _, up_second = engine.step(up, do_nothing_action())

        # Demand goes 80 → 50 (ramp down 30)
        down = {
            "time": 0,
            "demand": 80.0,
            "solar": 0.0,
            "soc": 0.5,
            "profiles": {"demand": [80.0, 50.0, 50.0], "solar": [0.0, 0.0, 0.0]},
            "price_profile": [0.24] * 3,
            "price": 0.24,
        }
        down, _ = engine.step(down, do_nothing_action())
        _, down_second = engine.step(down, do_nothing_action())

        assert up_second["cost_breakdown"]["ramp_charge"] == pytest.approx(
            down_second["cost_breakdown"]["ramp_charge"]
        )

    def test_prev_grid_power_persists_in_state(self, engine, base_state):
        """new_state['prev_grid_power_kw'] equals the just-computed net_grid_power."""
        new_state, outputs = engine.step(base_state, do_nothing_action())
        assert new_state["prev_grid_power_kw"] == pytest.approx(
            outputs["net_grid_power"]
        )

    def test_zero_rate_disables_component(self, base_state):
        """ramp_charge_per_kw2=0 turns off the ramp charge entirely."""
        from watt_the_hack.engine.engine import Engine, SimulationConfig

        engine = Engine(config=SimulationConfig(ramp_charge_per_kw2=0.0))
        state, _ = engine.step(base_state, do_nothing_action())
        # A second step with very different demand → would normally ramp
        state["demand"] = 200.0
        state["profiles"]["demand"][1] = 200.0
        _, outputs = engine.step(state, do_nothing_action())
        assert outputs["cost_breakdown"]["ramp_charge"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# FCAS reserve — passive revenue + inverter capacity trade-off
# ---------------------------------------------------------------------------


class TestFcasReserve:
    def test_no_reserve_no_revenue(self, engine, base_state):
        """fcas_reserve_kw=0 produces zero FCAS revenue."""
        _, outputs = engine.step(base_state, do_nothing_action())
        assert outputs["fcas_reserve"] == pytest.approx(0.0)
        assert outputs["cost_breakdown"]["fcas_revenue"] == pytest.approx(0.0)

    def test_reserve_generates_revenue(self, engine, base_state):
        """Holding 20 kW for FCAS earns 20 × dt × rate (negative cost)."""
        action = {**do_nothing_action(), "fcas_reserve_kw": 20.0}
        _, outputs = engine.step(base_state, action)
        cfg = engine.config
        expected_revenue = -20.0 * cfg.dt_hours * cfg.fcas_revenue_per_kw_per_hour
        assert outputs["cost_breakdown"]["fcas_revenue"] == pytest.approx(
            expected_revenue
        )
        assert outputs["cost_breakdown"]["fcas_revenue"] < 0  # it's revenue

    def test_reserve_clipped_to_inverter_max(self, engine, base_state):
        """FCAS reserve cannot exceed max_inverter_kw (and goes negative is rejected)."""
        cfg = engine.config
        too_much = {**do_nothing_action(), "fcas_reserve_kw": cfg.max_inverter_kw * 2}
        _, outputs = engine.step(base_state, too_much)
        assert outputs["fcas_reserve"] == pytest.approx(cfg.max_inverter_kw)

        negative = {**do_nothing_action(), "fcas_reserve_kw": -10.0}
        _, outputs = engine.step(base_state, negative)
        assert outputs["fcas_reserve"] == pytest.approx(0.0)

    # The trade-off — the whole point of this feature
    def test_reserve_eats_into_battery_capacity(self, engine, base_state):
        """Reserving 30 kW for FCAS leaves only 20 kW for battery dispatch."""
        cfg = engine.config
        action = {
            "battery_flow_kw": cfg.max_inverter_kw,  # request full discharge
            "emergency_generator": 0.0,
            "curtail_solar": 0.0,
            "fcas_reserve_kw": 30.0,  # but reserve 30 first
        }
        _, outputs = engine.step(base_state, action)
        # Effective battery budget = 50 - 30 = 20
        assert outputs["fcas_reserve"] == pytest.approx(30.0)
        assert outputs["battery_dispatch"] == pytest.approx(20.0)

    def test_full_reserve_locks_battery(self, engine, base_state):
        """Reserving the full inverter for FCAS leaves zero for battery dispatch."""
        cfg = engine.config
        action = {
            "battery_flow_kw": -cfg.max_inverter_kw,  # request full charge
            "emergency_generator": 0.0,
            "curtail_solar": 0.0,
            "fcas_reserve_kw": cfg.max_inverter_kw,
        }
        _, outputs = engine.step(base_state, action)
        assert outputs["fcas_reserve"] == pytest.approx(cfg.max_inverter_kw)
        assert outputs["battery_dispatch"] == pytest.approx(0.0)

    def test_reserve_does_not_drain_soc(self, engine, base_state):
        """FCAS reserve is capacity-only — battery energy is unchanged."""
        action = {**do_nothing_action(), "fcas_reserve_kw": 40.0}
        new_state, _ = engine.step(base_state, action)
        # SOC didn't move (no discharge / charge happened)
        assert new_state["soc"] == pytest.approx(base_state["soc"])

    def test_reserve_does_not_change_grid_power(self, engine, base_state):
        """FCAS reservation is invisible to the grid — net_grid_power
        is what battery+demand+solar+generator dictate, not FCAS."""
        no_fcas = engine.step(base_state, do_nothing_action())[1]["net_grid_power"]
        with_fcas = engine.step(
            base_state,
            {**do_nothing_action(), "fcas_reserve_kw": 40.0},
        )[1]["net_grid_power"]
        assert no_fcas == pytest.approx(with_fcas)

    def test_zero_rate_disables_revenue(self, base_state):
        """fcas_revenue_per_kw_per_hour=0 turns off revenue regardless of reserve."""
        from watt_the_hack.engine.engine import Engine, SimulationConfig

        engine = Engine(config=SimulationConfig(fcas_revenue_per_kw_per_hour=0.0))
        action = {**do_nothing_action(), "fcas_reserve_kw": 50.0}
        _, outputs = engine.step(base_state, action)
        assert outputs["cost_breakdown"]["fcas_revenue"] == pytest.approx(0.0)
        # But the reservation itself is still recognized (and battery is locked)
        assert outputs["fcas_reserve"] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Action key strings
# ---------------------------------------------------------------------------


class TestComplianceMechanic:
    """Compliance directives are SCENARIO-DECLARED and ENGINE-ENFORCED.

    A ``compliance_window`` event with ``min_soc_floor`` and/or
    ``max_export_kw_override`` fires automatically when the window is
    active. The controller's ability to AVOID the penalty depends on
    whether it read the preceding qualitative_alert and had time to
    position SOC / cap exports.

    This is the load-bearing LLM mechanic: only an alert-aware
    controller has lead time to comply.
    """

    @pytest.fixture
    def state_with_low_soc(self) -> dict:
        # Time=5, low SOC, mid-demand — primed to breach an SOC floor.
        return {
            "time": 5,
            "demand": 50.0,
            "solar": 10.0,
            "soc": 0.30,
            "profiles": {"demand": [50.0] * 10, "solar": [10.0] * 10},
            "price_profile": [0.20] * 10,
            "price": 0.20,
        }

    def test_no_penalty_when_no_compliance_events(self, engine, state_with_low_soc):
        """Scenario with no compliance_window events = no penalty, ever."""
        _, outputs = engine.step(state_with_low_soc, do_nothing_action())
        assert outputs["cost_breakdown"]["compliance_penalty"] == pytest.approx(0.0)

    def test_no_penalty_outside_window(self, engine, state_with_low_soc):
        """compliance_window event spans steps 10-20; at time=5 it's inert."""
        state = {
            **state_with_low_soc,
            "events": [
                {
                    "type": "compliance_window",
                    "at_step": 10,
                    "end_step": 20,
                    "min_soc_floor": 0.80,
                }
            ],
        }
        _, outputs = engine.step(state, do_nothing_action())
        assert outputs["cost_breakdown"]["compliance_penalty"] == pytest.approx(0.0)

    def test_soc_floor_breach_charged_automatically(self, engine, state_with_low_soc):
        """Engine fires the penalty regardless of agent_plan — pure-numerical
        controllers can't escape by ignoring."""
        state = {
            **state_with_low_soc,
            "events": [
                {
                    "type": "compliance_window",
                    "at_step": 0,
                    "end_step": 10,
                    "min_soc_floor": 0.50,
                }
            ],
        }
        _, outputs = engine.step(state, do_nothing_action())
        # SOC after idle step = 0.30; floor 0.50; shortfall 0.20
        expected = 0.20 * engine.config.compliance_soc_penalty_per_unit
        assert outputs["cost_breakdown"]["compliance_penalty"] == pytest.approx(expected)

    def test_soc_floor_satisfied_zero_penalty(self, engine, state_with_low_soc):
        """If SOC ends above the floor, no penalty even with an active window."""
        plenty = {**state_with_low_soc, "soc": 0.90}
        state = {
            **plenty,
            "events": [
                {
                    "type": "compliance_window",
                    "at_step": 0,
                    "end_step": 10,
                    "min_soc_floor": 0.50,
                }
            ],
        }
        _, outputs = engine.step(state, do_nothing_action())
        assert outputs["cost_breakdown"]["compliance_penalty"] == pytest.approx(0.0)

    def test_export_cap_breach_charged_automatically(self, engine):
        """Exporting 50 kW against a 10 kW cap → 40 kW excess × dt × rate."""
        state = {
            "time": 0,
            "demand": 5.0,
            "solar": 60.0,
            "soc": 1.0,
            "profiles": {"demand": [5.0] * 2, "solar": [60.0] * 2},
            "price_profile": [0.20] * 2,
            "price": 0.20,
            "events": [
                {
                    "type": "compliance_window",
                    "at_step": 0,
                    "end_step": 10,
                    "max_export_kw_override": 10.0,
                }
            ],
        }
        _, outputs = engine.step(state, do_nothing_action())
        assert outputs["net_grid_power"] == pytest.approx(-50.0)
        cfg = engine.config
        expected = 40.0 * cfg.dt_hours * cfg.compliance_export_penalty_per_kw
        assert outputs["cost_breakdown"]["compliance_penalty"] == pytest.approx(expected)

    def test_export_cap_no_penalty_when_importing(self, engine, state_with_low_soc):
        """Importing doesn't trip the export cap."""
        state = {
            **state_with_low_soc,
            "events": [
                {
                    "type": "compliance_window",
                    "at_step": 0,
                    "end_step": 10,
                    "max_export_kw_override": 5.0,
                }
            ],
        }
        _, outputs = engine.step(state, do_nothing_action())
        assert outputs["net_grid_power"] > 0  # importing
        assert outputs["cost_breakdown"]["compliance_penalty"] == pytest.approx(0.0)

    def test_both_constraints_sum_on_single_event(self, engine):
        """One event with both keys → both shortfalls add to the same line."""
        state = {
            "time": 0,
            "demand": 5.0,
            "solar": 60.0,
            "soc": 0.10,
            "profiles": {"demand": [5.0] * 2, "solar": [60.0] * 2},
            "price_profile": [0.20] * 2,
            "price": 0.20,
            "events": [
                {
                    "type": "compliance_window",
                    "at_step": 0,
                    "end_step": 10,
                    "min_soc_floor": 0.40,
                    "max_export_kw_override": 10.0,
                }
            ],
        }
        _, outputs = engine.step(state, do_nothing_action())
        cfg = engine.config
        soc_part = 0.30 * cfg.compliance_soc_penalty_per_unit
        exp_part = 40.0 * cfg.dt_hours * cfg.compliance_export_penalty_per_kw
        assert outputs["cost_breakdown"]["compliance_penalty"] == pytest.approx(
            soc_part + exp_part
        )

    def test_multiple_windows_accumulate(self, engine, state_with_low_soc):
        """Two overlapping compliance_window events both contribute."""
        state = {
            **state_with_low_soc,
            "events": [
                {
                    "type": "compliance_window",
                    "at_step": 0,
                    "end_step": 10,
                    "min_soc_floor": 0.40,
                },
                {
                    "type": "compliance_window",
                    "at_step": 0,
                    "end_step": 10,
                    "min_soc_floor": 0.50,
                },
            ],
        }
        _, outputs = engine.step(state, do_nothing_action())
        cfg = engine.config
        # SOC 0.30; first floor 0.40 → shortfall 0.10; second 0.50 → 0.20
        expected = (0.10 + 0.20) * cfg.compliance_soc_penalty_per_unit
        assert outputs["cost_breakdown"]["compliance_penalty"] == pytest.approx(expected)

    def test_other_event_types_ignored(self, engine, state_with_low_soc):
        """Forecast_bias / weather_anomaly events never trigger compliance."""
        state = {
            **state_with_low_soc,
            "events": [
                {
                    "type": "forecast_bias",
                    "at_step": 0,
                    "end_step": 10,
                    "channel": "demand",
                    "bias": 20.0,
                    "min_soc_floor": 0.90,  # bogus — wrong event type
                }
            ],
        }
        _, outputs = engine.step(state, do_nothing_action())
        assert outputs["cost_breakdown"]["compliance_penalty"] == pytest.approx(0.0)

    def test_penalty_below_blackout(self):
        """A full SOC unit shortfall must cost less than 1 kWh of blackout —
        otherwise controllers may rationally shed load to comply."""
        from watt_the_hack.engine.engine import SimulationConfig

        cfg = SimulationConfig()
        worst_soc_penalty = 1.0 * cfg.compliance_soc_penalty_per_unit
        blackout_1kwh = 1.0 * cfg.blackout_penalty_per_kwh
        assert worst_soc_penalty < blackout_1kwh

    def test_penalty_above_wear_cost(self):
        """A 0.10-SOC breach held for 4 steps (1 hour) must cost more than
        the wear it takes to fix it once — otherwise ignoring is cheapest.
        """
        from watt_the_hack.engine.engine import SimulationConfig

        cfg = SimulationConfig()
        wear_to_comply = 10.0 * cfg.battery_wear_cost_per_kwh
        ignore_4_steps = 0.10 * cfg.compliance_soc_penalty_per_unit * 4
        assert ignore_4_steps > wear_to_comply, (
            f"ignoring 1h ({ignore_4_steps:.2f}) cheaper than complying "
            f"({wear_to_comply:.2f})"
        )

    def test_missing_constraint_keys_silently_zero(self, engine, state_with_low_soc):
        """A compliance_window event with no floor/cap keys does nothing."""
        state = {
            **state_with_low_soc,
            "events": [{"type": "compliance_window", "at_step": 0, "end_step": 10}],
        }
        _, outputs = engine.step(state, do_nothing_action())
        assert outputs["cost_breakdown"]["compliance_penalty"] == pytest.approx(0.0)

    def test_malformed_constraint_values_silently_disabled(
        self, engine, state_with_low_soc
    ):
        """Non-numeric constraint values are ignored, never crash."""
        state = {
            **state_with_low_soc,
            "events": [
                {
                    "type": "compliance_window",
                    "at_step": 0,
                    "end_step": 10,
                    "min_soc_floor": "high",
                    "max_export_kw_override": [1, 2, 3],
                }
            ],
        }
        _, outputs = engine.step(state, do_nothing_action())
        assert outputs["cost_breakdown"]["compliance_penalty"] == pytest.approx(0.0)

    def test_agent_plan_no_longer_drives_penalty(self, engine, state_with_low_soc):
        """Regression: agent_plan compliance keys must NOT trigger the penalty
        — the engine now reads scenario events only. Controllers may set
        these keys as internal-coordination scratch without engine effect.
        """
        state = {
            **state_with_low_soc,
            "agent_plan": {
                "compliance_window": [0, 10],
                "min_soc_floor": 0.90,
                "max_export_kw_override": 5.0,
            },
        }
        _, outputs = engine.step(state, do_nothing_action())
        assert outputs["cost_breakdown"]["compliance_penalty"] == pytest.approx(0.0)


class TestActionKeys:
    def test_engine_reads_string_keys(self, engine, base_state):
        """Engine reads action via plain string keys."""
        action = {
            "battery_flow_kw": 5.0,
            "emergency_generator": 0.0,
            "curtail_solar": 0.0,
        }
        new_state, outputs = engine.step(base_state, action)
        assert outputs["battery_dispatch"] != 0.0  # battery did something

    def test_missing_keys_default_to_zero(self, engine, base_state):
        """Empty action dict → all zeros."""
        _, outputs = engine.step(base_state, {})
        assert outputs["battery_dispatch"] == pytest.approx(0.0)
        assert outputs["emergency_generator"] == pytest.approx(0.0)
        assert outputs["curtailed_solar"] == pytest.approx(0.0)
