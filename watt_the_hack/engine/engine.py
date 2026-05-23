import random
from dataclasses import dataclass, field
from typing import Any

from watt_the_hack.engine.base_engine import SimulationEngine


@dataclass(slots=True)
class PhysicsResult:
    """Output of one physics step. Named fields >>> tuple unpacking."""

    next_soc: float
    battery_kw: float  # actual dispatch after inverter + SOC + FCAS clipping
    emergency_generator_kw: float  # actual diesel after [0, max] clip
    curtailed_solar_kw: float  # actual curtailment after clamp to available solar
    net_grid_power: float  # +import / -export, after grid limit clipping
    unmet_demand: float  # kW above the import limit (blackout)
    overvoltage_kw: float  # kW below the negative export limit (overvoltage)
    fcas_reserve_kw: float  # actual capacity reserved for FCAS revenue


@dataclass(slots=True)
class SimulationConfig:
    # 1. Rebalanced Game Board
    battery_capacity_kwh: float = 100.0
    max_inverter_kw: float = 50.0
    grid_max_import_kw: float = 120.0
    grid_max_export_kw: float = 50.0  # NEW: Export limit

    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    dt_hours: float = 0.25

    # Phase 3 Realism: Split Pricing & Penalties
    export_tariff: float = 0.05  # $0.05 flat rate for exporting solar
    blackout_penalty_per_kwh: float = 10.00
    emergency_generator_cost_per_kwh: float = 1.00
    max_emergency_generator_kw: float = 50.0
    overvoltage_penalty_per_kwh: float = 5.00  # NEW: Penalty for exporting too much

    # Battery wear: each kWh moved through the battery (charge or discharge)
    # eats a fraction of its lifetime. Calibrated to ~$0.05/kWh throughput,
    # matching real Li-ion replacement cost (~$400/kWh capital, ~4000 cycles).
    # Forces controllers to value cycles, not just spam them.
    battery_wear_cost_per_kwh: float = 0.05

    # Demand charge: $/kW based on the HIGHEST single-step import seen in the
    # run. Real commercial bills do this monthly — one big spike costs you
    # for the whole period, not just the spike step. Forces peak-shaving
    # discipline distinct from "just don't blackout".
    # Billed incrementally: each step, only the *new* peak above the prior
    # peak is charged, so total = peak_import_kw * demand_charge_per_kw.
    demand_charge_per_kw: float = 1.0

    # Carbon price: charges every kg of CO2 emitted from imports + diesel.
    # Real-world calibration:
    #   - AU carbon price ~$50/tonne AUD = $0.05/kg
    #   - NSW/QLD grid intensity ~0.7 kg CO2/kWh (fossil-heavy)
    #   - Diesel intensity ~0.27 kg CO2/kWh (fixed by chemistry)
    # Exports earn nothing on carbon — you're sending clean power TO the grid.
    # Scenarios can override grid_co2_intensity via state["grid_co2_intensity"]
    # (e.g., 0.05 for Tasmania hydro, 0.8 for QLD coal).
    carbon_price_per_kg: float = 0.05
    grid_co2_intensity_kg_per_kwh: float = 0.7
    diesel_co2_intensity_kg_per_kwh: float = 0.27

    # Ramp charge: quadratic penalty on changes in net grid power between
    # steps. Real-world equivalent: AEMO's FCAS markets pay for smoothness.
    # cost = (grid_power[t] - grid_power[t-1])^2 * rate
    #   - 50 kW ramp → 2500 × 0.001 = $2.50
    #   - 100 kW ramp → 10000 × 0.001 = $10
    #   - 10 kW ramp → 100 × 0.001 = $0.10
    # Quadratic shape rewards smooth dispatch disproportionately over jagged.
    # First step has no prior grid power, so its ramp charge is zero.
    ramp_charge_per_kw2: float = 0.001

    # FCAS (Frequency Control Ancillary Services) reserve: passive revenue for
    # holding inverter capacity available for the grid. Real-world: AEMO pays
    # batteries to be ready to respond to frequency events whether or not
    # they're called. Hornsdale (Tesla SA) earns ~$10M/year on FCAS alone.
    #
    # The trade-off, made obvious to the controller:
    #   |battery_flow_kw| + fcas_reserve_kw <= max_inverter_kw
    # Every kW you commit to FCAS is a kW you cannot use for arbitrage.
    #
    # Calibration: $0.04/kW/hour ≈ real AEMO contingency-FCAS rates. Caps the
    # max revenue from full FCAS reservation at ~$48/day on a 50 kW inverter,
    # comparable to good arbitrage. Neither strategy dominates the other.
    fcas_revenue_per_kw_per_hour: float = 0.04

    # Forecast configuration (lookahead with growing noise)
    forecast_horizon: int = 16  # how many future steps the controller sees
    forecast_sigma_demand: float = 3.0  # additive noise std (kW)
    forecast_sigma_price: float = 0.02  # additive noise std ($/kWh)
    forecast_solar_noise_pct: float = (
        0.12  # multiplicative noise std (fraction of actual solar)
    )
    forecast_mu_demand: float = 0.0  # persistent bias (kW)
    forecast_mu_price: float = 0.0  # persistent bias ($/kWh)
    forecast_solar_mu_pct: float = 0.0  # persistent bias (fraction of actual solar)
    forecast_ar1_rho: float = 0.7  # AR(1) autocorrelation coefficient for error drift
    forecast_seed: int | None = None  # set for reproducible noise; None = random


@dataclass(slots=True)
class Engine(SimulationEngine):
    """Single node MVP engine with local storage, solar, emergency generation, and curtailment."""

    config: SimulationConfig = field(default_factory=SimulationConfig)

    def step(self, state: dict, action: dict) -> tuple[dict, dict]:
        """Run one timestep.

        Five phases, each a single helper call:
            1. Read inputs from state (demand, solar, soc, price)
            2. Feature gate — zero out actions for disabled features
            3. Physics — apply battery/generator/curtailment, compute net grid power
            4. Market — compute the cost breakdown for this step
            5. Build outputs dict
            6. Advance state to t+1 (forecast, peak tracking, profile lookup)
        """
        time = int(state.get("time", state.get("t", 0)))

        # 1. Inputs
        demand_kw, solar_kw, soc, import_price = self._read_inputs(state, time)

        # 2. Feature gate — scenarios declare which actions are available.
        #    Missing features dict = all features on (backwards compatible).
        action = self._gate_features(state, action)

        # 3. Physics
        physics = self._physics_step(
            action,
            demand_kw,
            solar_kw,
            soc,
            battery_throughput_remaining_kwh=state.get(
                "battery_throughput_remaining_kwh"
            ),
        )

        # 4. Market
        cost_breakdown = self._compute_market(state, time, import_price, physics)

        # 5. Outputs
        outputs = self._build_outputs(physics, import_price, cost_breakdown)

        # 6. State for t+1
        new_state = self._advance_state(state, time, physics, import_price)

        return new_state, outputs

    @staticmethod
    def _gate_features(state: dict, action: dict) -> dict:
        """Zero out action keys for features that are disabled in this scenario.

        Scenarios declare available features via ``state["features"]``:
            {"battery": true, "fcas": false, "flexible_loads": false, ...}

        If the features dict is absent, ALL features are enabled (backwards
        compatible with existing scenarios and tests that don't set it).
        """
        features = state.get("features")
        if features is None:
            return action  # no gating — everything allowed

        gated = dict(action)  # shallow copy so we don't mutate the caller's dict

        if not features.get("battery", True):
            gated["battery_flow_kw"] = 0.0

        if not features.get("curtailment", True):
            gated["curtail_solar"] = 0.0

        if not features.get("emergency_generator", True):
            gated["emergency_generator"] = 0.0

        if not features.get("fcas", True):
            gated["fcas_reserve_kw"] = 0.0

        # Future features (flexible_loads, forecast_purchasing) will be
        # gated here once their engine logic is implemented.

        return gated

    # ------------------------------------------------------------------
    # Phase helpers — each does one thing
    # ------------------------------------------------------------------

    def _read_inputs(self, state: dict, time: int) -> tuple[float, float, float, float]:
        """Pull the four scalars the engine needs at time t."""
        profiles = state.get("profiles", {})
        demand_kw = self._profile_at(profiles, "demand", time, state.get("demand", 0.0))
        solar_kw = self._profile_at(profiles, "solar", time, state.get("solar", 0.0))
        soc = self._clip(float(state.get("soc", 0.0)), 0.0, 1.0)
        import_price = self._state_value(state, "price", "price_profile", time, 0.0)
        return demand_kw, solar_kw, soc, import_price

    def _compute_market(
        self,
        state: dict,
        time: int,
        import_price: float,
        physics: PhysicsResult,
    ) -> dict:
        """Resolve per-scenario market params and run the cost calculation."""
        prev_peak = float(state.get("peak_import_kw", 0.0))
        new_peak = max(prev_peak, max(0.0, physics.net_grid_power))
        grid_co2 = float(
            state.get(
                "grid_co2_intensity",
                self.config.grid_co2_intensity_kg_per_kwh,
            )
        )
        # Sentinel: missing on first step → ramp charge is 0
        prev_grid_power = state.get("prev_grid_power_kw")

        return self._market_step(
            net_grid_power=physics.net_grid_power,
            import_price=import_price,
            unmet_demand=physics.unmet_demand,
            emergency_generator_kw=physics.emergency_generator_kw,
            overvoltage_kw=physics.overvoltage_kw,
            battery_kw=physics.battery_kw,
            fcas_reserve_kw=physics.fcas_reserve_kw,
            new_peak_import_kw=new_peak,
            prev_peak_import_kw=prev_peak,
            grid_co2_intensity=grid_co2,
            prev_grid_power_kw=prev_grid_power,
        )

    def _build_outputs(
        self,
        physics: PhysicsResult,
        import_price: float,
        cost_breakdown: dict,
    ) -> dict:
        return {
            "net_grid_power": physics.net_grid_power,
            "unmet_demand": physics.unmet_demand,
            "overvoltage_kw": physics.overvoltage_kw,
            "battery_dispatch": physics.battery_kw,
            "emergency_generator": physics.emergency_generator_kw,
            "curtailed_solar": physics.curtailed_solar_kw,
            "fcas_reserve": physics.fcas_reserve_kw,
            "import_price": float(import_price),
            "export_price": float(self.config.export_tariff),
            "step_cost": float(cost_breakdown["total"]),
            "cost_breakdown": cost_breakdown,
        }

    def _advance_state(
        self,
        state: dict,
        time: int,
        physics: PhysicsResult,
        import_price: float,
    ) -> dict:
        """Return a NEW state dict for t+1. Carries forward bookkeeping
        (peak import, prev grid power) and aligns the scalar mirrors
        (demand/solar/price) with the profile at t+1.
        """
        next_time = time + 1
        new_state = dict(state)
        new_state["time"] = next_time
        new_state["soc"] = float(physics.next_soc)

        # Bookkeeping for cost components that span steps
        new_state["peak_import_kw"] = max(
            float(state.get("peak_import_kw", 0.0)),
            max(0.0, physics.net_grid_power),
        )
        new_state["prev_grid_power_kw"] = float(physics.net_grid_power)

        # Throughput budget: decrement by |kWh| moved through the battery
        # this step. Only tracked when the scenario opted in (initial value
        # is not None).
        budget = state.get("battery_throughput_remaining_kwh")
        if budget is not None:
            new_state["battery_throughput_remaining_kwh"] = max(
                0.0,
                float(budget) - abs(physics.battery_kw) * self.config.dt_hours,
            )

        # Mirror profiles → top-level scalars for controllers to read
        profiles = state.get("profiles", {})
        if "demand" in profiles and next_time < len(profiles["demand"]):
            new_state["demand"] = float(profiles["demand"][next_time])
        if "solar" in profiles and next_time < len(profiles["solar"]):
            new_state["solar"] = float(profiles["solar"][next_time])

        # Keep state["price"] == price_profile[t+1] for the controller view.
        # _state_value() still prefers the profile row directly when present.
        new_state["price"] = self._price_at_timestep(new_state, next_time, import_price)

        # Refresh the forecast only for scenarios that opted into forecasts.
        if self._forecast_enabled(new_state):
            new_state["forecast"] = self._build_forecast(new_state, next_time)
        else:
            new_state.pop("forecast", None)

        return new_state

    @staticmethod
    def _profile_at(profiles: dict, key: str, time: int, fallback: float) -> float:
        """Look up profiles[key][time] with a scalar fallback."""
        series = profiles.get(key)
        if series is not None and time < len(series):
            return float(series[time])
        return float(fallback)

    def add_forecast_to_state(self, state: dict) -> dict:
        """Inject state["forecast"] for the current timestep. Call once at init."""
        time = int(state.get("time", 0))
        state["price"] = self._price_at_timestep(
            state, time, float(state.get("price", 0.2))
        )
        if self._forecast_enabled(state):
            state["forecast"] = self._build_forecast(state, time)
        else:
            state.pop("forecast", None)
        return state

    @staticmethod
    def _forecast_enabled(state: dict) -> bool:
        """Scenario-loaded states set forecast_config=None to disable forecasts.

        Missing forecast_config remains enabled for backwards-compatible tests
        and ad-hoc engine use.
        """
        return state.get("forecast_config", {}) is not None

    def _build_forecast(self, state: dict, time: int) -> dict:
        """Return a noisy view of the next H steps of demand, solar, and price.

        Noise is an AR(1) process over absolute time, meaning the error for
        timestep T is correlated with the error at T-1. This ensures errors
        drift smoothly instead of jittering, giving ML-style controllers a
        consistent bias to learn and correct.
        """
        seed = self.config.forecast_seed
        if seed is None:
            # Fallback for reproducible runs if config seed isn't set.
            # Use scenario_id to ensure the same scenario gets the same noise
            # profile across runs, or a default random base.
            scenario_id = state.get("scenario_id", "default")
            seed = hash(scenario_id) % 1_000_000

        profiles = state.get("profiles", {}) or {}
        sources = {
            "demand": profiles.get("demand"),
            "solar": profiles.get("solar"),
            "price": state.get("price_profile"),
        }

        forecast: dict[str, list[float]] = {}
        forecast_config = state.get("forecast_config", {})
        if forecast_config is None:
            return forecast
        events = state.get("events", [])

        horizon = forecast_config.get("horizon_steps", self.config.forecast_horizon)
        rho = forecast_config.get("ar1_rho", self.config.forecast_ar1_rho)

        import math

        for key, profile in sources.items():
            if not profile:
                continue

            if key == "demand":
                sigma = forecast_config.get(
                    "sigma_demand", self.config.forecast_sigma_demand
                )
                mu = forecast_config.get("mu_demand", self.config.forecast_mu_demand)
                is_mult = False
            elif key == "price":
                sigma = forecast_config.get(
                    "sigma_price", self.config.forecast_sigma_price
                )
                mu = forecast_config.get("mu_price", self.config.forecast_mu_price)
                is_mult = False
            elif key == "solar":
                sigma = forecast_config.get(
                    "solar_noise_pct", self.config.forecast_solar_noise_pct
                )
                mu = forecast_config.get(
                    "solar_mu_pct", self.config.forecast_solar_mu_pct
                )
                is_mult = True
            else:
                sigma = 0.0
                mu = 0.0
                is_mult = False

            # Isolate RNG per profile so they don't interfere
            rng = random.Random(f"{seed}_{key}")

            def get_sigma_eps(t_step: int) -> float:
                current_sigma = sigma
                for ev in events:
                    if ev.get("type") in (
                        "forecast_error",
                        "weather_anomaly",
                    ) and ev.get("at_step", 0) <= t_step <= ev.get("end_step", 0):
                        current_sigma *= ev.get("sigma_multiplier", 2.0)
                return (
                    current_sigma * math.sqrt(1.0 - rho**2)
                    if rho < 1.0
                    else current_sigma
                )

            # Smart fast-forward using state cache to avoid O(time) iteration
            ar1_cache = state.get("_ar1_cache", {})
            cache_key = f"{seed}_{key}"

            cached_time, cached_err, rng_state = ar1_cache.get(
                cache_key, (0, 0.0, None)
            )

            if rng_state is not None and cached_time <= time:
                rng.setstate(rng_state)
                err = cached_err
                start_t = cached_time
            else:
                err = 0.0
                start_t = 0

            for t_past in range(start_t, time):
                err = rho * err + rng.gauss(0.0, get_sigma_eps(t_past))

            # Save the RNG state at the exact current time `time` (before horizon generation)
            # We copy the dictionary to prevent mutating a parent state if branched.
            new_ar1_cache = dict(ar1_cache)
            new_ar1_cache[cache_key] = (time, err, rng.getstate())
            state["_ar1_cache"] = new_ar1_cache

            future = []
            for h in range(horizon):
                t_future = time + h
                if t_future >= len(profile):
                    break

                err = rho * err + rng.gauss(0.0, get_sigma_eps(t_future))
                base_val = float(profile[t_future])

                if is_mult:
                    noise = base_val * (err + mu)
                else:
                    noise = err + mu

                val = base_val + noise
                if key in ("demand", "solar"):
                    val = max(0.0, val)

                # Forecast corruption during attack windows (cybersecurity scenario)
                attack_windows = state.get("attack_windows", [])
                for window in attack_windows:
                    if window["start_step"] <= t_future <= window["end_step"]:
                        scale = window["corruption_scale"]
                        attack_rng = random.Random(f"{seed}_attack_{key}_{t_future}")
                        if key == "demand":
                            val = val * (1.0 - attack_rng.uniform(scale * 0.5, scale))
                        elif key == "solar":
                            val = val * (
                                1.0 + attack_rng.uniform(scale * 0.5, scale * 1.5)
                            )
                        elif key == "price":
                            val = val * (1.0 - attack_rng.uniform(0, scale * 0.5))
                        val = max(0.0, val)
                        break

                # Forecast-bias events: deterministic, systematic forecast error
                # applied over a window. Used to model "the forecast was just
                # wrong" — cloud bank that wasn't predicted, demand regime
                # change, price spike that didn't materialise — as distinct
                # from random noise (sigma) or adversarial corruption.
                #
                # Each event declares one channel + a bias value:
                #   - solar: multiplicative (`bias` is a fraction of actual).
                #     +0.5 → forecast shows 50% more solar than will arrive
                #     ("looks sunny but won't be")
                #   - demand: additive in kW.
                #     -20 → forecast under-predicts by 20 kW
                #   - price: additive in $/kWh.
                #     +0.10 → forecast over-predicts a price spike
                #
                # Bias stacks with the AR(1) noise — it is a persistent shift
                # the controller cannot fit out by tracking residuals, because
                # it only applies during the event window.
                for ev in events:
                    if ev.get("type") != "forecast_bias":
                        continue
                    if ev.get("channel") != key:
                        continue
                    if not (
                        ev.get("at_step", 0) <= t_future <= ev.get("end_step", 0)
                    ):
                        continue
                    bias = float(ev.get("bias", 0.0))
                    if key == "solar":
                        val = val * (1.0 + bias)
                    else:
                        val = val + bias
                    if key in ("demand", "solar", "price"):
                        val = max(0.0, val)

                future.append(val)

            forecast[key] = future

        return forecast

    def _physics_step(
        self,
        action: dict,
        demand_kw: float,
        solar_kw: float,
        soc: float,
        battery_throughput_remaining_kwh: float | None = None,
    ) -> PhysicsResult:
        """Apply battery + generator + curtailment, then clip the resulting
        net grid power against the import/export limits. Returns named
        fields rather than a positional tuple — much easier to read at
        the call site.
        """
        cfg = self.config

        # FCAS reserve gets first claim on the inverter. Any kW reserved
        # for FCAS is unavailable for arbitrage this step.
        fcas_reserve_kw = self._clip(
            float(action.get("fcas_reserve_kw", 0.0)),
            0.0,
            cfg.max_inverter_kw,
        )
        battery_inverter_budget = cfg.max_inverter_kw - fcas_reserve_kw

        # Throughput budget: scenarios may cap the total |kWh| moved
        # through the battery across the run. When set, the remaining
        # budget further clips this step's dispatch magnitude.
        if battery_throughput_remaining_kwh is not None:
            budget_kw_cap = max(
                0.0, float(battery_throughput_remaining_kwh) / cfg.dt_hours
            )
            battery_inverter_budget = min(battery_inverter_budget, budget_kw_cap)

        # Battery: clip to remaining inverter capacity (after FCAS) and SOC bounds
        requested_battery_kw = float(action.get("battery_flow_kw", 0.0))
        battery_kw = self._feasible_battery_power(
            requested_battery_kw,
            soc,
            inverter_limit=battery_inverter_budget,
        )
        next_soc = self._next_soc(soc, battery_kw)

        # Diesel: simple [0, max] clip
        emergency_generator_kw = self._clip(
            float(action.get("emergency_generator", 0.0)),
            0.0,
            cfg.max_emergency_generator_kw,
        )

        # Curtailment: can't curtail more than the available solar
        curtailed_solar_kw = self._clip(
            float(action.get("curtail_solar", 0.0)),
            0.0,
            solar_kw,
        )
        actual_solar_kw = solar_kw - curtailed_solar_kw

        # Power balance — what the grid has to make up
        net_grid_power = (
            demand_kw - actual_solar_kw - battery_kw - emergency_generator_kw
        )

        # Clip against grid import/export limits, capturing any overflow
        unmet_demand = 0.0
        overvoltage_kw = 0.0
        if net_grid_power > cfg.grid_max_import_kw:
            unmet_demand = net_grid_power - cfg.grid_max_import_kw
            net_grid_power = cfg.grid_max_import_kw
        elif net_grid_power < -cfg.grid_max_export_kw:
            overvoltage_kw = abs(net_grid_power) - cfg.grid_max_export_kw
            net_grid_power = -cfg.grid_max_export_kw

        return PhysicsResult(
            next_soc=next_soc,
            battery_kw=battery_kw,
            emergency_generator_kw=emergency_generator_kw,
            curtailed_solar_kw=curtailed_solar_kw,
            net_grid_power=net_grid_power,
            unmet_demand=unmet_demand,
            overvoltage_kw=overvoltage_kw,
            fcas_reserve_kw=fcas_reserve_kw,
        )

    def _market_step(
        self,
        *,
        net_grid_power: float,
        import_price: float,
        unmet_demand: float,
        emergency_generator_kw: float,
        overvoltage_kw: float,
        battery_kw: float,
        fcas_reserve_kw: float,
        prev_peak_import_kw: float,
        new_peak_import_kw: float,
        grid_co2_intensity: float,
        prev_grid_power_kw: float | None,
    ) -> dict:
        """Calculate every cost component for this timestep.

        Returns a breakdown dict whose ``total`` key is the headline step
        cost (negative = revenue). The other keys are exposed for the UI
        breakdown panel and for diagnostics.

        Each component is a one-line pure function of config + step
        physics. To add a new cost (carbon, ramp charge, FCAS, etc.):
            1. Add the rate to SimulationConfig.
            2. Add a line to the components dict below.
            3. Done — total + breakdown handle themselves.
        """
        dt = self.config.dt_hours
        cfg = self.config
        energy_kwh = net_grid_power * dt

        # Tariff is split into import and export lines so the dashboard can
        # tell a player how much they earned in exports vs paid in imports
        # (a single net value masks the partition for any mixed day).
        if energy_kwh > 0:
            tariff_import = energy_kwh * import_price  # positive, cost
            tariff_export = 0.0
        else:
            tariff_import = 0.0
            tariff_export = energy_kwh * cfg.export_tariff  # negative, revenue

        # Carbon: imports (positive grid power) and diesel both emit. Exports
        # are clean power leaving the city — they don't earn carbon credit
        # here, just the export tariff.
        import_kwh = max(0.0, energy_kwh)
        diesel_kwh = emergency_generator_kw * dt
        co2_kg = (
            import_kwh * grid_co2_intensity
            + diesel_kwh * cfg.diesel_co2_intensity_kg_per_kwh
        )

        # Ramp: quadratic penalty on the change in net grid power. First
        # step has no prior reference, so its ramp charge is zero.
        if prev_grid_power_kw is None:
            ramp_charge = 0.0
        else:
            ramp_kw = net_grid_power - prev_grid_power_kw
            ramp_charge = (ramp_kw**2) * cfg.ramp_charge_per_kw2

        components = {
            "tariff_import": tariff_import,
            "tariff_export": tariff_export,
            "generator_fuel": emergency_generator_kw
            * dt
            * cfg.emergency_generator_cost_per_kwh,
            "blackout_penalty": unmet_demand * dt * cfg.blackout_penalty_per_kwh,
            "overvoltage_penalty": overvoltage_kw
            * dt
            * cfg.overvoltage_penalty_per_kwh,
            "battery_wear": abs(battery_kw) * dt * cfg.battery_wear_cost_per_kwh,
            # Demand charge: incremental — only the rise above prior peak is billed
            # this step, so the running cumulative charge equals (peak * rate).
            "demand_charge": max(0.0, new_peak_import_kw - prev_peak_import_kw)
            * cfg.demand_charge_per_kw,
            "carbon_cost": co2_kg * cfg.carbon_price_per_kg,
            "ramp_charge": ramp_charge,
            # FCAS revenue: NEGATIVE cost (income) for capacity held available.
            "fcas_revenue": -fcas_reserve_kw * dt * cfg.fcas_revenue_per_kw_per_hour,
        }

        return {
            **{k: float(v) for k, v in components.items()},
            "total": float(sum(components.values())),
        }

    def _feasible_battery_power(
        self,
        requested_kw: float,
        soc: float,
        inverter_limit: float | None = None,
    ) -> float:
        """Clip a requested battery dispatch to:
        1. The inverter limit (default = max_inverter_kw, but FCAS reserve
           can shrink the effective budget for this step).
        2. The energy available in the battery (discharge can't exceed
           what's stored; charge can't exceed remaining headroom).
        """
        cfg = self.config
        limit = (
            cfg.max_inverter_kw if inverter_limit is None else max(0.0, inverter_limit)
        )
        clipped_kw = self._clip(requested_kw, -limit, limit)

        if clipped_kw > 0.0:
            max_discharge_kw = (
                soc * cfg.battery_capacity_kwh * cfg.discharge_efficiency
            ) / cfg.dt_hours
            return min(clipped_kw, max_discharge_kw)

        if clipped_kw < 0.0:
            headroom_kwh = (1.0 - soc) * cfg.battery_capacity_kwh
            max_charge_kw = headroom_kwh / (cfg.charge_efficiency * cfg.dt_hours)
            return max(clipped_kw, -max_charge_kw)

        return 0.0

    def _next_soc(self, soc: float, battery_kw: float) -> float:
        if battery_kw > 0.0:
            next_soc = soc - (battery_kw * self.config.dt_hours) / (
                self.config.battery_capacity_kwh * self.config.discharge_efficiency
            )
        elif battery_kw < 0.0:
            next_soc = (
                soc
                - (battery_kw * self.config.charge_efficiency * self.config.dt_hours)
                / self.config.battery_capacity_kwh
            )
        else:
            next_soc = soc

        return self._clip(next_soc, 0.0, 1.0)

    @staticmethod
    def _price_at_timestep(
        state: dict[str, Any], timestep: int, fallback: float
    ) -> float:
        """Expose the same tariff the engine uses so state['price'] matches price_profile[t]."""
        profile = state.get("price_profile")
        if isinstance(profile, list) and len(profile) > 0:
            idx = max(0, min(timestep, len(profile) - 1))
            return float(profile[idx])
        return float(state.get("price", fallback))

    @staticmethod
    def _state_value(
        state: dict[str, Any],
        scalar_key: str,
        profile_key: str,
        time: int,
        default: float,
    ) -> float:
        """Look up a top-level profile value with a strict integrity check.

        Used for the import_price lookup. Three-step resolution:
          1. If `scalar_key` is set and there's no profile, use the scalar.
          2. If `profile_key` is a list and `time` is in range, use it.
          3. If `profile_key` exists but `time` is past its end, raise — this
             is a programming error (state advanced past the profile horizon).
          4. Otherwise fall back to the scalar / default.

        Differs from `_profile_at`: this one raises on out-of-range, while
        `_profile_at` silently falls back. Use this when you want bugs to
        surface; use `_profile_at` when graceful degradation is fine.
        """
        if scalar_key in state and profile_key not in state:
            return float(state[scalar_key])
        profile = state.get(profile_key)
        if profile is None:
            return float(state.get(scalar_key, default))
        if time >= len(profile):
            raise IndexError(f"{profile_key} does not contain timestep {time}")
        return float(profile[time])

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))


# Backwards-compatible alias so existing imports still work during migration.
NetworkEngine = Engine
