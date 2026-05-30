import random
from dataclasses import dataclass, field
from typing import Any

from watt_the_hack.engine.base_engine import SimulationEngine


@dataclass(slots=True)
class PhysicsResult:
    """Output of one physics step. Named fields >>> tuple unpacking."""

    next_soc: float
    battery_mw: float  # actual dispatch after inverter + SOC + FCAS clipping
    emergency_generator_mw: float  # actual diesel after [0, max] clip
    curtailed_solar_mw: float  # actual curtailment after clamp to available solar
    net_grid_power: float  # +import / -export, after grid limit clipping
    unmet_demand: float  # MW above the import limit (blackout)
    overvoltage_mw: float  # MW below the negative export limit (overvoltage)
    fcas_reserve_mw: float  # actual capacity reserved for FCAS revenue
    fcas_dispatch_required_mw: float = 0.0
    fcas_dispatch_delivered_mw: float = 0.0
    fcas_shortfall_mw: float = 0.0


@dataclass(slots=True)
class SimulationConfig:
    # 1. Rebalanced Game Board
    battery_capacity_mwh: float = 100.0
    max_inverter_mw: float = 50.0
    grid_max_import_mw: float = 120.0
    grid_max_export_mw: float = 50.0  # NEW: Export limit

    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    dt_hours: float = 0.25

    # FCAS Dispatch Penalties and Bonuses
    fcas_shortfall_penalty_per_mwh: float = 20000.0
    fcas_dispatch_bonus_per_mwh: float = 200.0
    fcas_micro_cycling_factor: float = 0.01  # MWh of throughput per MW of reserve per hour
    fcas_ramp_penalty_per_mw: float = 500.0


    # Phase 3 Realism: Split Pricing & Penalties
    export_tariff: float = 50.0  # $50.00 flat rate for exporting solar (was 0.05)
    blackout_penalty_per_mwh: float = 10000.00
    emergency_generator_cost_per_mwh: float = 1000.00
    max_emergency_generator_mw: float = 50.0
    overvoltage_penalty_per_mwh: float = 5000.00  # NEW: Penalty for exporting too much

    # Battery wear: each MWh moved through the battery (charge or discharge)
    # eats a fraction of its lifetime. Calibrated to ~$50/MWh throughput,
    # matching real Li-ion replacement cost (~$400,000/MWh capital, ~4000 cycles).
    # Forces controllers to value cycles, not just spam them.
    battery_wear_cost_per_mwh: float = 50.0

    # Demand charge: $/MW based on the HIGHEST single-step import seen in the
    # run. Real commercial bills do this monthly — one big spike costs you
    # for the whole period, not just the spike step. Forces peak-shaving
    # discipline distinct from "just don't blackout".
    # Billed incrementally: each step, only the *new* peak above the prior
    # peak is charged, so total = peak_import_mw * demand_charge_per_mw.
    demand_charge_per_mw: float = 1000.0

    # Carbon price: charges every kg of CO2 emitted from imports + diesel.
    # Real-world calibration:
    #   - AU carbon price ~$50/tonne AUD = $0.05/kg -> $50/kg in city-scale
    #   - NSW/QLD grid intensity ~0.7 kg CO2/MWh (fossil-heavy)
    #   - Diesel intensity ~0.27 kg CO2/MWh (fixed by chemistry)
    # Exports earn nothing on carbon — you're sending clean power TO the grid.
    # Scenarios can override grid_co2_intensity via state["grid_co2_intensity"]
    # (e.g., 0.05 for Tasmania hydro, 0.8 for QLD coal).
    carbon_price_per_kg: float = 50.0
    grid_co2_intensity_kg_per_mwh: float = 0.7
    diesel_co2_intensity_kg_per_mwh: float = 0.27

    # Ramp charge: quadratic penalty on changes in net grid power between
    # steps. Real-world equivalent: AEMO's FCAS markets pay for smoothness.
    # cost = (grid_power[t] - grid_power[t-1])^2 * rate
    #   - 50 MW ramp → 2500 × 1.0 = $2500
    #   - 100 MW ramp → 10000 × 1.0 = $10000
    #   - 10 MW ramp → 100 × 1.0 = $100
    # Quadratic shape rewards smooth dispatch disproportionately over jagged.
    # First step has no prior grid power, so its ramp charge is zero.
    ramp_charge_per_kw2: float = 1.0

    # FCAS (Frequency Control Ancillary Services) reserve: passive revenue for
    # holding inverter capacity available for the grid. Real-world: AEMO pays
    # batteries to be ready to respond to frequency events whether or not
    # they're called. Hornsdale (Tesla SA) earns ~$10M/year on FCAS alone.
    #
    # The trade-off, made obvious to the controller:
    #   |battery_flow_mw| + fcas_reserve_mw <= max_inverter_mw
    # Every MW you commit to FCAS is a MW you cannot use for arbitrage.
    #
    # Calibration: $40.00/MW/hour ≈ real AEMO contingency-FCAS rates at city scale. Caps the
    # max revenue from full FCAS reservation at ~$48,000/day on a 50 MW inverter,
    # comparable to good arbitrage. Neither strategy dominates the other.
    fcas_revenue_per_kw_per_hour: float = 40.0

    # ---------------- Compliance mechanic (S5+) ----------------
    # Operators occasionally issue directives that constrain dispatch
    # (e.g. AEMO Reserve Trader: "hold ≥30% SOC reserve until 22:00").
    # Real-world: these directives arrive as prose to a human operator,
    # who interprets them into operational limits. In the simulator,
    # qualitative_alert events carry the prose; the controller's
    # agent_plan (typically populated by an LLM strategy layer) carries
    # the extracted numerical constraints. The engine reads the latter.
    #
    # Three constraint keys are recognised under state["agent_plan"]:
    #   compliance_window: [start_step, end_step] — when the constraints
    #     are active. If absent, the other keys are ignored (no
    #     compliance enforcement that step).
    #   min_soc_floor: float in [0,1] — penalise per-step SOC shortfall
    #     below this level.
    #   max_export_kw_override: float — penalise net exports above this
    #     ceiling (i.e. abs(net_grid_power) above this when exporting).
    #
    # Penalties are MODERATE — well above battery wear (~$50/MWh) so
    # ignoring them is costly, but well below blackouts ($10000/MWh) so a
    # controller can prefer compliance breach to load shedding in a
    # genuine emergency. This is the design: compliance is a soft
    # operational constraint, not a hard physics one.
    compliance_soc_penalty_per_unit: float = 200000.00  # $/SOC-unit short, per step
    compliance_export_penalty_per_mw: float = 50000.00  # $/MW exceeded, per step

    # ---------------- Diesel ban + exemption mechanic ----------------
    # Scenarios may declare ``diesel_ban_window`` events (with a
    # ``directive_id`` and an active step range). During an active ban
    # the engine still lets the controller fire diesel — physics is
    # physics — but charges a per-MWh penalty unless the controller
    # has submitted a valid emergency exemption in ``agent_plan``.
    #
    # The exemption itself is a small structured document the LLM must
    # COMPOSE (not just parse). The acceptor validates:
    #   * ``directive_id`` exactly matches an active ban event
    #   * ``reason`` is a string with ≥ ``min_exemption_reason_chars``
    #     non-whitespace characters (forces an actual justification,
    #     not a single-token "yes")
    #   * ``expected_duration_steps`` is a positive int ≤ ``max_exemption_duration_steps``
    #
    # This is the "creative-reasoning" mechanic: a regex extractor can
    # pull the ``directive_id`` out of the announcing alert text, but
    # can't compose a credible 60-char justification anchored to the
    # current operational context. An LLM does both naturally.
    diesel_ban_penalty_per_mwh: float = 3000.00  # $/MWh of diesel during a ban with no exemption
    min_exemption_reason_chars: int = 60
    max_exemption_duration_steps: int = 12

    # Forecast configuration (lookahead with growing noise)
    forecast_horizon: int = 16  # how many future steps the controller sees
    forecast_sigma_demand: float = 3.0  # additive noise std (MW)
    forecast_sigma_price: float = 20.0  # additive noise std ($/MWh)
    forecast_solar_noise_pct: float = (
        0.12  # multiplicative noise std (fraction of actual solar)
    )
    forecast_mu_demand: float = 0.0  # persistent bias (MW)
    forecast_mu_price: float = 0.0  # persistent bias ($/MWh)
    forecast_solar_mu_pct: float = 0.0  # persistent bias (fraction of actual solar)
    forecast_ar1_rho: float = 0.7  # AR(1) autocorrelation coefficient for error drift
    forecast_seed: int | None = None  # set for reproducible noise; None = random


# Controller-visible state surface. Engine.controller_view filters every
# state dict down to these keys before it reaches participant code.
#
# Anything outside this set is engine bookkeeping or scenario ground truth.
# Adding a key here is a deliberate decision to expose new information to
# controllers; never widen this set without thinking through the cheat path
# (e.g. exposing a full profile lets a controller perfectly forecast the run).
#
# Companion private keys carry the same data for engine internals:
#   _profiles_full         — full demand/solar series
#   _price_profile_full    — full price series
#   _events_full           — every scenario event with all fields
#   _attack_windows_full   — every cyber attack window with corruption_scale
#   _forecast_config_full  — sigma/mu/seed/horizon (mu and seed are spoilers)
#   _ar1_cache             — forecast RNG state for fast-forwarding
_CONTROLLER_PUBLIC_KEYS: frozenset[str] = frozenset(
    {
        # Clock
        "time",
        # Current realised values
        "demand",
        "solar",
        "price",
        "soc",
        # Bounded forecast (already horizon-limited by _build_forecast)
        "forecast",
        # Rules of the game
        "features",
        "scenario_id",
        "grid_co2_intensity",
        # Bookkeeping the controller earned and may want to read
        "peak_import_mw",
        "prev_grid_power_mw",
        "prev_fcas_reserve_mw",
        "battery_throughput_remaining_mwh",
        "battery_throughput_budget_mwh",
        "fcas_events_upcoming",

        # Channels for agentic strategies
        "agent_plan",
        # Currently-firing qualitative alerts (redacted; no bias/sigma payload)
        "alerts",
        # Numerical probability signal indicating attack likelihood if subscribed
        "ids_signal",
    }
)


@dataclass(slots=True)
class Engine(SimulationEngine):
    """Single node MVP engine with local storage, solar, emergency generation, and curtailment."""

    config: SimulationConfig = field(default_factory=SimulationConfig)

    @staticmethod
    def controller_view(state: dict) -> dict:
        """Return the subset of ``state`` safe to expose to controllers.

        Drops full demand/solar/price profiles, the unredacted event list,
        attack windows, forecast noise parameters, AR(1) cache, and any
        other engine-internal bookkeeping that would leak future intent.

        Use this at every controller boundary (admin runtime, browser
        sandbox, headless runner). Engine.step continues to read from the
        full state dict — never pass a controller_view back to step().
        """
        return {k: v for k, v in state.items() if k in _CONTROLLER_PUBLIC_KEYS}

    # ------------------------------------------------------------------
    # State accessors — prefer the private `_*_full` keys (set by the
    # scenario loader) but fall back to public legacy keys so ad-hoc
    # tests and notebooks that hand-build a state dict still work.
    # ------------------------------------------------------------------

    @staticmethod
    def _full_profile_series(state: dict, key: str) -> list[float] | None:
        profiles = state.get("_profiles_full") or state.get("profiles") or {}
        series = profiles.get(key)
        return list(series) if series is not None else None

    @staticmethod
    def _full_price_profile(state: dict) -> list[float] | None:
        series = state.get("_price_profile_full")
        if series is None:
            series = state.get("price_profile")
        return list(series) if series is not None else None

    @staticmethod
    def _full_events(state: dict) -> list[dict]:
        events = state.get("_events_full")
        if events is None:
            events = state.get("events")
        return list(events or [])

    @staticmethod
    def _full_attack_windows(state: dict) -> list[dict]:
        windows = state.get("_attack_windows_full")
        if windows is None:
            windows = state.get("attack_windows")
        return list(windows or [])

    @staticmethod
    def _full_forecast_config(state: dict) -> dict | None:
        if "_forecast_config_full" in state:
            return state["_forecast_config_full"]
        return state.get("forecast_config")

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
        demand_mw, solar_mw, soc, import_price = self._read_inputs(state, time)

        # 2. Feature gate — scenarios declare which actions are available.
        #    Missing features dict = all features on (backwards compatible).
        action = self._gate_features(state, action)

        # 3. Physics
        physics = self._physics_step(
            action,
            demand_mw,
            solar_mw,
            soc,
            battery_throughput_remaining_mwh=state.get(
                "battery_throughput_remaining_mwh"
            ),
        )

        # 4. Market
        cost_breakdown = self._compute_market(
            state, time, import_price, physics, action
        )

        # 5. Outputs
        outputs = self._build_outputs(physics, import_price, cost_breakdown)

        # 6. State for t+1
        new_state = self._advance_state(state, time, physics, import_price, action)

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
            gated["battery_flow_mw"] = 0.0

        if not features.get("curtailment", True):
            gated["curtail_solar"] = 0.0

        if not features.get("emergency_generator", True):
            gated["emergency_generator"] = 0.0

        if not features.get("fcas", True):
            gated["fcas_reserve_mw"] = 0.0

        if not features.get("ids", True):
            gated["subscribe_ids"] = (
                False  # Control disable/enable of IDS (Intrusion Detection System) subscription in cybersecurity scenario
            )

        # Future features (flexible_loads, forecast_purchasing) will be
        # gated here once their engine logic is implemented.

        return gated

    # ------------------------------------------------------------------
    # Phase helpers — each does one thing
    # ------------------------------------------------------------------

    def _read_inputs(self, state: dict, time: int) -> tuple[float, float, float, float]:
        """Pull the four scalars the engine needs at time t."""
        demand_mw = self._series_at(
            self._full_profile_series(state, "demand"),
            time,
            state.get("_demand_true", state.get("demand", 0.0)),
        )
        solar_mw = self._series_at(
            self._full_profile_series(state, "solar"),
            time,
            state.get("_solar_true", state.get("solar", 0.0)),
        )
        soc = self._clip(float(state.get("_soc_true", state.get("soc", 0.0))), 0.0, 1.0)
        import_price = self._resolve_import_price(state, time)
        return demand_mw, solar_mw, soc, import_price

    def _resolve_import_price(self, state: dict, time: int) -> float:
        """Look up the price for ``time``, preferring the full price profile.

        Mirrors the old ``_state_value`` semantics: raise on out-of-range
        when a profile is present (programming error), fall back to the
        scalar otherwise.
        """
        profile = self._full_price_profile(state)
        if profile is None:
            return float(state.get("price", 0.0))
        if time >= len(profile):
            raise IndexError(
                f"price profile does not contain timestep {time}"
            )
        return float(profile[time])

    def _compute_market(
        self,
        state: dict,
        time: int,
        import_price: float,
        physics: PhysicsResult,
        action: dict,
    ) -> dict:
        """Resolve per-scenario market params and run the cost calculation."""
        prev_peak = float(state.get("peak_import_mw", 0.0))
        new_peak = max(prev_peak, max(0.0, physics.net_grid_power))
        grid_co2 = float(
            state.get(
                "grid_co2_intensity",
                self.config.grid_co2_intensity_kg_per_mwh,
            )
        )
        # Sentinel: missing on first step → ramp charge is 0
        prev_grid_power = state.get("prev_grid_power_mw")

        soc_after = float(physics.next_soc)
        events_full = self._full_events(state)
        compliance = self._compliance_breach(
            events_full,
            time,
            soc_after,
            physics.net_grid_power,
        )
        subscribe_ids = bool(action.get("subscribe_ids", False))
        ids_cost_per_step = float(state.get("ids_cost_per_step", 0.0))

        # Diesel-ban: charge a per-MWh penalty if diesel ran inside an
        # active ban window AND no valid exemption is held in agent_plan.
        diesel_ban_penalty_mwh = self._diesel_ban_penalty_mwh(
            events_full,
            time,
            state.get("agent_plan"),
            physics.emergency_generator_mw,
        )

        phishing_fine = self._phishing_fine(
            events_full,
            time,
            state.get("agent_plan"),
            state.setdefault("_phishing_traps_charged", set()),
        )
        
        # Calculate FCAS Dispatch Logic (mutate physics next_soc)
        required_mw = 0.0
        for ev in events_full:
            if ev.get("type") == "fcas_dispatch":
                at_step = int(ev.get("at_step", -1))
                end_step = int(ev.get("end_step", at_step))
                if at_step <= time <= end_step:
                    required_mw += float(ev.get("magnitude_mw", 0.0))
        
        deliverable_mw = 0.0
        actual_delivery = 0.0
        shortfall = 0.0
        
        if required_mw > 0:
            # How much can we sustain for 1 hour from current SOC?
            soc_backed_capacity = (soc_after * self.config.battery_capacity_mwh * self.config.discharge_efficiency) / 1.0 
            deliverable_mw = min(physics.fcas_reserve_mw, soc_backed_capacity)
            actual_delivery = min(deliverable_mw, required_mw)
            shortfall = max(0.0, required_mw - actual_delivery)
            
            # Reduce SOC by the energy delivered
            if actual_delivery > 0:
                physics.next_soc -= (actual_delivery * self.config.dt_hours) / (
                    self.config.battery_capacity_mwh * self.config.discharge_efficiency
                )
                physics.next_soc = max(0.0, physics.next_soc)
                
        physics.fcas_dispatch_required_mw = required_mw
        physics.fcas_dispatch_delivered_mw = actual_delivery
        physics.fcas_shortfall_mw = shortfall

        return self._market_step(
            net_grid_power=physics.net_grid_power,
            import_price=import_price,
            unmet_demand=physics.unmet_demand,
            emergency_generator_mw=physics.emergency_generator_mw,
            overvoltage_mw=physics.overvoltage_mw,
            battery_mw=physics.battery_mw,
            fcas_reserve_mw=physics.fcas_reserve_mw,
            fcas_dispatch_delivered_mw=actual_delivery,
            fcas_shortfall_mw=shortfall,
            new_peak_import_mw=new_peak,
            prev_peak_import_mw=prev_peak,
            grid_co2_intensity=grid_co2,
            prev_grid_power_mw=prev_grid_power,
            compliance_soc_shortfall=compliance["soc_shortfall"],
            compliance_export_excess_mw=compliance["export_excess_mw"],
            subscribe_ids=subscribe_ids,
            ids_cost_per_step=ids_cost_per_step,
            diesel_ban_penalty_mwh=diesel_ban_penalty_mwh,
            phishing_fine=phishing_fine,
            prev_fcas_reserve_mw=float(state.get("prev_fcas_reserve_mw", 0.0)),
        )

    def _diesel_ban_penalty_mwh(
        self,
        events: list[dict],
        time: int,
        agent_plan: dict | None,
        diesel_mw: float,
    ) -> float:
        """Return the diesel-ban-penalty MWh quantity to charge this step.

        Returns 0 unless ALL of:
          - a ``diesel_ban_window`` event is active at ``time``,
          - the controller ran diesel (``diesel_mw > 0``),
          - and ``agent_plan["emergency_exemption"]`` does NOT match the
            acceptor criteria for the active ban's ``directive_id``.

        The amount returned is ``diesel_mw * dt_hours`` so the caller can
        multiply by ``$/MWh`` and produce a clean cost line.
        """
        if diesel_mw <= 0:
            return 0.0
        active_id: str | None = None
        for ev in events:
            if ev.get("type") != "diesel_ban_window":
                continue
            at_step = int(ev.get("at_step", -1))
            end_step = int(ev.get("end_step", at_step))
            if at_step <= time <= end_step:
                active_id = ev.get("directive_id")
                break
        if active_id is None:
            return 0.0  # no ban active — diesel is free to run

        if self._exemption_valid(agent_plan, active_id):
            return 0.0

        return diesel_mw * self.config.dt_hours

    @staticmethod
    def _phishing_fine(
        events: list[dict],
        time: int,
        agent_plan: dict | None,
        traps_charged: set[str],
    ) -> float:
        """Return one-shot fine for falling into a phishing trap at this step.
        Mutates traps_charged to avoid double-charging.
        """
        fine = 0.0
        if not isinstance(agent_plan, dict):
            return fine
        for ev in events:
            if ev.get("type") != "phishing_trap":
                continue
            at_step = int(ev.get("at_step", -1))
            if time != at_step:
                continue
            trap_id = ev.get("id", f"trap_{at_step}")
            if trap_id in traps_charged:
                continue
            bait_key = ev.get("bait_key")
            if bait_key and bait_key in agent_plan:
                bait_val = ev.get("bait_value")
                if bait_val is None or agent_plan[bait_key] == bait_val:
                    fine += float(ev.get("penalty", 0.0))
                    traps_charged.add(trap_id)
        return fine

    # Operational vocabulary the acceptor expects to see in the
    # ``reason`` field. Lowercase substring match — case-insensitive
    # by lowering the reason before testing. A credible justification
    # mentions at least one of these. Pure canned text without
    # operational vocabulary is rejected.
    _EXEMPTION_OPERATIONAL_KEYWORDS: tuple[str, ...] = (
        "mw",
        "soc",
        "demand",
        "deficit",
        "import",
        "capacity",
        "peak",
        "battery",
        "generation",
    )

    def _exemption_valid(self, agent_plan: dict | None, directive_id: str) -> bool:
        """Check ``agent_plan["emergency_exemption"]`` against acceptor
        criteria. Returns True only if every field is present, well-typed,
        and within bounds. Defensive parsing — never raises on malformed
        controller output.

        Acceptance criteria for the ``reason`` field, in order:
          1. Non-empty string.
          2. ≥ ``min_exemption_reason_chars`` non-whitespace characters.
          3. Contains at least one digit (a quantitative reference).
          4. Contains at least one operational keyword
             (case-insensitive substring match — see
             ``_EXEMPTION_OPERATIONAL_KEYWORDS``).

        Criteria (3)+(4) together make it impractical to satisfy with a
        static template. They force the reason to anchor to specific
        operational vocabulary AND a numeric quantity, which is the
        marker of a credible operator submission.
        """
        if not isinstance(agent_plan, dict):
            return False
        ex = agent_plan.get("emergency_exemption")
        if not isinstance(ex, dict):
            return False
        if ex.get("directive_id") != directive_id:
            return False
        reason = ex.get("reason")
        if not isinstance(reason, str):
            return False
        non_ws_chars = sum(1 for c in reason if not c.isspace())
        if non_ws_chars < self.config.min_exemption_reason_chars:
            return False
        if not any(c.isdigit() for c in reason):
            return False
        reason_lower = reason.lower()
        if not any(mw in reason_lower for mw in self._EXEMPTION_OPERATIONAL_KEYWORDS):
            return False
        duration = ex.get("expected_duration_steps")
        # Reject booleans — they're a subclass of int in Python and we
        # don't want True/False to sneak through as "1"/"0".
        if isinstance(duration, bool) or not isinstance(duration, int):
            return False
        if duration < 1 or duration > self.config.max_exemption_duration_steps:
            return False
        return True

    @staticmethod
    def _compliance_breach(
        events: list[dict],
        time: int,
        soc_after: float,
        net_grid_power_mw: float,
    ) -> dict[str, float]:
        """Return per-step compliance shortfalls implied by any active
        ``compliance_window`` events.

        Compliance directives are SCENARIO-DECLARED (in ``_events_full``)
        and ENGINE-ENFORCED — they fire whether or not the controller
        opts in. This is the load-bearing LLM mechanic:

          * The corresponding qualitative_alert announces the directive
            (and its values, in prose) ahead of when the
            compliance_window itself activates. A controller that reads
            the alert has many timesteps' notice to position SOC.
          * A controller that ignores the alert sees nothing different
            in state["alerts"] on the window's first step — the engine
            just starts charging penalties.

        Penalty values are per-step:
          * SOC shortfall: ``(floor - soc) * compliance_soc_penalty_per_unit``
          * Export excess: ``(export_mw - cap) * dt * compliance_export_penalty_per_mw``

        Multiple active windows accumulate.
        """
        soc_shortfall = 0.0
        soc_excess = 0.0
        export_excess = 0.0
        export_mw = max(0.0, -net_grid_power_mw)

        for ev in events:
            if ev.get("type") != "compliance_window":
                continue
            at_step = int(ev.get("at_step", -1))
            end_step = int(ev.get("end_step", at_step))
            if not (at_step <= time <= end_step):
                continue

            floor = ev.get("min_soc_floor")
            multiplier = float(ev.get("penalty_multiplier", 1.0))
            if isinstance(floor, (int, float)):
                soc_shortfall += max(0.0, float(floor) - soc_after) * multiplier

            ceiling = ev.get("max_soc_ceiling")
            if isinstance(ceiling, (int, float)):
                soc_excess += max(0.0, soc_after - float(ceiling)) * multiplier

            cap = ev.get("max_export_kw_override")
            if isinstance(cap, (int, float)) and export_mw > 0:
                export_excess += max(0.0, export_mw - float(cap)) * multiplier

        return {
            "soc_shortfall": soc_shortfall + soc_excess,
            "export_excess_mw": export_excess,
        }

    def _build_outputs(
        self,
        physics: PhysicsResult,
        import_price: float,
        cost_breakdown: dict,
    ) -> dict:
        return {
            "net_grid_power": physics.net_grid_power,
            "unmet_demand": physics.unmet_demand,
            "overvoltage_mw": physics.overvoltage_mw,
            "battery_dispatch": physics.battery_mw,
            "emergency_generator": physics.emergency_generator_mw,
            "curtailed_solar": physics.curtailed_solar_mw,
            "fcas_reserve": physics.fcas_reserve_mw,
            "fcas_dispatch_required": physics.fcas_dispatch_required_mw,
            "fcas_dispatch_delivered": physics.fcas_dispatch_delivered_mw,
            "fcas_shortfall": physics.fcas_shortfall_mw,
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
        action: dict,
    ) -> dict:
        """Return a NEW state dict for t+1. Carries forward bookkeeping
        (peak import, prev grid power) and aligns the scalar mirrors
        (demand/solar/price) with the profile at t+1.
        """
        next_time = time + 1
        new_state = dict(state)
        new_state["time"] = next_time
        new_state["_soc_true"] = float(physics.next_soc)
        new_state["soc"] = self._corrupt_sensor(new_state, "soc", float(physics.next_soc), next_time)

        # Bookkeeping for cost components that span steps
        new_state["peak_import_mw"] = max(
            float(state.get("peak_import_mw", 0.0)),
            max(0.0, physics.net_grid_power),
        )
        new_state["prev_grid_power_mw"] = float(physics.net_grid_power)
        new_state["prev_fcas_reserve_mw"] = float(physics.fcas_reserve_mw)

        # Throughput budget: decrement by |MWh| moved through the battery
        # this step. Only tracked when the scenario opted in (initial value
        # is not None).
        dt = self.config.dt_hours
        total_throughput_this_step = (
            abs(physics.battery_mw) +
            (physics.fcas_reserve_mw * self.config.fcas_micro_cycling_factor) +
            physics.fcas_dispatch_delivered_mw
        ) * dt

        budget = state.get("battery_throughput_remaining_mwh")
        if budget is not None:
            new_state["battery_throughput_remaining_mwh"] = max(
                0.0,
                float(budget) - total_throughput_this_step,
            )

        # Mirror profiles → top-level scalars for controllers to read.
        # Read from the private full series; the public `profiles` key is
        # no longer present in scenario-loaded states.
        demand_series = self._full_profile_series(state, "demand")
        if demand_series is not None and next_time < len(demand_series):
            true_demand = float(demand_series[next_time])
            new_state["_demand_true"] = true_demand
            new_state["demand"] = self._corrupt_sensor(new_state, "demand", true_demand, next_time)

        solar_series = self._full_profile_series(state, "solar")
        if solar_series is not None and next_time < len(solar_series):
            true_solar = float(solar_series[next_time])
            new_state["_solar_true"] = true_solar
            new_state["solar"] = self._corrupt_sensor(new_state, "solar", true_solar, next_time)

        # Keep state["price"] aligned with price_profile[t+1] for the
        # controller view. The engine itself still reads from the full
        # profile directly inside _read_inputs.
        new_state["price"] = self._price_at_timestep(new_state, next_time, import_price)

        # Refresh the forecast only for scenarios that opted into forecasts.
        if self._forecast_enabled(new_state):
            new_state["forecast"] = self._build_forecast(new_state, next_time)
        else:
            new_state.pop("forecast", None)

        # Refresh currently-firing qualitative alerts. This is the ONLY
        # channel through which controllers see event content — the full
        # event list lives in `_events_full` and is engine-only.
        new_state["alerts"] = self._current_alerts(state, next_time)

        # IDS signal — noisy attack probability hint if controller subscribed
        attack_windows = self._full_attack_windows(state)
        in_attack = any(
            w["start_step"] <= next_time <= w["end_step"] for w in attack_windows
        )
        if action.get("subscribe_ids", False):
            ids_rng = random.Random(f"{next_time}_ids_{state.get('scenario_id', '')}")
            if in_attack:
                raw = ids_rng.gauss(0.75, 0.15)
            else:
                raw = ids_rng.gauss(0.15, 0.10)
            new_state["ids_signal"] = max(0.0, min(1.0, raw))
        else:
            new_state["ids_signal"] = None
            
        new_state["fcas_events_upcoming"] = self._future_fcas_events(state, next_time)

        return new_state

    # Event types whose prose is exposed to the controller via
    # state["alerts"]. The set EXCLUDES engine-internal enforcement
    # types (compliance_window, phishing_trap, diesel_ban_window) that
    # exist solely to drive penalties — leaking those would let a
    # controller read the structured constraint values that the LLM
    # is supposed to extract from the prose. All narrative event
    # types — qualitative_alert, forecast_bias announcements, weather
    # notes, demand spikes — go through.
    _CONTROLLER_VISIBLE_EVENT_TYPES: frozenset[str] = frozenset(
        {
            "qualitative_alert",
            "forecast_bias",
            "forecast_error",
            "weather_anomaly",
            "weather",
            "demand_spike",
            "demand",
            "price_signal",
            "price_peak",
            "signal",
            "info",
            "other",
        }
    )

    def _current_alerts(self, state: dict, time: int) -> list[dict]:
        """Return events ACTIVE at ``time``, redacted to narrative fields.

        Includes every scenario event type on the controller-visible
        allowlist (qualitative alerts, forecast_bias announcements,
        weather notes, demand spikes). Excludes engine-internal types
        (compliance_window, phishing_trap, diesel_ban_window).

        Strips every per-event spoiler field on the way out: ``bias``,
        ``channel``, ``sigma_multiplier``, ``corruption_scale``,
        ``min_soc_floor``, ``max_export_kw_override``, ``directive_id``,
        ``bait_key``, ``bait_value``, ``penalty``. Only narrative
        metadata survives the strip.
        """
        visible_fields = (
            "id",
            "type",
            "severity",
            "at_step",
            "end_step",
            "title",
            "description",
            "icon",
        )
        out: list[dict] = []
        for ev in self._full_events(state):
            if ev.get("type") not in self._CONTROLLER_VISIBLE_EVENT_TYPES:
                continue
            at_step = int(ev.get("at_step", -1))
            end_step = int(ev.get("end_step", at_step))
            if at_step <= time <= end_step:
                out.append({k: ev.get(k) for k in visible_fields if k in ev})
        return out
        
    def _future_fcas_events(self, state: dict, time: int) -> list[dict]:
        """Return upcoming fcas_dispatch events to controllers."""
        out = []
        for ev in self._full_events(state):
            if ev.get("type") == "fcas_dispatch":
                at_step = int(ev.get("at_step", -1))
                end_step = int(ev.get("end_step", at_step))
                if end_step >= time:
                    out.append({
                        "at_step": at_step,
                        "end_step": end_step,
                        "magnitude_mw": float(ev.get("magnitude_mw", 0.0))
                    })
        return sorted(out, key=lambda x: x["at_step"])

    def add_forecast_to_state(self, state: dict) -> dict:
        """Inject state["forecast"] and state["alerts"] for the current
        timestep. Call once at init, before the first engine.step().
        """
        time = int(state.get("time", 0))
        
        # Initialize true state values and apply sensor corruption for t=0
        if "_soc_true" not in state:
            state["_soc_true"] = float(state.get("soc", 0.0))
        state["soc"] = self._corrupt_sensor(state, "soc", state["_soc_true"], time)
        
        demand_series = self._full_profile_series(state, "demand")
        if demand_series is not None and time < len(demand_series):
            true_demand = float(demand_series[time])
            state["_demand_true"] = true_demand
            state["demand"] = self._corrupt_sensor(state, "demand", true_demand, time)
            
        solar_series = self._full_profile_series(state, "solar")
        if solar_series is not None and time < len(solar_series):
            true_solar = float(solar_series[time])
            state["_solar_true"] = true_solar
            state["solar"] = self._corrupt_sensor(state, "solar", true_solar, time)

        state["price"] = self._price_at_timestep(
            state, time, float(state.get("price", 0.2))
        )
        if self._forecast_enabled(state):
            state["forecast"] = self._build_forecast(state, time)
        else:
            state.pop("forecast", None)
        # Surface any qualitative alerts firing at t=0 so the controller's
        # plan()/first step() call can react to them.
        state["alerts"] = self._current_alerts(state, time)
        state["fcas_events_upcoming"] = self._future_fcas_events(state, time)
        return state

    def _corrupt_sensor(self, state: dict, channel: str, val: float, time: int) -> float:
        """Applies False Data Injection (FDI) or noise to sensor telemetry."""
        events = self._full_events(state)
        seed = self.config.forecast_seed
        if seed is None:
            seed = hash(state.get("scenario_id", "default")) % 1_000_000

        corrupted_val = val
        for ev in events:
            if ev.get("type") == "sensor_fdi" and ev.get("channel") == channel:
                if ev.get("at_step", 0) <= time <= ev.get("end_step", 0):
                    bias = float(ev.get("bias", 0.0))
                    noise_sigma = float(ev.get("noise_sigma", 0.0))
                    
                    if noise_sigma > 0:
                        import random
                        rng = random.Random(f"{seed}_fdi_{channel}_{time}")
                        corrupted_val += rng.gauss(0.0, noise_sigma)
                        
                    corrupted_val += bias
                    
                    scale = float(ev.get("scale", 1.0))
                    corrupted_val *= scale

        if channel == "soc":
            return self._clip(corrupted_val, 0.0, 1.0)
        return max(0.0, corrupted_val)

    @classmethod
    def _forecast_enabled(cls, state: dict) -> bool:
        """Scenario-loaded states set forecast_config=None to disable forecasts.

        Missing forecast_config remains enabled for backwards-compatible tests
        and ad-hoc engine use.
        """
        # Sentinel-aware: a key present and set to None means "explicitly off".
        if "_forecast_config_full" in state:
            return state["_forecast_config_full"] is not None
        if "forecast_config" in state:
            return state["forecast_config"] is not None
        return True

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

        sources = {
            "demand": self._full_profile_series(state, "demand"),
            "solar": self._full_profile_series(state, "solar"),
            "price": self._full_price_profile(state),
        }

        forecast: dict[str, list[float]] = {}
        forecast_config = self._full_forecast_config(state)
        if forecast_config is None:
            return forecast
        if not isinstance(forecast_config, dict):
            forecast_config = {}
        events = self._full_events(state)
        attack_windows = self._full_attack_windows(state)

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
                #   - demand: additive in MW.
                #     -20 → forecast under-predicts by 20 MW
                #   - price: additive in $/MWh.
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
        demand_mw: float,
        solar_mw: float,
        soc: float,
        battery_throughput_remaining_mwh: float | None = None,
    ) -> PhysicsResult:
        """Apply battery + generator + curtailment, then clip the resulting
        net grid power against the import/export limits. Returns named
        fields rather than a positional tuple — much easier to read at
        the call site.
        """
        cfg = self.config

        # FCAS reserve gets first claim on the inverter. Any MW reserved
        # for FCAS is unavailable for arbitrage this step.
        fcas_reserve_mw = self._clip(
            float(action.get("fcas_reserve_mw", 0.0)),
            0.0,
            cfg.max_inverter_mw,
        )
        battery_inverter_budget = cfg.max_inverter_mw - fcas_reserve_mw

        # Throughput budget: scenarios may cap the total |MWh| moved
        # through the battery across the run. When set, the remaining
        # budget further clips this step's dispatch magnitude.
        if battery_throughput_remaining_mwh is not None:
            # We subtract the FCAS micro-cycling portion of the budget
            # because even if the battery doesn't dispatch for arbitrage,
            # holding reserve cycles the inverter/cells.
            fcas_cycling_cost = (
                fcas_reserve_mw * cfg.fcas_micro_cycling_factor * cfg.dt_hours
            )
            budget_kw_cap = max(
                0.0, (float(battery_throughput_remaining_mwh) - fcas_cycling_cost) / cfg.dt_hours
            )
            battery_inverter_budget = min(battery_inverter_budget, budget_kw_cap)

        # Battery: clip to remaining inverter capacity (after FCAS) and SOC bounds
        requested_battery_mw = float(action.get("battery_flow_mw", 0.0))
        battery_mw = self._feasible_battery_power(
            requested_battery_mw,
            soc,
            inverter_limit=battery_inverter_budget,
        )
        next_soc = self._next_soc(soc, battery_mw)

        # Diesel: simple [0, max] clip
        emergency_generator_mw = self._clip(
            float(action.get("emergency_generator", 0.0)),
            0.0,
            cfg.max_emergency_generator_mw,
        )

        # Curtailment: can't curtail more than the available solar
        curtailed_solar_mw = self._clip(
            float(action.get("curtail_solar", 0.0)),
            0.0,
            solar_mw,
        )
        actual_solar_mw = solar_mw - curtailed_solar_mw

        # Power balance — what the grid has to make up
        net_grid_power = (
            demand_mw - actual_solar_mw - battery_mw - emergency_generator_mw
        )

        # Clip against grid import/export limits, capturing any overflow
        unmet_demand = 0.0
        overvoltage_mw = 0.0
        if net_grid_power > cfg.grid_max_import_mw:
            unmet_demand = net_grid_power - cfg.grid_max_import_mw
            net_grid_power = cfg.grid_max_import_mw
        elif net_grid_power < -cfg.grid_max_export_mw:
            overvoltage_mw = abs(net_grid_power) - cfg.grid_max_export_mw
            net_grid_power = -cfg.grid_max_export_mw

        return PhysicsResult(
            next_soc=next_soc,
            battery_mw=battery_mw,
            emergency_generator_mw=emergency_generator_mw,
            curtailed_solar_mw=curtailed_solar_mw,
            net_grid_power=net_grid_power,
            unmet_demand=unmet_demand,
            overvoltage_mw=overvoltage_mw,
            fcas_reserve_mw=fcas_reserve_mw,
        )

    def _market_step(
        self,
        *,
        net_grid_power: float,
        import_price: float,
        unmet_demand: float,
        emergency_generator_mw: float,
        overvoltage_mw: float,
        battery_mw: float,
        fcas_reserve_mw: float,
        prev_fcas_reserve_mw: float,
        fcas_dispatch_delivered_mw: float = 0.0,
        fcas_shortfall_mw: float = 0.0,
        prev_peak_import_mw: float,
        new_peak_import_mw: float,
        grid_co2_intensity: float,
        prev_grid_power_mw: float | None,
        compliance_soc_shortfall: float = 0.0,
        compliance_export_excess_mw: float = 0.0,
        subscribe_ids: bool = False,
        ids_cost_per_step: float = 0.0,
        diesel_ban_penalty_mwh: float = 0.0,
        phishing_fine: float = 0.0,
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
        energy_mwh = net_grid_power * dt

        # Tariff is split into import and export lines so the dashboard can
        # tell a player how much they earned in exports vs paid in imports
        # (a single net value masks the partition for any mixed day).
        if energy_mwh > 0:
            tariff_import = energy_mwh * import_price  # positive, cost
            tariff_export = 0.0
        else:
            tariff_import = 0.0
            tariff_export = energy_mwh * cfg.export_tariff  # negative, revenue

        # Carbon: imports (positive grid power) and diesel both emit. Exports
        # are clean power leaving the city — they don't earn carbon credit
        # here, just the export tariff.
        import_mwh = max(0.0, energy_mwh)
        diesel_mwh = emergency_generator_mw * dt
        co2_kg = (
            import_mwh * grid_co2_intensity
            + diesel_mwh * cfg.diesel_co2_intensity_kg_per_mwh
        )

        # Ramp: quadratic penalty on the change in net grid power. First
        # step has no prior reference, so its ramp charge is zero.
        if prev_grid_power_mw is None:
            ramp_charge = 0.0
        else:
            ramp_mw = net_grid_power - prev_grid_power_mw
            ramp_charge = (ramp_mw**2) * cfg.ramp_charge_per_kw2

        components = {
            "tariff_import": tariff_import,
            "tariff_export": tariff_export,
            "generator_fuel": emergency_generator_mw
            * dt
            * cfg.emergency_generator_cost_per_mwh,
            "blackout_penalty": unmet_demand * dt * cfg.blackout_penalty_per_mwh,
            "overvoltage_penalty": overvoltage_mw
            * dt
            * cfg.overvoltage_penalty_per_mwh,
            "battery_wear": abs(battery_mw) * dt * cfg.battery_wear_cost_per_mwh,
            # Demand charge: incremental — only the rise above prior peak is billed
            # this step, so the running cumulative charge equals (peak * rate).
            "demand_charge": max(0.0, new_peak_import_mw - prev_peak_import_mw)
            * cfg.demand_charge_per_mw,
            "carbon_cost": co2_kg * cfg.carbon_price_per_kg,
            "ramp_charge": ramp_charge,
            # FCAS revenue: NEGATIVE cost (income) for capacity held available.
            "fcas_revenue": -fcas_reserve_mw * dt * cfg.fcas_revenue_per_kw_per_hour,
            # FCAS ramp: penalty for volatility in reserve capacity
            "fcas_ramp_charge": abs(fcas_reserve_mw - prev_fcas_reserve_mw) * cfg.fcas_ramp_penalty_per_mw,
            # Compliance: zero by default; only positive when the controller
            # opted in via agent_plan AND breached the bound it set.
            "compliance_penalty": (
                compliance_soc_shortfall * cfg.compliance_soc_penalty_per_unit
                + compliance_export_excess_mw
                * dt
                * cfg.compliance_export_penalty_per_mw
            ),
            # IDS cost: flat fee per step when controller subscribes to the
            # intrusion detection signal. Only active on cybersecurity scenario.
            "ids_cost": ids_cost_per_step if subscribe_ids else 0.0,
            # Diesel-ban penalty: per-MWh charge when diesel runs inside an
            # active ban window without a valid agent_plan exemption.
            # Zero everywhere else.
            "diesel_ban_penalty": (
                diesel_ban_penalty_mwh * cfg.diesel_ban_penalty_per_mwh
            ),
            "fcas_dispatch_bonus": -fcas_dispatch_delivered_mw * dt * cfg.fcas_dispatch_bonus_per_mwh,
            "fcas_shortfall_penalty": fcas_shortfall_mw * dt * cfg.fcas_shortfall_penalty_per_mwh,
            "phishing_fine": phishing_fine,
        }

        return {
            **{k: float(v) for k, v in components.items()},
            "total": float(sum(components.values())),
        }

    def _feasible_battery_power(
        self,
        requested_mw: float,
        soc: float,
        inverter_limit: float | None = None,
    ) -> float:
        """Clip a requested battery dispatch to:
        1. The inverter limit (default = max_inverter_mw, but FCAS reserve
           can shrink the effective budget for this step).
        2. The energy available in the battery (discharge can't exceed
           what's stored; charge can't exceed remaining headroom).
        """
        cfg = self.config
        limit = (
            cfg.max_inverter_mw if inverter_limit is None else max(0.0, inverter_limit)
        )
        clipped_mw = self._clip(requested_mw, -limit, limit)

        if clipped_mw > 0.0:
            max_discharge_mw = (
                soc * cfg.battery_capacity_mwh * cfg.discharge_efficiency
            ) / cfg.dt_hours
            return min(clipped_mw, max_discharge_mw)

        if clipped_mw < 0.0:
            headroom_mwh = (1.0 - soc) * cfg.battery_capacity_mwh
            max_charge_mw = headroom_mwh / (cfg.charge_efficiency * cfg.dt_hours)
            return max(clipped_mw, -max_charge_mw)

        return 0.0

    def _next_soc(self, soc: float, battery_mw: float) -> float:
        if battery_mw > 0.0:
            next_soc = soc - (battery_mw * self.config.dt_hours) / (
                self.config.battery_capacity_mwh * self.config.discharge_efficiency
            )
        elif battery_mw < 0.0:
            next_soc = (
                soc
                - (battery_mw * self.config.charge_efficiency * self.config.dt_hours)
                / self.config.battery_capacity_mwh
            )
        else:
            next_soc = soc

        return self._clip(next_soc, 0.0, 1.0)

    @classmethod
    def _price_at_timestep(
        cls, state: dict[str, Any], timestep: int, fallback: float
    ) -> float:
        """Expose the same tariff the engine uses so state['price'] matches
        the full price profile at ``timestep``. Falls back to the scalar
        when no profile is present (legacy tests).
        """
        profile = cls._full_price_profile(state)
        if profile:
            idx = max(0, min(timestep, len(profile) - 1))
            return float(profile[idx])
        return float(state.get("price", fallback))

    @staticmethod
    def _series_at(series: list[float] | None, time: int, fallback: float) -> float:
        """Look up ``series[time]`` with a scalar fallback when missing."""
        if series is not None and time < len(series):
            return float(series[time])
        return float(fallback)

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))


# Backwards-compatible alias so existing imports still work during migration.
NetworkEngine = Engine
