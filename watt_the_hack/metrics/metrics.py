"""Metrics accumulation and scoring.

The headline leaderboard score is raw run cost: lower cost wins, so
``summary()["final_score"]`` equals ``cost_sum``. Each scenario may
override the baselines via its `scoring` block (see
docs/creating-scenarios.md); when no override is provided we fall back
to DEFAULT_BASELINES.

``component_scores`` is kept as a pure helper (pinned by the contract
test) for ad-hoc tooling, but the live pipeline only reports the raw
cost sum.
"""

from dataclasses import dataclass, field

# Generous defaults for the playground / unscored runs. Per-scenario
# baselines computed by `scripts/freeze_baselines.py` should override these.
DEFAULT_BASELINES = {
    "cost": 1200.0,
    "stability_abs": 8000.0,
    "unmet": 50.0,
    "renewable": 0.35,
}


@dataclass
class Metrics:
    """Accumulates raw metrics each step, then derives a final score.

    `grid_stability_sum` is `-Σ(Δgrid_power)²` — the same shape as the
    engine's priced `ramp_charge` component (just unsigned and unscaled).
    It is kept separate so the contract test's stability vector and the
    cost-based ramp charge can evolve independently.
    """

    dt_hours: float = 0.25
    baselines: dict = field(default_factory=lambda: dict(DEFAULT_BASELINES))

    demand_served_locally_sum: float = 0.0
    demand_sum: float = 0.0
    grid_stability_sum: float = 0.0
    cost_sum: float = 0.0
    unmet_demand_sum: float = 0.0
    previous_grid_power: float | None = None

    def update(self, state: dict, outputs: dict) -> None:
        demand_kw = float(state.get("demand", 0.0))
        grid_power_kw = float(outputs.get("net_grid_power", 0.0))
        emergency_kw = float(outputs.get("emergency_generator", 0.0))
        unmet_demand_kw = float(outputs.get("unmet_demand", 0.0))
        step_cost = float(outputs.get("step_cost", 0.0))

        # Renewable serving = demand met by clean local sources (solar +
        # battery). Diesel is local but dirty, so it's excluded just like
        # grid imports.
        clean_local_kw = max(0.0, demand_kw - max(0.0, grid_power_kw) - emergency_kw)
        self.demand_served_locally_sum += clean_local_kw
        self.demand_sum += demand_kw

        if self.previous_grid_power is not None:
            ramp_kw = grid_power_kw - self.previous_grid_power
            self.grid_stability_sum -= ramp_kw**2
        self.previous_grid_power = grid_power_kw

        self.cost_sum += step_cost
        self.unmet_demand_sum += unmet_demand_kw * self.dt_hours

    def summary(self) -> dict:
        """Return raw metrics. ``final_score`` is the accumulated dollar cost."""
        renewable_ratio = (
            self.demand_served_locally_sum / self.demand_sum
            if self.demand_sum > 0.0
            else 0.0
        )
        return {
            "renewable_ratio": float(renewable_ratio),
            "grid_stability": float(self.grid_stability_sum),
            "cost": float(self.cost_sum),
            "unmet_demand_total": float(self.unmet_demand_sum),
            "final_score": float(self.cost_sum),
        }


def component_scores(
    *,
    renewable_ratio: float,
    grid_stability: float,
    cost: float,
    unmet_demand_total: float,
    baselines: dict,
) -> dict[str, float]:
    """All component scores in [0, 1]. 1.0 = perfect, 0.0 = baseline (do-nothing).

    Not used by the live pipeline (lowest-cost-wins). Retained for the
    contract test and any future leaderboard tooling that wants
    normalised per-axis scores.
    """
    cost_baseline = max(float(baselines.get("cost", DEFAULT_BASELINES["cost"])), 1.0)
    stab_baseline = max(
        float(baselines.get("stability_abs", DEFAULT_BASELINES["stability_abs"])), 1.0
    )
    unmet_baseline = max(float(baselines.get("unmet", DEFAULT_BASELINES["unmet"])), 1.0)
    renew_baseline = float(baselines.get("renewable", DEFAULT_BASELINES["renewable"]))

    # Renewable score is what the controller *added* on top of the natural
    # solar passthrough: 0 if you matched the do-nothing renewable share,
    # 1 if you reached 100% renewable serving.
    renew_headroom = max(1.0 - renew_baseline, 1e-6)
    renew_score = (renewable_ratio - renew_baseline) / renew_headroom

    return {
        "cost": _clamp01(1.0 - cost / cost_baseline),
        "renewable": _clamp01(renew_score),
        "stability": _clamp01(1.0 - abs(grid_stability) / stab_baseline),
        "reliability": _clamp01(1.0 - unmet_demand_total / unmet_baseline),
    }


def _clamp01(value: float) -> float:
    if value != value:  # NaN check
        return 0.0
    return max(0.0, min(1.0, value))
