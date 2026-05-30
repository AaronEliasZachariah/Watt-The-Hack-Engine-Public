"""Scenario loading.

A scenario file is JSON (will become YAML once we add PyYAML) that follows
the schema documented in docs/creating-scenarios.md. This loader resolves
the spec into an `initial_state` dict ready for the engine, plus the spec
itself for the API layer.

For this MVP only synthetic scenarios are wired up. AEMO loading is a
stub that the data team will fill in (download/preprocess/inspect commands
in watt_the_hack/data_loaders/aemo.py).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from watt_the_hack.engine.engine import SimulationConfig

SCENARIOS_ROOT = (
    Path(os.environ["SCENARIOS_DATA_DIR"])
    if "SCENARIOS_DATA_DIR" in os.environ
    else Path(__file__).resolve().parents[2] / "scenarios"
)

UNIT_SCALE = 1000.0

# Feature-key → mechanic-id mapping, with the engine's default-when-absent.
# Mirrors `_gate_features` in engine.py: omit the features dict and everything
# is enabled. New backend mechanics that gate via `features` get one row.
_FEATURE_MECHANICS: tuple[tuple[str, str, bool], ...] = (
    # (feature_key, mechanic_id, default_when_absent)
    ("battery", "battery", True),
    ("emergency_generator", "diesel", True),
    ("curtailment", "curtailment", True),
    ("fcas", "fcas", True),
)


def load_scenario(path: str | Path) -> tuple[dict, dict]:
    """Load a scenario file and produce (spec, initial_state).

    spec          — the full parsed scenario (id, title, narrative, forecast config, ...)
    initial_state — ready to hand to the engine; contains profiles + price_profile
    """
    spec = _read_spec(path)

    data_source = spec.get("data_source", "synthetic")
    if data_source == "synthetic":
        profiles = _load_synthetic(spec["synthetic"])
    elif data_source == "aemo":
        profiles = _load_aemo(spec["aemo"])
    else:
        raise ValueError(f"Unknown data_source: {data_source!r}")

    demand = profiles["demand"]
    solar = profiles["solar"]
    price = [float(p) * UNIT_SCALE for p in profiles["price"]] if profiles.get("price") else []

    # IMPORTANT: keys prefixed with `_` are ENGINE-INTERNAL. They carry the
    # full demand/solar/price profiles, the unredacted event list, attack
    # windows with corruption magnitudes, and the forecast noise parameters
    # (including the persistent bias `mu_*` and the RNG `seed`). Exposing any
    # of these to a controller leaks future intent and breaks the game.
    #
    # Controllers receive a filtered view via Engine.controller_view(state),
    # which keeps only the public surface (current scalars, bounded forecast,
    # features, agent_plan, alerts, bookkeeping).
    initial_state = {
        "time": 0,
        "demand": demand[0] if demand else 0.0,
        "solar": solar[0] if solar else 0.0,
        "soc": 0.50,
        "price": float(price[0]) if price else 0.0,
        "scenario_id": spec.get("id"),
        "alerts": [],  # populated per step by the engine; never spoilers
        # Engine-internal — DO NOT add these to controller_view's allowlist
        "_profiles_full": {"demand": demand, "solar": solar, "price": price},
        "_price_profile_full": price,
        "_events_full": _normalise_events(spec.get("events", [])),
        "_forecast_config_full": dict(spec["forecast"]) if "forecast" in spec else None,
        "_attack_windows_full": spec.get("attack_windows", []),
        "ids_cost_per_step": float(spec.get("ids_cost_per_step", 0.0)) * UNIT_SCALE,
    }
    # Optional throughput budget: total |MWh| the battery may move across
    # the whole run. Absent (or null) = unlimited (engine ignores it).
    throughput_budget = spec.get("battery_throughput_kwh_budget")
    if throughput_budget is not None:
        initial_state["battery_throughput_remaining_mwh"] = float(throughput_budget)
        initial_state["battery_throughput_budget_mwh"] = float(throughput_budget)
    # Inject feature flags if the scenario declares them.  When absent,
    # the engine treats all features as enabled (backwards compatible).
    features = spec.get("features")
    if features is not None:
        initial_state["features"] = dict(features)
    return spec, initial_state


def public_metadata(spec: dict) -> dict:
    """Return the subset of a scenario spec that is safe to expose to the
    frontend / spectator UI. Strips internal data-source bodies and any
    private forecast settings (e.g. seed for judging scenarios).

    Frontend developers consume this — it's the contract for the live UI.
    The frontend renders entirely from this payload, so adding a new
    scenario (or changing duration / start hour / mechanics on an existing
    one) requires no frontend changes.
    """
    forecast = dict(spec.get("forecast", {}))
    forecast.pop("seed", None)  # seeds are secret, even in metadata

    # Resolve the engine config the scenario will actually run under, so the
    # frontend can show correct clock cadence, hardware caps, and penalty
    # rates instead of hardcoded defaults.
    cfg = SimulationConfig(**config_overrides(spec))

    return {
        "id": spec.get("id"),
        "title": spec.get("title"),
        "archetype": spec.get("archetype"),
        "pool": spec.get("pool"),
        "narrative": spec.get("narrative", {}),
        "events": _normalise_events(spec.get("events", [])),
        "forecast": {
            "horizon_steps": forecast.get("horizon_steps"),
        },
        "scoring": scoring_config(spec),
        "mechanics": scenario_mechanics(spec),
        "controller_modes": spec.get("controller_modes"),
        "duration_steps": _scenario_duration_steps(spec),
        "start_hour": float(spec.get("start_hour", 0.0)),
        "dt_hours": cfg.dt_hours,
        "limits": {
            "battery_capacity_mwh": cfg.battery_capacity_mwh,
            "max_inverter_mw": cfg.max_inverter_mw,
            "grid_max_import_mw": cfg.grid_max_import_mw,
            "grid_max_export_mw": cfg.grid_max_export_mw,
            "max_emergency_generator_mw": cfg.max_emergency_generator_mw,
        },
        "penalties": {
            "blackout_per_mwh": cfg.blackout_penalty_per_mwh,
            "overvoltage_per_mwh": cfg.overvoltage_penalty_per_mwh,
            "diesel_per_mwh": cfg.emergency_generator_cost_per_mwh,
            "battery_wear_per_mwh": cfg.battery_wear_cost_per_mwh,
            "demand_charge_per_mw": cfg.demand_charge_per_mw,
            "export_tariff_per_mwh": cfg.export_tariff,
            "fcas_revenue_per_kw_per_hour": cfg.fcas_revenue_per_kw_per_hour,
        },
    }


def scenario_mechanics(spec: dict) -> list[dict]:
    """Return the ordered list of mechanics active in this scenario.

    Derived from the scenario's existing declarations (`features`,
    `forecast`, `battery_throughput_kwh_budget`, `attack_windows`).
    No parallel schema field to maintain — adding a new scenario that
    reuses existing mechanics requires only the standard JSON edit.

    Frontend renders each via its mechanic catalog (lib/mechanics.ts).
    Unknown mechanic ids are tolerated by the frontend (rendered with a
    minimal default) so a backend-side WIP feature flag doesn't break
    the UI for other scenarios.

    Order is display order. Keep it stable: feature-gated mechanics
    first, then scenario-config-derived ones.
    """
    out: list[dict[str, Any]] = []
    features = spec.get("features")
    features_dict = features if isinstance(features, dict) else {}

    # Engine convention: an absent `features` dict means everything enabled.
    # Match that here so the UI reflects what the engine will actually accept.
    features_absent = not isinstance(features, dict)

    for feature_key, mechanic_id, default_on in _FEATURE_MECHANICS:
        if features_absent:
            active = default_on
        else:
            active = bool(features_dict.get(feature_key, default_on))
        if active:
            out.append({"id": mechanic_id})

    # Any unknown feature flag the scenario sets to true gets surfaced too.
    # Lets the other devs prototype new mechanics (e.g. `price_lag`) and
    # have the frontend at least acknowledge them.
    known_keys = {k for k, _, _ in _FEATURE_MECHANICS}
    for key, value in features_dict.items():
        if key not in known_keys and value:
            out.append({"id": key})

    if spec.get("forecast"):
        cfg = spec["forecast"]
        out.append(
            {
                "id": "forecast",
                "config": {"horizon_steps": cfg.get("horizon_steps")},
            }
        )

    if spec.get("battery_throughput_kwh_budget") is not None:
        out.append(
            {
                "id": "throughput_budget",
                "config": {"mwh": float(spec["battery_throughput_kwh_budget"])},
            }
        )

    if spec.get("attack_windows"):
        out.append({"id": "cyber_attack"})

    # Surface `forecast_bias` as a mechanic when the scenario declares any
    # event of that type. Lets the frontend briefing card flag "the forecast
    # will lie to you" without inspecting individual events.
    if any(
        (ev.get("type") == "forecast_bias")
        for ev in (spec.get("events") or [])
    ):
        out.append({"id": "forecast_bias"})

    return out


def _scenario_duration_steps(spec: dict) -> int:
    """Resolve the run length in timesteps.

    Prefer the inline profile length (the engine's source of truth);
    fall back to the declared `steps` field for non-inline scenarios.
    """
    synthetic = spec.get("synthetic", {}) or {}
    if synthetic.get("mode") == "inline":
        profiles = synthetic.get("profiles", {}) or {}
        demand = profiles.get("demand") or []
        if demand:
            return len(demand)
    declared = synthetic.get("steps")
    if isinstance(declared, int) and declared > 0:
        return declared
    # Last-resort default matches the legacy 24h/15min assumption.
    return 96


def scoring_config(spec: dict) -> dict:
    """Extract the scenario's scoring block.

    ``baselines`` is the only field consumed by the live scoring pipeline
    (the leaderboard sorts by raw cost, so only ``baselines.cost`` is
    actively meaningful — the other baselines feed ``component_scores``
    for ad-hoc tooling).

    ``baseline_breakdown`` is per-component do-nothing totals written by
    ``scripts/freeze_baselines.py``. It is no longer displayed in the
    UI but is preserved as historical data and for any future tooling.

    Missing keys fall back to ``DEFAULT_BASELINES`` in the Metrics layer.
    """
    raw = spec.get("scoring", {}) or {}
    return {
        "baselines": dict(raw.get("baselines", {})),
        "baseline_breakdown": dict(raw.get("baseline_breakdown", {})),
    }


def config_overrides(spec: dict) -> dict:
    """Extract per-scenario SimulationConfig field overrides.

    Scenarios may include a ``config_overrides`` block to tune the engine
    physics for that specific scenario (e.g. raising demand_charge_per_mw
    or zeroing fcas_revenue_per_kw_per_hour).  Returns an empty dict if
    the block is absent — safe to unpack directly into SimulationConfig.

    Example usage in server.py::

        overrides = config_overrides(spec)
        engine = Engine(config=SimulationConfig(**overrides))
    """
    return dict(spec.get("config_overrides", {}) or {})


_EVENT_CORE_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "type",
        "severity",
        "at_step",
        "end_step",
        "title",
        "description",
        "icon",
    }
)


def _normalise_events(raw_events: list[dict]) -> list[dict]:
    """Validate and normalise the events list. Defaults end_step = at_step
    (point event) and severity = 'info' if missing.

    Unknown / mechanic-specific fields (e.g. ``sigma_multiplier`` on
    ``weather_anomaly`` events, ``channel`` + ``bias`` on ``forecast_bias``
    events) are passed through verbatim. The engine reads them out of
    ``state["_events_full"]`` when processing the corresponding mechanic
    (this list is engine-private and never reaches controllers). Without
    this passthrough, scenario-authored overrides would be silently dropped.
    """
    out = []
    for ev in raw_events or []:
        at_step = int(ev["at_step"])
        end_step = int(ev.get("end_step", at_step))
        normalised: dict[str, Any] = {
            "id": ev.get("id", f"event_{at_step}"),
            "type": ev.get("type", "other"),
            "severity": ev.get("severity", "info"),
            "at_step": at_step,
            "end_step": end_step,
            "title": ev.get("title", ""),
            "description": ev.get("description", ""),
            "icon": ev.get("icon"),  # optional — frontend maps to its icon set
        }
        for key, value in ev.items():
            if key not in _EVENT_CORE_FIELDS:
                if ev.get("type") == "phishing_trap" and key == "penalty":
                    normalised[key] = float(value) * UNIT_SCALE
                else:
                    normalised[key] = value
        out.append(normalised)
    return out


def list_scenarios(
    root: Path | None = None, include_judging: bool = False
) -> list[dict]:
    """Return a summary list of all scenarios under scenarios/. Used by the API."""
    root = root or SCENARIOS_ROOT
    if not root.exists():
        return []
    summaries = []
    for path in sorted(root.rglob("*.json")):
        try:
            spec = _read_spec(path)
        except Exception:
            continue

        pool = spec.get("pool", "synthetic")
        if not include_judging and pool == "judging":
            continue

        summaries.append(
            {
                "id": spec.get("id", path.stem),
                "title": spec.get("title", path.stem),
                "pool": pool,
                "archetype": spec.get("archetype", ""),
                "one_liner": spec.get("narrative", {}).get("one_liner", ""),
                "path": str(path.relative_to(root.parent)),
                # Lightweight mechanic ids so the frontend can derive the
                # "unlocked in X" progression in a single round trip rather
                # than calling /sim/init per scenario.
                "mechanics": [m["id"] for m in scenario_mechanics(spec)],
            }
        )
    return summaries


def find_scenario_by_id(scenario_id: str, root: Path | None = None) -> Path | None:
    """Look up a scenario file by its `id` field."""
    root = root or SCENARIOS_ROOT
    for summary in list_scenarios(root, include_judging=True):
        if summary["id"] == scenario_id:
            return root.parent / summary["path"]
    return None


# ---------- internals ----------


def _read_spec(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_synthetic(synthetic: dict) -> dict[str, list[float]]:
    mode = synthetic.get("mode", "inline")
    if mode == "inline":
        profiles = synthetic["profiles"]
        return {
            "demand": list(profiles["demand"]),
            "solar": list(profiles["solar"]),
            "price": list(profiles["price"]),
        }
    if mode == "sidecar":
        sidecar_path = Path(synthetic["file"])
        with sidecar_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "demand": list(data["demand"]),
            "solar": list(data["solar"]),
            "price": list(data["price"]),
        }
    if mode == "generator":
        # Reserved for parametric synthetic scenarios. Wire to
        # watt_the_hack.data_loaders.synthetic_generators.SYNTHETIC_GENERATORS
        # once that registry exists.
        raise NotImplementedError(
            "Generator-mode synthetic scenarios not yet implemented. "
            "See docs/creating-scenarios.md §4.3 for the design."
        )
    raise ValueError(f"Unknown synthetic.mode: {mode!r}")


def _load_aemo(aemo: dict) -> dict[str, list[float]]:
    """Stub. The data team will fill this in.

    Expected behaviour: read watt_the_hack/data/aemo/{region}_{date}.parquet,
    return {'demand': [...], 'solar': [...], 'price': [...]}.

    For now this raises so missing implementation is loud, not silent.
    """
    raise NotImplementedError(
        "AEMO loader is not yet implemented. See watt_the_hack/data_loaders/aemo.py "
        "for the planned download / preprocess / inspect commands. Until that "
        "lands, use data_source='synthetic' in your scenario JSON."
    )
