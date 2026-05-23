"""FastAPI HTTP layer for the energy grid simulation."""

from __future__ import annotations

from collections import defaultdict
import os
from typing import Annotated, Any, Literal, Union

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from watt_the_hack.api.sandbox import ControllerCompileError, compile_controller_source
from watt_the_hack.constants import DEFAULT_STEPS
from watt_the_hack.controllers.parametric import (
    ParametricControllerParams,
    make_parametric_controller,
)

from watt_the_hack.data_loaders.scenarios import (
    config_overrides as scenario_config_overrides,
    find_scenario_by_id,
    list_scenarios,
    load_scenario,
    public_metadata,
    scoring_config,
)
from watt_the_hack.engine.engine import Engine, SimulationConfig
from watt_the_hack.metrics.metrics import Metrics


class ParametricControllerParamsModel(BaseModel):
    battery_flow_kw: float = 0.0
    emergency_generator: float = 0.0
    curtail_solar: float = 0.0
    fcas_reserve_kw: float = 0.0


class SimpleControllerSpec(BaseModel):
    kind: Literal["simple"] = "simple"
    params: ParametricControllerParamsModel = Field(
        default_factory=ParametricControllerParamsModel
    )


class CodeControllerSpec(BaseModel):
    kind: Literal["code"] = "code"
    source: str


ControllerSpec = Annotated[
    Union[SimpleControllerSpec, CodeControllerSpec],
    Field(discriminator="kind"),
]


class InitRequest(BaseModel):
    steps: int = DEFAULT_STEPS
    scenario_id: str | None = None  # if set, loads scenario instead of defaults


class InitResponse(BaseModel):
    state: dict[str, Any]
    steps: int
    scenario: dict[str, Any] | None = None  # spec metadata if a scenario was loaded


class ScenarioSummary(BaseModel):
    id: str
    title: str
    pool: str
    archetype: str
    one_liner: str
    path: str
    mechanics: list[str] = Field(default_factory=list)


class StepRequest(BaseModel):
    state: dict[str, Any]
    controller: ControllerSpec = Field(default_factory=SimpleControllerSpec)


class StepResponse(BaseModel):
    state: dict[str, Any]
    outputs: dict[str, Any]
    controller_error: str | None = None


class RunRequest(BaseModel):
    state: dict[str, Any]
    controller: ControllerSpec = Field(default_factory=SimpleControllerSpec)
    steps: int = DEFAULT_STEPS
    scenario_id: str | None = None  # for scoring with the scenario's baselines
    team_id: str | None = None  # used for rate limiting judging scenarios
    team_token: str | None = None  # secret token to prevent team spoofing


class MetricsSummary(BaseModel):
    renewable_ratio: float
    grid_stability: float
    cost: float
    unmet_demand_total: float = 0.0
    final_score: float


class RunResponse(BaseModel):
    final_state: dict[str, Any]
    states: list[dict[str, Any]]
    outputs: list[dict[str, Any]]
    metrics: MetricsSummary
    controller_error: str | None = None


app = FastAPI(title="Watt The Hack Simulation API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


_engine = Engine()
_submission_counts: dict[str, int] = defaultdict(int)

# In production (e.g. on Render), if you want the public to be able to playtest
# the synthetic/tutorial scenarios before the event, set AUTO_UNLOCK=true in your environment variables.
# For the actual hackathon where you incrementally release scenarios, set AUTO_UNLOCK=false.
_auto_unlock = os.getenv("AUTO_UNLOCK", "true").lower() == "true"

_unlocked_scenarios: set[str] = {"t1_welcome"}
if _auto_unlock:
    _unlocked_scenarios.update(s["id"] for s in list_scenarios(include_judging=False))

# Simple API Authentication for Judging
# Replace these with your actual hackathon teams and random passwords
REGISTERED_TEAMS = {"team_alpha": "secret_abc123", "team_beta": "secret_xyz987"}

ADMIN_TOKEN = "hackathon_admin_secret"


class ScenarioUnlockRequest(BaseModel):
    scenario_id: str
    admin_token: str


@app.post("/admin/scenarios/unlock")
def admin_unlock_scenario(req: ScenarioUnlockRequest):
    if req.admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    _unlocked_scenarios.add(req.scenario_id)
    return {"status": "ok", "unlocked": list(_unlocked_scenarios)}


@app.post("/admin/scenarios/lock")
def admin_lock_scenario(req: ScenarioUnlockRequest):
    if req.admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    _unlocked_scenarios.discard(req.scenario_id)
    return {"status": "ok", "unlocked": list(_unlocked_scenarios)}


@app.get("/admin/scenarios/status")
def admin_scenarios_status(admin_token: str):
    if admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")
    all_scenarios = [s["id"] for s in list_scenarios(include_judging=True)]
    return {
        "unlocked": list(_unlocked_scenarios),
        "locked": list(set(all_scenarios) - _unlocked_scenarios),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sim/scenarios", response_model=list[ScenarioSummary])
def sim_scenarios() -> list[ScenarioSummary]:
    # Only return scenarios that are explicitly unlocked by the admin
    return [
        ScenarioSummary(**s) for s in list_scenarios() if s["id"] in _unlocked_scenarios
    ]


@app.post("/sim/init", response_model=InitResponse)
def sim_init(req: InitRequest = InitRequest()) -> InitResponse:
    spec_meta: dict[str, Any] | None = None
    if not req.scenario_id:
        raise HTTPException(
            status_code=400,
            detail="scenario_id is required. The default profile has been removed.",
        )

    if req.scenario_id not in _unlocked_scenarios:
        raise HTTPException(
            status_code=403,
            detail="This scenario is locked or has not been released yet.",
        )

    path = find_scenario_by_id(req.scenario_id)
    if path is None:
        raise ValueError(f"Unknown scenario_id: {req.scenario_id!r}")

    spec, state = load_scenario(path)
    spec_meta = public_metadata(spec)
    # Number of steps comes from the loaded profiles, not the request
    steps = len(state["profiles"]["demand"])

    _engine.add_forecast_to_state(state)

    if _is_judging_scenario(spec_meta.get("pool") if spec_meta else None):
        raise HTTPException(
            status_code=403,
            detail="Judging scenarios cannot be initialized step-by-step. Use /sim/run for full evaluation.",
        )

    _prepare_state_for_response(state)
    return InitResponse(state=state, steps=steps, scenario=spec_meta)


@app.post("/sim/step", response_model=StepResponse)
def sim_step(req: StepRequest) -> StepResponse:
    controller_fn, controller_error = _resolve_controller(req.controller)
    controller_state = _state_visible_to_controller(req.state)
    try:
        action = controller_fn(controller_state)
    except Exception as exc:
        action = _fallback_controller()(controller_state)
        controller_error = controller_error or f"Runtime error: {exc}"

    engine_state = _rehydrate_state_for_engine(req.state)
    new_state, outputs = _engine.step(engine_state, action)

    scenario_id = req.state.get("scenario_id")
    if scenario_id and _is_judging_scenario_by_id(scenario_id):
        raise HTTPException(
            status_code=403,
            detail="Judging scenarios cannot be stepped. Use /sim/run for full evaluation.",
        )

    _prepare_state_for_response(new_state)
    return StepResponse(
        state=new_state, outputs=outputs, controller_error=controller_error
    )


@app.post("/sim/run", response_model=RunResponse)
def sim_run(req: RunRequest) -> RunResponse:
    controller_fn, controller_error = _resolve_controller(req.controller)
    fallback = _fallback_controller()

    is_judging = False
    scoring = {}
    overrides = {}
    path = None
    scenario_id = req.scenario_id or req.state.get("scenario_id")
    if scenario_id:
        if scenario_id not in _unlocked_scenarios:
            raise HTTPException(
                status_code=403,
                detail="This scenario is locked or has not been released yet.",
            )
        path = find_scenario_by_id(scenario_id)
        if path is not None:
            spec, _ = load_scenario(path)
            scoring = scoring_config(spec)
            overrides = scenario_config_overrides(spec)
            is_judging = spec.get("pool") == "judging"

    # Build a per-run engine if the scenario has physics overrides, otherwise
    # reuse the shared global instance (no object allocation cost for normal runs).
    run_engine = Engine(config=SimulationConfig(**overrides)) if overrides else _engine

    dt_hours = getattr(run_engine, "dt_hours", run_engine.config.dt_hours)
    metrics = Metrics(
        dt_hours=dt_hours,
        baselines={**Metrics().baselines, **scoring.get("baselines", {})},
    )

    if is_judging:
        team = req.team_id
        if not team:
            raise HTTPException(
                status_code=401, detail="team_id is required for judging scenarios."
            )

        expected_token = REGISTERED_TEAMS.get(team)
        if not expected_token:
            raise HTTPException(
                status_code=401,
                detail=f"Unregistered team_id: '{team}'. Please contact the organizers.",
            )

        if req.team_token != expected_token:
            raise HTTPException(
                status_code=401, detail="Invalid team_token. Authentication failed."
            )

        if _submission_counts[team] >= 3:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded: max 3 judging submissions per team.",
            )
        _submission_counts[team] += 1

    state = _rehydrate_state_for_engine(req.state)
    # The browser may send a controller-visible state with future traces hidden.
    # Rehydrate engine-only scenario fields before physics, then strip them
    # again from every response snapshot.
    if is_judging and path is not None:
        _, full_state = load_scenario(path)
        run_engine.add_forecast_to_state(full_state)
        state = full_state

    states: list[dict[str, Any]] = []
    outputs_list: list[dict[str, Any]] = []

    for _ in range(req.steps):
        controller_state = _state_visible_to_controller(state)
        try:
            action = controller_fn(controller_state)
        except Exception as exc:
            controller_error = controller_error or f"Runtime error: {exc}"
            action = fallback(controller_state)

        state, outputs = run_engine.step(state, action)
        metrics.update(state, outputs)

        # Save copies for the response
        out_state = dict(state)
        out_outputs = dict(outputs)

        if is_judging:
            _strip_judging_data(out_state, out_outputs)

        _prepare_state_for_response(out_state)
        states.append(out_state)
        outputs_list.append(out_outputs)

    final_state = dict(state)
    if is_judging:
        _strip_judging_data(final_state)
        # Security: Do not return step-by-step history for judging scenarios
        states = []
        outputs_list = []

    _prepare_state_for_response(final_state)

    return RunResponse(
        final_state=final_state,
        states=states,
        outputs=outputs_list,
        metrics=MetricsSummary(**metrics.summary()),
        controller_error=controller_error,
    )


def _fallback_controller():
    return make_parametric_controller(ParametricControllerParams())


def _resolve_controller(spec: ControllerSpec):
    if isinstance(spec, SimpleControllerSpec):
        params = ParametricControllerParams(
            battery_flow_kw=spec.params.battery_flow_kw,
            emergency_generator=spec.params.emergency_generator,
            curtail_solar=spec.params.curtail_solar,
            fcas_reserve_kw=spec.params.fcas_reserve_kw,
        )
        return make_parametric_controller(params), None

    try:
        return compile_controller_source(spec.source), None
    except ControllerCompileError as exc:
        return _fallback_controller(), str(exc)


def _is_judging_scenario(pool: str | None) -> bool:
    return pool == "judging"


def _is_judging_scenario_by_id(scenario_id: str) -> bool:
    path = find_scenario_by_id(scenario_id)
    if path:
        spec, _ = load_scenario(path)
        return _is_judging_scenario(spec.get("pool"))
    return False


def _strip_internal_state(state: dict[str, Any]) -> None:
    """Strip internal fields that should never be sent to the client."""
    state.pop("_ar1_cache", None)
    state.pop("forecast_config", None)
    state.pop("attack_windows", None)


def _prepare_state_for_response(state: dict[str, Any]) -> None:
    """Strip internal and scenario-hidden fields before returning state."""
    _strip_internal_state(state)
    
    # ALWAYS strip the ground-truth profiles so controllers cannot cheat by reading future steps.
    state.pop("profiles", None)
    state.pop("price_profile", None)
    
    visibility = _controller_visibility(state)
    if visibility.get("events") is False:
        state.pop("events", None)


def _state_visible_to_controller(state: dict[str, Any]) -> dict[str, Any]:
    """Return exactly what participant code should see this step."""
    visible = dict(state)
    _prepare_state_for_response(visible)
    return visible


def _rehydrate_state_for_engine(state: dict[str, Any]) -> dict[str, Any]:
    """Restore scenario internals stripped from controller-visible state."""
    scenario_id = state.get("scenario_id")
    if not scenario_id:
        return state

    path = find_scenario_by_id(str(scenario_id))
    if path is None:
        return state

    _, scenario_state = load_scenario(path)
    engine_state = dict(state)
    for key in (
        "profiles",
        "price_profile",
        "events",
        "forecast_config",
        "attack_windows",
        "features",
    ):
        if key in scenario_state:
            engine_state[key] = scenario_state[key]
    return engine_state


def _controller_visibility(state: dict[str, Any]) -> dict[str, Any]:
    scenario_id = state.get("scenario_id")
    if not scenario_id:
        return {}

    path = find_scenario_by_id(str(scenario_id))
    if path is None:
        return {}

    spec, _ = load_scenario(path)
    return dict(spec.get("controller_visibility", {}) or {})


def _strip_judging_data(
    state: dict[str, Any], outputs: dict[str, Any] | None = None
) -> None:
    """Strip scenario data from judging scenarios to prevent probing."""
    state.pop("profiles", None)
    state.pop("price_profile", None)
    state.pop("forecast", None)
    if outputs is not None:
        outputs.pop("import_price", None)
        outputs.pop("export_price", None)
