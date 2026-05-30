from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ParametricControllerParams:
    """Direct action knobs for the dev/debug 'simple' controller.

    Each field maps 1:1 to a key in the action dict the engine consumes.
    Values are passed through as constants every timestep.
    """

    battery_flow_mw: float = 0.0  # MW, + discharge / - charge
    emergency_generator: float = 0.0  # MW, clipped to [0, max_emergency_generator_mw]
    curtail_solar: float = 0.0  # MW of solar to disconnect this step
    fcas_reserve_mw: float = 0.0  # MW of inverter capacity held for FCAS revenue
    subscribe_ids: bool = (
        False  # whether to subscribe to IDS events (if enabled in cybersecurity scenario)
    )


def make_parametric_controller(params: ParametricControllerParams):
    """Return a controller that emits the parameter values verbatim each step."""

    battery_flow_mw = float(params.battery_flow_mw)
    emergency_generator = float(params.emergency_generator)
    curtail_solar = max(0.0, float(params.curtail_solar))
    fcas_reserve_mw = max(0.0, float(params.fcas_reserve_mw))
    subscribe_ids = bool(params.subscribe_ids)

    def controller(_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "battery_flow_mw": battery_flow_mw,
            "emergency_generator": emergency_generator,
            "curtail_solar": curtail_solar,
            "fcas_reserve_mw": fcas_reserve_mw,
            "subscribe_ids": subscribe_ids,
        }

    return controller
