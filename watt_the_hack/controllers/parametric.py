from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ParametricControllerParams:
    """Direct action knobs for the dev/debug 'simple' controller.

    Each field maps 1:1 to a key in the action dict the engine consumes.
    Values are passed through as constants every timestep.
    """

    battery_flow_kw: float = 0.0  # kW, + discharge / - charge
    emergency_generator: float = 0.0  # kW, clipped to [0, max_emergency_generator_kw]
    curtail_solar: float = 0.0  # kW of solar to disconnect this step
    fcas_reserve_kw: float = 0.0  # kW of inverter capacity held for FCAS revenue


def make_parametric_controller(params: ParametricControllerParams):
    """Return a controller that emits the parameter values verbatim each step."""

    battery_flow_kw = float(params.battery_flow_kw)
    emergency_generator = float(params.emergency_generator)
    curtail_solar = max(0.0, float(params.curtail_solar))
    fcas_reserve_kw = max(0.0, float(params.fcas_reserve_kw))

    def controller(_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "battery_flow_kw": battery_flow_kw,
            "emergency_generator": emergency_generator,
            "curtail_solar": curtail_solar,
            "fcas_reserve_kw": fcas_reserve_kw,
        }

    return controller
