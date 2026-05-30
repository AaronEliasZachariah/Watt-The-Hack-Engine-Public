from dataclasses import dataclass
from typing import Any

from watt_the_hack.engine.base_engine import SimulationEngine


@dataclass(slots=True)
class SimpleEnergyEngine(SimulationEngine):
    """MVP energy-system engine with battery and grid-import constraints."""

    battery_capacity_mwh: float = 20.0
    max_inverter_mw: float = 10.0
    grid_max_import_mw: float = 25.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    dt_hours: float = 0.25

    def step(self, state: dict, action: dict) -> tuple[dict, dict]:
        t = int(state.get("t", 0))
        soc = self._clip(float(state.get("soc", 0.5)), 0.0, 1.0)

        demand_mw = self._state_value(state, "demand", "demand_profile", t, 0.0)
        solar_mw = self._state_value(state, "solar", "solar_profile", t, 0.0)
        price = self._state_value(state, "price", "price_profile", t, 0.0)

        requested_battery_mw = float(action.get("battery_flow_mw", 0.0))
        battery_mw = self._feasible_battery_power(requested_battery_mw, soc)
        next_soc = self._next_soc(soc, battery_mw)

        grid_power_mw = demand_mw - solar_mw - battery_mw
        unmet_demand_mw = 0.0
        if grid_power_mw > self.grid_max_import_mw:
            grid_power_mw = self.grid_max_import_mw
            unmet_demand_mw = demand_mw - (
                solar_mw + battery_mw + self.grid_max_import_mw
            )

        outputs = {
            "grid_power": float(grid_power_mw),
            "battery_flow_mw": float(battery_mw),
            "solar_used": float(max(0.0, min(solar_mw, demand_mw - unmet_demand_mw))),
            "unmet_demand": float(max(0.0, unmet_demand_mw)),
            "price": float(price),
        }

        new_state = dict(state)
        new_state.update(
            {
                "t": t + 1,
                "soc": float(next_soc),
                "last_demand": float(demand_mw),
                "last_solar": float(solar_mw),
                "last_price": float(price),
            }
        )

        return new_state, outputs

    def _feasible_battery_power(self, requested_mw: float, soc: float) -> float:
        clipped_mw = self._clip(
            requested_mw,
            -self.max_inverter_mw,
            self.max_inverter_mw,
        )

        if clipped_mw > 0.0:
            max_discharge_mw = (
                soc * self.battery_capacity_mwh * self.discharge_efficiency
            ) / self.dt_hours
            return min(clipped_mw, max_discharge_mw)

        if clipped_mw < 0.0:
            headroom_mwh = (1.0 - soc) * self.battery_capacity_mwh
            max_charge_mw = headroom_mwh / (self.charge_efficiency * self.dt_hours)
            return max(clipped_mw, -max_charge_mw)

        return 0.0

    def _next_soc(self, soc: float, battery_mw: float) -> float:
        if battery_mw > 0.0:
            next_soc = soc - (battery_mw * self.dt_hours) / (
                self.battery_capacity_mwh * self.discharge_efficiency
            )
        elif battery_mw < 0.0:
            next_soc = (
                soc
                - (battery_mw * self.charge_efficiency * self.dt_hours)
                / self.battery_capacity_mwh
            )
        else:
            next_soc = soc

        return self._clip(next_soc, 0.0, 1.0)

    @staticmethod
    def _state_value(
        state: dict[str, Any],
        scalar_key: str,
        profile_key: str,
        t: int,
        default: float,
    ) -> float:
        if scalar_key in state:
            return float(state[scalar_key])

        profile = state.get(profile_key)
        if profile is None:
            return float(default)

        if t >= len(profile):
            raise IndexError(f"{profile_key} does not contain timestep {t}")

        return float(profile[t])

    @staticmethod
    def _clip(value: float, lower: float, upper: float) -> float:
        return max(lower, min(upper, value))
