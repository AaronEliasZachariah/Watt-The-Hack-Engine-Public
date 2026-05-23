from dataclasses import dataclass
from typing import Any

from watt_the_hack.engine.base_engine import SimulationEngine


@dataclass(slots=True)
class SimpleEnergyEngine(SimulationEngine):
    """MVP energy-system engine with battery and grid-import constraints."""

    battery_capacity_kwh: float = 20.0
    max_inverter_kw: float = 10.0
    grid_max_import_kw: float = 25.0
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    dt_hours: float = 0.25

    def step(self, state: dict, action: dict) -> tuple[dict, dict]:
        t = int(state.get("t", 0))
        soc = self._clip(float(state.get("soc", 0.5)), 0.0, 1.0)

        demand_kw = self._state_value(state, "demand", "demand_profile", t, 0.0)
        solar_kw = self._state_value(state, "solar", "solar_profile", t, 0.0)
        price = self._state_value(state, "price", "price_profile", t, 0.0)

        requested_battery_kw = float(action.get("battery_flow_kw", 0.0))
        battery_kw = self._feasible_battery_power(requested_battery_kw, soc)
        next_soc = self._next_soc(soc, battery_kw)

        grid_power_kw = demand_kw - solar_kw - battery_kw
        unmet_demand_kw = 0.0
        if grid_power_kw > self.grid_max_import_kw:
            grid_power_kw = self.grid_max_import_kw
            unmet_demand_kw = demand_kw - (
                solar_kw + battery_kw + self.grid_max_import_kw
            )

        outputs = {
            "grid_power": float(grid_power_kw),
            "battery_flow_kw": float(battery_kw),
            "solar_used": float(max(0.0, min(solar_kw, demand_kw - unmet_demand_kw))),
            "unmet_demand": float(max(0.0, unmet_demand_kw)),
            "price": float(price),
        }

        new_state = dict(state)
        new_state.update(
            {
                "t": t + 1,
                "soc": float(next_soc),
                "last_demand": float(demand_kw),
                "last_solar": float(solar_kw),
                "last_price": float(price),
            }
        )

        return new_state, outputs

    def _feasible_battery_power(self, requested_kw: float, soc: float) -> float:
        clipped_kw = self._clip(
            requested_kw,
            -self.max_inverter_kw,
            self.max_inverter_kw,
        )

        if clipped_kw > 0.0:
            max_discharge_kw = (
                soc * self.battery_capacity_kwh * self.discharge_efficiency
            ) / self.dt_hours
            return min(clipped_kw, max_discharge_kw)

        if clipped_kw < 0.0:
            headroom_kwh = (1.0 - soc) * self.battery_capacity_kwh
            max_charge_kw = headroom_kwh / (self.charge_efficiency * self.dt_hours)
            return max(clipped_kw, -max_charge_kw)

        return 0.0

    def _next_soc(self, soc: float, battery_kw: float) -> float:
        if battery_kw > 0.0:
            next_soc = soc - (battery_kw * self.dt_hours) / (
                self.battery_capacity_kwh * self.discharge_efficiency
            )
        elif battery_kw < 0.0:
            next_soc = (
                soc
                - (battery_kw * self.charge_efficiency * self.dt_hours)
                / self.battery_capacity_kwh
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
