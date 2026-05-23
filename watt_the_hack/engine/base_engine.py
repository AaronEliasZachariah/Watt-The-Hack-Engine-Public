from abc import ABC, abstractmethod


class SimulationEngine(ABC):
    """Base interface for discrete-time simulation engines."""

    @abstractmethod
    def step(self, state: dict, action: dict) -> tuple[dict, dict]:
        """
        Input:
            state: current system state
            action: control inputs

        Returns:
            new_state: updated state
            outputs: observable outputs for this timestep
        """
        raise NotImplementedError
