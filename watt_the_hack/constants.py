"""Shared timing constants.

These two values are used across the backend (engine, runner, server,
metrics) and must agree.  Defined here to avoid duplication.
"""

DEFAULT_STEPS: int = 96
"""Number of 15-minute steps in a standard 24-hour simulation."""

DT_HOURS: float = 0.25
"""Duration of a single simulation timestep in hours (15 minutes)."""
