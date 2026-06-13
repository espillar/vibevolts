import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from constants import *


def safe_normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Safely normalizes an array of vectors along the specified axis,
    preventing division by zero.
    """
    norms = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.where(norms == 0, 0.0, v / norms)


class SimulationState(dict):
    """
    SimulationState models the main simulation data structure.
    It inherits from dict to maintain 100% backward compatibility
    with dict-style lookup/subscription and isinstance(..., dict) checks,
    but supports attribute-style access for better structure.
    """
    start_time: datetime
    time: datetime
    delta_time: float
    counts: Dict[str, int]
    pointing_spheres: Dict[int, np.ndarray]
    detector: Any
    satellites: Dict[str, Any]
    observatories: Dict[str, Any]
    fixedpoints: Dict[str, Any]
    celestial: Dict[str, Any]
    cadenceStructure: Dict[str, Any]

    def __init__(self, start_time: datetime, time: datetime, delta_time: float = 60.0):
        super().__init__()
        self['start_time'] = start_time
        self['time'] = time
        self['delta_time'] = delta_time
        self['counts'] = {}
        self['pointing_spheres'] = {}

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'SimulationState' object has no attribute '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __delattr__(self, key: str) -> None:
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"'SimulationState' object has no attribute '{key}'")


def create_empty_simulation(start_time: datetime, delta_time: float = 60.0) -> SimulationState:
    """
    Initializes a minimal, empty data structure for a space simulation.

    Args:
        start_time: The starting time and date of the simulation. This must be a
                    timezone-aware datetime object set to UTC.
        delta_time: The time step for the simulation in seconds.

    Returns:
        A SimulationState object representing the basic simulation state, not yet filled except
        for date and time and counts and pointing spheres.
    """
    if not isinstance(start_time, datetime):
        raise TypeError("start_time must be a datetime object.")
    if start_time.tzinfo is None:
        raise ValueError("start_time must be timezone-aware. Please set tzinfo.")

    return SimulationState(
        start_time=start_time,
        time=start_time,
        delta_time=delta_time
    )

