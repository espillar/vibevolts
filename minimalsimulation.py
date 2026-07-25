import numpy as np
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from constants import *


def safe_normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Safely normalizes an array of vectors along the specified axis,
    preventing division by zero.
    """
    norms = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.where(norms == 0, 0.0, v / norms)


class DotDict(dict):
    """
    A clean dictionary subclass that supports dot-notation attribute access.
    Unifies attribute access and dictionary keys into a single storage mechanism.
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


class SchemaDict(DotDict):
    """
    A DotDict that ensures default attributes are always initialized
    while seamlessly accepting any extra user-defined keys.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class CountsState(SchemaDict):
    def __init__(self, **kwargs):
        defaults = {
            'celestial': 0,
            'satellites': 0,
            'observatories': 0,
            'fixedpoints': 0,
        }
        super().__init__(**{**defaults, **kwargs})


class CelestialState(SchemaDict):
    def __init__(self, **kwargs):
        defaults = {
            'position': np.zeros((2, 3)),
            'velocity': np.zeros((2, 3)),
            'acceleration': np.zeros((2, 3)),
        }
        super().__init__(**{**defaults, **kwargs})


class SatellitesState(SchemaDict):
    def __init__(self, **kwargs):
        defaults = {
            'position': np.zeros((0, 3)),
            'velocity': np.zeros((0, 3)),
            'acceleration': np.zeros((0, 3)),
            'orbital_elements': np.zeros((0, 6)),
            'epochs': [],
        }
        super().__init__(**{**defaults, **kwargs})


class FixedPointsState(SchemaDict):
    def __init__(self, **kwargs):
        defaults = {
            'position': np.zeros((0, 3)),
            'exclusion': np.zeros((0,)),
            'size': np.zeros((0,)),
            'albedo': np.zeros((0,)),
        }
        super().__init__(**{**defaults, **kwargs})


class ObservatoriesState(SchemaDict):
    def __init__(self, **kwargs):
        defaults = {
            'position': np.zeros((0, 3)),
            'velocity': np.zeros((0, 3)),
            'acceleration': np.zeros((0, 3)),
            'pointing': np.zeros((0, 3)),
        }
        super().__init__(**{**defaults, **kwargs})


@dataclass
class CadenceGroup:
    """
    Holds scheduling information for one group of detectors that share
    an identical integration time.

    Attributes:
        scanInterval: Integration period in seconds for this group.
        scanMask:     Boolean array (length = total detectors) selecting
                      the detectors that belong to this group.
        scanNext:     The datetime at which the next scan for this group
                      is due.  Initialised to sim_data.time so that every
                      group fires on the first call to nextIntegration.
    """
    scanInterval: float
    scanMask: np.ndarray
    scanNext: datetime


class CadenceState(SchemaDict):
    """
    Top-level cadence schedule stored in sim_data.cadenceStructure.

    Attributes:
        cadenceList: Ordered list of CadenceGroup objects, one per unique
                     integration time found in sim_data.detector.
        nextTime:    The datetime of the earliest upcoming scan across all
                     groups.
        nextGroup:   Index into cadenceList identifying which group fires
                     next.
    """
    def __init__(self, **kwargs):
        defaults = {
            'cadenceList': [],
            'nextTime': None,
            'nextGroup': 0,
        }
        super().__init__(**{**defaults, **kwargs})


class SimulationState(SchemaDict):
    """
    SimulationState models the main simulation data structure.
    It inherits from SchemaDict to maintain 100% backward compatibility
    with dict-style lookup/subscription and isinstance(..., dict) checks,
    but supports attribute-style access for better structure.
    """
    _FIELD_SCHEMAS = {
        'counts': CountsState,
        'celestial': CelestialState,
        'satellites': SatellitesState,
        'fixedpoints': FixedPointsState,
        'observatories': ObservatoriesState,
        'cadenceStructure': CadenceState,
    }

    def __init__(self, **kwargs):
        if 'start_time' not in kwargs:
            raise TypeError("SimulationState requires a 'start_time' parameter.")
        start = kwargs['start_time']
        
        defaults = {
            'time': start,
            'delta_time': 60.0,
            'counts': CountsState(),
            'pointing_spheres': {},
            'detector': None,
            'satellites': None,
            'observatories': None,
            'fixedpoints': None,
            'celestial': None,
            'cadenceStructure': None,
        }
        super().__init__(**{**defaults, **kwargs})

    def __setitem__(self, key, value):
        if key == 'detector':
            from detector import DetectorArray
            cls = DetectorArray
        else:
            cls = self._FIELD_SCHEMAS.get(key)

        if cls is not None and isinstance(value, dict) and not isinstance(value, cls):
            value = cls(**value)
        super().__setitem__(key, value)


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
