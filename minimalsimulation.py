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


def instantiate_dataclass(cls, data_dict):
    field_names = set(cls.__dataclass_fields__.keys()) if hasattr(cls, '__dataclass_fields__') else set()
    init_args = {}
    extra_args = {}
    for k, v in data_dict.items():
        if k in field_names:
            init_args[k] = v
        else:
            extra_args[k] = v
    obj = cls(**init_args)
    for k, v in extra_args.items():
        obj[k] = v
    return obj


def get_state_class(key: str, field_type: Any) -> Optional[type]:
    if isinstance(field_type, type) and issubclass(field_type, DictDataclass):
        return field_type
    if hasattr(field_type, '__origin__'):
        from typing import Union
        import types
        if field_type.__origin__ is Union or (hasattr(types, 'UnionType') and field_type.__origin__ is types.UnionType):
            for arg in field_type.__args__:
                if isinstance(arg, type) and issubclass(arg, DictDataclass):
                    return arg
    key_lower = key.lower()
    if key_lower == 'satellites' or key_lower.endswith('satellites'):
        return SatellitesState
    elif key_lower == 'observatories' or key_lower.endswith('observatories'):
        return ObservatoriesState
    elif key_lower == 'fixedpoints' or key_lower.endswith('fixedpoints'):
        return FixedPointsState
    elif key_lower == 'celestial':
        return CelestialState
    elif key_lower == 'counts':
        return CountsState
    elif key_lower == 'cadencestructure':
        return CadenceState
    elif key_lower == 'detector':
        from detector import DetectorArray  # lazy import avoids circular dependency
        return DetectorArray
    return None


class DictDataclass(dict):
    """
    A base class that allows standard dictionary-based classes decorated
    with @dataclass to support both attribute-style (dot notation) access 
    and dictionary subscript lookup/assignment.
    """
    def __getitem__(self, key):
        if hasattr(self, '__dataclass_fields__') and key in self.__dataclass_fields__:
            val = getattr(self, key)
            if val is None:
                raise KeyError(key)
            return val
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        field_type = None
        if hasattr(self, '__dataclass_fields__') and key in self.__dataclass_fields__:
            field_type = self.__dataclass_fields__[key].type
        cls = get_state_class(key, field_type)
        if cls is not None and isinstance(value, dict) and not isinstance(value, cls):
            value = instantiate_dataclass(cls, value)

        if hasattr(self, '__dataclass_fields__') and key in self.__dataclass_fields__:
            super().__setattr__(key, value)
        else:
            super().__setitem__(key, value)

    def __contains__(self, key):
        if hasattr(self, '__dataclass_fields__') and key in self.__dataclass_fields__:
            return getattr(self, key) is not None
        return super().__contains__(key)

    def get(self, key, default=None):
        if hasattr(self, '__dataclass_fields__') and key in self.__dataclass_fields__:
            val = getattr(self, key)
            if val is None:
                return default
            return val
        return super().get(key, default)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        field_type = None
        if hasattr(self, '__dataclass_fields__') and key in self.__dataclass_fields__:
            field_type = self.__dataclass_fields__[key].type
        cls = get_state_class(key, field_type)
        if cls is not None and isinstance(value, dict) and not isinstance(value, cls):
            value = instantiate_dataclass(cls, value)

        if hasattr(self, '__dataclass_fields__') and key in self.__dataclass_fields__:
            super().__setattr__(key, value)
        else:
            self[key] = value

    def __delitem__(self, key):
        if hasattr(self, '__dataclass_fields__') and key in self.__dataclass_fields__:
            raise KeyError(f"Cannot delete dataclass field '{key}'")
        super().__delitem__(key)

    def __iter__(self):
        fields = [f for f in self.__dataclass_fields__.keys() if getattr(self, f) is not None] if hasattr(self, '__dataclass_fields__') else []
        seen = set()
        for f in fields:
            seen.add(f)
            yield f
        for k in super().__iter__():
            if k not in seen:
                yield k

    def __len__(self):
        fields = {f for f in self.__dataclass_fields__.keys() if getattr(self, f) is not None} if hasattr(self, '__dataclass_fields__') else set()
        return len(fields | set(super().keys()))

    def keys(self):
        return list(self)

    def values(self):
        return [self[k] for k in self]

    def items(self):
        return [(k, self[k]) for k in self]


@dataclass
class CountsState(DictDataclass):
    celestial: int = 0
    satellites: int = 0
    observatories: int = 0
    fixedpoints: int = 0


@dataclass
class CelestialState(DictDataclass):
    position: np.ndarray = field(default_factory=lambda: np.zeros((2, 3)))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros((2, 3)))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros((2, 3)))


@dataclass
class SatellitesState(DictDataclass):
    position: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    orbital_elements: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    epochs: List[datetime] = field(default_factory=list)
    pointing: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    pointing_state: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))


@dataclass
class FixedPointsState(DictDataclass):
    position: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    exclusion: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    size: np.ndarray = field(default_factory=lambda: np.zeros((0,)))
    albedo: np.ndarray = field(default_factory=lambda: np.zeros((0,)))


@dataclass
class ObservatoriesState(DictDataclass):
    position: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))
    pointing: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))



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


@dataclass
class CadenceState(DictDataclass):
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
    cadenceList: List['CadenceGroup'] = field(default_factory=list)
    nextTime: Optional[datetime] = None
    nextGroup: int = 0


@dataclass
class SimulationState(DictDataclass):
    """
    SimulationState models the main simulation data structure.
    It inherits from DictDataclass to maintain 100% backward compatibility
    with dict-style lookup/subscription and isinstance(..., dict) checks,
    but supports attribute-style access for better structure.
    """
    start_time: datetime
    time: datetime
    delta_time: float = 60.0
    counts: CountsState = field(default_factory=CountsState)
    pointing_spheres: Dict[int, np.ndarray] = field(default_factory=dict)
    detector: Optional[Any] = None  # DetectorArray; typed Any to avoid circular import
    satellites: Optional[SatellitesState] = None
    observatories: Optional[ObservatoriesState] = None
    fixedpoints: Optional[FixedPointsState] = None
    celestial: Optional[CelestialState] = None
    cadenceStructure: Optional[CadenceState] = None



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

